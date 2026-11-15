"""U-Net, SegGBC refinement, and feature-entropy weighting experiment."""

from __future__ import annotations

import copy
import csv
import json
import re
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


N_MASK_CHANNELS = 3
PCA_CHANNELS = 4
ECE_BINS = 15
UNET_MODEL_CONFIG = {"in_channels": 4, "out_channels": N_MASK_CHANNELS, "width": 8}
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
THRESHOLDS = tuple(float(t) for t in np.arange(0.25, 0.751, 0.05))


class BraTSSliceDataset(Dataset):
    def __init__(self, files: list[Path]) -> None:
        self.files = list(files)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        path = self.files[index]
        with h5py.File(path, "r") as h5_file:
            image = np.asarray(h5_file["image"][...], dtype=np.float32)
            mask = np.asarray(h5_file["mask"][...], dtype=np.float32)

        if image.ndim != 3 or image.shape[-1] != 4:
            raise ValueError(f"{path.name}: expected image shape (H, W, 4), got {image.shape}")
        if mask.shape != (*image.shape[:2], N_MASK_CHANNELS):
            raise ValueError(f"{path.name}: unexpected mask shape {mask.shape}")

        scale = np.max(np.abs(image), axis=(0, 1), keepdims=True)
        scale[scale < 1e-6] = 1.0
        image = np.clip(image / scale, -1.0, 1.0)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).permute(2, 0, 1)
        return image_tensor, mask_tensor, path.name


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet2D(nn.Module):
    """Small 4-input, 3-output U-Net that also exposes its bottleneck features."""

    def __init__(self, in_channels: int = 4, out_channels: int = 3, width: int = 8) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.enc1 = DoubleConv(in_channels, width)
        self.enc2 = DoubleConv(width, width * 2)
        self.enc3 = DoubleConv(width * 2, width * 4)
        self.enc4 = DoubleConv(width * 4, width * 8)
        self.bottleneck = DoubleConv(width * 8, width * 16)
        self.up4 = nn.ConvTranspose2d(width * 16, width * 8, 2, stride=2)
        self.dec4 = DoubleConv(width * 16, width * 8)
        self.up3 = nn.ConvTranspose2d(width * 8, width * 4, 2, stride=2)
        self.dec3 = DoubleConv(width * 8, width * 4)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.dec2 = DoubleConv(width * 4, width * 2)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.dec1 = DoubleConv(width * 2, width)
        self.output = nn.Conv2d(width, out_channels, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        bottleneck = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(bottleneck), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.output(d1), bottleneck


def split_by_volume(
    files: list[Path], seed: int
) -> tuple[list[Path], list[Path], list[Path]]:
    volume_pattern = re.compile(r"volume_(\d+)_slice_")
    volume_files: dict[str, list[Path]] = {}
    for path in files:
        match = volume_pattern.search(path.name)
        if match is None:
            raise ValueError(f"Unexpected slice filename: {path.name}")
        volume_files.setdefault(match.group(1), []).append(path)

    volume_ids = np.asarray(sorted(volume_files))
    rng = np.random.default_rng(seed)
    rng.shuffle(volume_ids)
    n_volumes = len(volume_ids)
    n_train = int(n_volumes * 0.70)
    n_val = int(n_volumes * 0.15)
    train_ids = set(volume_ids[:n_train])
    val_ids = set(volume_ids[n_train:n_train + n_val])
    test_ids = set(volume_ids[n_train + n_val:])

    train = [p for v in train_ids for p in volume_files[v]]
    val = [p for v in val_ids for p in volume_files[v]]
    test = [p for v in test_ids for p in volume_files[v]]
    for group in (train, val, test):
        group.sort()
    if train_ids & val_ids or train_ids & test_ids or val_ids & test_ids:
        raise RuntimeError("Volume split contains leakage.")
    return train, val, test


def make_loader(
    files: list[Path], batch_size: int, shuffle: bool
) -> DataLoader:
    return DataLoader(
        BraTSSliceDataset(files),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def save_model_checkpoint(
    model: UNet2D,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    *,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "model_config": UNET_MODEL_CONFIG,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            **metadata,
        },
        temporary_path,
    )
    temporary_path.replace(checkpoint_path)


def mask_positive_weights(files: list[Path], device: torch.device) -> torch.Tensor:
    positives = np.zeros(N_MASK_CHANNELS, dtype=np.float64)
    total_pixels = 0
    for path in tqdm(files, desc="Counting training mask pixels", unit="slice"):
        with h5py.File(path, "r") as h5_file:
            mask = np.asarray(h5_file["mask"][...]) > 0
        positives += mask.sum(axis=(0, 1))
        total_pixels += mask.shape[0] * mask.shape[1]

    negatives = total_pixels - positives
    weights = negatives / np.maximum(positives, 1.0)
    return torch.as_tensor(np.clip(weights, 1.0, 20.0), dtype=torch.float32, device=device)


def segmentation_loss(
    logits: torch.Tensor,
    masks: torch.Tensor,
    pos_weight: torch.Tensor,
    sample_class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        masks,
        pos_weight=pos_weight.view(1, -1, 1, 1),
        reduction="none",
    ).mean(dim=(2, 3))

    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * masks).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    per_sample_class_loss = bce + dice_loss

    if sample_class_weights is not None:
        per_sample_class_loss = per_sample_class_loss * sample_class_weights
    return per_sample_class_loss.mean()


def train_model(
    model: UNet2D,
    train_files: list[Path],
    val_files: list[Path],
    device: torch.device,
    epochs: int,
    batch_size: int,
    pos_weight: torch.Tensor,
    checkpoint_path: Path,
    sample_weights: dict[str, np.ndarray] | None = None,
    learning_rate: float = 1e-3,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> UNet2D:
    train_loader = make_loader(train_files, batch_size, shuffle=True)
    val_loader = make_loader(val_files, batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        for images, masks, names in tqdm(
            train_loader, desc=f"Training epoch {epoch + 1}/{epochs}", unit="batch"
        ):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            class_weights = None
            if sample_weights is not None:
                class_weights = torch.as_tensor(
                    np.stack([sample_weights[name] for name in names]),
                    dtype=torch.float32,
                    device=device,
                )

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(images)
            loss = segmentation_loss(logits, masks, pos_weight, class_weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.inference_mode():            
            for images, masks, _ in val_loader:
                logits, _ = model(images.to(device, non_blocking=True))
                val_loss += segmentation_loss(
                    logits, masks.to(device, non_blocking=True), pos_weight
                ).item()
                val_batches += 1

        train_loss /= max(train_batches, 1)
        val_loss /= max(val_batches, 1)
        print(
            f"Epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            save_model_checkpoint(
                model,
                checkpoint_path,
                checkpoint_metadata or {},
                best_val_loss=best_val_loss,
                best_epoch=epoch + 1,
            )

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    return model


def class_semantic_scales(
    features: torch.Tensor, masks: torch.Tensor
) -> np.ndarray:
    """Paper-inspired squared-singular-value entropy per class and slice."""
    resized_mask = F.adaptive_avg_pool2d(
        masks.unsqueeze(0), output_size=features.shape[-2:]
    )[0] > 0.5
    scales = np.full(N_MASK_CHANNELS, np.nan, dtype=np.float32)
    channels = features.shape[0]

    for class_index in range(N_MASK_CHANNELS):
        selected = resized_mask[class_index].flatten()
        if int(selected.sum()) < 2:
            continue
        matrix = features.float().flatten(1)[:, selected]
        matrix = matrix - matrix.mean(dim=1, keepdim=True)
        singular_values = torch.linalg.svdvals(matrix)
        energy = singular_values.square()
        energy_sum = energy.sum()
        rank = energy.numel()
        if rank <= 1 or energy_sum <= 1e-12:
            scales[class_index] = 0.0
            continue
        probabilities = energy / energy_sum
        probabilities = probabilities.clamp_min(1e-12)
        entropy = -(probabilities * probabilities.log()).sum()
        scales[class_index] = (entropy / np.log(rank)).item()

    return scales


def derive_uncertainty_weights(
    model: UNet2D,
    train_files: list[Path],
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Estimate training-only AUVs from class-conditioned U-Net bottleneck features."""
    model.eval()
    scales_by_name: dict[str, np.ndarray] = {}
    all_scales: list[list[float]] = [[] for _ in range(N_MASK_CHANNELS)]
    loader = make_loader(train_files, batch_size=1, shuffle=False)

    with torch.inference_mode():
        for images, masks, names in tqdm(
            loader, desc="Estimating training feature entropy", unit="slice"
        ):
            _, features = model(images.to(device))
            scales = class_semantic_scales(features[0], masks[0].to(device))
            scales_by_name[names[0]] = scales
            for class_index, value in enumerate(scales):
                if np.isfinite(value):
                    all_scales[class_index].append(float(value))

    # High AUV (low semantic scale) gets a bounded lower loss weight.
    result: dict[str, np.ndarray] = {}
    for name, scales in scales_by_name.items():
        weights = np.ones(N_MASK_CHANNELS, dtype=np.float32)
        for class_index, values in enumerate(all_scales):
            value = scales[class_index]
            if not np.isfinite(value) or not values:
                continue
            log_values = np.log(np.maximum(np.asarray(values), 1e-8))
            low, high = float(log_values.min()), float(log_values.max())
            if high - low <= 1e-8:
                auv = 0.0
            else:
                auv = 1.0 - (np.log(max(float(value), 1e-8)) - low) / (high - low)
            weights[class_index] = np.clip(1.0 - 0.75 * auv, 0.25, 1.0)
        result[name] = weights
    return result


def pca_four_channels(feature_tensor: torch.Tensor) -> np.ndarray:
    feature_map = feature_tensor.detach().float().cpu().numpy()
    channels, height, width = feature_map.shape
    vectors = feature_map.reshape(channels, -1).T.astype(np.float64)
    vectors -= vectors.mean(axis=0, keepdims=True)
    u, singular_values, _ = np.linalg.svd(vectors, full_matrices=False)
    reduced_channels = min(PCA_CHANNELS, u.shape[1])
    reduced = u[:, :reduced_channels] * singular_values[:reduced_channels]
    if reduced_channels < PCA_CHANNELS:
        reduced = np.pad(reduced, ((0, 0), (0, PCA_CHANNELS - reduced_channels)))
    return reduced.reshape(height, width, PCA_CHANNELS).astype(np.float32)


def gbc_refined_probabilities(
    feature_tensor: torch.Tensor,
    probabilities: torch.Tensor,
    segmenter_class: type,
) -> torch.Tensor:
    """Average U-Net probabilities inside GBC clusters and resize to input resolution."""
    latent_features = pca_four_channels(feature_tensor)
    cluster_map = segmenter_class().fit(
        latent_features, spatial_weight=0.1, show_progress=False
    )
    cluster_ids = np.unique(cluster_map)
    cluster_tensor = torch.as_tensor(
        cluster_map, dtype=torch.float32, device=probabilities.device
    )[None, None]
    coarse_probabilities = F.adaptive_avg_pool2d(
        probabilities.unsqueeze(0), output_size=cluster_map.shape
    )[0]
    coarse_refined = torch.zeros_like(coarse_probabilities)

    for cluster_id in cluster_ids:
        region = cluster_tensor[0, 0] == int(cluster_id)
        coarse_refined[:, region] = coarse_probabilities[:, region].mean(
            dim=1, keepdim=True
        )

    return F.interpolate(
        coarse_refined.unsqueeze(0),
        size=probabilities.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0]


def tune_on_validation(
    model: UNet2D,
    val_files: list[Path],
    device: torch.device,
    segmenter_class: type,
    max_slices: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose class thresholds and GBC blending strengths using validation only."""
    if len(val_files) > max_slices:
        rng = np.random.default_rng(2026)
        indices = np.sort(rng.choice(len(val_files), size=max_slices, replace=False))
        val_files = [val_files[int(i)] for i in indices]

    thresholds = np.asarray(THRESHOLDS, dtype=np.float32)
    alphas = np.asarray(ALPHAS, dtype=np.float32)
    intersections = np.zeros((len(alphas), len(thresholds), N_MASK_CHANNELS))
    predicted_counts = np.zeros_like(intersections)
    target_counts = np.zeros(N_MASK_CHANNELS)
    model.eval()

    for path in tqdm(val_files, desc="Tuning on validation volumes", unit="slice"):
        image_tensor, target_tensor, _ = BraTSSliceDataset([path])[0]
        with torch.inference_mode():
            logits, features = model(image_tensor.unsqueeze(0).to(device))
            probabilities = torch.sigmoid(logits[0])
            refined = gbc_refined_probabilities(
                features[0], probabilities, segmenter_class
            )

        target = target_tensor.numpy().astype(bool)
        target_counts += target.sum(axis=(1, 2))
        for alpha_index, alpha in enumerate(alphas):
            blended = ((1.0 - alpha) * probabilities + alpha * refined).cpu().numpy()
            for threshold_index, threshold in enumerate(thresholds):
                prediction = blended >= threshold
                intersections[alpha_index, threshold_index] += np.count_nonzero(
                    prediction & target, axis=(1, 2)
                )
                predicted_counts[alpha_index, threshold_index] += prediction.sum(
                    axis=(1, 2)
                )

    dice = (2.0 * intersections + 1e-8) / (
        predicted_counts + target_counts[None, None, :] + 1e-8
    )
    best_alpha = np.zeros(N_MASK_CHANNELS, dtype=np.float32)
    best_threshold = np.full(N_MASK_CHANNELS, 0.5, dtype=np.float32)
    best_dice = np.zeros(N_MASK_CHANNELS, dtype=np.float32)
    plain_unet_threshold = np.full(N_MASK_CHANNELS, 0.5, dtype=np.float32)
    for class_index in range(N_MASK_CHANNELS):
        alpha_index, threshold_index = np.unravel_index(
            np.argmax(dice[:, :, class_index]), dice[:, :, class_index].shape
        )
        best_alpha[class_index] = alphas[alpha_index]
        best_threshold[class_index] = thresholds[threshold_index]
        best_dice[class_index] = dice[alpha_index, threshold_index, class_index]
        plain_threshold_index = int(np.argmax(dice[0, :, class_index]))
        plain_unet_threshold[class_index] = thresholds[plain_threshold_index]
    return best_alpha, best_threshold, best_dice, plain_unet_threshold


def evaluate_on_test(
    baseline_model: UNet2D,
    weighted_model: UNet2D,
    test_files: list[Path],
    device: torch.device,
    segmenter_class: type,
    baseline_params: tuple[np.ndarray, np.ndarray, np.ndarray],
    weighted_params: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    model_specs = {
        "unet": (baseline_model, np.zeros(N_MASK_CHANNELS), baseline_params[2]),
        "unet_gbc": (baseline_model, baseline_params[0], baseline_params[1]),
        "unet_uncertainty": (
            weighted_model, np.zeros(N_MASK_CHANNELS), weighted_params[2]
        ),
        "unet_uncertainty_gbc": (
            weighted_model, weighted_params[0], weighted_params[1]
        ),
    }
    counts = {
        name: {
            "intersection": np.zeros(N_MASK_CHANNELS, dtype=np.int64),
            "predicted": np.zeros(N_MASK_CHANNELS, dtype=np.int64),
            "target": np.zeros(N_MASK_CHANNELS, dtype=np.int64),
            "pixel_count": np.zeros(N_MASK_CHANNELS, dtype=np.int64),
            "bce_sum": np.zeros(N_MASK_CHANNELS, dtype=np.float64),
            "brier_sum": np.zeros(N_MASK_CHANNELS, dtype=np.float64),
            "ece_bin_count": np.zeros((N_MASK_CHANNELS, ECE_BINS), dtype=np.int64),
            "ece_bin_probability_sum": np.zeros(
                (N_MASK_CHANNELS, ECE_BINS), dtype=np.float64
            ),
            "ece_bin_target_sum": np.zeros(
                (N_MASK_CHANNELS, ECE_BINS), dtype=np.float64
            ),
        }
        for name in model_specs
    }
    previews: list[dict[str, Any]] = []
    predicted_union_total = {name: 0 for name in model_specs}
    combined_intersection_total = {name: 0 for name in model_specs}
    total_pixels = 0
    target_union_total = 0
    for model, _, _ in model_specs.values():
        model.eval()

    for path in tqdm(test_files, desc="Final held-out test", unit="slice"):
        image_tensor, target_tensor, _ = BraTSSliceDataset([path])[0]
        target = target_tensor.numpy().astype(bool)
        total_pixels += target.shape[1] * target.shape[2]
        target_union_total += int(target.any(axis=0).sum())
        probabilities_by_model: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        for model, _, _ in model_specs.values():
            model_key = id(model)
            if model_key not in probabilities_by_model:
                with torch.inference_mode():
                    logits, features = model(image_tensor.unsqueeze(0).to(device))
                    probabilities = torch.sigmoid(logits[0])
                    refined = gbc_refined_probabilities(
                        features[0], probabilities, segmenter_class
                    )
                probabilities_by_model[model_key] = (probabilities, refined)

        predictions_for_preview: dict[str, np.ndarray] = {}
        for name, (model, alpha, threshold) in model_specs.items():
            probabilities, refined = probabilities_by_model[id(model)]
            alpha_tensor = torch.as_tensor(
                alpha, dtype=probabilities.dtype, device=probabilities.device
            )[:, None, None]
            blended = (1.0 - alpha_tensor) * probabilities + alpha_tensor * refined
            blended_np = blended.detach().cpu().numpy()
            prediction = (blended >= torch.as_tensor(
                threshold, dtype=blended.dtype, device=blended.device
            )[:, None, None]).cpu().numpy()
            predicted_union = prediction.any(axis=0)
            target_union = target.any(axis=0)
            predicted_union_total[name] += int(predicted_union.sum())
            combined_intersection_total[name] += int(
                np.count_nonzero(predicted_union & target_union)
            )

            counts[name]["intersection"] += np.count_nonzero(
                prediction & target, axis=(1, 2)
            )
            counts[name]["predicted"] += prediction.sum(axis=(1, 2))
            counts[name]["target"] += target.sum(axis=(1, 2))
            target_float = target.astype(np.float64)
            for channel in range(N_MASK_CHANNELS):
                channel_probabilities = np.clip(
                    blended_np[channel].reshape(-1), 1e-7, 1.0 - 1e-7
                )
                channel_targets = target_float[channel].reshape(-1)
                bin_ids = np.minimum(
                    (channel_probabilities * ECE_BINS).astype(np.int64),
                    ECE_BINS - 1,
                )
                counts[name]["pixel_count"][channel] += channel_targets.size
                counts[name]["bce_sum"][channel] += -np.sum(
                    channel_targets * np.log(channel_probabilities)
                    + (1.0 - channel_targets) * np.log1p(-channel_probabilities)
                )
                counts[name]["brier_sum"][channel] += np.sum(
                    (channel_probabilities - channel_targets) ** 2
                )
                counts[name]["ece_bin_count"][channel] += np.bincount(
                    bin_ids, minlength=ECE_BINS
                )
                counts[name]["ece_bin_probability_sum"][channel] += np.bincount(
                    bin_ids,
                    weights=channel_probabilities,
                    minlength=ECE_BINS,
                )
                counts[name]["ece_bin_target_sum"][channel] += np.bincount(
                    bin_ids, weights=channel_targets, minlength=ECE_BINS
                )
            if name in ("unet", "unet_gbc", "unet_uncertainty", "unet_uncertainty_gbc"):
                predictions_for_preview[name] = prediction.any(axis=0)

        if target.any() and len(previews) < 4:
            previews.append({
                "name": path.name,
                "image": image_tensor[0].numpy().copy(),
                "target": target.any(axis=0),
                "predictions": predictions_for_preview,
            })

    results: dict[str, Any] = {}
    for name, values in counts.items():
        intersection = values["intersection"]
        predicted = values["predicted"]
        target = values["target"]
        dice = np.divide(
            2 * intersection,
            predicted + target,
            out=np.ones(N_MASK_CHANNELS, dtype=np.float64),
            where=(predicted + target) > 0,
        )
        iou = np.divide(
            intersection,
            predicted + target - intersection,
            out=np.ones(N_MASK_CHANNELS, dtype=np.float64),
            where=(predicted + target - intersection) > 0,
        )
        pixel_count = values["pixel_count"]
        bce_per_channel = values["bce_sum"] / pixel_count
        brier_per_channel = values["brier_sum"] / pixel_count
        ece_per_channel = np.zeros(N_MASK_CHANNELS, dtype=np.float64)
        for channel in range(N_MASK_CHANNELS):
            nonempty_bins = values["ece_bin_count"][channel] > 0
            bin_counts = values["ece_bin_count"][channel, nonempty_bins]
            mean_probability = (
                values["ece_bin_probability_sum"][channel, nonempty_bins] / bin_counts
            )
            mean_target = values["ece_bin_target_sum"][channel, nonempty_bins] / bin_counts
            ece_per_channel[channel] = np.sum(
                bin_counts / pixel_count[channel]
                * np.abs(mean_probability - mean_target)
            )

        calibration = {
            "ece_bins": ECE_BINS,
            "bce_per_channel": bce_per_channel.tolist(),
            "brier_per_channel": brier_per_channel.tolist(),
            "ece_per_channel": ece_per_channel.tolist(),
            "macro_bce": float(bce_per_channel.mean()),
            "macro_brier": float(brier_per_channel.mean()),
            "macro_ece": float(ece_per_channel.mean()),
        }
        combined_intersection = combined_intersection_total[name]
        combined_predicted = predicted_union_total[name]
        combined_target = target_union_total
        combined_union = combined_predicted + combined_target - combined_intersection
        results[name] = {
            "dice_per_channel": dice.tolist(),
            "iou_per_channel": iou.tolist(),
            "mean_channel_dice": float(dice.mean()),
            "combined_dice": (
                2 * combined_intersection / (combined_predicted + combined_target)
                if combined_predicted + combined_target else 1.0
            ),
            "combined_iou": (
                combined_intersection / combined_union if combined_union else 1.0
            ),
            "calibration": calibration,
            "predicted_positive_fraction_per_channel": (
                predicted / total_pixels
            ).tolist(),
            "reference_positive_fraction_per_channel": (
                target / total_pixels
            ).tolist(),
            "predicted_combined_positive_fraction": (
                predicted_union_total[name] / total_pixels
            ),
            "reference_combined_positive_fraction": target_union_total / total_pixels,
        }
    return {"metrics": results, "previews": previews}


def plot_test_previews(previews: list[dict[str, Any]]) -> None:
    if not previews:
        print("No positive-mask test slices available for preview.")
        return

    methods = ("unet", "unet_gbc", "unet_uncertainty", "unet_uncertainty_gbc")
    fig, axes = plt.subplots(
        len(previews), len(methods) + 2,
        figsize=(4 * (len(methods) + 2), 4 * len(previews)),
        squeeze=False,
    )
    for row, sample in enumerate(previews):
        image = sample["image"]
        low, high = np.percentile(image, [1, 99])
        axes[row, 0].imshow(image, cmap="gray", vmin=low, vmax=high)
        axes[row, 0].set_title(f"{sample['name']} — MRI channel 0")
        axes[row, 1].imshow(sample["target"], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Combined reference mask")
        for col, method in enumerate(methods, start=2):
            axes[row, col].imshow(image, cmap="gray", vmin=low, vmax=high)
            axes[row, col].imshow(
                np.ma.masked_where(~sample["predictions"][method],
                                   sample["predictions"][method]),
                cmap="Reds", alpha=0.55, vmin=0, vmax=1,
            )
            axes[row, col].set_title(method)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    plt.show()


def run_experiment(
    data_dir: str | Path,
    segmenter_class: type,
    *,
    seed: int = 42,
    batch_size: int = 4,
    baseline_epochs: int = 5,
    uncertainty_epochs: int = 2,
    tuning_slices: int = 250,
    checkpoint_dir: str | Path = "checkpoints",
) -> dict[str, Any]:
    """Train and evaluate the four requested variants using volume-separated data."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("volume_*_slice_*.h5"))
    if not files:
        raise FileNotFoundError(f"No BraTS HDF5 slice files found in {data_dir}")

    train_files, val_files, test_files = split_by_volume(files, seed)
    checkpoint_dir = Path(checkpoint_dir)
    test_files_relative = [str(path.relative_to(data_dir)) for path in test_files]
    print(
        f"Split by volume: train={len(train_files)} slices, "
        f"validation={len(val_files)}, test={len(test_files)}"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    pos_weight = mask_positive_weights(train_files, device)
    baseline = UNet2D(**UNET_MODEL_CONFIG).to(device)
    baseline_checkpoint = checkpoint_dir / "unet_baseline_best.pth"
    baseline = train_model(
        baseline,
        train_files,
        val_files,
        device,
        baseline_epochs,
        batch_size,
        pos_weight,
        baseline_checkpoint,
        checkpoint_metadata={
            "variant": "unet_baseline",
            "data_dir": str(data_dir.resolve()),
            "test_files": test_files_relative,
            "split_seed": seed,
            "batch_size": batch_size,
            "epochs": baseline_epochs,
            "learning_rate": 1e-3,
            "positive_weights": pos_weight.detach().cpu().tolist(),
        },
    )

    uncertainty_weights = derive_uncertainty_weights(
        baseline, train_files, device
    )
    weighted = copy.deepcopy(baseline)
    uncertainty_checkpoint = checkpoint_dir / "unet_uncertainty_weighted_best.pth"
    weighted = train_model(
        weighted,
        train_files,
        val_files,
        device,
        uncertainty_epochs,
        batch_size,
        pos_weight,
        uncertainty_checkpoint,
        sample_weights=uncertainty_weights,
        learning_rate=1e-4,
        checkpoint_metadata={
            "variant": "unet_uncertainty_weighted",
            "data_dir": str(data_dir.resolve()),
            "test_files": test_files_relative,
            "split_seed": seed,
            "batch_size": batch_size,
            "epochs": uncertainty_epochs,
            "learning_rate": 1e-4,
            "positive_weights": pos_weight.detach().cpu().tolist(),
        },
    )
    print(f"Saved baseline checkpoint to {baseline_checkpoint.resolve()}")
    print(f"Saved uncertainty-weighted checkpoint to {uncertainty_checkpoint.resolve()}")

    (
        baseline_alpha,
        baseline_threshold,
        baseline_val_dice,
        baseline_unet_threshold,
    ) = tune_on_validation(
        baseline, val_files, device, segmenter_class, tuning_slices
    )
    (
        weighted_alpha,
        weighted_threshold,
        weighted_val_dice,
        weighted_unet_threshold,
    ) = tune_on_validation(
        weighted, val_files, device, segmenter_class, tuning_slices
    )
    print("Validation-selected U-Net-only thresholds:", baseline_unet_threshold)
    print("Validation-selected per-channel alpha (baseline):", baseline_alpha)
    print("Validation-selected per-channel thresholds (baseline):", baseline_threshold)
    print("Validation Dice at selected settings (baseline):", baseline_val_dice)
    print("Validation-selected uncertainty-U-Net-only thresholds:", weighted_unet_threshold)
    print("Validation-selected per-channel alpha (uncertainty):", weighted_alpha)
    print("Validation-selected per-channel thresholds (uncertainty):", weighted_threshold)
    print("Validation Dice at selected settings (uncertainty):", weighted_val_dice)

    outcome = evaluate_on_test(
        baseline,
        weighted,
        test_files,
        device,
        segmenter_class,
        (baseline_alpha, baseline_threshold, baseline_unet_threshold),
        (weighted_alpha, weighted_threshold, weighted_unet_threshold),
    )
    summary = {
        "train_slices": len(train_files),
        "validation_slices": len(val_files),
        "test_slices": len(test_files),
        "calibration_protocol": {
            "probabilities": "before thresholding, after configured GBC blending",
            "bce": "unweighted binary cross-entropy, clipped probabilities",
            "brier": "mean squared probability error",
            "ece_bins": ECE_BINS,
            "ece_binning": "equal-width bins over [0, 1]",
            "macro_metrics": "unweighted mean across the three mask channels",
        },
        "baseline_alpha": baseline_alpha.tolist(),
        "baseline_threshold": baseline_threshold.tolist(),
        "baseline_unet_threshold": baseline_unet_threshold.tolist(),
        "uncertainty_alpha": weighted_alpha.tolist(),
        "uncertainty_threshold": weighted_threshold.tolist(),
        "uncertainty_unet_threshold": weighted_unet_threshold.tolist(),
        "test_metrics": outcome["metrics"],
    }
    results_path = Path("unet_seggbc_experiment_results.json")
    results_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved metrics to {results_path.resolve()}")
    for method, metrics in outcome["metrics"].items():
        print(
            f"{method}: combined Dice={metrics['combined_dice']:.4f}, "
            f"combined IoU={metrics['combined_iou']:.4f}, "
            f"per-channel Dice={np.round(metrics['dice_per_channel'], 4)}"
        )
        calibration = metrics["calibration"]
        print(
            f"{method}: macro BCE={calibration['macro_bce']:.6f}, "
            f"macro Brier={calibration['macro_brier']:.6f}, "
            f"macro ECE={calibration['macro_ece']:.6f}"
        )
    plot_test_previews(outcome["previews"])
    return summary
