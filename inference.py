"""Compare saved U-Net, SegGBC, and AUV-weighted segmentation predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from unet_seggbc_experiment import (
    N_MASK_CHANNELS,
    UNET_MODEL_CONFIG,
    UNet2D,
    pca_four_channels,
)


DEFAULT_DATA_DIR = Path("data/3/BraTS2020_training_data/content/data")
DEFAULT_CHECKPOINT_DIR = Path("checkpoints")
DEFAULT_RESULTS_PATH = Path("unet_seggbc_experiment_results.json")
MASK_COLORS = ("#ff3b30", "#34c759", "#1688ff")
ECE_BINS = 15
METHOD_LABELS = {
    "unet": "U-Net",
    "unet_seggbc": "U-Net + SegGBC",
    "unet_seggbc_auv": "U-Net + SegGBC + AUV",
}


class GranularBall:
    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.dim = self.points.shape[1]
        self.center = np.mean(self.points, axis=0)
        self.radius_max = 0.0
        self.radius_avg = 0.0
        self.radius_sta = 0.0
        self.mu = np.zeros(len(self.points), dtype=np.float64)
        self.nu = np.zeros(len(self.points), dtype=np.float64)
        self.pi = np.zeros(len(self.points), dtype=np.float64)
        self.scm = 1.0

    def compute_fuzzy_parameters(self, alpha: float = 0.7) -> None:
        count = len(self.points)
        if not count:
            return

        differences = self.points - self.center
        distance_squared = np.sum(differences**2, axis=1)
        maximum_distance = np.max(distance_squared) if count > 1 else 1e-5
        sigma_membership = max(alpha * np.sqrt(maximum_distance + 1e-8), 1e-5)
        sigma_nonmembership = max(
            (1.0 - alpha) * np.sqrt(maximum_distance + 1e-8), 1e-5
        )

        self.mu = np.exp(-distance_squared / sigma_membership**2)
        self.nu = 1.0 - np.exp(-distance_squared / sigma_nonmembership**2)
        self.pi = np.clip(1.0 - self.mu - self.nu, 0.0, 1.0)
        self.center = np.mean(self.points + self.pi[:, None], axis=0)

        differences = self.points - self.center
        distance_with_uncertainty = np.sum(differences**2, axis=1) + self.pi
        self.radius_max = (
            np.max(distance_with_uncertainty) if count > 1 else 1e-5
        )
        self.radius_avg = (
            np.mean(distance_with_uncertainty) if count > 1 else 1e-5
        )


class SegGBC:
    """SegGBC implementation used by the U-Net notebook."""

    def __init__(
        self, beta: float = 0.8, epsilon: float = 0.75, alpha: float = 0.7
    ) -> None:
        self.beta = beta
        self.epsilon = epsilon
        self.alpha = alpha
        self.balls: list[GranularBall] = []

    def fit(self, image: np.ndarray, spatial_weight: float = 0.1) -> np.ndarray:
        if image.ndim != 3 or image.shape[-1] != 4:
            raise ValueError(f"Expected image shape (H, W, 4); got {image.shape}")
        if spatial_weight <= 0:
            raise ValueError("spatial_weight must be greater than 0")

        height, width, channels = image.shape
        image = image.astype(np.float64)
        scales = np.max(np.abs(image), axis=(0, 1), keepdims=True)
        scales[scales == 0] = 1.0
        normalized_image = image / scales
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        features = np.column_stack(
            (
                normalized_image.reshape(-1, channels),
                xx.reshape(-1).astype(np.float64) / width * spatial_weight,
                yy.reshape(-1).astype(np.float64) / height * spatial_weight,
            )
        )

        initial_ball = GranularBall(features)
        initial_ball.compute_fuzzy_parameters(self.alpha)
        self.balls = [initial_ball]

        for _ in range(30):
            new_balls: list[GranularBall] = []
            split_occurred = False
            for ball in self.balls:
                ball.compute_fuzzy_parameters(self.alpha)
                ball.radius_sta = ball.radius_avg * 0.5

                dimension = ball.dim
                max_volume = max(ball.radius_max**dimension, 1e-8)
                average_volume = max(ball.radius_avg**dimension, 1e-8)
                stable_volume = max(ball.radius_sta**dimension, 1e-8)
                max_density = len(ball.points) / max_volume

                differences = ball.points - ball.center
                distance_with_uncertainty = (
                    np.sum(differences**2, axis=1) + ball.pi
                )
                average_density = (
                    np.sum(distance_with_uncertainty <= ball.radius_avg)
                    / average_volume
                )
                stable_density = (
                    np.sum(distance_with_uncertainty <= ball.radius_sta)
                    / stable_volume
                )
                numerator = min(average_density, max_density) * min(
                    stable_density, max_density
                )
                denominator = (
                    max(average_density, max_density)
                    * max(stable_density, max_density)
                    + 1e-8
                )
                ball.scm = numerator / denominator

                if ball.scm < self.beta and len(ball.points) > 20:
                    split_occurred = True
                    midpoint = len(ball.points) // 2
                    new_balls.extend(
                        (
                            GranularBall(ball.points[:midpoint]),
                            GranularBall(ball.points[midpoint:]),
                        )
                    )
                else:
                    new_balls.append(ball)

            self.balls = new_balls
            if not split_occurred:
                break

        self._merge_clusters()
        return self._generate_mask(height, width, spatial_weight)

    def _merge_clusters(self) -> None:
        if len(self.balls) <= 1:
            return

        centers = np.asarray([ball.center for ball in self.balls])
        tree = cKDTree(centers)
        average_radius = np.mean([ball.radius_max for ball in self.balls])
        neighbor_radius = max(average_radius * 3.0, 1.0)
        pairs = tree.query_pairs(r=neighbor_radius)
        merged_indices: set[int] = set()

        for first, second in pairs:
            if first in merged_indices or second in merged_indices:
                continue
            ball_a, ball_b = self.balls[first], self.balls[second]
            distance = np.linalg.norm(ball_a.center - ball_b.center)
            similarity = 1.0 / (1.0 + distance)
            if similarity > self.epsilon:
                merged_indices.add(second)
                ball_a.points = np.vstack((ball_a.points, ball_b.points))
                ball_a.compute_fuzzy_parameters(self.alpha)

        self.balls = [
            ball for index, ball in enumerate(self.balls)
            if index not in merged_indices
        ]

    def _generate_mask(
        self, height: int, width: int, spatial_weight: float
    ) -> np.ndarray:
        mask = np.zeros((height, width), dtype=np.int32)
        for cluster_id, ball in enumerate(self.balls):
            for point in ball.points:
                x = int(round(point[4] * width / spatial_weight))
                y = int(round(point[5] * height / spatial_weight))
                if 0 <= y < height and 0 <= x < width:
                    mask[y, x] = cluster_id
        return mask


def load_h5_slice(path: Path) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5_file:
        image = np.asarray(h5_file["image"][...], dtype=np.float32)
        mask = np.asarray(h5_file["mask"][...], dtype=np.float32)

    if image.ndim != 3 or image.shape[-1] != 4:
        raise ValueError(f"{path.name}: expected image shape (H, W, 4), got {image.shape}")
    if mask.shape != (*image.shape[:2], N_MASK_CHANNELS):
        raise ValueError(f"{path.name}: unexpected mask shape {mask.shape}")

    scale = np.max(np.abs(image), axis=(0, 1), keepdims=True)
    scale[scale < 1e-6] = 1.0
    normalized_image = np.clip(image / scale, -1.0, 1.0)
    image_tensor = torch.from_numpy(normalized_image).permute(2, 0, 1)
    binary_mask = (mask > 0).transpose(2, 0, 1)
    return image_tensor, normalized_image, binary_mask


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[UNet2D, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    config = checkpoint.get("model_config", UNET_MODEL_CONFIG)
    model = UNet2D(**config).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, checkpoint


def resolve_samples(
    data_dir: Path, checkpoint: dict[str, Any], count: int
) -> list[Path]:
    recorded_files = checkpoint.get("test_files", [])
    if recorded_files:
        files = []
        for recorded_path in recorded_files:
            path = Path(recorded_path)
            path = path if path.is_absolute() else data_dir / path
            if not path.is_file():
                raise FileNotFoundError(
                    f"Test slice recorded in checkpoint is missing: {path}"
                )
            files.append(path)
    else:
        files = sorted(data_dir.glob("volume_*_slice_*.h5"))

    if len(files) < count:
        raise ValueError(f"Requested {count} images, but only found {len(files)}.")
    return files[:count]


def thresholds_from_results(
    results_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not results_path.is_file():
        raise FileNotFoundError(f"Experiment results file not found: {results_path}")
    results = json.loads(results_path.read_text())
    return (
        np.asarray(results["baseline_alpha"], dtype=np.float32),
        np.asarray(results["baseline_threshold"], dtype=np.float32),
        np.asarray(results["baseline_unet_threshold"], dtype=np.float32),
        np.asarray(results["uncertainty_alpha"], dtype=np.float32),
        np.asarray(results["uncertainty_threshold"], dtype=np.float32),
    )


def new_metric_totals() -> dict[str, dict[str, Any]]:
    return {
        method: {
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
            "combined_intersection": 0,
            "combined_predicted": 0,
            "combined_target": 0,
        }
        for method in METHOD_LABELS
    }


def accumulate_metrics(
    totals: dict[str, dict[str, Any]],
    probabilities: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
) -> None:
    target_float = target.astype(np.float64)
    target_union = target.any(axis=0)

    for method in METHOD_LABELS:
        prediction = predictions[method]
        probability = np.clip(probabilities[method], 1e-7, 1.0 - 1e-7)
        values = totals[method]
        values["intersection"] += np.count_nonzero(
            prediction & target, axis=(1, 2)
        )
        values["predicted"] += prediction.sum(axis=(1, 2))
        values["target"] += target.sum(axis=(1, 2))

        predicted_union = prediction.any(axis=0)
        values["combined_intersection"] += int(
            np.count_nonzero(predicted_union & target_union)
        )
        values["combined_predicted"] += int(predicted_union.sum())
        values["combined_target"] += int(target_union.sum())

        for channel in range(N_MASK_CHANNELS):
            channel_probability = probability[channel].reshape(-1)
            channel_target = target_float[channel].reshape(-1)
            bin_ids = np.minimum(
                (channel_probability * ECE_BINS).astype(np.int64),
                ECE_BINS - 1,
            )
            values["pixel_count"][channel] += channel_target.size
            values["bce_sum"][channel] += -np.sum(
                channel_target * np.log(channel_probability)
                + (1.0 - channel_target) * np.log1p(-channel_probability)
            )
            values["brier_sum"][channel] += np.sum(
                (channel_probability - channel_target) ** 2
            )
            values["ece_bin_count"][channel] += np.bincount(
                bin_ids, minlength=ECE_BINS
            )
            values["ece_bin_probability_sum"][channel] += np.bincount(
                bin_ids,
                weights=channel_probability,
                minlength=ECE_BINS,
            )
            values["ece_bin_target_sum"][channel] += np.bincount(
                bin_ids,
                weights=channel_target,
                minlength=ECE_BINS,
            )


def finalize_metrics(totals: dict[str, dict[str, Any]]) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for method, values in totals.items():
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
        bce = values["bce_sum"] / pixel_count
        brier = values["brier_sum"] / pixel_count
        ece = np.zeros(N_MASK_CHANNELS, dtype=np.float64)
        for channel in range(N_MASK_CHANNELS):
            nonempty_bins = values["ece_bin_count"][channel] > 0
            bin_counts = values["ece_bin_count"][channel, nonempty_bins]
            mean_probability = (
                values["ece_bin_probability_sum"][channel, nonempty_bins]
                / bin_counts
            )
            mean_target = (
                values["ece_bin_target_sum"][channel, nonempty_bins] / bin_counts
            )
            ece[channel] = np.sum(
                bin_counts / pixel_count[channel]
                * np.abs(mean_probability - mean_target)
            )

        combined_union = (
            values["combined_predicted"]
            + values["combined_target"]
            - values["combined_intersection"]
        )
        combined_denominator = (
            values["combined_predicted"] + values["combined_target"]
        )
        results[method] = {
            "label": METHOD_LABELS[method],
            "dice_per_channel": dice.tolist(),
            "iou_per_channel": iou.tolist(),
            "mean_channel_dice": float(dice.mean()),
            "mean_channel_iou": float(iou.mean()),
            "combined_dice": (
                float(2 * values["combined_intersection"] / combined_denominator)
                if combined_denominator
                else 1.0
            ),
            "combined_iou": (
                float(values["combined_intersection"] / combined_union)
                if combined_union
                else 1.0
            ),
            "bce_per_channel": bce.tolist(),
            "macro_bce": float(bce.mean()),
            "brier_per_channel": brier.tolist(),
            "macro_brier": float(brier.mean()),
            "ece_bins": ECE_BINS,
            "ece_per_channel": ece.tolist(),
            "macro_ece": float(ece.mean()),
        }
    return results


def refine_probabilities(
    features: torch.Tensor, probabilities: torch.Tensor
) -> torch.Tensor:
    latent_features = pca_four_channels(features)
    cluster_map = SegGBC().fit(latent_features, spatial_weight=0.1)
    cluster_ids = np.unique(cluster_map)
    cluster_tensor = torch.as_tensor(
        cluster_map, dtype=torch.float32, device=probabilities.device
    )[None, None]
    coarse_probabilities = F.adaptive_avg_pool2d(
        probabilities.unsqueeze(0), output_size=cluster_map.shape
    )[0]
    refined = torch.zeros_like(coarse_probabilities)

    for cluster_id in cluster_ids:
        region = cluster_tensor[0, 0] == int(cluster_id)
        refined[:, region] = coarse_probabilities[:, region].mean(
            dim=1, keepdim=True
        )

    return F.interpolate(
        refined.unsqueeze(0),
        size=probabilities.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0]


def overlay_mask(axis: Any, image: np.ndarray, mask: np.ndarray, title: str) -> None:
    low, high = np.percentile(image, [1, 99])
    axis.imshow(image, cmap="gray", vmin=low, vmax=high)
    for channel, color in enumerate(MASK_COLORS):
        if mask[channel].any():
            axis.contourf(
                mask[channel],
                levels=(0.5, 1.5),
                colors=(color,),
                alpha=0.55,
            )
    axis.set_title(title)
    axis.axis("off")


def save_comparison(
    path: Path,
    image: np.ndarray,
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), squeeze=False)
    axes = axes[0]
    low, high = np.percentile(image[..., 0], [1, 99])
    axes[0].imshow(image[..., 0], cmap="gray", vmin=low, vmax=high)
    axes[0].set_title("MRI channel 0")
    axes[0].axis("off")
    overlay_mask(axes[1], image[..., 0], target, "Ground-truth mask")
    for axis, (title, prediction) in zip(axes[2:], predictions.items()):
        overlay_mask(axis, image[..., 0], prediction, title)
    fig.suptitle(f"{path.name} - mask colors: 0=red, 1=green, 2=blue")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_inference(
    data_dir: Path,
    checkpoint_dir: Path,
    results_path: Path,
    output_dir: Path,
    count: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline, baseline_checkpoint = load_model(
        checkpoint_dir / "unet_baseline_best.pth", device
    )
    auv_model, _ = load_model(
        checkpoint_dir / "unet_uncertainty_weighted_best.pth", device
    )
    files = resolve_samples(data_dir, baseline_checkpoint, count)
    (
        baseline_alpha,
        baseline_threshold,
        baseline_unet_threshold,
        uncertainty_alpha,
        uncertainty_threshold,
    ) = thresholds_from_results(results_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    metric_totals = new_metric_totals()
    print(f"Running {len(files)} slices on {device}; saving results to {output_dir}")
    for index, path in enumerate(files, start=1):
        image_tensor, image, target = load_h5_slice(path)
        model_outputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        with torch.inference_mode():
            for model in (baseline, auv_model):
                logits, features = model(image_tensor.unsqueeze(0).to(device))
                probabilities = torch.sigmoid(logits[0])
                refined = refine_probabilities(features[0], probabilities)
                model_outputs[id(model)] = probabilities, refined

        baseline_probabilities, baseline_refined = model_outputs[id(baseline)]
        auv_probabilities, auv_refined = model_outputs[id(auv_model)]

        baseline_alpha_tensor = torch.as_tensor(
            baseline_alpha, dtype=baseline_probabilities.dtype, device=device
        )[:, None, None]
        auv_alpha_tensor = torch.as_tensor(
            uncertainty_alpha, dtype=auv_probabilities.dtype, device=device
        )[:, None, None]
        plain_prediction = baseline_probabilities >= torch.as_tensor(
            baseline_unet_threshold, dtype=baseline_probabilities.dtype, device=device
        )[:, None, None]
        baseline_blended = (
            (1.0 - baseline_alpha_tensor) * baseline_probabilities
            + baseline_alpha_tensor * baseline_refined
        )
        auv_blended = (
            (1.0 - auv_alpha_tensor) * auv_probabilities
            + auv_alpha_tensor * auv_refined
        )
        baseline_threshold_tensor = torch.as_tensor(
            baseline_threshold,
            dtype=baseline_probabilities.dtype,
            device=device,
        )[:, None, None]
        gbc_prediction = (
            baseline_blended >= baseline_threshold_tensor
        )
        auv_prediction = (
            auv_blended
            >= torch.as_tensor(
                uncertainty_threshold,
                dtype=auv_probabilities.dtype,
                device=device,
            )[:, None, None]
        )
        predictions = {
            "unet": plain_prediction.cpu().numpy(),
            "unet_seggbc": gbc_prediction.cpu().numpy(),
            "unet_seggbc_auv": auv_prediction.cpu().numpy(),
        }
        probabilities = {
            "unet": baseline_probabilities.cpu().numpy(),
            "unet_seggbc": baseline_blended.cpu().numpy(),
            "unet_seggbc_auv": auv_blended.cpu().numpy(),
        }
        accumulate_metrics(metric_totals, probabilities, predictions, target)

        destination = output_dir / f"{path.stem}_comparison.png"
        save_comparison(
            destination,
            image,
            target,
            {
                METHOD_LABELS[method]: prediction
                for method, prediction in predictions.items()
            },
        )
        print(f"[{index}/{len(files)}] {path.name} -> {destination}")

    metrics_path = output_dir / "inference_metrics.json"
    report = {
        "images_evaluated": len(files),
        "image_files": [path.name for path in files],
        "metric_protocol": {
            "dice_iou": "global overlap counts across all evaluated images",
            "overlap_predictions": "per-channel validation-selected thresholds",
            "combined": "union of the three mask channels",
            "calibration_probabilities": (
                "sigmoid probabilities; GBC methods use the configured blend"
            ),
            "bce": "unweighted binary cross-entropy with probabilities clipped to [1e-7, 1-1e-7]",
            "brier": "mean squared probability error",
            "ece_bins": ECE_BINS,
            "ece_binning": "equal-width bins over [0, 1]",
            "macro_metrics": "unweighted mean across the three mask channels",
        },
        "metrics": finalize_metrics(metric_totals),
    }
    metrics_path.write_text(json.dumps(report, indent=2))
    print(f"Saved metrics to {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare U-Net, U-Net + SegGBC, and AUV-weighted predictions."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("inference_results"))
    parser.add_argument("--num-images", type=int, default=10)
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num-images must be greater than zero")
    run_inference(
        args.data_dir,
        args.checkpoint_dir,
        args.results,
        args.output_dir,
        args.num_images,
    )


if __name__ == "__main__":
    main()
