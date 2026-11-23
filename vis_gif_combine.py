"""Generate a single video containing multiple Argoverse scenarios.

This script mirrors the behavior of `vis_gif.py` but concatenates the frames
from a set of scenarios into one video file. Each scenario segment is labeled
at the top of the frame. By default, it renders three predefined scenarios.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Final, Iterable, List, Optional, Sequence

import click
from joblib import Parallel, delayed
import numpy as np
import torch
from rich.progress import track

from av2.datasets.motion_forecasting.data_schema import TrackCategory

from av2.datasets.motion_forecasting import scenario_serialization
from av2.datasets.motion_forecasting.viz.scenario_visualization import (
    _OBS_DURATION_TIMESTEPS,
    _PLOT_BOUNDS_BUFFER_M,
    _PRED_DURATION_TIMESTEPS,
    _plot_actor_tracks,
    _plot_static_map_elements,
)
from av2.map.map_api import ArgoverseStaticMap
import cv2
from PIL import Image as img
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from src.model.trainer_forecast import Trainer
from src.datamodule.av2_dataset import Av2Dataset

_DEFAULT_N_JOBS: Final[int] = -2  # Use all but one CPUs
_DEFAULT_SCENE_IDS: Final[Sequence[str]] = (
    "6a28ba10-4ba7-4664-b85f-9adb89729225",
    "55c99f35-8a3f-4fe0-ac54-c78fd02e622f",
    "fb8a1e37-8e07-4e7c-99cb-414eb7b5dd58",
)

_FOCAL_AGENT_COLOR: Final[str] = "#ECA25B"  # orange
_FOCAL_AGENT_FUTURE_COLOR: Final[str] = "#FF6B6B"  # red
_AV_COLOR: Final[str] = "#007672"  # green
_DEFAULT_ACTOR_COLOR: Final[str] = "#D3E8EF"  # light gray
_PREDICTION_COLOR: Final[str] = "#1E88E5"  # blue


@dataclass
class PredictionContext:
    """Holds model and dataset context for prediction overlay."""

    model: Trainer
    dataset: Av2Dataset
    scenario_id_to_idx: Dict[str, int]
    device: torch.device


def _load_prediction_model(checkpoint_path: Path, device: torch.device) -> Trainer:
    """Load DeMo model from checkpoint and move to device."""
    from omegaconf import OmegaConf

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]
    model_cfg = (
        OmegaConf.to_container(hparams["model"], resolve=True)
        if hasattr(hparams["model"], "_metadata")
        else dict(hparams["model"])
    )

    if "type" not in model_cfg:
        model_cfg["type"] = "ModelForecast"

    hparams_dict = {
        "model": model_cfg,
        "pretrained_weights": hparams.get("pretrained_weights", None),
        "lr": hparams["lr"],
        "warmup_epochs": hparams["warmup_epochs"],
        "epochs": hparams["epochs"],
        "weight_decay": hparams["weight_decay"],
    }

    model = Trainer(**hparams_dict)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.to(device)
    return model


def _build_scenario_index_map(dataset: Av2Dataset) -> Dict[str, int]:
    """Map scenario_id -> dataset index for fast lookup."""
    scenario_id_to_idx: Dict[str, int] = {}
    for idx, file_path in enumerate(track(dataset.file_list, description="Indexing processed scenarios")):
        raw_data = torch.load(file_path, map_location="cpu")
        scenario_id = raw_data.get("scenario_id")
        if scenario_id is not None:
            scenario_id_to_idx[scenario_id] = idx
    return scenario_id_to_idx


def _predict_trajectories(model: Trainer, scene_data: dict, device: torch.device) -> np.ndarray:
    """Run prediction and return refined trajectories (y_new_hat or y_hat)."""
    with torch.no_grad():
        scene_data_device = {}
        for key, value in scene_data.items():
            if isinstance(value, torch.Tensor):
                scene_data_device[key] = value.to(device).unsqueeze(0)
            else:
                scene_data_device[key] = value

        scene_data_device["x_key_valid_mask"] = (
            scene_data["x_valid_mask"].any(-1).to(device).unsqueeze(0)
        )
        scene_data_device["lane_key_valid_mask"] = (
            scene_data["lane_valid_mask"].any(-1).to(device).unsqueeze(0)
        )

        out = model(scene_data_device)
        if "new_y_hat" in out and out["new_y_hat"] is not None:
            y_pred = out["new_y_hat"].detach().cpu().numpy()[0]
        else:
            y_pred = out["y_hat"].detach().cpu().numpy()[0]

        return y_pred


def _transform_to_global(predictions: np.ndarray, raw_data: dict, step: int = 49) -> np.ndarray:
    """Transform local predictions to global coordinates."""
    focal_idx = raw_data["focal_idx"]
    last_obs_global = raw_data["x_positions"][focal_idx, step].numpy()
    theta = raw_data["x_angles"][focal_idx, step].item()

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotate_mat = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    predictions_global = []
    for traj in predictions:
        traj_global = np.matmul(traj, rotate_mat.T) + last_obs_global
        predictions_global.append(traj_global)

    return np.array(predictions_global)


def _get_predictions_for_scenario(
    scenario_id: str,
    prediction_context: PredictionContext,
) -> Optional[np.ndarray]:
    """Return predicted trajectories in global coordinates for a scenario_id."""
    idx = prediction_context.scenario_id_to_idx.get(scenario_id)
    if idx is None:
        return None

    scene_data = prediction_context.dataset[idx]
    if isinstance(scene_data, list):
        scene_data = scene_data[-1]

    raw_data = torch.load(prediction_context.dataset.file_list[idx], map_location="cpu")
    predictions_local = _predict_trajectories(prediction_context.model, scene_data, prediction_context.device)
    predictions_global = _transform_to_global(predictions_local, raw_data)
    return predictions_global


def _plot_prediction_trajectories(
    predictions_global: np.ndarray,
    timestep: int,
    *,
    prediction_color: str = _PREDICTION_COLOR,
) -> None:
    """Plot predicted trajectories up to the current prediction timestep."""
    if predictions_global is None:
        return

    pred_step = max(0, timestep - _OBS_DURATION_TIMESTEPS)
    for traj in predictions_global:
        traj_slice = traj[: pred_step + 1]
        if traj_slice.shape[0] < 2:
            continue
        plt.plot(
            traj_slice[:, 0],
            traj_slice[:, 1],
            color=prediction_color,
            linewidth=2.5,
            alpha=0.85,
            zorder=10,
        )

        dx = traj_slice[-1, 0] - traj_slice[-2, 0]
        dy = traj_slice[-1, 1] - traj_slice[-2, 1]
        if np.hypot(dx, dy) > 0.05:
            plt.arrow(
                traj_slice[-2, 0],
                traj_slice[-2, 1],
                dx,
                dy,
                head_width=1.2,
                head_length=1.6,
                fc=prediction_color,
                ec=prediction_color,
                linewidth=1.6,
                alpha=0.85,
                zorder=11,
            )


def _generate_frames_for_scenario(
    *,
    scenario_path: Path,
    scenario_id: str,
    scenario_label: str,
    static_map_path: Path,
    prediction_context: Optional[PredictionContext],
    fig_size: tuple = (8, 8),
) -> List[img.Image]:
    """Render frames for a single scenario with an overlay label."""
    scenario = scenario_serialization.load_argoverse_scenario_parquet(scenario_path)
    static_map = ArgoverseStaticMap.from_json(static_map_path)
    predictions_global = None
    if prediction_context is not None:
        predictions_global = _get_predictions_for_scenario(scenario_id, prediction_context)

    frames: List[img.Image] = []
    plot_bounds = (0, 0, 0, 0)
    total_steps = _OBS_DURATION_TIMESTEPS + _PRED_DURATION_TIMESTEPS

    for timestep in range(total_steps):
        fig, ax = plt.subplots(figsize=fig_size)

        _plot_static_map_elements(static_map)
        cur_plot_bounds = _plot_actor_tracks(ax, scenario, timestep)
        if cur_plot_bounds:
            plot_bounds = cur_plot_bounds

        # Overlay focal agent history (orange) and ground truth future (red) incrementally as timesteps advance.
        focal_track = next((t for t in scenario.tracks if t.category == TrackCategory.FOCAL_TRACK), None)
        if focal_track is not None:
            timesteps = np.array([state.timestep for state in focal_track.object_states])
            positions = np.array([list(state.position) for state in focal_track.object_states])
            hist_mask = (timesteps <= timestep) & (timesteps < _OBS_DURATION_TIMESTEPS)
            fut_mask = (timesteps >= _OBS_DURATION_TIMESTEPS) & (timesteps <= timestep)

            if hist_mask.any():
                ax.plot(
                    positions[hist_mask, 0],
                    positions[hist_mask, 1],
                    color=_FOCAL_AGENT_COLOR,
                    linewidth=2.8,
                    alpha=0.9,
                    zorder=7,
                )

            if fut_mask.any():
                ax.plot(
                    positions[fut_mask, 0],
                    positions[fut_mask, 1],
                    color=_FOCAL_AGENT_FUTURE_COLOR,
                    linewidth=2.8,
                    alpha=0.9,
                    zorder=8,
                )

        if predictions_global is not None and timestep >= _OBS_DURATION_TIMESTEPS:
            _plot_prediction_trajectories(predictions_global, timestep)

        plt.xlim(plot_bounds[0] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[1] + _PLOT_BOUNDS_BUFFER_M)
        plt.ylim(plot_bounds[2] - _PLOT_BOUNDS_BUFFER_M, plot_bounds[3] + _PLOT_BOUNDS_BUFFER_M)
        plt.gca().set_aspect("equal", adjustable="box")
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        # Scene label displayed in a fixed position at the top center of the frame.
        fig.text(
            0.5,
            0.975,
            scenario_label,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )

        # Legend placed consistently in the lower area of the frame.
        legend_handles = [
            Rectangle((0, 0), 1.8, 0.8, color=_FOCAL_AGENT_COLOR, label="Target agent"),
            Rectangle((0, 0), 1.8, 0.8, color=_AV_COLOR, label="Ego agent"),
            Rectangle((0, 0), 1.8, 0.8, color=_DEFAULT_ACTOR_COLOR, label="Other agents"),
            Line2D([0], [0], color=_FOCAL_AGENT_COLOR, lw=3, label="Historical trajectory"),
            Line2D([0], [0], color=_FOCAL_AGENT_FUTURE_COLOR, lw=3, label="Ground truth"),
            Line2D([0], [0], color=_PREDICTION_COLOR, lw=3, label="Predicted trajectory"),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper right",
            bbox_to_anchor=(0.98, 0.98),
            ncol=1,
            frameon=True,
            fontsize=9,
            columnspacing=1.2,
            handlelength=1.8,
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close()
        buf.seek(0)
        frame = img.open(buf)
        frames.append(frame)

    return frames


def _collect_scenario_paths(
    argoverse_scenario_dir: Path, scene_ids: Iterable[str]
) -> List[Path]:
    """Return parquet paths for the requested scene_ids, preserving order."""
    paths: List[Path] = []
    for scene_id in scene_ids:
        matches = sorted(argoverse_scenario_dir.rglob(f"scenario_{scene_id}.parquet"))
        if not matches:
            raise ValueError(f"Scenario ID '{scene_id}' not found under {argoverse_scenario_dir}")
        paths.append(matches[0])
    return paths


def _write_video(frames: List[img.Image], output_path: Path, fps: int = 10) -> None:
    """Write frames to an mp4 file."""
    if not frames:
        raise ValueError("No frames to write")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_size = frames[0].size
    video = cv2.VideoWriter(str(output_path), fourcc, fps=fps, frameSize=frame_size)
    for frame in frames:
        frame_temp = frame.copy()
        video.write(cv2.cvtColor(np.array(frame_temp), cv2.COLOR_RGB2BGR))
    video.release()


def _write_gif(frames: List[img.Image], output_path: Path, fps: int = 10) -> None:
    """Write frames to a gif file."""
    if not frames:
        raise ValueError("No frames to write")

    duration_ms = int(1000 / max(1, fps))
    first, *rest = [frame.convert("P", palette=img.ADAPTIVE) for frame in frames]
    first.save(
        output_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def generate_combined_visualization(
    argoverse_scenario_dir: Path,
    output_path: Path,
    scene_ids: Sequence[str],
    prediction_context: Optional[PredictionContext] = None,
    *,
    debug: bool = False,
    gif: bool = False,
) -> None:
    """Generate a single video containing the requested scenarios in order."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_paths = _collect_scenario_paths(argoverse_scenario_dir, scene_ids)

    def render_one(idx: int, scenario_path: Path) -> List[img.Image]:
        scene_id = scene_ids[idx]
        scenario_label = f"Scene {idx + 1}"
        static_map_path = scenario_path.parents[0] / f"log_map_archive_{scene_id}.json"
        return _generate_frames_for_scenario(
            scenario_path=scenario_path,
            scenario_id=scene_id,
            scenario_label=scenario_label,
            static_map_path=static_map_path,
            prediction_context=prediction_context,
        )

    if debug:
        frame_blocks = [render_one(i, path) for i, path in enumerate(track(scenario_paths))]
    else:
        frame_blocks = Parallel(n_jobs=_DEFAULT_N_JOBS)(
            delayed(render_one)(i, path)
            for i, path in enumerate(track(scenario_paths))
        )

    combined_frames: List[img.Image] = []
    for frames in frame_blocks:
        combined_frames.extend(frames)

    if gif:
        if output_path.suffix.lower() != ".gif":
            output_path = output_path.with_suffix(".gif")
        _write_gif(combined_frames, output_path)
    else:
        if output_path.suffix.lower() != ".mp4":
            output_path = output_path.with_suffix(".mp4")
        _write_video(combined_frames, output_path)


@click.command(help="Generate a single video from multiple Argoverse scenarios.")
@click.option(
    "--argoverse-scenario-dir",
    required=True,
    help="Path to local directory where Argoverse scenarios are stored.",
    type=click.Path(exists=True),
)
@click.option(
    "--viz-output-path",
    required=True,
    help="Path to the combined output file to write (mp4 or gif).",
    type=click.Path(),
)
@click.option(
    "--scene-ids",
    default=",".join(_DEFAULT_SCENE_IDS),
    help="Comma-separated list of scenario IDs to combine (order preserved).",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Runs preprocessing in single-threaded mode when enabled.",
)
@click.option(
    "--checkpoint",
    default=None,
    help="Optional DeMo checkpoint path to overlay predicted trajectories.",
    type=click.Path(exists=True),
)
@click.option(
    "--data-root",
    default="data/DeMo_processed",
    help="Root directory of processed DeMo data.",
    type=click.Path(),
)
@click.option(
    "--split",
    default="val",
    help="Processed dataset split to use for predictions.",
    type=click.Choice(["train", "val", "test"], case_sensitive=False),
)
@click.option(
    "--device",
    default=None,
    help="Device for model inference (e.g., 'cuda' or 'cpu').",
)
@click.option(
    "--gif",
    is_flag=True,
    default=False,
    help="Write output as gif instead of mp4.",
)

def run_generate_combined_visualization(
    argoverse_scenario_dir: str,
    viz_output_path: str,
    scene_ids: str,
    debug: bool,
    checkpoint: Optional[str],
    data_root: str,
    split: str,
    device: Optional[str],
    gif: bool,
) -> None:
    """Click entry point for combined scenario visualization generation."""
    parsed_scene_ids = [s.strip() for s in scene_ids.split(",") if s.strip()]
    if not parsed_scene_ids:
        raise ValueError("At least one scene ID must be provided")

    prediction_context = None
    if checkpoint:
        device_obj = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        dataset = Av2Dataset(
            data_root=data_root,
            split=split.lower(),
            num_historical_steps=50,
            sequence_origins=[50],
            radius=150.0,
            train_mode="only_focal",
        )
        model = _load_prediction_model(Path(checkpoint), device_obj)
        scenario_id_to_idx = _build_scenario_index_map(dataset)
        prediction_context = PredictionContext(
            model=model,
            dataset=dataset,
            scenario_id_to_idx=scenario_id_to_idx,
            device=device_obj,
        )

    generate_combined_visualization(
        Path(argoverse_scenario_dir),
        Path(viz_output_path),
        parsed_scene_ids,
        prediction_context,
        debug=debug,
        gif=gif,
    )


if __name__ == "__main__":
    run_generate_combined_visualization()
