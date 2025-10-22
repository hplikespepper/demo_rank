"""
Visualization script for trajectory prediction using trained DeMo model.
可视化DeMo模型的轨迹预测结果
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# Import model and dataset
from src.model.trainer_forecast import Trainer
from src.datamodule.av2_dataset import Av2Dataset

# Import av2 visualization
from av2.datasets.motion_forecasting.viz.scenario_visualization import visualize_scenario
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet


def load_model_and_data(checkpoint_path, scene_idx=0):
    """Load trained model and specified validation scene.
    
    Args:
        checkpoint_path: Path to the model checkpoint file.
        scene_idx: Index of the validation scene to visualize (0-based).
    """
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create model from checkpoint
    from omegaconf import OmegaConf
    hparams = checkpoint['hyper_parameters']
    model_cfg = OmegaConf.to_container(hparams['model'], resolve=True) if hasattr(hparams['model'], '_metadata') else dict(hparams['model'])
    
    if 'type' not in model_cfg:
        model_cfg['type'] = 'ModelForecast'
    
    hparams_dict = {
        'model': model_cfg,
        'pretrained_weights': hparams.get('pretrained_weights', None),
        'lr': hparams['lr'],
        'warmup_epochs': hparams['warmup_epochs'],
        'epochs': hparams['epochs'],
        'weight_decay': hparams['weight_decay']
    }
    
    model = Trainer(**hparams_dict)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.cuda()
    print("Model loaded successfully!")
    
    # Load validation dataset
    val_dataset = Av2Dataset(
        data_root="data/DeMo_processed",
        split="val",
        num_historical_steps=50,
        sequence_origins=[50],
        radius=150.0,
        train_mode='only_focal'
    )
    
    print(f"Total validation scenes: {len(val_dataset)}")
    
    # Check if scene index is valid
    if scene_idx < 0 or scene_idx >= len(val_dataset):
        raise ValueError(f"Scene index {scene_idx} is out of range. Valid range: 0-{len(val_dataset)-1}")
    
    # Get specified scene
    print(f"Loading scene at index: {scene_idx}")
    scene = val_dataset[scene_idx]
    if isinstance(scene, list):
        scene = scene[-1]
    
    # Get scenario_id from raw data
    raw_data = torch.load(val_dataset.file_list[scene_idx])
    scenario_id = raw_data['scenario_id']
    print(f"Processing scenario: {scenario_id}")
    
    return model, scene, scenario_id, raw_data


def predict_trajectories(model, scene_data):
    """Run prediction and return refined trajectories (y_new_hat)."""
    
    with torch.no_grad():
        # Add required key_valid_mask fields
        scene_data['x_key_valid_mask'] = scene_data['x_valid_mask'].any(-1)
        scene_data['lane_key_valid_mask'] = scene_data['lane_valid_mask'].any(-1)
        
        # Move data to GPU and add batch dimension
        for key in scene_data:
            if isinstance(scene_data[key], torch.Tensor):
                scene_data[key] = scene_data[key].cuda().unsqueeze(0)
        
        # Forward pass
        out = model(scene_data)
        
        # Get refined predictions (state query results)
        if 'new_y_hat' in out and out['new_y_hat'] is not None:
            y_pred = out['new_y_hat'].cpu().numpy()[0]  # [6, 60, 2]
            pi = out['new_pi'].cpu().numpy()[0]  # [6]
            print("Using refined predictions (y_new_hat)")
        else:
            y_pred = out['y_hat'].cpu().numpy()[0]  # [6, 60, 2]
            pi = out['pi'].cpu().numpy()[0]  # [6]
            print("Using standard predictions (y_hat)")
        
        print(f"Prediction shape: {y_pred.shape}")
        print(f"Probability scores: {pi}")
        
        return y_pred, pi


def transform_to_global(predictions, raw_data, step=49):
    """Transform local predictions to global coordinates.
    
    The predictions from the model are relative displacements from the last observed position
    in the local coordinate frame. We need to:
    1. Convert relative displacements to absolute positions in local frame
    2. Transform from local frame to global frame
    """
    
    focal_idx = raw_data['focal_idx']
    
    # Get the last observed position in global coordinates (at timestep 49)
    last_obs_global = raw_data['x_positions'][focal_idx, step].numpy()  # [2]
    theta = raw_data['x_angles'][focal_idx, step].item()
    
    # Inverse rotation matrix (from local to global)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotate_mat = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    
    # Transform each trajectory
    predictions_global = []
    for traj in predictions:
        # traj is relative displacement from last observed position
        # Transform relative displacements to global coordinates
        traj_global = np.matmul(traj, rotate_mat.T) + last_obs_global
        predictions_global.append(traj_global)
    
    predictions_global = np.array(predictions_global)  # [6, 60, 2]
    print(f"Transformed to global coordinates: {predictions_global.shape}")
    
    return predictions_global


def visualize_with_av2(scenario_id, predictions_global, probabilities, save_dir='./viz_traj'):
    """Visualize predictions using official av2 visualization style."""
    from av2.map.map_api import ArgoverseStaticMap
    from av2.datasets.motion_forecasting.data_schema import ObjectType, TrackCategory
    from matplotlib.patches import Rectangle
    import math
    
    # Constants from vis_argo2_static.py
    _OBS_DURATION_TIMESTEPS = 50
    _PRED_DURATION_TIMESTEPS = 60
    _ESTIMATED_VEHICLE_LENGTH_M = 4.0
    _ESTIMATED_VEHICLE_WIDTH_M = 2.0
    _ESTIMATED_CYCLIST_LENGTH_M = 2.0
    _ESTIMATED_CYCLIST_WIDTH_M = 0.7
    _PLOT_BOUNDS_BUFFER_M = 30.0
    
    _DRIVABLE_AREA_COLOR = "#7A7A7A"
    _LANE_SEGMENT_COLOR = "#E0E0E0"
    _DEFAULT_ACTOR_COLOR = "#D3E8EF"
    _FOCAL_AGENT_COLOR = "#ECA25B"
    _FOCAL_AGENT_FUTURE_COLOR = "#FF6B6B"
    _AV_COLOR = "#007672"
    _BOUNDING_BOX_ZORDER = 100
    
    _STATIC_OBJECT_TYPES = {
        ObjectType.STATIC,
        ObjectType.BACKGROUND,
        ObjectType.CONSTRUCTION,
        ObjectType.RIDERLESS_BICYCLE,
    }
    
    # Find original scenario data
    scenario_dir = Path("../argo2data/val") / scenario_id
    scenario_file = scenario_dir / f"scenario_{scenario_id}.parquet"
    map_file = scenario_dir / f"log_map_archive_{scenario_id}.json"
    
    if not scenario_file.exists():
        print(f"Error: Scenario file not found at {scenario_file}")
        return
    
    print(f"Loading scenario from: {scenario_file}")
    
    # Load scenario and static map
    scenario = load_argoverse_scenario_parquet(scenario_file)
    static_map = ArgoverseStaticMap.from_json(map_file)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 16))
    
    # Plot drivable areas (official style)
    for drivable_area in static_map.vector_drivable_areas.values():
        plt.fill(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], 
                color=_DRIVABLE_AREA_COLOR, alpha=0.5, zorder=1)
    
    # Plot lane segments (official style)
    for lane_segment in static_map.vector_lane_segments.values():
        plt.plot(lane_segment.left_lane_boundary.xyz[:, 0], 
                lane_segment.left_lane_boundary.xyz[:, 1],
                color=_LANE_SEGMENT_COLOR, linewidth=0.5, zorder=2)
        plt.plot(lane_segment.right_lane_boundary.xyz[:, 0], 
                lane_segment.right_lane_boundary.xyz[:, 1],
                color=_LANE_SEGMENT_COLOR, linewidth=0.5, zorder=2)
    
    # Plot agent tracks with official style
    track_bounds = None
    for track in scenario.tracks:
        # Get all valid timesteps
        actor_timesteps = np.array([state.timestep for state in track.object_states])
        if len(actor_timesteps) < 1:
            continue
        
        # Get full trajectory and heading
        actor_trajectory = np.array([list(state.position) for state in track.object_states])
        actor_headings = np.array([state.heading for state in track.object_states])
        
        # Split into history and future
        history_mask = actor_timesteps < _OBS_DURATION_TIMESTEPS
        future_mask = actor_timesteps >= _OBS_DURATION_TIMESTEPS
        
        history_trajectory = actor_trajectory[history_mask]
        future_trajectory = actor_trajectory[future_mask]
        
        # Determine track properties
        is_focal = track.category == TrackCategory.FOCAL_TRACK
        is_av = track.track_id == "AV"
        is_static = track.object_type in _STATIC_OBJECT_TYPES
        
        if is_static:
            continue
        
        if is_focal:
            # Calculate bounds from full trajectory
            x_min, x_max = actor_trajectory[:, 0].min(), actor_trajectory[:, 0].max()
            y_min, y_max = actor_trajectory[:, 1].min(), actor_trajectory[:, 1].max()
            track_bounds = (x_min, x_max, y_min, y_max)
            
            # Plot history in orange
            if len(history_trajectory) > 0:
                plt.plot(history_trajectory[:, 0], history_trajectory[:, 1], 
                        color=_FOCAL_AGENT_COLOR, linewidth=3, linestyle='-', 
                        label='Historical Trajectory', zorder=5)
            
            # Plot future ground truth in red (solid line)
            if len(future_trajectory) > 0:
                plt.plot(future_trajectory[:, 0], future_trajectory[:, 1], 
                        color=_FOCAL_AGENT_FUTURE_COLOR, linewidth=3, linestyle='-', 
                        label='Ground Truth', zorder=5)
            
            track_color = _FOCAL_AGENT_COLOR
        elif is_av:
            track_color = _AV_COLOR
            if len(history_trajectory) > 0:
                plt.plot(history_trajectory[:, 0], history_trajectory[:, 1], 
                        color=track_color, linewidth=1.5, alpha=0.6, zorder=3)
        else:
            track_color = _DEFAULT_ACTOR_COLOR
            if len(history_trajectory) > 0:
                plt.plot(history_trajectory[:, 0], history_trajectory[:, 1], 
                        color=track_color, linewidth=1, alpha=0.3, zorder=2)
        
        # Plot bounding box at last observed position
        if len(history_trajectory) > 0:
            last_pos = history_trajectory[-1]
            last_heading = actor_headings[history_mask][-1]
            
            if track.object_type == ObjectType.VEHICLE:
                bbox_length, bbox_width = _ESTIMATED_VEHICLE_LENGTH_M, _ESTIMATED_VEHICLE_WIDTH_M
            elif track.object_type in [ObjectType.CYCLIST, ObjectType.MOTORCYCLIST]:
                bbox_length, bbox_width = _ESTIMATED_CYCLIST_LENGTH_M, _ESTIMATED_CYCLIST_WIDTH_M
            else:
                plt.plot(last_pos[0], last_pos[1], "o", color=track_color, markersize=4, zorder=_BOUNDING_BOX_ZORDER)
                continue
            
            # Draw bounding box
            d = np.hypot(bbox_length, bbox_width)
            theta_2 = math.atan2(bbox_width, bbox_length)
            pivot_x = last_pos[0] - (d / 2) * math.cos(last_heading + theta_2)
            pivot_y = last_pos[1] - (d / 2) * math.sin(last_heading + theta_2)
            
            vehicle_bbox = Rectangle(
                (pivot_x, pivot_y), bbox_length, bbox_width, 
                np.degrees(last_heading), color=track_color, zorder=_BOUNDING_BOX_ZORDER
            )
            ax.add_patch(vehicle_bbox)
    
    # Add predicted trajectories with blue color and arrows
    prediction_color = '#1E88E5'  # Blue color for predictions
    prediction_plotted = False  # Flag to add legend only once
    
    for i, traj in enumerate(predictions_global):
        # Plot trajectory line
        line_label = 'Predicted Trajectories' if not prediction_plotted else None
        plt.plot(traj[:, 0], traj[:, 1], 
                color=prediction_color, linewidth=2.5, alpha=0.8,
                linestyle='-', label=line_label, zorder=10)
        
        # Add arrow at the end point
        if len(traj) >= 2:
            # Calculate direction from second-to-last to last point
            dx = traj[-1, 0] - traj[-2, 0]
            dy = traj[-1, 1] - traj[-2, 1]
            plt.arrow(traj[-2, 0], traj[-2, 1], dx, dy,
                     head_width=1.5, head_length=2.0, fc=prediction_color, 
                     ec=prediction_color, linewidth=2, alpha=0.8, zorder=11)
        
        prediction_plotted = True
    
    # Set map bounds
    if track_bounds:
        plt.xlim(track_bounds[0] - _PLOT_BOUNDS_BUFFER_M, track_bounds[1] + _PLOT_BOUNDS_BUFFER_M)
        plt.ylim(track_bounds[2] - _PLOT_BOUNDS_BUFFER_M, track_bounds[3] + _PLOT_BOUNDS_BUFFER_M)
    
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend(loc='upper right', fontsize=14, framealpha=0.95)
    plt.title(f'Multimodal Trajectory Predictions\nScenario: {scenario_id}', 
             fontsize=16, fontweight='bold')
    
    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save figure
    save_path = Path(save_dir) / f"vis_pred_{scenario_id}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()
    print("Done!")





def main():
    """Main function with command-line argument support."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Visualize DeMo trajectory predictions')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='outputs/lowrank_r32_simplify_seed42/20251013-223147/checkpoints/epoch=59.ckpt',
        help='Path to the model checkpoint file'
    )
    parser.add_argument(
        '--idx',
        type=int,
        default=0,
        help='Index of the validation scene to visualize (0-based, e.g., --idx 2 for 3rd scene)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default='data/DeMo_processed',
        help='Root directory of processed DeMo data'
    )
    parser.add_argument(
        '--argo2-data-dir',
        type=str,
        default='../argo2data/val',
        help='Directory containing original Argoverse 2 data'
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./viz_traj',
        help='Directory to save visualization images'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DeMo Trajectory Prediction Visualization")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Scene Index: {args.idx}")
    print(f"Data Root: {args.data_root}")
    print(f"Save Directory: {args.save_dir}")
    print("="*70)
    
    # Step 1: Load model and data
    print("\n[1/4] Loading model and validation data...")
    model, scene_data, scenario_id, raw_data = load_model_and_data(args.checkpoint, args.idx)
    
    # Step 2: Run prediction
    print("\n[2/4] Running model prediction...")
    predictions_local, probabilities = predict_trajectories(model, scene_data)
    
    # Step 3: Transform to global coordinates
    print("\n[3/4] Transforming to global coordinates...")
    predictions_global = transform_to_global(predictions_local, raw_data)
    
    # Step 4: Visualize with av2
    print("\n[4/4] Generating visualization with av2...")
    visualize_with_av2(scenario_id, predictions_global, probabilities, args.save_dir)
    
    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()
