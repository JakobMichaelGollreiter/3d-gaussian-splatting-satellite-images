# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
# VGGT Bundle Adjustment and Visualization Module
#
# This module provides bundle adjustment optimization and 3D visualization capabilities for the VGGT repository.
# Key components:
# - bundle_adjustment(): Refines camera poses and 3D point positions using non-linear optimization
# - integrate_bundle_adjustment(): Integrates bundle adjustment into the VGGT pipeline
# - viser_wrapper(): Visualizes 3D reconstructions with interactive controls using viser
# - Sky segmentation functionality to filter out sky points
#
# The module supports:
# - Multi-stage optimization with confidence thresholds
# - Interactive visualization of point clouds and camera frustums
# - Point filtering based on confidence scores
# - Support for various compute devices (CUDA, MPS, CPU)
#
# Used in the VGGT repository for refining and visualizing 3D geometry from multi-view images
"""

import os
import glob
import time
import threading
import argparse
from typing import List, Optional

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2


try:
    import onnxruntime
except ImportError:
    print("onnxruntime not found. Sky segmentation may not work.")

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
import torch
import torch.optim as optim
import time
import numpy as np

def bundle_adjustment(
    world_points: torch.Tensor, 
    image_points: torch.Tensor,
    extrinsics: torch.Tensor, 
    intrinsics: torch.Tensor,
    conf_map: torch.Tensor,
    num_iterations: int = 50,
    learning_rate: float = 0.0051,
    confidence_threshold: float = 0.3,
    verbose: bool = True
):
    """
    Perform bundle adjustment to refine camera poses and 3D point positions.
    
    Args:
        world_points: Tensor of shape (S, H, W, 3) containing 3D point positions
        image_points: Tensor of shape (S, H, W, 2) containing 2D image coordinates
        extrinsics: Tensor of shape (S, 3, 4) containing camera extrinsics
        intrinsics: Tensor of shape (S, 3, 3) containing camera intrinsics
        conf_map: Tensor of shape (S, H, W) containing confidence values
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        confidence_threshold: Base threshold for filtering low-confidence points
        verbose: Whether to print progress information
        
    Returns:
        Tuple of refined (world_points, extrinsics)
    """
    device = world_points.device
    
    # Extract valid points based on confidence
    S, H, W, _ = world_points.shape
    
    # Start with higher threshold and progressively lower it
    start_threshold = confidence_threshold * 3
    
    # Huber loss for robustness to outliers
    def huber_loss(error, delta=1.0):
        abs_error = torch.abs(error)
        quadratic = torch.min(abs_error, torch.tensor(delta, device=error.device))
        linear = abs_error - quadratic
        return 0.5 * quadratic**2 + delta * linear
    
    # Orthogonality loss for rotation matrices
    def orthogonality_loss(extrinsics):
        loss = 0.0
        for i in range(extrinsics.shape[0]):
            R = extrinsics[i, :3, :3]
            # R·Rᵀ should be identity for a proper rotation matrix
            RRT = torch.matmul(R, R.transpose(0, 1))
            identity = torch.eye(3, device=R.device)
            loss += torch.sum((RRT - identity)**2)
        return loss / extrinsics.shape[0]
    
    # Reprojection function
    def reproject(points_3d, cam_indices, extrinsics_param, intrinsics_fixed):
        """Project 3D points into camera coordinates with compatible tensor shapes"""
        batch_size = points_3d.shape[0]
        
        # Initialize output tensors
        points_2d = torch.zeros((batch_size, 2), device=device)
        depths = torch.zeros((batch_size, 1), device=device)
        
        # Process each camera separately to avoid batching issues
        for cam_idx in torch.unique(cam_indices):
            # Get indices for this camera
            cam_mask = cam_indices == cam_idx
            cam_points = points_3d[cam_mask]
            
            # Get camera parameters
            R = extrinsics_param[cam_idx, :3, :3]  # 3x3 rotation matrix
            t = extrinsics_param[cam_idx, :3, 3]   # 3x1 translation vector
            K = intrinsics_fixed[cam_idx]          # 3x3 intrinsic matrix
            
            # Apply camera extrinsics (world to camera)
            cam_points_homo = torch.ones((cam_points.shape[0], 4), device=device)
            cam_points_homo[:, :3] = cam_points
            
            # Reshape for matrix multiplication
            ext_matrix = torch.zeros((4, 4), device=device)
            ext_matrix[:3, :3] = R
            ext_matrix[:3, 3] = t
            ext_matrix[3, 3] = 1.0
            
            # Transform points
            points_cam = torch.mm(cam_points_homo, ext_matrix.t())[:, :3]
            
            # Store depths
            depths[cam_mask] = points_cam[:, 2:3]
            
            # Apply projection (camera to image)
            points_norm = torch.zeros((points_cam.shape[0], 2), device=device)
            mask = points_cam[:, 2] > 1e-10
            points_norm[mask, 0] = points_cam[mask, 0] / points_cam[mask, 2]
            points_norm[mask, 1] = points_cam[mask, 1] / points_cam[mask, 2]
            
            # Apply camera matrix
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            
            points_2d_cam = torch.zeros((points_cam.shape[0], 2), device=device)
            points_2d_cam[:, 0] = fx * points_norm[:, 0] + cx
            points_2d_cam[:, 1] = fy * points_norm[:, 1] + cy
            
            # Store projected points
            points_2d[cam_mask] = points_2d_cam
        
        return points_2d, depths
    
    # Main optimization loop
    if verbose:
        print("Starting bundle adjustment...")
    
    start_time = time.time()
    losses = []
    
    # Perform bundle adjustment in stages with decreasing confidence thresholds
    stages = [
        {"threshold": start_threshold, "iterations": num_iterations // 3},
        {"threshold": start_threshold * 0.5, "iterations": num_iterations // 3},
        {"threshold": confidence_threshold, "iterations": num_iterations - 2*(num_iterations // 3)}
    ]
    
    # Function to collect valid points based on threshold
    def collect_valid_points(threshold):
        mask = conf_map > threshold
        valid_indices = []
        valid_image_points = []
        valid_world_points = []
        camera_indices = []
        
        for s in range(S):
            points_mask = mask[s]
            indices = torch.nonzero(points_mask, as_tuple=True)
            
            if len(indices[0]) > 0:
                h_indices, w_indices = indices
                
                # Get corresponding points
                img_pts = image_points[s, h_indices, w_indices]
                world_pts = world_points[s, h_indices, w_indices]
                
                # Record camera index
                cam_idx = torch.full((len(h_indices),), s, dtype=torch.long, device=device)
                
                valid_image_points.append(img_pts)
                valid_world_points.append(world_pts)
                camera_indices.append(cam_idx)
                
                # Store indices for updating
                valid_indices.append((s, h_indices, w_indices))
        
        if len(valid_world_points) == 0:
            return None, None, None, None
        
        return (
            torch.cat(valid_image_points, dim=0),
            torch.cat(valid_world_points, dim=0),
            torch.cat(camera_indices, dim=0),
            valid_indices
        )
    
    # Initialize optimization with high-confidence points
    valid_image_points, valid_world_points, camera_indices, valid_indices = collect_valid_points(stages[0]["threshold"])
    
    if valid_world_points is None:
        print("No valid points found with initial threshold. Skipping bundle adjustment.")
        return world_points, extrinsics
    
    # Create parameters to optimize
    opt_world_points = torch.nn.Parameter(valid_world_points.clone())
    opt_extrinsics = torch.nn.Parameter(extrinsics.clone())
    
    # Fixed intrinsics
    fixed_intrinsics = intrinsics.detach()
    
    # Setup optimizer
    optimizer = optim.Adam([opt_world_points, opt_extrinsics], lr=learning_rate)
    
    # Run optimization in stages
    iteration_counter = 0
    for stage_idx, stage in enumerate(stages):
        current_threshold = stage["threshold"]
        stage_iterations = stage["iterations"]
        
        if verbose:
            print(f"Stage {stage_idx+1}: Threshold = {current_threshold:.4f}, Iterations = {stage_iterations}")
        
        # Collect new points if not the first stage
        if stage_idx > 0:
            new_img_points, new_world_points, new_cam_indices, new_indices = collect_valid_points(current_threshold)
            
            if new_world_points is None:
                print(f"No valid points for threshold {current_threshold}. Using previous threshold.")
                continue
            
            # Update data for optimization
            valid_image_points = new_img_points
            valid_indices = new_indices
            camera_indices = new_cam_indices
            
            # Transfer optimized values
            with torch.no_grad():
                new_opt_world_points = torch.nn.Parameter(new_world_points.clone())
                opt_world_points = new_opt_world_points
                
                # Update optimizer
                optimizer = optim.Adam([opt_world_points, opt_extrinsics], lr=learning_rate)
        
        # Run iterations for this stage
        for i in range(stage_iterations):
            iteration_counter += 1
            
            optimizer.zero_grad()
            
            # Project 3D points
            projected_points, depths = reproject(
                opt_world_points, camera_indices, opt_extrinsics, fixed_intrinsics
            )
            
            # Calculate errors with robust loss
            reprojection_distances = torch.sqrt(torch.sum((projected_points - valid_image_points) ** 2, dim=1))
            reproj_error = torch.mean(huber_loss(reprojection_distances, delta=2.0))
            
            # Regularization terms
            depth_penalty = torch.sum(torch.relu(-depths) * 10.0)
            ortho_loss = orthogonality_loss(opt_extrinsics)
            
            # Combined loss
            loss = reproj_error + depth_penalty + 0.1 * ortho_loss
            losses.append(loss.item())
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            if verbose and (iteration_counter) % 10 == 0:
                print(f"Iteration {iteration_counter}/{num_iterations}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Reprojection: {reproj_error.item():.4f}, "
                      f"Ortho: {ortho_loss.item():.4f}")
    
    end_time = time.time()
    if verbose:
        print(f"Bundle adjustment completed in {end_time - start_time:.2f} seconds")
        print(f"Initial loss: {losses[0]:.4f}, Final loss: {losses[-1]:.4f}")
    
    # Update world points with optimized values
    refined_world_points = world_points.clone().to(device=opt_world_points.device, dtype=opt_world_points.dtype)

    # Safe update approach
    point_idx = 0
    for s, h_indices, w_indices in valid_indices:
        # Convert indices to CPU numpy for safer indexing
        h_indices_np = h_indices.cpu().numpy() if isinstance(h_indices, torch.Tensor) else h_indices
        w_indices_np = w_indices.cpu().numpy() if isinstance(w_indices, torch.Tensor) else w_indices
        
        for h_idx, w_idx in zip(h_indices_np, w_indices_np):
            if point_idx < opt_world_points.shape[0]:
                try:
                    # Use safer tensor assignment
                    with torch.no_grad():
                        refined_world_points[s, h_idx, w_idx] = opt_world_points[point_idx]
                    point_idx += 1
                except IndexError as e:
                    print(f"Index error at s={s}, h={h_idx}, w={w_idx}, point_idx={point_idx}")
                    # Skip this point
                    point_idx += 1
                    continue

    # Move tensors to CPU
    refined_world_points = refined_world_points.cpu()
    refined_extrinsics = opt_extrinsics.cpu()

    # Memory cleanup
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()

    return refined_world_points.detach(), refined_extrinsics.detach()

def integrate_bundle_adjustment(pred_dict, num_iterations=100, confidence_threshold=0.5):
    """
    Integrate bundle adjustment into the VGGT pipeline.
    
    Args:
        pred_dict: Dictionary containing the VGGT predictions
        num_iterations: Number of bundle adjustment iterations
        confidence_threshold: Confidence threshold for point filtering
        
    Returns:
        Updated prediction dictionary
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Using device {device} for bundle adjustment")
    
    # Check if inputs are numpy arrays or PyTorch tensors
    if isinstance(pred_dict["world_points"], np.ndarray):
        # Convert numpy arrays to PyTorch tensors
        world_points = torch.tensor(pred_dict["world_points"], device=device)
        extrinsics = torch.tensor(pred_dict["extrinsic"], device=device)
        intrinsics = torch.tensor(pred_dict["intrinsic"], device=device)
        conf_map = torch.tensor(pred_dict["world_points_conf"], device=device)
    else:
        # Already PyTorch tensors, just ensure they're on the right device
        world_points = pred_dict["world_points"].to(device)
        extrinsics = pred_dict["extrinsic"].to(device)
        intrinsics = pred_dict["intrinsic"].to(device)
        conf_map = pred_dict["world_points_conf"].to(device)
    
    # Create 2D image points from pixel coordinates
    S, H, W, _ = world_points.shape
    h_coords, w_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device),
                                        indexing='ij')
    image_points = torch.stack([w_coords, h_coords], dim=-1).unsqueeze(0).repeat(S, 1, 1, 1).float()
    
    print("Running bundle adjustment to refine camera poses and 3D points...")
    refined_world_points, refined_extrinsics = bundle_adjustment(
        world_points, image_points, extrinsics, intrinsics, conf_map,
        num_iterations=num_iterations,
        confidence_threshold=confidence_threshold
    )
    
    # Check if refined outputs are tensors and convert to numpy
    if isinstance(refined_world_points, torch.Tensor):
        pred_dict["world_points"] = refined_world_points.detach().cpu().numpy()
    else:
        pred_dict["world_points"] = refined_world_points
        
    if isinstance(refined_extrinsics, torch.Tensor):
        pred_dict["extrinsic"] = refined_extrinsics.detach().cpu().numpy()
    else:
        pred_dict["extrinsic"] = refined_extrinsics
    
    # Clean up memory
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()
    
    import gc
    gc.collect()
    
    print("Bundle adjustment completed successfully")
    return pred_dict


def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,  # represents percentage (e.g., 50 means filter lowest 50%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):

    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    world_points_map = pred_dict["world_points"]  # (S, H, W, 3)
    conf_map = pred_dict["world_points_conf"]  # (S, H, W)

    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)

    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if not use_point_map:
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox(
        "Show Cameras",
        initial_value=True,
    )

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent",
        min=0,
        max=100,
        step=0.1,
        initial_value=init_conf_threshold,
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames",
        options=["All"] + [str(i) for i in range(S)],
        initial_value="All",
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=1.0,
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server


# Helper functions for sky segmentation


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url("https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx")

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):  # Limit to the number of images in the batch
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match H×W if needed
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape S×H×W
    sky_mask_array = np.array(sky_mask_list)
    # Apply sky mask to confidence scores
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf


parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument(
    "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
)
parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
parser.add_argument(
    "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
)
parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")


def main():
    """
    Main function for the VGGT demo with viser for 3D visualization.

    This function:
    1. Loads the VGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser

    Command-line arguments:
    --image_folder: Path to folder containing input images
    --use_point_map: Use point map instead of depth-based points
    --background_mode: Run the viser server in background mode
    --port: Port number for the viser server
    --conf_threshold: Initial percentage of low-confidence points to filter out
    --mask_sky: Apply sky segmentation to filter out sky points
    """
    args = parser.parse_args()
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    device = "mps"
    print(f"Using device: {device}")

    print("Initializing and loading VGGT model...")
    # model = VGGT.from_pretrained("facebook/VGGT-1B")

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

    model.eval()
    model = model.to(device)

    # Use the provided image folder path
    print(f"Loading images from {args.image_folder}...")
    image_names = glob.glob(os.path.join(args.image_folder, "*"))
    print(f"Found {len(image_names)} images")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    print("Running inference...")    
    # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    dtype = torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # remove batch dimension and convert to numpy
   
    # Add bundle adjustment
    print("Using device mps for bundle adjustment")
    predictions = integrate_bundle_adjustment(
        predictions, 
        num_iterations=100,
        confidence_threshold=0.3
    )

    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    print("Starting viser visualization...")

    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("Visualization complete")



if __name__ == "__main__":
    main()
