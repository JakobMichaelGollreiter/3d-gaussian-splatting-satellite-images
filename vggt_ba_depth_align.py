import os
import glob
import time
import threading
import argparse
from typing import List

import numpy as np
import torch
from tqdm.auto import tqdm
import viser
import viser.transforms as viser_tf
import cv2
from scipy.optimize import least_squares

import kornia as K
from kornia.feature import LightGlue, DISK

from visual_util import segment_sky, download_file_from_url
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# disk + lightglue matcher to extract 2D-2D matches and initialize 3D points for BA
def extract_superglue_matches(images_tensor, depths, extrinsics, intrinsics, device):
    print("\nExtracting DISK features and matching with LightGlue...")
    
    # Initialize Kornia's native DISK and LightGlue models
    disk = DISK.from_pretrained("depth").eval().to(device)
    lightglue = LightGlue("disk").eval().to(device)
    
    S = images_tensor.shape[0]
    
    # DISK expects RGB images in the [0, 1] range. 
    # normalize safely just in case VGGT shifted the values.
    images_norm = images_tensor - images_tensor.min()
    images_norm = images_norm / (images_norm.max() + 1e-8)

    keypoints_list = []
    descriptors_list = []
    image_sizes = []

    with torch.no_grad():
        for i in range(S):
            img = images_norm[i:i+1]
            
            # Extract 2048 neural features per image
            features = disk(img, 2048, pad_if_not_divisible=True)[0]
            keypoints_list.append(features.keypoints)
            descriptors_list.append(features.descriptors)
            
            # LightGlue <- original image dimensions [width, height]
            h, w = img.shape[-2], img.shape[-1]
            image_sizes.append(torch.tensor([w, h], device=device).view(1, 2))

    # Build tracks
    camera_indices = []
    point_indices = []
    points_2d = []
    global_3d_points = []
    
    current_point_idx = 0

    print("Building track graph across images...")
    with torch.no_grad():
        for i in range(S - 1):
            j = i + 1
            
            # Format dictionaries for LightGlue 
            in_dict = {
                "image0": {
                    "keypoints": keypoints_list[i].unsqueeze(0),
                    "descriptors": descriptors_list[i].unsqueeze(0),
                    "image_size": image_sizes[i]
                },
                "image1": {
                    "keypoints": keypoints_list[j].unsqueeze(0),
                    "descriptors": descriptors_list[j].unsqueeze(0),
                    "image_size": image_sizes[j]
                }
            }
            
            # Perform the cross-attention matching
            matches = lightglue(in_dict)
            
            # Extract match indices 
            match_idx0 = matches['matches'][0][:, 0].cpu().numpy()
            match_idx1 = matches['matches'][0][:, 1].cpu().numpy()
            
            pts0 = keypoints_list[i][match_idx0].cpu().numpy()
            pts1 = keypoints_list[j][match_idx1].cpu().numpy()
            
            # Process matches to initialize 3D points based on VGGT's depth
            H, W = depths.shape[1], depths.shape[2]
            
            for m in range(len(pts0)):
                u_i, v_i = int(pts0[m][0]), int(pts0[m][1])
                u_j, v_j = int(pts1[m][0]), int(pts1[m][1])
                
                # Ensure it stays within image bounds
                if not (0 <= u_i < W and 0 <= v_i < H and 0 <= u_j < W and 0 <= v_j < H):
                    continue
                    
                depth_i = depths[i, v_i, u_i, 0]
                if depth_i <= 0.1: 
                    continue
                    
                # Unproject
                K_i = intrinsics[i]
                R_i = extrinsics[i, :3, :3]
                t_i = extrinsics[i, :3, 3]
                
                x_c = (u_i - K_i[0, 2]) * depth_i / K_i[0, 0]
                y_c = (v_i - K_i[1, 2]) * depth_i / K_i[1, 1]
                p_cam = np.array([x_c, y_c, depth_i])
                p_world = R_i.T @ (p_cam - t_i)
                
                global_3d_points.append(p_world)
                
                # Record dual observations
                camera_indices.extend([i, j])
                point_indices.extend([current_point_idx, current_point_idx])
                points_2d.extend([pts0[m], pts1[m]])
                
                current_point_idx += 1

    return (
        np.array(global_3d_points),
        np.array(camera_indices),
        np.array(point_indices),
        np.array(points_2d)
    )


# scipy ba levenverg-marquardt implementation 
def project_points(points_3d, camera_params, Ks, camera_indices, point_indices):
    """Projects 3D points into 2D cameras using given parameters."""
    cams = camera_params[camera_indices]
    pts = points_3d[point_indices]
    K_matrices = Ks[camera_indices]
    
    points_2d_pred = np.zeros((len(pts), 2))
    
    for i in range(len(pts)):
        R, _ = cv2.Rodrigues(cams[i, :3])
        t = cams[i, 3:6]
        K = K_matrices[i]
        
        # P_cam = R * P_world + t
        p_cam = R @ pts[i] + t
        
        # Perspective division
        z = p_cam[2] if p_cam[2] > 1e-5 else 1e-5
        x_norm = p_cam[0] / z
        y_norm = p_cam[1] / z
        
        # Apply intrinsics
        points_2d_pred[i, 0] = x_norm * K[0, 0] + K[0, 2]
        points_2d_pred[i, 1] = y_norm * K[1, 1] + K[1, 2]
        
    return points_2d_pred

def ba_objective_function(params, n_cameras, n_points, camera_indices, point_indices, points_2d, Ks):
    """The residual function for Levenberg-Marquardt optimizer."""
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    
    points_2d_pred = project_points(points_3d, camera_params, Ks, camera_indices, point_indices)
    
    # Return the residuals (flattened)
    return (points_2d_pred - points_2d).ravel()
def run_true_bundle_adjustment(images_tensor, depths, extrinsics, intrinsics, device):
    init_pts, cam_idx, pt_idx, target_2d = extract_superglue_matches(images_tensor, depths, extrinsics, intrinsics, device)
    
    if len(init_pts) < 10:
        print("Not enough robust matches. Skipping BA.")
        return extrinsics, depths  # Return original if it fails
        
    n_cameras = extrinsics.shape[0]
    n_points = len(init_pts)
    
    print(f"Running Levenberg-Marquardt Bundle Adjustment on {n_points} points...")
    
    camera_params = np.zeros((n_cameras, 6))
    for i in range(n_cameras):
        rvec, _ = cv2.Rodrigues(extrinsics[i, :3, :3])
        camera_params[i, :3] = rvec.squeeze()
        camera_params[i, 3:6] = extrinsics[i, :3, 3]
        
    x0 = np.hstack((camera_params.ravel(), init_pts.ravel()))
    
    res = least_squares(
        ba_objective_function, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
        args=(n_cameras, n_points, cam_idx, pt_idx, target_2d, intrinsics),
        loss='huber', f_scale=1.0, max_nfev=50
    )
    
    optimized_params = res.x
    opt_camera_params = optimized_params[:n_cameras * 6].reshape((n_cameras, 6))
    opt_points_3d = optimized_params[n_cameras * 6:].reshape((n_points, 3))
    
    optimized_extrinsics = np.zeros_like(extrinsics)
    for i in range(n_cameras):
        if i == 0:
            optimized_extrinsics[i] = extrinsics[0]
            continue
            
        R, _ = cv2.Rodrigues(opt_camera_params[i, :3])
        optimized_extrinsics[i, :3, :3] = R
        optimized_extrinsics[i, :3, 3] = opt_camera_params[i, 3:6]
        
    print("Bundle Adjustment Complete.")
    return optimized_extrinsics, opt_points_3d, cam_idx, pt_idx

def scale_align_depth_maps(depths, optimized_extrinsics, intrinsics, sparse_points_3d, camera_indices, point_indices):
    print("\nAligning dense depth maps to optimized sparse points (Fixing scale ambiguity)...")
    S, H, W, _ = depths.shape
    aligned_depths = depths.copy()

    for i in range(S):
        # Find which sparse points are visible in Camera i
        mask = (camera_indices == i)
        if not np.any(mask):
            continue
            
        pts_idx = point_indices[mask]
        pts_3d = sparse_points_3d[pts_idx]
        
        # Project 3D points into Camera i's view
        R = optimized_extrinsics[i, :3, :3]
        t = optimized_extrinsics[i, :3, 3]
        
        pts_cam = (R @ pts_3d.T).T + t
        z_expected = pts_cam[:, 2]
        
        # Filter points that fall behind the camera
        valid_z = z_expected > 0.1
        pts_cam = pts_cam[valid_z]
        z_expected = z_expected[valid_z]
        
        if len(z_expected) < 10:
            continue
            
        # Convert to pixel coordinates
        K = intrinsics[i]
        u = np.round(pts_cam[:, 0] * K[0, 0] / z_expected + K[0, 2]).astype(int)
        v = np.round(pts_cam[:, 1] * K[1, 1] / z_expected + K[1, 2]).astype(int)
        
        # Keep only pixels that fall inside the image frame
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z_expected = u[in_bounds], v[in_bounds], z_expected[in_bounds]
        
        if len(u) < 10:
            continue
            
        # Sample the raw VGGT depth map at those exact pixels
        z_vggt = depths[i, v, u, 0]
        
        valid_vggt = z_vggt > 0.1
        z_vggt = z_vggt[valid_vggt]
        z_expected = z_expected[valid_vggt]
        
        if len(z_vggt) < 10:
            continue
            
        # Calculate the Median Scale Multiplier
        scale = np.median(z_expected / z_vggt)
        
        # Apply the scale if it's reasonable
        if 0.1 < scale < 10.0:
            print(f"  -> Camera {i} Depth scaled by: {scale:.3f}")
            aligned_depths[i] = depths[i] * scale
        else:
            print(f"  -> Camera {i} Scale {scale:.3f} rejected, keeping original.")
            
    return aligned_depths

# ============================================================================
# VISUALIZATION WRAPPER
# ============================================================================
def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 50.0,
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: str = None,
):
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    images = pred_dict["images"]
    world_points_map = pred_dict["world_points"]
    conf_map = pred_dict["world_points_conf"]

    depth_map = pred_dict["depth"]
    depth_conf = pred_dict["depth_conf"]

    extrinsics_cam = pred_dict["extrinsic"]
    intrinsics_cam = pred_dict["intrinsic"]

    if not use_point_map:
        ext_t = torch.tensor(extrinsics_cam)
        int_t = torch.tensor(intrinsics_cam)
        depth_t = torch.tensor(depth_map)
        world_points_t = unproject_depth_map_to_point_map(depth_t, ext_t, int_t)
        
        if isinstance(world_points_t, torch.Tensor):
            world_points = world_points_t.cpu().numpy()
        else:
            world_points = world_points_t
            
        conf = depth_conf
    else:
        world_points = world_points_map
        conf = conf_map

    colors = images.transpose(0, 2, 3, 1)
    S, H, W, _ = world_points.shape

    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)
    cam_to_world = cam_to_world_mat[:, :3, :]

    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    frame_indices = np.repeat(np.arange(S), H * W)

    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)
    gui_points_conf = server.gui.add_slider("Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold)
    gui_frame_selector = server.gui.add_dropdown("Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All")

    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        for f in frames: f.remove()
        frames.clear()
        for fr in frustums: fr.remove()
        frustums.clear()

        img_ids = range(S)
        for img_id in img_ids:
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            img = images_[img_id]
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]
            fov = 2 * np.arctan2(h / 2, 1.1 * h)

            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum",
                fov=fov,
                aspect=w / h,
                scale=0.05,
                image=img,
                line_width=1.0,
            )
            frustums.append(frustum_cam)

    def update_point_cloud() -> None:
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)
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
    def _(_) -> None: update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None: update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        for f in frames: f.visible = gui_show_frames.value
        for fr in frustums: fr.visible = gui_show_frames.value

    visualize_frames(cam_to_world, images)

    if background_mode:
        def server_loop():
            while True: time.sleep(0.001)
        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True: time.sleep(0.01)

    return server

parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
parser.add_argument("--image_folder", type=str, default="examples/kitchen/images/")
parser.add_argument("--use_point_map", action="store_true")
parser.add_argument("--background_mode", action="store_true")
parser.add_argument("--port", type=int, default=8080)
parser.add_argument("--conf_threshold", type=float, default=25.0)

def main():
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("Initializing VGGT model...")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval().to(device)

    print(f"Loading images from {args.image_folder}...")
    image_names = sorted(glob.glob(os.path.join(args.image_folder, "*")))
    
    # Keep the raw PyTorch tensor for SuperGlue before it gets squeezed into numpy
    images_tensor_raw = load_and_preprocess_images(image_names).to(device)

    dtype = torch.float16 if device != "cpu" else torch.float32
    with torch.no_grad():
        autocast_device = "cuda" if "cuda" in device else "cpu"
        with torch.autocast(device_type=autocast_device, dtype=dtype):
            predictions = model(images_tensor_raw)

    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images_tensor_raw.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Squeeze to numpy for classical processing
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            
    # run bundle adjustment using the SuperGlue 
    optimized_extrinsics, opt_points_3d, cam_idx, pt_idx = run_true_bundle_adjustment(
        images_tensor_raw, 
        predictions["depth"], 
        predictions["extrinsic"], 
        predictions["intrinsic"], 
        device
    )
    predictions["extrinsic"] = optimized_extrinsics

    # scale align the dense depth maps to the optimized sparse points (fixing scale ambiguity)
    aligned_depths = scale_align_depth_maps(
        predictions["depth"],
        optimized_extrinsics,
        predictions["intrinsic"],
        opt_points_3d,
        cam_idx,
        pt_idx
    )
    predictions["depth"] = aligned_depths

    viser_server = viser_wrapper(
        predictions, port=args.port, init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map, background_mode=args.background_mode,
        image_folder=args.image_folder,
    )

if __name__ == "__main__":
    main()
