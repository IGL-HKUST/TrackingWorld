import time
import argparse
import numpy as np
import imageio
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

import viser
import viser.transforms as tf

# ================= Data Loading and Processing Utilities =================

def save_numpy_list_to_gif(frames, output_path, fps=10):
    """Saves a list of numpy arrays (frames) to a GIF file."""
    frames_uint8 = [np.uint8(frame) for frame in frames]
    imageio.mimsave(output_path, frames_uint8, fps=fps)
    print(f"Saved GIF to {output_path}")

def load_data(filepath):
    """Loads 4D tracking data from a .npz file."""
    print(f"Loading data from {filepath}...")
    track_4d = np.load(filepath)
    
    # Static points and colors
    static_points = track_4d['static_points'].astype(np.float32)
    static_rgbs = track_4d['static_rgbs'].astype(np.float32) / 255.0
    # Dynamic points and colors
    dyn_points = track_4d['dyn_points'].astype(np.float32)
    dyn_rgbs = track_4d['dyn_rgbs'].astype(np.float32) / 255.0
    # Dynamic visibility (T, N_dyn)
    dyn_vis = track_4d['dyn_vis']
    
    # Static track initialization frames (N_static,)
    track_init_frames_static = track_4d['track_init_frames_static'].astype(np.int32)
    
    # Camera poses (c2w), images (rgbs), and intrinsic matrix (K)
    c2w = track_4d['c2w']
    rgbs = track_4d['rgbs']
    K = track_4d['Ks'][0]
    
    # Dynamic track initialization frames (N_dyn,). np.argmax finds the first visible frame.
    track_init_frames_dyn = np.argmax(dyn_vis, axis=0)
    
    return static_points, static_rgbs, dyn_points, dyn_rgbs, dyn_vis, track_init_frames_static, track_init_frames_dyn, c2w, rgbs, K

def compute_knn_avg_distance(dyn_pts: np.ndarray, k: int = 10):
    """Computes the average distance to the k nearest neighbors for outlier detection."""
    if dyn_pts.shape[0] <= k:
        return np.zeros(dyn_pts.shape[0])
    # k+1 because the point itself is the nearest neighbor (distance 0)
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(dyn_pts)
    distances, _ = nbrs.kneighbors(dyn_pts)
    # Exclude the distance to the point itself (index 0) and average the rest
    avg_dists = distances[:, 1:].mean(axis=1)
    return avg_dists

def preprocess_dynamic_points(dyn_pts, dyn_vis, frames_dyn, z_threshold=30.5):
    """Performs geometric filtering and KNN outlier removal on dynamic points."""
    # (Keeping the original logic unchanged)
    N, T, _ = dyn_pts.shape
    dyn_vis_T = dyn_vis.T 

    # Compute Z-mean for geometric outlier removal (based on mean depth)
    z_vals = dyn_pts[..., 2]
    vis_sum = np.sum(dyn_vis_T, axis=1)
    vis_sum[vis_sum == 0] = 1
    z_mean = np.sum(z_vals * dyn_vis_T, axis=1) / vis_sum
    
    mask = z_mean > z_threshold # Identify points with a mean Z (depth) greater than threshold
    
    # Replace outliers with their mean position across visible frames
    for i in np.where(mask)[0]:
        vis_i = dyn_vis_T[i]
        if np.sum(vis_i) == 0:
            continue
        mean_xyz = np.mean(dyn_pts[i][vis_i], axis=0)
        dyn_pts[i][vis_i] = mean_xyz

    print("Running KNN outlier removal...")
    # KNN filtering on a per-frame basis
    for i in tqdm(range(T), desc="KNN Filtering"):
        current_frame_mask = (frames_dyn == i) # Points initialized in the current frame
        if np.sum(current_frame_mask) > 0:
            current_pts = dyn_pts[current_frame_mask, i]
            dis = compute_knn_avg_distance(current_pts)
            
            # Use KNN distance to update visibility mask
            dyn_vis1 = dyn_vis[:, current_frame_mask]
            dyn_vis1[:, dis > 0.1] = False # Mark points with high average KNN distance as invisible
            dyn_vis[:, current_frame_mask] = dyn_vis1
            
    return dyn_pts, dyn_vis

# ================= Main Program =================

def main(
    filepath="",
    port=8080,
    downsample_factor=1,  # Downsampling factor for the point clouds
    sample_points=100, # Note: Not used in current logic, replaced by step_size logic
    trail_length=10,
    frame_window=1,
    render_save_path="./rendered_frames"
):
    server = viser.ViserServer(port=port)
    print(f"Viser server started at http://localhost:{port}")
    
    # Wait for a client to connect
    while not server.get_clients():
        time.sleep(1.0)
    
    client = list(server.get_clients().values())[0]

    # 1. Load data
    data = load_data(filepath)
    static_pts, static_rgbs, dyn_pts, dyn_rgbs, dyn_vis, frames_static, frames_dyn, c2w, rgbs, K = data

    # ------------------ Downsampling Logic ------------------
    if downsample_factor > 1:
        print(f"Downsampling point clouds by factor {downsample_factor}...")
        
        # Static Points: Shape (N, 3). Sample along the N dimension (axis 0).
        static_pts = static_pts[::downsample_factor]
        static_rgbs = static_rgbs[::downsample_factor]
        frames_static = frames_static[::downsample_factor] # Indices must be sampled synchronously

        # Dynamic Points: Shape (N, T, 3). Sample along the N dimension (axis 0).
        dyn_pts = dyn_pts[::downsample_factor]
        dyn_rgbs = dyn_rgbs[::downsample_factor]
        
        # Visibility: Shape (T, N). Sample along the N dimension (axis 1).
        dyn_vis = dyn_vis[:, ::downsample_factor]
        
        # Init frames: Shape (N,). Sample along the N dimension (axis 0).
        frames_dyn = frames_dyn[::downsample_factor]
    # -------------------------------------------------------------

    # 2. Preprocess dynamic points (KNN time complexity is significantly reduced after downsampling)
    dyn_pts, dyn_vis = preprocess_dynamic_points(dyn_pts, dyn_vis, frames_dyn)

    # Combine static and dynamic points for a unified coordinate system (T, N_static + N_dyn, 3)
    coords = np.concatenate([static_pts, dyn_pts], axis=0).transpose(1, 0, 2) 
    
    num_frames = coords.shape[0]
    num_static = static_pts.shape[0]
    num_points = coords.shape[1]
    
    print(f"Data Loaded (After Downsampling): {num_frames} frames, {num_points} total points")

    # 3. Sample trajectory points
    # Note: This logic automatically adapts to the new num_points
    dyn_ids = np.arange(num_static, num_points)
    chosen_ids = []
    num_segments = 15
    
    # Stratified sampling of trajectories across time segments
    for seg in range(num_segments):
        start_frame = seg * num_frames // num_segments
        end_frame = (seg + 1) * num_frames // num_segments
        # Select dynamic points whose initial frame falls within the segment and is before frame 20
        mask = (frames_dyn >= start_frame) & (frames_dyn < end_frame) & (frames_dyn < 20)
        seg_ids = dyn_ids[mask]
        
        if len(seg_ids) == 0:
            continue
            
        # Dynamically adjust trajectory sampling density based on point cloud quantity
        step_size = max(1, 300 // downsample_factor) 
        chosen_seg_ids = seg_ids[::step_size]  
        chosen_ids.extend(chosen_seg_ids)
        
    # Assign distinct colors to the sampled trajectories
    traj_colors = [cm.get_cmap("hsv")(i / max(1, len(chosen_ids)))[:3] for i in range(len(chosen_ids))]

    # 4. Create Viser scene objects
    frame_nodes = []
    static_clouds = []
    dyn_clouds = []
    camera_frames = {}
    axes_frames = {}
    
    # Only visualize a subset of frames for efficiency
    display_frames = min(num_frames, 60)

    print("Creating scene nodes...")
    for i in tqdm(range(display_frames)):
        # Main frame node for the current timestep
        frame_node = server.scene.add_frame(f"/frame_{i}", show_axes=False, visible=(i==0))
        frame_nodes.append(frame_node)

        # Static Cloud
        # A static point is visible if the current frame is its initial frame or later
        is_static_visible = i >= frames_static
        if np.any(is_static_visible):
            # Select points and colors that are visible in the current frame
            s_pts = static_pts.transpose(1, 0, 2)[i, is_static_visible]
            s_clr = static_rgbs[is_static_visible]
            
            s_cloud = server.scene.add_point_cloud(
                name=f"/frame_{i}/static_cloud",
                points=s_pts,
                colors=s_clr,
                point_size=0.03,
                point_shape="rounded",
            )
            static_clouds.append(s_cloud)

        # Dynamic Cloud
        # A dynamic point is visible if its corresponding flag in dyn_vis[i] is True
        is_dyn_visible = dyn_vis[i]
        if np.any(is_dyn_visible):
            d_pts = dyn_pts.transpose(1, 0, 2)[i, is_dyn_visible]
            d_clr = dyn_rgbs[is_dyn_visible]
            
            d_cloud = server.scene.add_point_cloud(
                name=f"/frame_{i}/dyn_cloud",
                points=d_pts,
                colors=d_clr,
                point_size=0.01,
                point_shape="rounded",
            )
            dyn_clouds.append(d_cloud)

        # Camera Frustum
        cam_pose = c2w[i]
        R_mat = cam_pose[:3, :3]
        t = cam_pose[:3, 3]
        
        # Calculate FOV and aspect ratio from intrinsic matrix K
        fov = 2 * np.arctan2(rgbs.shape[1] / 2, K[0, 0])
        aspect = rgbs.shape[2] / rgbs.shape[1]
        # Assign color based on time
        color_rgb = cm.viridis(i / (display_frames - 1))[:3] if display_frames > 1 else (1,0,0)

        cam_frustum = server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.1,
            # Downsample image for faster loading
            image=rgbs[i][::4, ::4], 
            wxyz=tf.SO3.from_matrix(R_mat).wxyz,
            position=t,
            color=color_rgb,
            visible=(i==0) # Only visible at frame 0 initially
        )
        camera_frames[i] = cam_frustum
        
        # Add local axes for the camera frame
        axes_frame = server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.1,
            axes_radius=0.01,
            visible=(i==0)
        )
        axes_frames[i] = axes_frame

    # Trajectories (Splines)
    splines = {}
    # Unified visibility mask for static and dynamic points (T, N_total)
    # Static visibility: True if current frame T >= init_frame_static
    init_frame_mask = np.concatenate([(np.arange(num_frames)[:, None] >= frames_static), dyn_vis], axis=1)

    for idx, pid in enumerate(chosen_ids):
        # Create an empty spline object for each sampled trajectory
        splines[pid] = server.scene.add_spline_catmull_rom(
            name=f"/traj_{pid}",
            positions=np.zeros((2, 3)),
            color=traj_colors[idx],
            line_width=1.5,
            tension=0.5,
        )

    # ... [Keeping helper logic, GUI controls, event callbacks unchanged] ...
    
    # ------------------ Original Helper Functions and GUI Code ------------------
    
    def update_scene_state(current_t, prev_visible_set):
        """Updates the visibility and position of scene objects for the current timestep."""
        with server.atomic():
            # Hide previous frames outside the current window
            for i in prev_visible_set:
                frame_nodes[i].visible = False
                if i in camera_frames: camera_frames[i].visible = False
                if i in axes_frames: axes_frames[i].visible = False
            
            # Determine the current frame window to display
            start = max(0, current_t - frame_window_slider.value + 1)
            current_visible = set(range(start, current_t + 1))

            # Show current frames within the window
            for i in current_visible:
                frame_nodes[i].visible = True
                if i in camera_frames: camera_frames[i].visible = True
                if i in axes_frames: axes_frames[i].visible = True

            # Update trajectories
            for idx, pid in enumerate(chosen_ids):
                # Get frames where the point was visible up to the current frame
                active_frames = np.where(init_frame_mask[:current_t + 1, pid])[0]
                # Apply trail length limitation
                traj_start = max(0, current_t - trail_len.value + 1)
                active_frames = active_frames[active_frames >= traj_start]
                if len(active_frames) >= 2:
                    # Update spline positions
                    splines[pid].positions = coords[active_frames, pid].tolist()
                else:
                    # Reset spline if not enough visible points
                    splines[pid].positions = np.zeros((2, 3)).tolist()
                    
        return current_visible

    with server.gui.add_folder("Playback"):
        timestep = server.gui.add_slider("Frame", min=0, max=display_frames - 1, step=1, initial_value=0)
        fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=10)
        playing = server.gui.add_checkbox("Playing", False)
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_next_frame = server.gui.add_button("Next Frame")
            
    with server.gui.add_folder("Appearance"):
        static_pt_size_slider = server.gui.add_slider("Static Pt Size", min=0.001, max=0.2, step=0.001, initial_value=0.03)
        dyn_pt_size_slider = server.gui.add_slider("Dynamic Pt Size", min=0.001, max=0.1, step=0.001, initial_value=0.01)
        trail_len = server.gui.add_slider("Trail Length", min=1, max=50, step=1, initial_value=trail_length)
        frame_window_slider = server.gui.add_slider("Frame Window", min=1, max=30, step=1, initial_value=frame_window)

    with server.gui.add_folder("Export"):
        rendering = server.gui.add_checkbox("Start GIF Render", False)
        save_viser_btn = server.gui.add_button("Save .viser Recording")

    @static_pt_size_slider.on_update
    def _(_) -> None:
        """Updates the size of static point clouds."""
        for node in static_clouds:
            node.point_size = static_pt_size_slider.value

    @dyn_pt_size_slider.on_update
    def _(_) -> None:
        """Updates the size of dynamic point clouds."""
        for node in dyn_clouds:
            node.point_size = dyn_pt_size_slider.value

    @gui_next_frame.on_click
    def _(_) -> None:
        """Moves to the next frame."""
        timestep.value = (timestep.value + 1) % display_frames
        # Log camera properties for debugging/setup (optional)
        cam = client.camera
        print("Camera position:", cam.position)
        print("Camera rotation (wxyz):", cam.wxyz)
        print("Camera look_at:", cam.look_at)
        print("Camera up:", cam.up_direction)
        print("FOV:", cam.fov)
        print("Near/Far:", cam.near, cam.far)
        
    @gui_prev_frame.on_click
    def _(_) -> None:
        """Moves to the previous frame."""
        timestep.value = (timestep.value - 1) % display_frames

    @save_viser_btn.on_click
    def _(_):
        """Saves the scene trajectory to a .viser file for offline playback."""
        print("Initializing Viser Serialization...")
        was_playing = playing.value
        playing.value = False # Pause playback during recording
        serializer = server.get_scene_serializer()
        rec_visible_frames = set() 
        
        try:
            # Step through all frames and record the scene state
            for t in tqdm(range(display_frames), desc="Recording .viser"):
                rec_visible_frames = update_scene_state(t, rec_visible_frames)
                serializer.insert_sleep(1.0 / fps.value) # Insert pause between frames
            
            print("Serializing data...")
            data = serializer.serialize()
            output_filename = "track_recording.viser"
            # Write the serialized binary data to a file
            Path(output_filename).write_bytes(data)
            print(f"Successfully saved {output_filename} ({len(data)/1024/1024:.2f} MB)")
            
        except Exception as e:
            print(f"Serialization failed: {e}")
        finally:
            # Restore previous playback state
            playing.value = was_playing

    @rendering.on_update
    def _(_) -> None:
        """Starts rendering the scene to a GIF via client screenshots."""
        if rendering.value:
            render_imgs = []
            print("Starting GIF render...")
            was_playing = playing.value
            playing.value = False # Pause playback during rendering
            try:
                import os
                os.makedirs(render_save_path, exist_ok=True)
                for i in range(display_frames):
                    timestep.value = i
                    time.sleep(0.5) # Wait for scene to update and render
                    # Capture high-resolution screenshot from the client
                    img = client.get_render(height=1080, width=1920)
                    render_imgs.append(img)
                    print(f"Rendered frame {i}/{display_frames}")
                # Save the captured frames as a GIF
                save_numpy_list_to_gif(render_imgs, os.path.join(render_save_path, "rendered.gif"), fps=fps.value)
            except Exception as e:
                print(f"Rendering failed: {e}")
            finally:
                rendering.value = False
                playing.value = was_playing # Restore previous playback state

    prev_frame = -1
    visible_frames = set()

    # Main Viser server loop
    while True:
        # Disable manual frame controls when playing
        gui_next_frame.disabled = playing.value
        gui_prev_frame.disabled = playing.value 
        if playing.value:
            # Advance frame when playing is enabled
            timestep.value = (timestep.value + 1) % display_frames
        
        # Update scene if the frame slider value changed
        if timestep.value != prev_frame:
            visible_frames = update_scene_state(timestep.value, visible_frames)
            prev_frame = timestep.value
            
        time.sleep(1.0 / fps.value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="./demo_data/dog/uni4d/base/fused_track_4d_full.npz")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--downsample", type=int, default=10, help="Downsample factor (e.g., 5 means keep 1/5 of points)")
    args = parser.parse_args()
    
    main(filepath=args.filepath, port=args.port, downsample_factor=args.downsample)