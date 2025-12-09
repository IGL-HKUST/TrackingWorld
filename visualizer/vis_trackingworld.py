import time
import argparse
import numpy as np
import imageio
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

import viser
import viser.transforms as tf

# ================= Data Loading and Processing Utilities =================

def save_numpy_list_to_gif(frames, output_path, fps=10):
    """Save a list of numpy frames as a GIF"""
    frames_uint8 = [np.uint8(frame) for frame in frames]
    imageio.mimsave(output_path, frames_uint8, fps=fps)
    print(f"Saved GIF to {output_path}")

def load_data(filepath):
    """Load NPZ data"""
    print(f"Loading data from {filepath}...")
    track_4d = np.load(filepath)
    
    static_points = track_4d['static_points'].astype(np.float32)
    static_rgbs = track_4d['static_rgbs'].astype(np.float32) / 255.0
    dyn_points = track_4d['dyn_points'].astype(np.float32)
    dyn_rgbs = track_4d['dyn_rgbs'].astype(np.float32) / 255.0
    dyn_vis = track_4d['dyn_vis']
    
    # Note: track_init_frames_static is loaded but not mainly used later; keep it just in case
    track_init_frames_static = track_4d['track_init_frames_static'].astype(np.int32)
    
    c2w = track_4d['c2w']
    rgbs = track_4d['rgbs']
    K = track_4d['Ks'][0]
    
    # Compute initial frame indices for dynamic points
    track_init_frames_dyn = np.argmax(dyn_vis, axis=0)
    
    return static_points, static_rgbs, dyn_points, dyn_rgbs, dyn_vis, track_init_frames_static, track_init_frames_dyn, c2w, rgbs, K

def compute_knn_avg_distance(dyn_pts: np.ndarray, k: int = 10):
    """Compute KNN average distance for outlier removal"""
    if dyn_pts.shape[0] <= k:
        return np.zeros(dyn_pts.shape[0])
        
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(dyn_pts)
    distances, _ = nbrs.kneighbors(dyn_pts)
    # Exclude self (distance=0)
    avg_dists = distances[:, 1:].mean(axis=1)
    return avg_dists

def preprocess_dynamic_points(dyn_pts, dyn_vis, frames_dyn, z_threshold=30.5):
    """Preprocess dynamic points: smooth positions based on Z-axis threshold and remove outliers using KNN"""
    N, T, _ = dyn_pts.shape
    dyn_vis_T = dyn_vis.T  # (N, T)

    # 1. Smooth based on Z-axis mean
    z_vals = dyn_pts[..., 2]
    vis_sum = np.sum(dyn_vis_T, axis=1)
    vis_sum[vis_sum == 0] = 1
    z_mean = np.sum(z_vals * dyn_vis_T, axis=1) / vis_sum
    
    mask = z_mean > z_threshold
    
    # For points exceeding threshold, set all visible frames to mean center
    for i in np.where(mask)[0]:
        vis_i = dyn_vis_T[i]
        if np.sum(vis_i) == 0:
            continue
        mean_xyz = np.mean(dyn_pts[i][vis_i], axis=0)
        dyn_pts[i][vis_i] = mean_xyz

    # 2. Remove outliers using KNN
    print("Running KNN outlier removal...")
    for i in tqdm(range(T), desc="KNN Filtering"):
        # Select points initialized in the current frame
        current_frame_mask = (frames_dyn == i)
        if np.sum(current_frame_mask) > 0:
            current_pts = dyn_pts[current_frame_mask, i]
            dis = compute_knn_avg_distance(current_pts)
            
            # Update visibility: if distance too large, mark as invisible
            dyn_vis1 = dyn_vis[:, current_frame_mask]
            dyn_vis1[:, dis > 0.1] = False
            dyn_vis[:, current_frame_mask] = dyn_vis1
            
    return dyn_pts, dyn_vis

# ================= Main Program =================

def main(
    filepath="",
    sample_points=100,
    trail_length=10,
    port=8080,
    frame_window=1,
    render_save_path="./rendered_frames" # Default save path
):
    # Start Viser server
    server = viser.ViserServer(port=port)
    print(f"Viser server started at http://localhost:{port}")
    
    while not server.get_clients():
        time.sleep(1.0)
    
    # Get first client for rendering
    client = list(server.get_clients().values())[0]

    # 1. Load data
    data = load_data(filepath)
    static_pts, static_rgbs, dyn_pts, dyn_rgbs, dyn_vis, frames_static, frames_dyn, c2w, rgbs, K = data

    # 2. Preprocess dynamic points
    dyn_pts, dyn_vis = preprocess_dynamic_points(dyn_pts, dyn_vis, frames_dyn)

    # Prepare merged data for computation
    coords = np.concatenate([static_pts, dyn_pts], axis=0).transpose(1, 0, 2) # (T, N_total, 3)
    
    num_frames = coords.shape[0]
    num_static = static_pts.shape[0]
    num_points = coords.shape[1]
    
    print(f"Data Loaded: {num_frames} frames, {num_points} total points ({num_static} static)")

    # 3. Sample trajectory points (for drawing trails)
    dyn_ids = np.arange(num_static, num_points)
    chosen_ids = []
    num_segments = 15
    points_per_segment = sample_points // num_segments

    for seg in range(num_segments):
        start_frame = seg * num_frames // num_segments
        end_frame = (seg + 1) * num_frames // num_segments
        mask = (frames_dyn >= start_frame) & (frames_dyn < end_frame) & (frames_dyn < 20)
        seg_ids = dyn_ids[mask]
        
        if len(seg_ids) == 0:
            continue
            
        # Sampling strategy: slice step
        chosen_seg_ids = seg_ids[60::300] 
        chosen_ids.extend(chosen_seg_ids)
        
    traj_colors = [cm.get_cmap("hsv")(i / len(chosen_ids))[:3] for i in range(len(chosen_ids))]

    # 4. Create Viser scene objects
    frame_nodes = []     # Store frame root nodes
    static_clouds = []   # Store static point cloud handles (for size adjustment)
    dyn_clouds = []      # Store dynamic point cloud handles (for size adjustment)
    camera_frames = {}   # Store camera frustum
    axes_frames = {}
    
    display_frames = min(num_frames, 60) # Limit maximum frames to display

    print("Creating scene nodes...")
    for i in tqdm(range(display_frames)):
        frame_node = server.scene.add_frame(f"/frame_{i}", show_axes=False, visible=(i==0))
        frame_nodes.append(frame_node)

        # Add static point cloud
        is_static_visible = i >= frames_static
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

        # Add dynamic point cloud
        is_dyn_visible = dyn_vis[i]
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

        # Add camera frustum
        cam_pose = c2w[i]
        R_mat = cam_pose[:3, :3]
        t = cam_pose[:3, 3]
        
        fov = 2 * np.arctan2(rgbs.shape[1] / 2, K[0, 0])
        aspect = rgbs.shape[2] / rgbs.shape[1]
        
        color_rgb = cm.viridis(i / (display_frames - 1))[:3] if display_frames > 1 else (1,0,0)

        cam_frustum = server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.1,
            image=rgbs[i][::4, ::4], # downsample for performance
            wxyz=tf.SO3.from_matrix(R_mat).wxyz,
            position=t,
            color=color_rgb,
            visible=(i==0)
        )
        camera_frames[i] = cam_frustum
        
        axes_frame = server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.1,
            axes_radius=0.01,
            visible=(i==0)
        )
        axes_frames[i] = axes_frame

    # Add trajectory lines (initially empty)
    splines = {}
    init_frame_mask = np.concatenate([(np.arange(num_frames)[:, None] >= frames_static), dyn_vis], axis=1)

    for idx, pid in enumerate(chosen_ids):
        splines[pid] = server.scene.add_spline_catmull_rom(
            name=f"/traj_{pid}",
            positions=np.zeros((2, 3)),
            color=traj_colors[idx],
            line_width=1.5,
            tension=0.5,
        )

    # ================= GUI Controls =================
    
    with server.gui.add_folder("Playback"):
        timestep = server.gui.add_slider("Frame", min=0, max=display_frames - 1, step=1, initial_value=0)
        fps = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=10)
        playing = server.gui.add_checkbox("Playing", False)
        
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_next_frame = server.gui.add_button("Next Frame")
            
    with server.gui.add_folder("Appearance"):
        # === Add: Point size control ===
        static_pt_size_slider = server.gui.add_slider("Static Pt Size", min=0.001, max=0.2, step=0.001, initial_value=0.03)
        dyn_pt_size_slider = server.gui.add_slider("Dynamic Pt Size", min=0.001, max=0.1, step=0.001, initial_value=0.01)
        
        trail_len = server.gui.add_slider("Trail Length", min=1, max=50, step=1, initial_value=trail_length)
        frame_window_slider = server.gui.add_slider("Frame Window", min=1, max=30, step=1, initial_value=frame_window)

    with server.gui.add_folder("Rendering"):
        rendering = server.gui.add_checkbox("Start Render", False)

    # ================= Event Callbacks =================

    # 1. Point size update callbacks
    @static_pt_size_slider.on_update
    def _(_) -> None:
        for node in static_clouds:
            node.point_size = static_pt_size_slider.value

    @dyn_pt_size_slider.on_update
    def _(_) -> None:
        for node in dyn_clouds:
            node.point_size = dyn_pt_size_slider.value

    # 2. Playback control callbacks
    @gui_next_frame.on_click
    def _(_) -> None:
        timestep.value = (timestep.value + 1) % display_frames
        
    @gui_prev_frame.on_click
    def _(_) -> None:
        timestep.value = (timestep.value - 1) % display_frames

    # 3. Rendering callback
    @rendering.on_update
    def _(_) -> None:
        if rendering.value:
            render_imgs = []
            print("Starting render...")
            was_playing = playing.value
            playing.value = False
            
            try:
                import os
                os.makedirs(render_save_path, exist_ok=True)
                
                for i in range(display_frames):
                    timestep.value = i
                    time.sleep(0.5) 
                    img = client.get_render(height=1080, width=1920)
                    render_imgs.append(img)
                    print(f"Rendered frame {i}/{display_frames}")
                
                print("Saving GIFs...")
                save_numpy_list_to_gif(render_imgs, os.path.join(render_save_path, "rendered.gif"), fps=fps.value)
                print("Render done.")
            except Exception as e:
                print(f"Rendering failed: {e}")
            finally:
                rendering.value = False
                playing.value = was_playing

    # ================= Main Loop =================
    
    prev_frame = -1
    visible_frames = set()

    while True:
        gui_next_frame.disabled = playing.value
        gui_prev_frame.disabled = playing.value 
        
        if playing.value:
            timestep.value = (timestep.value + 1) % display_frames

        if timestep.value != prev_frame:
            with server.atomic():
                current_t = timestep.value
                
                # 1. Hide old frames
                for i in visible_frames:
                    frame_nodes[i].visible = False
                    if i in camera_frames: camera_frames[i].visible = False
                    if i in axes_frames: axes_frames[i].visible = False
                visible_frames.clear()

                # 2. Compute frames to show in the current window
                start = max(0, current_t - frame_window_slider.value + 1)
                current_visible = set(range(start, current_t + 1))

                # 3. Show new frames
                for i in current_visible:
                    frame_nodes[i].visible = True
                    if i in camera_frames: camera_frames[i].visible = True
                    if i in axes_frames: axes_frames[i].visible = True
                    visible_frames.add(i)

                # 4. Update trajectory lines
                for idx, pid in enumerate(chosen_ids):
                    active_frames = np.where(init_frame_mask[:timestep.value + 1, pid])[0]
                    traj_start = max(0, timestep.value - trail_len.value + 1)
                    active_frames = active_frames[active_frames >= traj_start]
                    if len(active_frames) >= 2:
                        splines[pid].positions = coords[active_frames, pid].tolist()
                    else:
                        splines[pid].positions = np.zeros((2, 3)).tolist()

            prev_frame = timestep.value

        time.sleep(1.0 / fps.value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str, default="./demo_data/dog/uni4d/base/fused_track_4d_full.npz")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    main(filepath=args.filepath, port=args.port)
