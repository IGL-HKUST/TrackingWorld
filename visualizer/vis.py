"""
Record3D Visualizer

Parse and stream Record3D captures with interactive point size control.
"""

import os
import sys
import time
import pickle
from pathlib import Path

import numpy as np
import tyro
import viser
import viser.transforms as tf
from tqdm.auto import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from densetrack3d.datasets.custom_data import read_data


def main(
    filepath: str = "./demo_data/dog/densetrack3d_efep/dense_3d_track.pkl",
    max_frames: int = 100,
    port: int = 8082,
    init_point_size: float = 0.005,
) -> None:
    server = viser.ViserServer(port=port)
    print("Loading 3D trajectory data...")

    # Load trajectory data
    with open(filepath, "rb") as f:
        trajs_3d_dict = pickle.load(f)

    coords = trajs_3d_dict["coords"].astype(np.float32)  # T N 3
    colors = trajs_3d_dict["colors"].astype(np.float32) / 255.0  # N 3
    vis = trajs_3d_dict["vis"].astype(np.float32)  # T N
    init_frame = trajs_3d_dict.get("dense_frame_e", np.zeros(coords.shape[1], dtype=int)).astype(np.int32)

    num_frames, num_points = coords.shape[:2]
    print(f"Num frames: {num_frames}, points per frame: {num_points}")

    # Load video frames if available
    filename = os.path.basename(filepath).split(".")[0]
    try:
        video, videodepth = read_data("demo_data", filename)
    except Exception:
        video, videodepth = None, None

    # ----------------- GUI Controls -----------------
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider("Timestep", min=0, max=num_frames - 1, step=1, initial_value=0)
        gui_next_frame = server.gui.add_button("Next Frame")
        gui_prev_frame = server.gui.add_button("Prev Frame")
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=12)
        gui_framerate_options = server.gui.add_button_group("FPS Options", ("10", "20", "30", "60"))

    with server.gui.add_folder("Point Cloud Settings"):
        gui_point_size = server.gui.add_slider("Point Size", min=0.001, max=0.1, step=0.001, initial_value=init_point_size)

    # ----------------- Frame Navigation -----------------
    @gui_next_frame.on_click
    def next_frame(_):
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def prev_frame(_):
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    @gui_playing.on_update
    def toggle_playing(_):
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    @gui_framerate_options.on_click
    def set_framerate(_):
        gui_framerate.value = int(gui_framerate_options.value)

    # ----------------- Scene Setup -----------------
    server.scene.add_frame("/frames", wxyz=(1, 0, 0, 0), position=(0, 0, 0), show_axes=False)
    frame_nodes = []
    point_cloud_handles = []

    for i in tqdm(range(min(num_frames, max_frames)), desc="Adding frames"):
        frame_node = server.scene.add_frame(f"/frames/t{i}", show_axes=False)
        frame_nodes.append(frame_node)

        # Select points that exist at current frame
        mask = init_frame <= i
        coords_i = coords[i, mask]
        colors_i = colors[mask]

        # Add point cloud
        point_cloud = server.scene.add_point_cloud(
            name=f"/frames/t{i}/pos",
            points=coords_i,
            colors=colors_i,
            point_size=gui_point_size.value,
            point_shape="rounded",
            wxyz=(1, 0, 0, 0),
            position=(0, 0, 0),
        )
        point_cloud_handles.append(point_cloud)

        # Add camera frustum if video exists
        if video is not None and i < len(video):
            img_i = video[i]
            h, w = img_i.shape[:2]
            fov = 2 * np.arctan2(h / 2, w)
            aspect = w / h
            server.scene.add_camera_frustum(
                f"/frames/t{i}/frustum",
                fov=fov,
                aspect=aspect,
                scale=0.5,
                image=img_i,
                wxyz=(1, 0, 0, 0),
                position=(0, 0, -2),
            )

    # Show only initial frame
    prev_timestep = gui_timestep.value
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = (i == prev_timestep)

    # ----------------- GUI Updates -----------------
    @gui_timestep.on_update
    def update_frame(_):
        nonlocal prev_timestep
        current = gui_timestep.value
        with server.atomic():
            frame_nodes[current].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current
        server.flush()

    @gui_point_size.on_update
    def update_point_size(_):
        new_size = gui_point_size.value
        for node in point_cloud_handles:
            node.point_size = new_size
        server.flush()

    # ----------------- Playback Loop -----------------
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    tyro.cli(main)
