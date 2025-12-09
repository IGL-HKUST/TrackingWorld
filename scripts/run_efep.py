import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")  # adjust if needed
sys.path.append(os.path.abspath(project_root))

import argparse
import pickle
import imageio
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from densetrack3d.datasets.custom_data import read_data, read_data_with_depthcrafter
from densetrack3d.models.densetrack3d.densetrack3d_epef import DenseTrack3D
from densetrack3d.models.geometry_utils import least_square_align
from densetrack3d.models.predictor.dense_predictor import DensePredictor3D
from densetrack3d.utils.depthcrafter_utils import read_video
from densetrack3d.utils.visualizer import Visualizer
from densetrack3d.models.model_utils import bilinear_sampler

BASE_DIR = os.getcwd()
device = torch.device("cuda")


@torch.inference_mode()
def predict_unidepth(video, model):
    video_torch = torch.from_numpy(video).permute(0, 3, 1, 2).to(device)
    depth_pred = []

    for chunk in torch.split(video_torch, 32, dim=0):
        predictions = model.infer(chunk)
        depth_pred.append(predictions["depth"].squeeze(1).cpu().numpy())

    return np.concatenate(depth_pred, axis=0)


@torch.inference_mode()
def predict_depthcrafter(video, pipe):
    frames, ori_h, ori_w = read_video(video, max_res=1024)
    res = pipe(
        frames,
        height=frames.shape[1],
        width=frames.shape[2],
        output_type="np",
        guidance_scale=1.2,
        num_inference_steps=25,
        window_size=110,
        overlap=25,
        track_time=False,
    ).frames[0]

    # convert to single-channel depth map and normalize
    res = res.mean(axis=-1)
    res = (res - res.min()) / (res.max() - res.min())
    res = F.interpolate(torch.from_numpy(res[:, None]), (ori_h, ori_w), mode="nearest").squeeze(1).numpy()

    return res


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="./checkpoints/densetrack3d.pth")
    parser.add_argument("--video_path", type=str, default="./demo_data", help="Root directory containing multiple video folders")
    parser.add_argument("--output_path", type=str, default="./results")
    parser.add_argument("--use_depthcrafter", action="store_true", help="use DepthCrafter as input depth")
    parser.add_argument("--viz_sparse", type=bool, default=True, help="visualize sparse tracking")
    parser.add_argument("--downsample", type=int, default=4, help="downsample factor for sparse tracking")
    parser.add_argument("--upsample_factor", type=int, default=4, help="model upsample factor")
    parser.add_argument("--T_end", type=int, default=31, help="number of frames to process")
    parser.add_argument("--use_fp16", action="store_true", help="use fp16 precision")
    parser.add_argument("--clip_size", type=int, default=10, help="clip size")
    parser.add_argument("--use_cotracker", type=bool, default=False, help="use cotracker")

    return parser


def main():
    args = get_args_parser().parse_args()

    # --- 1. Initialize Tracker Model (Load Once) ---
    print("Creating DenseTrack3D model...")
    model = DenseTrack3D(
        stride=4,
        window_len=16,
        add_space_attn=True,
        num_virtual_tracks=64,
        model_resolution=(384, 512),
        upsample_factor=args.upsample_factor
    )

    print(f"Loading checkpoint from {args.ckpt}...")
    state_dict = torch.load(args.ckpt, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict, strict=False)

    predictor = DensePredictor3D(model=model).eval().cuda()

    # --- 2. Initialize Depth Models (Load Once) ---
    unidepth_model = None
    depth_crafter_pipe = None

    # Load UniDepth (Always loaded as fallback or primary)
    os.sys.path.append(os.path.join(BASE_DIR, "submodules", "UniDepth"))
    from unidepth.models import UniDepthV2
    print("Loading UniDepth model...")
    unidepth_model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").eval().to(device)

    # Load DepthCrafter (If requested)
    if args.use_depthcrafter:
        os.sys.path.append(os.path.join(BASE_DIR, "submodules", "DepthCrafter"))
        from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
        from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter

        print("Loading DepthCrafter model...")
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            "tencent/DepthCrafter", low_cpu_mem_usage=True, torch_dtype=torch.float16
        )
        depth_crafter_pipe = DepthCrafterPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", unet=unet, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

        try:
            depth_crafter_pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("Xformers not enabled")
        depth_crafter_pipe.enable_attention_slicing()

    # --- 3. Iterate over all videos in the directory ---
    if not os.path.exists(args.video_path):
        print(f"Error: Video path {args.video_path} does not exist.")
        return

    # Get list of subdirectories (assuming each video is a folder)
    video_list = sorted([d for d in os.listdir(args.video_path) if os.path.isdir(os.path.join(args.video_path, d))])
    
    print(f"Found {len(video_list)} videos in {args.video_path}")

    for video_name in tqdm(video_list, desc="Processing Videos"):
      if video_name in ['rollerblade']:
        full_video_dir = os.path.join(args.video_path, video_name)
        
        # Define output directory for this specific video
        save_dir = os.path.join(args.output_path, video_name, 'densetrack3d_efep')
        if os.path.exists(os.path.join(save_dir, "results.npz")):
            print(f"Skipping {video_name}, results already exist.")
            continue
            
        print(f"\nProcessing: {video_name}")
        
        try:
            # --- Load Video Data ---
            video, videodepth, videodisp = read_data_with_depthcrafter(full_path=full_video_dir)

            if video is None:
                print(f"Skipping {video_name}: Could not read video data.")
                continue

            # --- Depth Processing ---
            if videodepth is None:
                print("Running UniDepth inference...")
                videodepth = predict_unidepth(video, unidepth_model)
                # Save cache
                os.makedirs(os.path.join(full_video_dir, "unidepth"), exist_ok=True)
                np.save(os.path.join(full_video_dir, "unidepth/depth.npy"), videodepth)

            if args.use_depthcrafter:
                if videodisp is None:
                    print("Running DepthCrafter inference...")
                    videodisp = predict_depthcrafter(video, depth_crafter_pipe)
                    np.save(os.path.join(full_video_dir, "depth_depthcrafter.npy"), videodisp)

                videodepth = least_square_align(videodepth, videodisp)

            # --- Convert to Torch ---
            # Ensure we reset inputs for the new video
            video_torch = torch.from_numpy(video).permute(0, 3, 1, 2).cuda()[None].float()
            videodepth_torch = torch.from_numpy(videodepth).cuda()[None].float()

            os.makedirs(save_dir, exist_ok=True)

            # --- Run DenseTrack3D ---
            print("Running DenseTrack3D tracking...")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.use_fp16):
                # Using T_end + 1 to include the end frame index
                out_dict = predictor.forward_epef(
                    video_torch[:, :args.T_end+1], 
                    videodepth_torch[:, :args.T_end+1], 
                    grid_query_frame=0, 
                    clip_size=args.clip_size, 
                    use_cotracker=args.use_cotracker
                )

            # --- Save 3D Trajectories ---
            trajs_3d_dict = {k: v[0].cpu().numpy() for k, v in out_dict["trajs_3d_dict"].items()}
            with open(os.path.join(save_dir, "dense_3d_track.pkl"), "wb") as f:
                pickle.dump(trajs_3d_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

            # --- Process Results ---
            pred_tracks = out_dict['trajs_uv']
            pred_visibility = out_dict['vis'][..., None]
            pred_confidence = out_dict['conf'][..., None]
            pred_tracks_depth = out_dict['trajs_depth']
            pred_tracks_colors = out_dict['trajs_3d_dict']['colors']
            init_frames = out_dict['dense_frame_e']

            dense_reso = out_dict["dense_reso"]
            interp_size = predictor.interp_shape[0] * predictor.interp_shape[1]

            # Helper for sparse conversion
            def to_sparse(tensor, shape, downsample):
                tensor = rearrange(tensor, f"b t (h w) c -> b t h w c", h=shape[0], w=shape[1])
                tensor = tensor[:, :, ::downsample, ::downsample]
                return rearrange(tensor, "b t h w c -> b t (h w) c")

            sparse_trajs_uv = to_sparse(pred_tracks[:, :, :interp_size], dense_reso, args.downsample)
            sparse_trajs_vis = to_sparse(pred_visibility[:, :, :interp_size], dense_reso, args.downsample)
            sparse_trajs_conf = to_sparse(pred_confidence[:, :, :interp_size], dense_reso, args.downsample)
            sparse_trajs_depth = to_sparse(pred_tracks_depth[:, :, :interp_size], dense_reso, args.downsample)

            sparse_trajs_colors = rearrange(pred_tracks_colors[:, :interp_size], "b (h w) c -> b h w c", h=dense_reso[0], w=dense_reso[1])
            sparse_trajs_colors = sparse_trajs_colors[:, ::args.downsample, ::args.downsample, :]
            sparse_trajs_colors = rearrange(sparse_trajs_colors, "b h w c -> b (h w) c")

            sparse_init_frames = rearrange(init_frames[:, :interp_size], "b (h w) -> b h w", h=dense_reso[0], w=dense_reso[1])
            sparse_init_frames = sparse_init_frames[:, ::args.downsample, ::args.downsample]
            sparse_init_frames = rearrange(sparse_init_frames, "b h w -> b (h w)")

            np.savez(
                os.path.join(save_dir, "results.npz"),
                all_confidences=pred_confidence[0].cpu().numpy(),
                all_tracks=pred_tracks[0].cpu().numpy(),
                all_tracks_depth=pred_tracks_depth[0].cpu().numpy(),
                all_tracks_color=pred_tracks_colors[0].cpu().numpy(),
                all_visibilities=pred_visibility[0].cpu().numpy(),
                init_frames=init_frames[0].cpu().numpy(),
                orig_shape=video.shape[-2:]
            )

            # --- Visualization ---
            if args.viz_sparse:
                print("Visualizing...")
                W = video.shape[-1]
                # Re-initialize visualizer per video to avoid stale state if any
                visualizer_2d = Visualizer(save_dir=os.path.join(save_dir, "viz"), fps=10, show_first_frame=0, linewidth=int(1 * W / 512), tracks_leave_trace=10)

                # 1. Sparse Viz
                video2d_viz = visualizer_2d.visualize(
                    video_torch[:, :args.T_end+1],
                    sparse_trajs_uv,
                    sparse_trajs_vis,
                    filename=f"{video_name}_sparse",
                    save_video=False,
                )
                video2d_viz = video2d_viz[0].permute(0, 2, 3, 1).cpu().numpy()

                save_path = os.path.join(save_dir, "first_2d_track.mp4")
                with imageio.get_writer(save_path, fps=10, codec="libx264") as writer:
                    for frame in video2d_viz:
                        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
                        writer.append_data(frame_uint8)

                # 2. Extra 2D Track Viz
                video2d_viz_extra = visualizer_2d.visualize(
                    video_torch[:, :args.T_end+1],
                    pred_tracks[:, :, interp_size::4],
                    pred_visibility[:, :, interp_size::4],
                    filename=f"{video_name}_extra",
                    save_video=False,
                )
                video2d_viz_extra = video2d_viz_extra[0].permute(0, 2, 3, 1).cpu().numpy()

                save_path_extra = os.path.join(save_dir, "extra_2d_track.mp4")
                with imageio.get_writer(save_path_extra, fps=10, codec="libx264") as writer:
                    for frame in video2d_viz_extra:
                        frame_uint8 = np.clip(frame, 0, 255).astype(np.uint8)
                        writer.append_data(frame_uint8)
            
            # Clean up GPU memory for this video
            del video_torch, videodepth_torch, out_dict
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"FAILED processing {video_name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()