import configargparse
import os
import torch
from engine import Engine

def parse_args(input_string=None):
    parser = configargparse.ArgParser()

    # -----------------------------------------------------------
    # Basic settings
    # -----------------------------------------------------------
    parser.add_argument('--config', is_config_file=True, default="./config/config.yaml",
                        help='Path to config file')
    parser.add_argument('--gpu', type=str, default='4', help='GPU id')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--workdir', type=str, default='workdir', help='Working directory')
    parser.add_argument('--experiment_name', type=str, help='Experiment name')

    # -----------------------------------------------------------
    # Learning rates
    # -----------------------------------------------------------
    parser.add_argument('--intrinsics_lr', type=float, default=1e-3, help='LR for intrinsics')
    parser.add_argument('--cp_translation_dyn_lr', type=float, default=1e-3,
                        help='LR for dynamic point translation')
    parser.add_argument('--uncertainty_lr', type=float, default=1e-4, help='LR for uncertainty estimation')
    parser.add_argument('--ba_lr', type=float, default=1e-2, help='LR for bundle adjustment')

    # -----------------------------------------------------------
    # Training epochs / schedule
    # -----------------------------------------------------------
    parser.add_argument('--num_init_epochs', type=int, default=100, help='Initial optimization epochs')
    parser.add_argument('--num_BA_epochs', type=int, default=100, help='Bundle adjustment epochs')
    parser.add_argument('--num_motion_start_epochs', type=int, default=100,
                        help='Epochs before motion optimization begins')
    parser.add_argument('--num_dyn_epochs', type=int, default=100, help='Dynamic optimization epochs')

    # -----------------------------------------------------------
    # Paths: depth / pose / mask / trackers
    # -----------------------------------------------------------
    parser.add_argument('--depth_dir', type=str, default="unidepth",
                        help='Directory for predicted depth')
    parser.add_argument('--pose_path', type=str, default="vggt",
                        help='Directory for predicted pose')
    parser.add_argument('--cotracker_path', type=str, default="cotracker",
                        help='CoTracker type or path')
    parser.add_argument('--dyn_mask_dir', type=str, help='Directory for dynamic masks')
    parser.add_argument('--deva_dir', type=str, default="deva", help='Directory for DEVA segmentation')
    parser.add_argument('--video', type=str, help='Which video file to process')

    # -----------------------------------------------------------
    # Loss weights
    # -----------------------------------------------------------
    parser.add_argument('--reproj_weight', type=float, help='Weight for reprojection error')
    parser.add_argument('--pose_smooth_weight_t', type=float, help='Pose smoothness weight for translation')
    parser.add_argument('--pose_smooth_weight_r', type=float, help='Pose smoothness weight for rotation')
    parser.add_argument('--dyn_smooth_weight_t', type=float, help='Dynamic translation smoothness weight')
    parser.add_argument('--dyn_laplacian_weight_t', type=float, help='Dynamic Laplacian weight')

    # -----------------------------------------------------------
    # Flags / Options
    # -----------------------------------------------------------
    parser.add_argument('--log', action='store_true', default=False, help='Log to file instead of console')
    parser.add_argument('--opt_intrinsics', action='store_true', default=True,
                        help='Optimize camera intrinsics')
    parser.add_argument('--vis_4d', action='store_true', default=False, help='Visualize 4D trajectory')
    parser.add_argument('--optimize_dyn_upsample', action='store_true',
                        help='Optimize dynamic upsample module')
    parser.add_argument('--save_upsample', action='store_true',
                        help='Save upsampled outputs')
    parser.add_argument('--use_sampson', action='store_true',
                        help='Use Sampson distance instead of reprojection error')

    # -----------------------------------------------------------
    # Randomness & Logging
    # -----------------------------------------------------------
    parser.add_argument('--print_every', type=int, default=20, help='Print frequency')
    parser.add_argument('--deterministic', action='store_true',
                        help='Deterministic mode for reproducibility')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # -----------------------------------------------------------
    # Other configs
    # -----------------------------------------------------------
    parser.add_argument('--loss_fn', type=str, help='Loss function type')
    parser.add_argument('--depth_version', type=str, default='v2', help='Depth version')
    parser.add_argument('--global_downsample_rate', type=int, default=1,
                        help='Global downsample rate')


    if input_string is not None:
        opt = parser.parse_args(input_string)
    else:
        opt = parser.parse_args()

    return opt

def train_from_opt(opt):

    engine = Engine(opt)
    engine.initialize()

    engine.optimize_init_sliding()
    engine.log_timer("init")

    engine.optimize_BA()
    engine.reinitialize_static()             # add more static points
    engine.log_timer("BA")

    if engine.num_points_dyn > 0:
        if opt.optimize_dyn_upsample and opt.save_upsample:
            engine.init_dyn_cp_upsample()
            engine.optimize_dyn()
            engine.filter_dyn()
        else:
            engine.init_dyn_cp()
            engine.optimize_dyn()
            if opt.save_upsample:
                engine.upsample_dyn()
                engine.filter_upsample_dyn()
            else:
                engine.filter_dyn()
            engine.log_timer("dyn") 

    if not opt.save_upsample:
        engine.save_tracks(save_upsample_staic=False) 
    elif opt.optimize_dyn_upsample:
        engine.save_tracks(save_upsample_staic=True) 
    else:
        engine.save_upsample_tracks()

    del engine

def main():
    opt = parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    os.environ['NUMEXPR_MAX_THREADS'] = '32'

    if opt.video == "None":
        videos = sorted(os.listdir(f"{opt.workdir}"))

        for i, video in enumerate(videos):
          if video in ['libby', 'dog']:
            print(f"Working on {video}", flush=True)

            opt.video_name = video
            
            train_from_opt(opt)

            torch.cuda.empty_cache()

    else:

        print(f"Working on {opt.video}", flush=True)

        opt.video_name = opt.video
        
        train_from_opt(opt)

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()