import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "submodules", "UniDepth")  # adjust if needed
sys.path.append(os.path.abspath(project_root))

import torch
import imageio
import numpy as np
import glob
from unidepth.models import UniDepthV2, UniDepthV2old
from unidepth.utils.camera import Pinhole, BatchCamera
import argparse
from tqdm import tqdm
import time
import cv2

parser = argparse.ArgumentParser(description='Running unidepth')
parser.add_argument('--workdir',
                    metavar='DIR',
                    help='path to dataset',
                    default="./data_preprocessed/sintel_test")
parser.add_argument('--v2',
                    action='store_true',
                    help='use UniDepthV2')
parser.add_argument('--use_gt_K',
                    action='store_true',
                    help='use ground truth intrinsics')

args = parser.parse_args()
BASE = args.workdir

videos = sorted([x for x in sorted(os.listdir(BASE))])

if args.v2:
    model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    print("using v2")
else:
    model = UniDepthV2old.from_pretrained(f"lpiccinelli/unidepth-v2old-vitl14")

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

start_time = time.time()

for video in tqdm(videos, desc="Predicting Unidepth"):
  if video in ['rollerblade']:
    OUTPUT_BASE = os.path.join(BASE, video, "unidepth")

    if args.use_gt_K:
        OUTPUT_BASE += "_gt_K"
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(BASE, video, "color", "*.png")) + glob.glob(os.path.join(BASE, video, "color", "*.jpg")))

    images = []

    for image_path in image_paths:
        
        im = imageio.imread(image_path)
        images.append(im)
        
    images = torch.tensor(np.array(images)) # F x H x W x 3
    f, h, w, c = images.shape
    batch_size = 8

    if not args.use_gt_K:

        with torch.no_grad():
                                
            frame_range = list(range(f))
            chuncks = [frame_range[i:i+batch_size] for i in range(0, len(frame_range), batch_size)]
            initial_intrinsics = []
            for chunk in chuncks:
                imgs = images[chunk, ...]
                imgs = torch.permute(imgs, (0, 3, 1, 2))
                preds = model.infer(imgs)
                initial_intrinsics.append(preds['intrinsics'])

        initial_intrinsics = torch.cat(initial_intrinsics, dim=0)
        initial_intrinsics = torch.mean(initial_intrinsics, dim=0)

        K = initial_intrinsics.detach().cpu()

        K[0][-1] = w / 2
        K[1][-1] = h / 2

    else:

        K = torch.from_numpy(np.load(os.path.join(BASE, video, "K.npy"))).float()
        if len(K.shape) == 2:
            K = K.unsqueeze(0).repeat(imgs.shape[0], 1, 1)
        
    intrinsics_save_path = os.path.join(OUTPUT_BASE, 'intrinsics.npy')
    depth_save_path_disp = os.path.join(OUTPUT_BASE, 'depth.npy')

    np.save(intrinsics_save_path, K)

    if len(K.shape) == 2:
        K = K.unsqueeze(0).repeat(images.shape[0], 1, 1)

    depths = []

    with torch.no_grad():
                        
        frame_range = list(range(f))
        chuncks = [frame_range[i:i+batch_size] for i in range(0, len(frame_range), batch_size)]
        for chunk in chuncks:
            imgs = images[chunk, ...]
            Ks = K[0:1, ...]
            # print(Ks.shape)
            if isinstance(model, (UniDepthV2)):
                Ks = Pinhole(K=Ks)
                cam = BatchCamera.from_camera(Ks)
            imgs = torch.permute(imgs, (0, 3, 1, 2))
            # print(imgs.shape)
            preds = model.infer(imgs, Ks)
            depth = preds['depth']                                         # B x 1 x H x W
            depths.append(depth)

        depths = torch.cat(depths, dim=0)
        depths = depths.detach().cpu().numpy()

    depth_vis = depths[:,0,:,:]
    disp_vis = 1/(depth_vis + 1e-12)
    disp_vis = cv2.normalize(disp_vis,  disp_vis, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    for f in range(len(disp_vis)):
        disp = disp_vis[f]
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)  
        imageio.imwrite(os.path.join(OUTPUT_BASE, str(f).zfill(4)+".png"), disp)
    np.save(depth_save_path_disp, depths)
