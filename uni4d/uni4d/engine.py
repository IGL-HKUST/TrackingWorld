import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import cv2
import itertools

import logging
import json

import imageio
import matplotlib.pyplot as plt
from importlib import reload
from collections import defaultdict
from torchvision import transforms
import torch.nn.functional as F

from pytorch3d.ops import knn_points, ball_query

from util import clean_depth_outliers, EarlyStopper, KeyFrameBuffer, l1_loss_with_uncertainty, flow_norm

from variables import CameraPoseDeltaCollection, ControlPoints, CameraIntrinsics, ControlPointsDynamic, ControlPointsMotion, UncertaintyCollection, _so3_log_map
import concurrent.futures
import os
import wandb

import random

from tqdm import tqdm
import time

import multiprocessing as mp
from functools import partial
#import madpose
# from vis.vis_cotracker import vis
mp.set_start_method('spawn', force=True)
class Engine():

    def __init__(self, opt) -> None:
        self.opt = opt
        self.dataset = opt.workdir.split('/')[-1]
        self.depth_version = opt.depth_version
        if opt.device == "cuda":
            self.device = torch.device("cuda")


    def init_vars(self) -> None:

        """
        Loads in the poses for optimization and initializes starting point using pred depth and pose
        """
        self.flow_norm_l1 = False

        if self.opt.loss_fn == "l1":
            self.loss_fn = l1_loss_with_uncertainty
        else:
            raise ValueError("Invalid loss type")

        self.controlpoints_static = ControlPoints(number_of_points=self.num_points_static)

        self.motion_static = ControlPointsDynamic(number_of_points=self.num_points_static, number_of_frames=self.num_frames, with_norm=False)
        self.controlpoints_dyn = ControlPointsDynamic(number_of_points=self.num_points_dyn, number_of_frames=self.num_frames, with_norm=False)
        self.poses = CameraPoseDeltaCollection(self.num_frames)

        self.conf_vars_static = UncertaintyCollection(number_of_points=self.num_points_static, number_of_frames=self.num_frames)
        self.conf_vars_static.set_confidence(self.tracking_confidences[:, self.all_labels==0])

        if self.opt.opt_intrinsics:
            self.intrinsics = CameraIntrinsics(self.K[0,0], self.K[1,1])
            self.intrinsics.register_shape(self.shape)
            
    
    @torch.no_grad()
    def get_track_depths(self):

        """
        Instantiates the depth for each track
        """
        _, _, H, W  = self.depth.shape

        coords_normalized_static = torch.zeros_like(self.all_tracks_static)
        coords_normalized_static[:, :, 0] = 2.0 * self.all_tracks_static[:, :, 0] / (W - 1) - 1.0
        coords_normalized_static[:, :, 1] = 2.0 * self.all_tracks_static[:, :, 1] / (H - 1) - 1.0

        coords_normalized_dyn = torch.zeros_like(self.all_tracks_dyn)
        coords_normalized_dyn[:, :, 0] = 2.0 * self.all_tracks_dyn[:, :, 0] / (W - 1) - 1.0
        coords_normalized_dyn[:, :, 1] = 2.0 * self.all_tracks_dyn[:, :, 1] / (H - 1) - 1.0

        coords_normalized_static_downsample = torch.zeros_like(self.all_tracks_static_downsample)
        coords_normalized_static_downsample[:, :, 0] = 2.0 * self.all_tracks_static_downsample[:, :, 0] / (W - 1) - 1.0
        coords_normalized_static_downsample[:, :, 1] = 2.0 * self.all_tracks_static_downsample[:, :, 1] / (H - 1) - 1.0

        coords_normalized_static = coords_normalized_static.unsqueeze(2)
        coords_normalized_dyn = coords_normalized_dyn.unsqueeze(2)
        coords_normalized_static_downsample = coords_normalized_static_downsample.unsqueeze(2)

        self.all_tracks_static_depth = F.grid_sample(self.depth, coords_normalized_static, align_corners=True).squeeze(1).squeeze(-1)     # F x N
        self.all_tracks_dyn_depth = F.grid_sample(self.depth, coords_normalized_dyn, align_corners=True).squeeze(1).squeeze(-1)           # F x N
        self.all_tracks_static_depth_downsample = F.grid_sample(self.depth, coords_normalized_static_downsample, align_corners=True).squeeze(1).squeeze(-1)           # F x N

    @torch.no_grad()
    def init_dyn_cp(self) -> None:

        """
        Initializes values for dynamic control points once we have some pose
        """

        for f in range(self.num_frames):   # initializes values for dynamic points

            R, t = self.get_poses([f])
            R = R[0]
            t = t[0]

            track_2d = self.all_tracks_dyn[f, :]
            vis_2d = self.all_vis_dyn[f, :]

            depth = self.get_depth(f)

            norm = None
            
            ts, ns = self.reproject(R, t, track_2d, vis_2d, depth, f, norm) # N x 3

            self.controlpoints_dyn.set_translation(f, ts)
    
    @torch.no_grad()
    def init_dyn_cp_upsample(self) -> None:

        """
        Initializes values for dynamic control points once we have some pose
        """
        all_dyn_filter = self.all_labels_full == 1
        self.dyn_mask_final = all_dyn_filter

        self.all_tracks_dyn = self.all_tracks_full[:, all_dyn_filter]
        self.track_init_frames_dyn = self.track_init_frames_full[all_dyn_filter]
        self.all_vis_dyn = self.all_vis_full[:, all_dyn_filter]
        self.num_points_dyn = self.all_tracks_dyn.shape[1]
        self.controlpoints_dyn = ControlPointsDynamic(number_of_points=self.num_points_dyn, number_of_frames=self.num_frames, with_norm=False).to(self.device) 
        translation_dyn_params, _ = self.controlpoints_dyn.get_params()
        self.t_dyn_optimizer = torch.optim.Adam([translation_dyn_params], lr=self.opt.cp_translation_dyn_lr, weight_decay=0, amsgrad=True)
        self.optimizers['points_dyn_t'] = self.t_dyn_optimizer
        if self.depth_version == 'v2':
            self.get_track_depths()
        elif self.depth_version == 'v1':
            self.all_tracks_dyn_depth = self.all_tracks_depth_full[:, all_dyn_filter, 0]

        for f in range(self.num_frames):   # initializes values for dynamic points

            R, t = self.get_poses([f])
            R = R[0]
            t = t[0]

            track_2d = self.all_tracks_dyn[f, :]
            vis_2d = self.all_vis_dyn[f, :]

            depth = self.get_depth(f)

            norm = None
            
            ts, ns = self.reproject(R, t, track_2d, vis_2d, depth, f, norm) # N x 3

            self.controlpoints_dyn.set_translation(f, ts)

    def get_depth(self, frame_idx=None):

        """
        Function to get depth, add scale here
        """
        depth = self.depth[[frame_idx]]
        depth = depth[0][0]

        assert(len(depth.shape) == 2)

        return depth

    def get_poses(self, frame_idx):

        """
        Function to get pose, used when gt is used
        """

        Rs, ts = self.poses(frame_idx)

        return Rs, ts

    def project(self, R, t, K, points_3d):

        """
        Projects 3d points into 2d points
        R -> 3 x 3
        t -> 3 x 1
        points_3d -> N x 3 -> points in world coordinate
        depth -> bool -> if True, returns depth as well
        """

        points_3d_cam = (torch.linalg.inv(R) @ (points_3d.T - t)).T
        points_3d_depth = points_3d_cam[:,2]

        points_2d_homo = (K @ points_3d_cam.T).T

        points_2d_homo = (points_2d_homo / (points_2d_homo[:,2:] + 1e-16))[:,:2]

        return points_2d_homo, points_3d_depth

    def reproject(self, R, t, track_2d, vis_2d, depth, f, norm = None):

        """
        Reprojects 2d points into 3d points
        R -> 3 x 3
        t -> 3 x 1
        track_2d -> N x 2, stored as x y values
        vis_2d -> N
        depth -> H x W
        """

        N,_ = track_2d.shape
        K = self.get_intrinsics_K([f])[0]

        points_2d_homo = torch.concatenate((track_2d, torch.ones((N, 1)).to(self.device)), axis=-1)

        ray = (torch.linalg.inv(K) @ points_2d_homo.T).T

        ray_norm = ray / ray[:,-1].unsqueeze(-1) # N x 3

        #if self.all_tracks_dyn_depth
        if self.depth_version == 'v2':
            points_depth = depth[track_2d[:,1].long(), track_2d[:,0].long()]
        elif self.depth_version == 'v1':
            points_depth = self.all_tracks_dyn_depth[f]

        # self.all_tracks_dyn_depth[f]
        points_3d_src = ray_norm * (points_depth.unsqueeze(-1)) # N x 3

        points_3d_world_t = (R @ points_3d_src.T).T + t[:,0]

        if norm is not None:
            points_3d_world_n = (R @ norm.T).T
        else:
            points_3d_world_n = None

        return points_3d_world_t, points_3d_world_n # N x 3

    def reset_optimizer(self, lr=1e-3, patience=10):

        for optimizer in self.active_optimizers:
            del self.optimizers[optimizer]
            if optimizer in self.schedulers:
                del self.schedulers[optimizer]

        self.init_optimization(lr=lr, patience=patience)

    def filter_depth_edges(self, depth):

        """
        Filter depth edges using gradients
        """

        depth_image = depth.detach().cpu().numpy()

        # Compute the gradients along the x and y axes
        grad_x = cv2.Sobel(depth_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient along x-axis
        grad_y = cv2.Sobel(depth_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient along y-axis

        # Optionally, compute the gradient magnitude (Euclidean distance)
        grad_magnitude = cv2.magnitude(grad_x, grad_y)

        threshold = np.percentile(grad_magnitude, 95)

        mask = grad_magnitude < threshold

        return mask

    def fuse_static_points(self, save_path):

        """
        Fuses depth point cloud into a static point cloud
        """

        import open3d as o3d

        X, Y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        coords = torch.tensor(np.stack([X,Y], axis=-1).reshape(-1, 2)).to(self.device)
        coords_2d_homo = torch.concatenate((coords, torch.ones((len(coords),1)).float().to(self.device)), dim=1)  # N x 3
        
        points = np.zeros((0, 3))
        rgbs = np.zeros((0, 3), np.uint8)

        dyn_points =[]
        dyn_rgbs = []
        Rs, ts = self.get_poses(torch.arange(self.num_frames))
        Ks = self.get_intrinsics_K(torch.arange(self.num_frames))
        
        for f in range(self.num_frames):

            R = Rs[f]
            t = ts[f]
            K = Ks[f]

            src_ray =  (torch.linalg.inv(K) @ coords_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            depth = self.reproj_depths[f]
            edge_mask = self.filter_depth_edges(depth)
            edge_mask = edge_mask.flatten()
            
            static_mask = (self.dyn_masks_filters[f] == 0).flatten()
            dyn_mask = (self.dyn_masks_filters[f] == 1).flatten()
            static_mask = np.logical_and(static_mask, edge_mask)
            dyn_mask = np.logical_and(dyn_mask, edge_mask)

            points_3d_src = src_ray_homo * depth.flatten().unsqueeze(-1)
            points_3d_world = (R @ points_3d_src.T + t).T
            rgb = self.images[f].reshape((-1, 3))

            points_3d_world_static = points_3d_world[static_mask]
            rgb_static = rgb[static_mask]
            points_3d_world_dyn = points_3d_world[dyn_mask]
            rgb_dyn = rgb[dyn_mask]

            points_3d_world_static = points_3d_world_static[::100]
            rgb_static = rgb_static[::100]

            points = np.vstack([points, points_3d_world_static.detach().cpu().numpy()])
            rgbs = np.vstack([rgbs, rgb_static.detach().cpu().numpy()])

            dyn_points.append(points_3d_world_dyn.detach().cpu().numpy())
            dyn_rgbs.append(rgb_dyn.detach().cpu().numpy())

        c2w = np.zeros((self.num_frames, 4, 4))
        c2w[:,:3,:3] = Rs.detach().cpu().numpy()
        c2w[:,:3,3] = ts.squeeze(-1).detach().cpu().numpy()

        output = {
            "static_points": points,
            "static_rgbs": rgbs,
            "dyn_points": np.array(dyn_points, dtype=object),
            "dyn_rgbs": np.array(dyn_rgbs, dtype=object),
            "c2w": c2w,
            "rgbs": self.images.detach().cpu().numpy(),
            "Ks": Ks.detach().cpu()
        }

        np.savez_compressed(save_path, **output)

    def save_results(self,save_fused_points=False):

        """
        Saves the results
        """

        self.depth_interpolate()

        if save_fused_points:
            self.fuse_static_points(os.path.join(self.opt.output_dir, "fused_4d.npz"))   # fuse depth maps into global point clouds and saves

        for f in range(self.num_frames):
            img = self.images[f].detach().cpu().numpy().squeeze()
            R,t = self.get_poses([f])
            R = R.detach().cpu().numpy().squeeze()
            t = t.detach().cpu().numpy().squeeze()
            K = self.get_intrinsics_K([f]).detach().cpu().numpy()[0]
            depth_interp = self.reproj_depths[f].detach().cpu().numpy()
            disp_aligned = (1 / (depth_interp + 1e-16))

            output_dict = {  # DISP is saved in opt_shape
                'disp': disp_aligned,
                'img': img,
                'R': R,
                't': t,
                'K': K
            }
            np.savez_compressed(os.path.join(self.opt.output_dir, f"{str(f).zfill(4)}.npz"), **output_dict)
    
    def save_upsample_tracks(self, save_fused_points=False):

        """
        Saves the results
        """

        #self.depth_interpolate()
        control_pts_static = self.controlpoints_static_full.clone()
        static_rgbs = self.all_tracks_color_full[self.all_static_filter_final_full.cpu().numpy()]
        control_pts_static = control_pts_static[~self.static_isolated_points_full.cpu().numpy()]
        static_rgbs = static_rgbs[~self.static_isolated_points_full.cpu().numpy()]

        control_pts_dyn = self.controlpoints_dynamic_full.clone()
        control_pts_dyn_3D_world = control_pts_dyn.squeeze(1)  
        dyn_rgbs = self.all_tracks_color_full[self.dyn_mask_final_full.cpu().numpy()]

        Rs, ts = self.get_poses(torch.arange(self.num_frames))
        Ks = self.get_intrinsics_K(torch.arange(self.num_frames))
        
        c2w = np.zeros((self.num_frames, 4, 4))
        c2w[:,:3,:3] = Rs.detach().cpu().numpy()
        c2w[:,:3,3] = ts.squeeze(-1).detach().cpu().numpy()
        output = {
            "static_points": control_pts_static.detach().cpu().numpy(),
            "static_rgbs": static_rgbs,
            "track_init_frames_static": self.track_init_frames_full_static[~self.static_isolated_points_full].detach().cpu().numpy(),
            "dyn_points": control_pts_dyn_3D_world.detach().cpu().numpy(),
            "dyn_rgbs": dyn_rgbs,
            "dyn_vis": self.all_vis_full_dyn.detach().cpu().numpy(),
            'track_init_frames_dyn': self.track_init_frames_dyn.detach().cpu().numpy(),
            "c2w": c2w,
            "rgbs": self.images.detach().cpu().numpy(),
            "Ks": Ks.detach().cpu().numpy()
        }

        np.savez_compressed(os.path.join(self.opt.output_dir, "fused_track_4d_full.npz"), **output)

    def save_tracks(self, save_upsample_staic=False):

        """
        Saves the results
        """

        if not save_upsample_staic: 
            control_pts_static = self.controlpoints_static.forward()
            motion_world, _ = self.motion_static()
            static_filter =  motion_world.abs().max(1)[0].max(-1)[0] < 0.5
            motion_world[static_filter] = motion_world[static_filter].mean(1)[:, None].repeat(1, motion_world.shape[1], 1)
            dyn_filter_vis = self.all_vis_static[:, self.prev_idx][:, ~static_filter]
            update_motion = propagate_motion(motion_world[~static_filter], dyn_filter_vis)
            motion_world[~static_filter] = update_motion
            control_pts_static = control_pts_static.unsqueeze(1).repeat(1, self.num_frames, 1)
            control_pts_static[self.prev_idx] = control_pts_static[self.prev_idx] + motion_world

            static_rgbs = self.all_tracks_color[self.all_static_filter_final.cpu().numpy()]
            control_pts_static = control_pts_static[~self.static_isolated_points.cpu().numpy()]
            static_rgbs = static_rgbs[~self.static_isolated_points.cpu().numpy()]
            track_init_frames_static = self.track_init_frames_static[~self.static_isolated_points].detach().cpu().numpy()
        else:
            control_pts_static = self.controlpoints_static_full.clone()
            static_rgbs = self.all_tracks_color_full[self.all_static_filter_final_full.cpu().numpy()]
            control_pts_static = control_pts_static[~self.static_isolated_points_full.cpu().numpy()]
            static_rgbs = static_rgbs[~self.static_isolated_points_full.cpu().numpy()]
            track_init_frames_static = self.track_init_frames_full_static[~self.static_isolated_points_full].detach().cpu().numpy()


        control_pts_dyn, _ = self.controlpoints_dyn.forward()
        control_pts_dyn_3D_world = control_pts_dyn.squeeze(1) 
        if self.opt.save_upsample:
            dyn_rgbs = self.all_tracks_color_full[self.dyn_mask_final.cpu().numpy()]
        else:
            dyn_rgbs = self.all_tracks_color[self.dyn_mask_final.cpu().numpy()]
        Rs, ts = self.get_poses(torch.arange(self.num_frames))
        Ks = self.get_intrinsics_K(torch.arange(self.num_frames))
        
        c2w = np.zeros((self.num_frames, 4, 4))
        c2w[:,:3,:3] = Rs.detach().cpu().numpy()
        c2w[:,:3,3] = ts.squeeze(-1).detach().cpu().numpy()
        output = {
            "static_points": control_pts_static.detach().cpu().numpy(),
            "static_rgbs": static_rgbs,
            "track_init_frames_static": track_init_frames_static,
            "dyn_points": control_pts_dyn_3D_world.detach().cpu().numpy(),
            "dyn_rgbs": dyn_rgbs,
            "dyn_vis": self.all_vis_dyn.detach().cpu().numpy(),
            'track_init_frames_dyn': self.track_init_frames_dyn.detach().cpu().numpy(),
            "c2w": c2w,
            "rgbs": self.images.detach().cpu().numpy(),
            "Ks": Ks.detach().cpu().numpy()
        }
        if not save_upsample_staic: 
          np.savez_compressed(os.path.join(self.opt.output_dir, "fused_track_4d.npz"), **output)
        else:
          np.savez_compressed(os.path.join(self.opt.output_dir, "fused_track_4d_full.npz"), **output)

    def statistical_outlier_removal(self, point_cloud, k=4, std_dev_multiplier=3.0):
        # point_cloud: (N, 3) tensor of points
        N = point_cloud.shape[0]
        
        # Compute k-nearest neighbors
        _, idx, _ = knn_points(point_cloud[None, ...], point_cloud[None, ...], K=k)
        idx = idx[0]  # (N, k)
        
        # Calculate mean distance to neighbors
        neighbor_points = point_cloud[idx]  # (N, k, 3)
        mean_distances = torch.mean(torch.norm(neighbor_points - point_cloud[:, None, :], dim=2), dim=1)
        
        # Calculate threshold distance
        mean = torch.mean(mean_distances)
        std_dev = torch.std(mean_distances)
        threshold = mean + std_dev_multiplier * std_dev
        
        # Filter out points with mean distance above threshold
        mask = mean_distances < threshold
        filtered_points = point_cloud[mask]
        
        return filtered_points, mask

    def fuse_helper(self, R, t, depth, label, depthmap_res, f, coords, src_ray_homo):

        """
        Fuses results using label
        """

        if label == 0:

            if self.opt.save_upsample:
                vis = self.all_vis_full_static[f]   
                track_sel = torch.where(vis)[0]
                control_pts_3D_world = self.controlpoints_static_full[track_sel, f]
            else:
                vis = self.all_vis_static[f]             # N
                track_sel = torch.where(vis)[0]
                motion_world, _ = self.motion_static()
                static_filter =  motion_world.abs().max(1)[0].max(-1)[0] < 0.3
                motion_world[static_filter] = motion_world[static_filter].mean(1)[:, None].repeat(1, motion_world.shape[1], 1)
                
                dyn_filter_vis = self.all_vis_static[:, self.prev_idx][:, ~static_filter]
                update_motion = propagate_motion(motion_world[~static_filter], dyn_filter_vis)
                motion_world[~static_filter] = update_motion
                control_pts_static = self.controlpoints_static.forward()
                control_pts_static[self.prev_idx] = control_pts_static[self.prev_idx] + motion_world[:, f]

                control_pts_3D_world = control_pts_static[track_sel]

        else:

            track_dyn = self.all_tracks_dyn[f]
            vis = self.all_vis_dyn[f]
            track_dyn = track_dyn[vis]
            track_sel_dyn = torch.where(vis)[0]
            control_pts, _  = self.controlpoints_dyn(frames=f, points=track_sel_dyn)    # N x 1 x 3
            control_pts_3D_world = control_pts.squeeze(1)                                    # N x 3

        mask = self.dyn_masks[f] == label

        if len(control_pts_3D_world) == 0:
            depthmap_res[mask] = depth[mask]     # no alignment if we have no control points
            return
    
        K = self.get_intrinsics_K([f])[0]
        control_pts_2D, control_points_depth = self.project(R, t, K, control_pts_3D_world)
        
        in_range = torch.logical_and(torch.logical_and(control_pts_2D[:,0] >= 0, control_pts_2D[:,0] < self.shape[1]), torch.logical_and(control_pts_2D[:,1] >= 0, control_pts_2D[:,1] < self.shape[0]))   # filter in-view after projection
        in_range = torch.logical_and(in_range, control_points_depth>0)

        control_pts_2D = control_pts_2D[in_range]
        control_points_depth = control_points_depth[in_range]
        unidepth_vals = self.interpolate(depth.unsqueeze(-1), control_pts_2D).squeeze(-1)

        if torch.sum(in_range) == 0:
            depthmap_res[mask] = depth[mask]     # no alignment if we have no control points
            return
        
        depth_src = depth[coords[:,1].long(), coords[:,0].long()]

        points_3d_src = (src_ray_homo * depth_src.unsqueeze(-1))   # N x 3
        points_3d_world = (R @ points_3d_src.T + t).T              # 3d reprojection of unidepth for this frame, used to find knn

        points_3d_world_grid = points_3d_world.reshape((self.shape[0], self.shape[1], 3))
        control_pts_3D_world_unidepth = points_3d_world_grid[control_pts_2D[:, 1].long(), control_pts_2D[:,0].long()]  # axis seems to be flipped

        points_3d_query = points_3d_world[mask.flatten()]

        knn_result = knn_points(points_3d_query.unsqueeze(0), control_pts_3D_world_unidepth.unsqueeze(0),  K=3)   # we find nn using reproject unidepth 3D instead

        idx = knn_result.idx[0]  # Shape: (num_points1, k)
        dists = knn_result.dists[0]

        weights = 1 / (dists + 1e-8) 

        weights_sum = weights.sum(dim=-1, keepdim=True)
        probabilities = weights / (weights_sum)
        scale = control_points_depth / unidepth_vals

        scales = torch.sum(scale[idx] * probabilities, axis=1)
        # scales[interpolation_mask] = 1.0
    
        depthmap_res[mask] = depth[mask] * scales

        if torch.sum(scales < 0) > 0:
            breakpoint()


    def depth_interpolate(self):

        """
        Scale interpolation using 3d knn in unidepth reprojection
        """

        self.reproj_depths = []

        X, Y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
        coords = torch.tensor(np.stack([X,Y], axis=-1).reshape(-1, 2)).to(self.device)
        coords_2d_homo = torch.concatenate((coords, torch.ones((len(coords),1)).float().to(self.device)), dim=1)  # N x 3
        
        if self.opt.save_upsample:
            scene_points = self.controlpoints_static_full.mean(1)
        else:
            scene_points = self.controlpoints_static()
        self.scene_scale = torch.mean(torch.quantile(scene_points, 0.9, dim=0) - torch.quantile(scene_points, 0.1, dim=0))

        for f in range(self.num_frames):

            Rs, ts = self.get_poses([f])
            R = Rs[0]
            t = ts[0]

            K = self.get_intrinsics_K([f])[0]
            src_ray =  (torch.linalg.inv(K) @ coords_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            depth = self.get_depth(f)

            depthmap_res = torch.zeros_like(depth)

            labels = torch.unique(self.dyn_masks[f])    # get unique labels for this frame

            for label in labels:

                self.fuse_helper(R, t, depth, label, depthmap_res, f, coords, src_ray_homo)

            self.reproj_depths.append(depthmap_res.detach())   # add res depthmap

        self.reproj_depths = torch.stack(self.reproj_depths)

        assert(torch.sum(torch.isnan(self.reproj_depths))==0)
        if torch.sum(self.reproj_depths==0)> 0:
            breakpoint()
        
    def get_masks(self):

        """
        Returns dyn_mask where
        0 -> static
        1 -> dynamic
        2 -> rejected

        Also erode and inflate dynamic masks for better boundaries
        """
        deva_anot_path = os.path.join(self.opt.BASE, self.opt.dyn_mask_dir, "Annotations")
        mask_paths = sorted(os.listdir(deva_anot_path))

        all_masks = []

        for mask_path in mask_paths:
            if mask_path.endswith(".npy"):
                mask_dyn = np.load(os.path.join(deva_anot_path, mask_path))
                all_masks.append(mask_dyn)
            elif mask_path.endswith(".png"):
                mask_dyn = imageio.imread(os.path.join(deva_anot_path, mask_path))
                if len(mask_dyn.shape) == 3:
                    mask_dyn = mask_dyn.astype(np.int64)
                    mask_dyn = mask_dyn[:,:,0] + mask_dyn[:,:,1] * 256 + mask_dyn[:,:,2] * 256 * 256
                all_masks.append(mask_dyn)

        masks = np.array(all_masks)

        _, masks_unique = np.unique(masks, return_inverse=True)
        dyn_masks = masks_unique.reshape(masks.shape)          # 0 -> static, 1+ -> dynamic

        dyn_masks_filters = []

        for mask_dyn in dyn_masks:

            mask_dyn_pro = np.zeros(masks.shape[1:]) + 2
            static_mask = cv2.erode((mask_dyn==0).astype(np.uint8), np.ones((4,4), np.uint8), iterations=3).astype(np.bool_)
            mask_dyn_pro[static_mask] = 0
            dyn_mask = cv2.erode((mask_dyn!=0).astype(np.uint8), np.ones((4,4), np.uint8), iterations=2).astype(np.bool_)
            mask_dyn_pro[dyn_mask] = 1
            dyn_masks_filters.append(mask_dyn_pro)

        dyn_masks_filters = np.array(dyn_masks_filters)
        dyn_masks_filters = dyn_masks_filters.astype(np.uint8)
        
        self.dyn_masks = torch.tensor(dyn_masks)
        self.dyn_masks_filters = dyn_masks_filters
    
    def float_to_heat_rgb(self, value):
        """
        Convert a float in the range 0-1 to an RGB color based on the heat colormap.
        
        Parameters:
        value (float): A float value between 0 and 1.
        
        Returns:
        tuple: A tuple representing the RGB color.
        """
        if not 0 <= value <= 1:
            raise ValueError("Value must be within the range [0, 1]")
        
        # Get the colormap
        colormap = plt.get_cmap('cool')
        
        # Get the color corresponding to the value
        rgba = colormap(value)
        
        # Convert the RGBA color to RGB
        rgb = tuple(int(255 * c) for c in rgba[:3])
        
        return rgb

    def load_depths(self):

        """
        Get depth map and intrinsics from unidepth
        """

        self.depth = []

        depth_save_base = os.path.join(self.opt.workdir, self.opt.video_name, self.opt.depth_dir)

        depth_save_path_depth =  os.path.join(depth_save_base, "depth.npy")
        intrinsics_save_path = os.path.join(depth_save_base, "intrinsics.npy")

        self.depth = torch.tensor(np.load(depth_save_path_depth))#[:32]

        if len(self.depth.shape)==3:
            self.depth = self.depth[:, None].float()

        if self.opt.opt_intrinsics:
            assert(os.path.exists(intrinsics_save_path))
            self.K = torch.tensor(np.load(intrinsics_save_path))
        else:
            self.K = self.K_gt

        if self.all_tracks.shape[-1] == 2:
            self.get_track_depths()
        else:
            if self.depth_version == 'v2':
                # v2
                self.all_tracks_static = self.all_tracks_static[..., :2]
                self.all_tracks_dyn = self.all_tracks_dyn[..., :2]
                self.all_tracks_static_downsample = self.all_tracks_static_downsample[..., :2]
                self.get_track_depths()
            elif self.depth_version == 'v1':
                # v1
                self.all_tracks_static_depth = self.all_tracks_static[..., 2]
                self.all_tracks_dyn_depth = self.all_tracks_dyn[..., 2]
                self.all_tracks_static = self.all_tracks_static[..., :2]
                self.all_tracks_dyn = self.all_tracks_dyn[..., :2]
                self.all_tracks_static_depth_downsample = self.all_tracks_static_downsample[..., 2]
                self.all_tracks_static_downsample = self.all_tracks_static_downsample[..., :2]


    def load_images(self):

        image_paths_base = os.path.join(self.opt.BASE, "color")

        image_paths = sorted([os.path.join(image_paths_base, x) for x in os.listdir(image_paths_base) if x.endswith(".png") or x.endswith(".jpg")])

        self.images = []

        for image_path in image_paths:
            
            im = imageio.imread(image_path)
            self.images.append(im)
            
        self.images = torch.tensor(np.array(self.images)) # F x H x W x 3

        _, self.height, self.width, _ = self.images.shape
        self.shape = (self.height, self.width)

        self.num_frames = self.images.shape[0]

    def load_tracklets(self):

        cotracker_dir = self.opt.cotracker_path
        global_downsample_rate = self.opt.global_downsample_rate

        cotrack_results = np.load(os.path.join(self.opt.BASE, cotracker_dir, "results.npz"))
        
        self.all_tracks_full = cotrack_results["all_tracks"]   # F x N x 2 (x,y)
        self.track_init_frames_full = cotrack_results["init_frames"] 
        self.all_vis_full = cotrack_results["all_visibilities"][..., 0]

        self.filter_cotrackerv2()

        if global_downsample_rate > 1:
            self.selected_indices, self.merged_dict = filter_integer_coords_v2((self.all_tracks_full.copy()/global_downsample_rate), self.track_init_frames_full)
        else:
            self.selected_indices = np.arange(self.all_tracks_full.shape[1])
        self.all_tracks = self.all_tracks_full[:, self.selected_indices]  
        self.track_init_frames = self.track_init_frames_full[self.selected_indices]
        self.all_vis = self.all_vis_full[:, self.selected_indices] #* (cotrack_results['all_confidences'][:, selected_indices]>0.2)          # F x N
        self.confidences = self.all_vis.copy().astype(np.float32) 
        self.all_labels = self.all_labels_full[self.selected_indices]

        if 'all_tracks_depth' in cotrack_results:
            self.all_tracks_depth_full = cotrack_results["all_tracks_depth"]
            #self.all_tracks_depth_full = self.all_tracks_depth_full[:, valid]
            self.all_tracks_depth = self.all_tracks_depth_full[:, self.selected_indices]   
            self.all_tracks = np.concatenate([self.all_tracks, self.all_tracks_depth], axis=-1)
            self.all_tracks_depth_full = torch.tensor(self.all_tracks_depth_full).cuda()
          
        if 'all_tracks_color' in cotrack_results:
            self.all_tracks_color_full = cotrack_results["all_tracks_color"]
            #self.all_tracks_color_full = self.all_tracks_color_full[valid]
            self.all_tracks_color = self.all_tracks_color_full[self.selected_indices]  

        self.tracking_confidences_full = cotrack_results['all_confidences'].astype(np.float32)
        #self.tracking_confidences_full = self.tracking_confidences_full[:, valid]
        self.tracking_confidences = self.tracking_confidences_full[:, self.selected_indices]

        self.all_tracks_full = torch.tensor(self.all_tracks_full).float()
        self.track_init_frames_full = torch.tensor(self.track_init_frames_full).int()
        self.all_vis_full = torch.tensor(self.all_vis_full).bool()
        self.all_labels_full = torch.tensor(self.all_labels_full).int()
        self.tracking_confidences = torch.tensor(self.tracking_confidences).float()


        save_path = os.path.join(self.opt.BASE, cotracker_dir, "filtered_results.npz")
        np.savez_compressed(save_path, tracks=self.all_tracks, vis=self.all_vis, labels=self.all_labels)

        self.all_tracks = torch.tensor(self.all_tracks).float()
        self.track_init_frames = torch.tensor(self.track_init_frames).int()
        self.all_vis = torch.tensor(self.all_vis).bool()
        self.all_labels = torch.tensor(self.all_labels).int()
        self.confidences = torch.tensor(self.confidences).float()

        self.all_tracks_static = self.all_tracks[:, self.all_labels==0, :]
        self.all_vis_static = self.all_vis[:, self.all_labels==0]
        self.all_tracks_static_init = self.track_init_frames[self.all_labels==0]
        self.track_init_frames_static = self.track_init_frames[self.all_labels==0]
        self.static_confidences = self.confidences[:, self.all_labels==0]
        
        stage1_downsample = 4
        self.all_tracks_static_downsample = self.all_tracks_static[:, ::stage1_downsample, :].clone()
        self.all_vis_static_downsample = self.all_vis_static[:, ::stage1_downsample].clone()
        self.all_tracks_static_init_downsample = self.all_tracks_static_init[::stage1_downsample].clone()
        self.track_init_frames_static_downsample = self.track_init_frames_static[::stage1_downsample].clone()
        self.static_confidences_downsample = self.static_confidences[:, ::stage1_downsample].clone()
        
        dyn_mask = self.all_labels==1
        self.dyn_mask_final = dyn_mask
        self.track_init_frames_dyn = self.track_init_frames[dyn_mask]
        self.all_tracks_dyn = self.all_tracks[:, dyn_mask, :]
        self.all_vis_dyn = self.all_vis[:, dyn_mask]
        self.dyn_confidences = self.confidences[:, dyn_mask]

        self.num_points_static = self.all_tracks_static.shape[1] 
        self.num_points_dyn = self.all_tracks_dyn.shape[1]

        logging.info(f"Number of static points: {self.num_points_static}")
        logging.info(f"Number of dynamic points: {self.num_points_dyn}")

    def load_data(self):

        """
        Method used to load in rgb, cotrackers, and predict / store depth from unidepth
        """

        self.opt.BASE = f"{self.opt.workdir}/{self.opt.video_name}"
        self.opt.output_dir = f"{self.opt.workdir}/{self.opt.video_name}/uni4d/{self.opt.experiment_name}"

        os.makedirs(self.opt.output_dir, exist_ok=True)
        if self.opt.log:
            reload(logging)
            logging.basicConfig(filename=os.path.join(self.opt.output_dir , f'training_info.log'), level=logging.INFO,
                                filemode='w', format='%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S')
        else:
            logging.disable(logging.CRITICAL + 1)
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%m-%d %H:%M:%S')

        arg_dict = vars(self.opt)
        json_string = json.dumps(arg_dict, indent=4)
        logging.info(json_string)

        self.load_images()
        self.get_masks()
        self.load_tracklets()
        self.load_gt()
        self.load_depths()
    
    def reset_schedulers(self, patience=10):
        
        if hasattr(self, 'schedulers'):
            for scheduler in self.schedulers.values():
                del scheduler

        self.schedulers = {}

        for opt in self.optimizers:
            if opt == "shift":  # Lets not use scheduler for shift
                continue

            self.schedulers[opt] = ReduceLROnPlateau(self.optimizers[opt], 'min', factor=0.5, patience=patience, min_lr=1e-4)


    def init_optimization(self, lr=1e-3, patience=10):

        """
        Initializes the optimization process
        """

        if not hasattr(self, 'active_optimizers'):
            self.active_optimizers=[]

        translation_params = self.controlpoints_static.get_param()
        
        self.translation_optimizer = torch.optim.Adam([translation_params], lr=self.opt.cp_translation_dyn_lr, weight_decay=0, amsgrad=True)

        rotation_params, translation_params = self.poses.get_rotation_and_translation_params()
        self.pose_rotation_optimizer = torch.optim.Adam(rotation_params, lr=lr, weight_decay=0, amsgrad=True)
        self.pose_translation_optimizer = torch.optim.Adam(translation_params, lr=lr, weight_decay=0, amsgrad=True)
        
        translation_dyn_params, _ = self.controlpoints_dyn.get_params()
        self.t_dyn_optimizer = torch.optim.Adam([translation_dyn_params], lr=self.opt.cp_translation_dyn_lr, weight_decay=0, amsgrad=True)

        motion_static_params, _ = self.motion_static.get_params()
        self.motion_static_optimizer = torch.optim.Adam([motion_static_params], lr=self.opt.cp_translation_dyn_lr, weight_decay=0, amsgrad=True)
        
        self.optimizers = {
            "points_stat_t": self.translation_optimizer,
            "rotation_pose": self.pose_rotation_optimizer,
            "translation_pose": self.pose_translation_optimizer,
            "points_dyn_t": self.t_dyn_optimizer,
            "motion_stat_t": self.motion_static_optimizer

        }

        if self.opt.opt_intrinsics:
            self.intrinsics_optimizer = torch.optim.Adam(self.intrinsics.parameters(), lr=self.opt.intrinsics_lr, weight_decay=0, amsgrad=True)
            self.optimizers["intrinsics"]= self.intrinsics_optimizer

        self.reset_schedulers(patience)
        

    def to_device(self):

        """
        Moves everything to device
        """
        self.all_tracks_full = self.all_tracks_full.to(self.device)
        self.track_init_frames_full = self.track_init_frames_full.to(self.device)
        self.all_vis_full = self.all_vis_full.to(self.device)
        self.all_labels_full = self.all_labels_full.to(self.device)
        self.tracking_confidences = self.tracking_confidences.to(self.device)

        self.all_tracks = self.all_tracks.to(self.device)
        self.all_vis = self.all_vis.to(self.device)
        self.confidences = self.confidences.to(self.device)

        self.all_tracks_static = self.all_tracks_static.to(self.device)
        self.all_tracks_static_init = self.all_tracks_static_init.to(self.device)
        self.static_confidences = self.static_confidences.to(self.device)
        self.dyn_confidences = self.dyn_confidences.to(self.device)

        self.all_tracks_static_downsample = self.all_tracks_static_downsample.to(self.device)
        self.all_vis_static_downsample = self.all_vis_static_downsample.to(self.device)
        self.all_tracks_static_init_downsample = self.all_tracks_static_init_downsample.to(self.device)
        self.track_init_frames_static_downsample = self.track_init_frames_static_downsample.to(self.device)
        self.static_confidences_downsample = self.static_confidences_downsample.to(self.device)

        self.track_init_frames = self.track_init_frames.to(self.device)
        self.all_tracks_dyn = self.all_tracks_dyn.to(self.device)
        self.all_vis_static = self.all_vis_static.to(self.device)
        self.all_vis_dyn = self.all_vis_dyn.to(self.device)
        self.conf_vars_static = self.conf_vars_static.to(self.device)

        self.all_tracks_static_depth = self.all_tracks_static_depth.to(self.device)
        self.all_tracks_static_depth_downsample = self.all_tracks_static_depth_downsample.to(self.device)
        self.all_tracks_dyn_depth = self.all_tracks_dyn_depth.to(self.device)

        self.depth = self.depth.to(self.device)

        self.poses = self.poses.to(self.device)
        self.controlpoints_static = self.controlpoints_static.to(self.device)
        self.controlpoints_dyn = self.controlpoints_dyn.to(self.device) 
        self.motion_static = self.motion_static.to(self.device)   
        
        self.all_labels = self.all_labels.to(self.device)

        if self.opt.opt_intrinsics:
            self.intrinsics = self.intrinsics.to(self.device)
        else:
            self.K_gt = self.K_gt.to(self.device)

    def load_gt(self):

        if not self.opt.opt_intrinsics:
            #intrinsic = np.load(os.path.join(self.opt.BASE, "K.npy"))
            intrinsic = np.load(os.path.join(self.opt.BASE, "unidepth", "intrinsics.npy"))
            self.K_gt = torch.tensor(intrinsic).float()
            #print(self.K_gt.shape)
            if len(self.K_gt.shape) == 1:
                self.K_gt = torch.tensor([[intrinsic[0], 0, intrinsic[2]],
                [0, intrinsic[1], intrinsic[3]],
                [0, 0, 1]]).float()

            if len(self.K_gt.shape) == 2:
                self.K_gt = self.K_gt.unsqueeze(0).repeat(self.num_frames, 1, 1).float()
                

    def set_all_seeds(self, seed=42, deterministic=True):
        """
        Set all seeds to make results reproducible.
        
        Args:
            seed: Integer seed for reproducibility.
            deterministic: If True, ensures PyTorch operations are deterministic.
                        May impact performance due to deterministic algorithms.
        
        Note: Setting deterministic=True may significantly impact performance.
            Only use it when absolute reproducibility is required.
        """
        # Python's random seed
        
        random.seed(seed)
        
        # NumPy seed
        np.random.seed(seed)
        
        # PyTorch seeds
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

        # PyTorch backend
        if deterministic:
            logging.info("Deterministic operations enabled.")
            # Configure backend for deterministic operations
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Set environment variable for deterministic operations
            os.environ['CUBLAS_WORKSPACE_CONFIG']= ":4096:8"
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            # Optional: Force PyTorch operations to be deterministic
            torch.use_deterministic_algorithms(True)
        else:
            logging.info("Deterministic operations disabled.")
            # Better performance, but not fully deterministic
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    def log_timer(self, stage):

        timer_file = os.path.join(self.opt.output_dir, "timer.txt")

        torch.cuda.synchronize()
        with open(timer_file, "a") as f:
            f.write(f"{stage}: {time.time() - self.start_time}\n")   # log time

    def initialize(self):

        self.start_time = time.time()
        self.set_all_seeds(seed=self.opt.seed, deterministic=self.opt.deterministic)   # seed for randomization

        self.load_data()
        self.init_vars()
        self.init_optimization()
        self.to_device()
    
    def freeze_frame(self, frame_index):

        pos_param = getattr(self.poses, f'delta_rotation_{frame_index}')
        pos_param.requires_grad = False
        trans_param = getattr(
            self.poses, f'delta_translation_{frame_index}')
        trans_param.requires_grad = False

    def init_pose(self, frame_index):  # initialize with previous frame
        so3_1, tr_1 = self.poses.get_raw_value(frame_index-1)
        self.poses.set_rotation_and_translation(frame_index, so3_1, tr_1)


    def smooth_reg(self, filter_outlier=False):

        """
        Temporal smoothness and spatial smoothness (neighbors)
        """
   

        # ------------------------------------------------- hard smooth reg -------------------------------------------

        ts, _ = self.controlpoints_dyn.forward() # -> for ts: N x F x 3
        vis_dyn = self.all_vis_dyn

        d = ts[:,1:,:] - ts[:,:-1,:]
        d_norm = torch.norm(d, dim=-1, p=2)

        mask = torch.logical_and(vis_dyn[1:], vis_dyn[:-1]).T               # visibility mask
        
        outliers = None

        if filter_outlier:

            d_norm[~mask] = 0.0
            outliers = d_norm

        # mask = torch.ones_like(mask)

        reg_t = torch.mean(d_norm[mask])

        return reg_t, None, outliers   

    def smooth_reg_upsample(self, filter_outlier=False):

        """
        Temporal smoothness and spatial smoothness (neighbors)
        """
   

        # ------------------------------------------------- hard smooth reg -------------------------------------------

        ts = self.controlpoints_dynamic_full # -> for ts: N x F x 3
        vis_dyn = self.all_vis_full_dyn

        d = ts[:,1:,:] - ts[:,:-1,:]
        d_norm = torch.norm(d, dim=-1, p=2)

        mask = torch.logical_and(vis_dyn[1:], vis_dyn[:-1]).T               # visibility mask
        
        outliers = None

        if filter_outlier:

            d_norm[~mask] = 0.0
            outliers = d_norm

        # mask = torch.ones_like(mask)

        reg_t = torch.mean(d_norm[mask])

        return reg_t, None, outliers   

    def ordered_ratio(self, disp_a, disp_b):
        ratio_a = torch.maximum(disp_a, disp_b) / \
            (torch.minimum(disp_a, disp_b)+1e-5)
        return ratio_a - 1

    def get_intrinsics_K(self, idxes=None):

        if self.opt.opt_intrinsics:
            if idxes is None:
                return self.intrinsics.get_K().unsqueeze(0)
            else:
                return self.intrinsics.get_K().unsqueeze(0).repeat(len(idxes), 1, 1)
        else:
            if idxes is None:
                return self.K_gt
            else:
                return self.K_gt[idxes]


    def interpolate(self, input, uv_orig, clone=True):

        """
        takes in an input, and uv
        Interpolate values at uv from input
        input: H x W x C
        uv: N x 2

        returns output: N x C
        """
        if clone:
            uv = uv_orig.clone()   
        else:
            uv = uv_orig                                         # clones uv to avoid changing original uv
        uv[:, 0] = (uv[:, 0] / (self.width-1)) * 2 - 1
        uv[:, 1] = (uv[:, 1] / (self.height-1)) * 2 - 1

        uv = uv.unsqueeze(0).unsqueeze(0)

        input = input.unsqueeze(0).permute(0, 3, 1, 2)

        output = F.grid_sample(input, uv, align_corners=True).squeeze(0).squeeze(1).T

        return output
    def compute_motion_over_time(self, motion_world, T, normalize_time=True):
        """
        Compute motion values based on motion_world and time points tforpoly, with optional time normalization.

        Args:
            motion_world (torch.Tensor): Shape [N, 9], containing linear, quadratic, and cubic coefficients.
            T (int): Maximum time point, tforpoly = 0, 1, ..., T.
            normalize_time (bool): Whether to normalize time to [0, 1]. Default is True.

        Returns:
            torch.Tensor: Shape [N, T+1, 3], motion values for N trajectories at T+1 time points.
        """
        # Input validation
        if motion_world.shape[1] != 9:
            raise ValueError("motion_world must have shape [N, 9]")
        if T < 0:
            raise ValueError("T must be non-negative")

        # Get dimensions and device
        N = motion_world.shape[0]
        device = motion_world.device
        
        # Create time points tensor t = [0, 1, ..., T]
        t = torch.arange(T, dtype=motion_world.dtype, device=device)  # Shape [T+1]
        
        # Normalize time to [0, 1]
        if normalize_time and T > 0:
            t = t / T  # Map to [0, 1]
        
        # Extract coefficients
        linear = motion_world[:, 0:3]  # Shape [N, 3]
        quadratic = motion_world[:, 3:6]  # Shape [N, 3]
        cubic = motion_world[:, 6:9]  # Shape [N, 3]
        
        # Reshape t for broadcasting
        t = t.view(1, T, 1)  # Shape [1, T+1, 1]
        
        # Compute t, t^2, t^3
        t_linear = t  # Shape [1, T+1, 1]
        t_quadratic = t ** 2  # Shape [1, T+1, 1]
        t_cubic = t ** 3  # Shape [1, T+1, 1]
        
        # Compute motion values
        # Broadcasting: (N, 3) * (1, T+1, 1) -> (N, T+1, 3)
        motion = (
            linear[:, None, :] * t_linear +
            quadratic[:, None, :] * t_quadratic +
            cubic[:, None, :] * t_cubic
        )  # Shape [N, T+1, 3]
        
        return motion

    def BA_fast(self, is_dyn=False, with_motion=False):

        """
        Parallel and faster implementation
        """

        if is_dyn:
            cp_world, _ = self.controlpoints_dyn()
            cp_world = cp_world.permute(1,0,2)
            vis_mask = self.all_vis_dyn
            gt_track = self.all_tracks_dyn
        else:
            cp_world = self.controlpoints_static().unsqueeze(0)     # 1 x N x 3
            vis_mask = self.all_vis_static
            gt_track = self.all_tracks_static
            motion_world, _ = self.motion_static()
            if with_motion:
                cp_world = cp_world + motion_world.permute(1,0,2)

        Rs, ts = self.get_poses(torch.arange(self.num_frames).to(self.device))    # F x 3 x 3, F x 3 x 1
        K = self.get_intrinsics_K(torch.arange(self.num_frames).to(self.device))                                               # 3x3

        cp_cam = torch.einsum("bin,bmi->bmn", Rs, cp_world - ts.permute(0,2,1))

        cp_coord = torch.einsum("bji,bni->bnj", K, cp_cam)
        cp_coord = cp_coord / (cp_coord[:,:,-1].unsqueeze(-1)+1e-16)
        cp_coord = cp_coord[:,:,:2]     # F x N x 2


        flow_error = flow_norm(gt_track - cp_coord, self.flow_norm_l1)   # F x N
        flow_error = torch.sum(flow_error * vis_mask, dim=1) / (torch.sum(vis_mask, dim=1)+1e-16)   # F
        flow_error = torch.mean(flow_error)
        if is_dyn:
            motion_norm = torch.tensor(0.0).cuda()
        else:
            motion_norm = torch.abs(motion_world.permute(1,0,2)[vis_mask]).mean()
        return flow_error + 5*motion_norm
    
    def Temp_depth_fast(self, is_dyn=False, with_motion=False):

        """
        Parallel and faster implementation
        """

        if is_dyn:
            cp_world, _ = self.controlpoints_dyn()
            cp_world = cp_world.permute(1,0,2)
            vis_mask = self.all_vis_dyn
            gt_track_depth = self.all_tracks_dyn_depth
        else:
            cp_world = self.controlpoints_static().unsqueeze(0)     # 1 x N x 3
            vis_mask = self.all_vis_static
            gt_track_depth = self.all_tracks_static_depth
            motion_world, _ = self.motion_static()
            if with_motion: 
                cp_world = cp_world + motion_world.permute(1,0,2)

        Rs, ts = self.get_poses(torch.arange(self.num_frames).to(self.device))    # F x 3 x 3, F x 3 x 1
        K = self.get_intrinsics_K(torch.arange(self.num_frames).to(self.device))                                               # 3x3

        cp_cam = torch.einsum("bin,bmi->bmn", Rs, cp_world - ts.permute(0,2,1))

        cp_cam_depth = cp_cam[..., -1]

        depth_error = flow_norm(gt_track_depth[..., None] - cp_cam_depth[..., None], self.flow_norm_l1)   # F x N
        depth_error = torch.sum(depth_error * vis_mask, dim=1) / (torch.sum(vis_mask, dim=1)+1e-16)   # F
        depth_error = torch.mean(depth_error)
        if is_dyn:
            motion_norm = torch.tensor(0.0).cuda()
        else:
            motion_norm = torch.abs(motion_world.permute(1,0,2)[vis_mask]).mean()
        return depth_error + 5*motion_norm

    def filter_static(self):


        points_3d_world = self.controlpoints_static()
        motion_world, _ = self.motion_static()
        static_filter =  motion_world.abs().max(1)[0].max(-1)[0] < 0.3
        motion_world[static_filter] = motion_world[static_filter].mean(1)[:, None].repeat(1, motion_world.shape[1], 1)
        
        dyn_filter_vis = self.all_vis_static[:, self.prev_idx][:, ~static_filter]
        update_motion = propagate_motion(motion_world[~static_filter], dyn_filter_vis)
        motion_world[~static_filter] = update_motion

        points_3d_world[self.prev_idx] = points_3d_world[self.prev_idx] + motion_world.mean(1)
        mask = self.filter_isolated_points(points_3d_world)
        self.static_isolated_points = mask
        self.all_vis_static[:, mask] = False

    def filter_upsample_static(self):
        points_3d_world = self.controlpoints_static_full.mean(1)
        mask = self.filter_isolated_points(points_3d_world)
        self.static_isolated_points_full = mask
        self.all_vis_full_static[:, mask] = False

    @torch.no_grad()
    def reinitialize_static(self):

        """
        Reinitialize static with all static points clouds
        """

        prev_3d_static = self.controlpoints_static.forward()

        all_static_filter = torch.logical_or(self.all_labels == 0, self.all_labels==2)
        vis_filter = torch.sum(self.all_vis, axis=0) > 0                                                                # things must be visible at least once
        all_static_filter = torch.logical_and(all_static_filter, vis_filter)

        self.all_static_filter_final = all_static_filter
        self.all_tracks_static = self.all_tracks[:, all_static_filter, :2]
        if self.depth_version == 'v1':
            self.all_tracks_static_depth = self.all_tracks[:, all_static_filter, 2]
        else:
            self.get_track_depths()
        self.all_vis_static = self.all_vis[:, all_static_filter]
        self.all_tracks_static_init = self.track_init_frames[all_static_filter]
        self.track_init_frames_static = self.track_init_frames[all_static_filter]
        self.static_confidences = self.confidences[:, all_static_filter]

        self.num_points_static = self.all_tracks_static.shape[1] 

        self.controlpoints_static = ControlPoints(number_of_points=self.num_points_static).to(self.device)

        self.prev_idx = prev_idx = torch.where(self.all_labels[all_static_filter] == 0)[0]
        self.static_control_new_points = torch.where(self.all_labels[all_static_filter] == 2)[0]

        self.controlpoints_static.set_translation(prev_idx, prev_3d_static)                                            # sets translation for old points

        logging.info(f"Number of static points: {self.num_points_static}")
        logging.info(f"Number of dynamic points: {self.num_points_dyn}")

        self.init_BA()
        if self.opt.save_upsample:
            self.upsample_static2()
            self.filter_upsample_static()
        else:
            self.filter_static()

    @torch.no_grad()
    def reinitialize_invalid(self):

        """
        Reinitialize invalid with all invalid points clouds
        """

        #prev_3d_static = self.controlpoints_static.forward()

        all_static_filter = torch.logical_or(self.all_labels == 0, self.all_labels==2)
        vis_filter = torch.sum(self.all_vis, axis=0) > 2                                                                # things must be visible at least once
        all_static_filter = torch.logical_and(all_static_filter, vis_filter)

        all_dynamic_filter = self.all_labels==1
        
        all_invalid_filter = (~all_static_filter) and (~all_dynamic_filter)

        self.all_tracks_invalid = self.all_tracks[:, all_invalid_filter, :2]
        self.all_vis_invalid = self.all_vis[:, all_invalid_filter]
        self.all_tracks_invalid_init = self.track_init_frames[all_invalid_filter]
        self.track_init_frames_invalid = self.track_init_frames[all_invalid_filter]
        self.invalid_confidences = self.confidences[:, all_invalid_filter]

        self.num_points_invalid = self.all_tracks_invalid.shape[1] 

        self.controlpoints_invalid = ControlPoints(number_of_points=self.num_points_invalid).to(self.device)

        logging.info(f"Number of static points: {self.num_points_static}")
        logging.info(f"Number of dynamic points: {self.num_points_dyn}")
        logging.info(f"Number of dynamic points: {self.invalid_confidences}")
        self.init_BA_invalid()
        self.filter_static()
        

    def get_init_center(self):

        """
        Get 3D points from static points at init frame
        """

        control_pts_center = torch.zeros((self.num_points_static, 3)).to(self.device)

        point_3d = torch.zeros((3,1)).to(self.device)

        for p in range(self.num_points_static):

            frame = self.all_tracks_static_init[p].item()
            control_pts_2d = self.all_tracks_static[frame, p]

            depth = self.get_depth(frame)

            depth = self.interpolate(depth.unsqueeze(-1), control_pts_2d[None,:])[0]

            Rs, ts = self.get_poses([frame])
            R = Rs[0]
            t = ts[0]

            point_3d[0] = control_pts_2d[0]
            point_3d[1] = control_pts_2d[1]
            point_3d[2] = 1

            K = self.get_intrinsics_K([frame])[0]

            src_ray =  (torch.linalg.inv(K) @ point_3d).T                     # 1 x 3
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            points_3d_src = (src_ray_homo * depth.unsqueeze(-1))              # 1 x 3

            points_3d_world = (R @ points_3d_src.T + t).T

            control_pts_center[p] = points_3d_world[0]

        return control_pts_center

    def filter_isolated_points(self, points, percentile=98, k=5):
        """
        Filter out isolated points using PyTorch3D's optimized KNN implementation.
        
        Args:
            points: (N, 3) tensor of 3D points on CUDA
            percentile: percentile threshold for filtering (e.g., 99 removes top 1% most isolated points)
            k: number of nearest neighbors to consider
        
        Returns:
            filtered_points: (M, 3) tensor with isolated points removed
            mask: boolean mask indicating which points were kept
        """
        # Add batch dimension if needed
        if len(points.shape) == 2:
            points = points.unsqueeze(0)  # (1, N, 3)
        
        # Compute k nearest neighbors
        # knn_points returns (dists, idx) where dists is (B, P1, K) tensor of squared distances
        results = knn_points(
            points,                     # (B, P1, D) query points
            points,                     # (B, P2, D) reference points
            K=k+1,                     # number of neighbors (k+1 because point is its own neighbor)
            return_sorted=False        # we don't need sorted distances
        )
        
        # Remove the first distance (distance to self = 0)
        neighbor_distances = results.dists[0, :, 1:]   # (N, K)
        
        # Calculate average distance to k nearest neighbors
        avg_distances = torch.mean(neighbor_distances, dim=1)  # (N,)
        
        # Calculate threshold based on percentile
        threshold = torch.quantile(avg_distances, percentile/100)
        
        # Create mask for points to keep
        mask = avg_distances > threshold
        
        return mask

    @torch.no_grad()
    def init_BA(self):

        """
        Initialize controlpoints by taking average of current pose and depth
        """

        control_pts_init = torch.zeros((self.num_points_static, 3)).to(self.device)
        all_conf = self.all_vis_static
        for f in range(self.num_frames):
            
            Rs, ts = self.get_poses([f])
            R = Rs[0]
            t = ts[0]

            track = self.all_tracks_static[f]
            vis = self.all_vis_static[f]
            conf = all_conf[f]
            track_sel = torch.where(vis)[0]                 # index of selected tracks

            N = len(track_sel)
            K = self.get_intrinsics_K([f])[0]

            control_pts_2d = track[track_sel]
            conf = conf[track_sel]

            control_pts_2d_homo = torch.concatenate((control_pts_2d, torch.ones((N,1)).float().to(control_pts_2d.device)), dim=1)  # N x 3

            src_ray =  (torch.linalg.inv(K) @ control_pts_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)

            if self.depth_version == 'v2':
                depth = self.get_depth(f)
                depth_src = depth[control_pts_2d[:,1].long(), control_pts_2d[:,0].long()]
            elif self.depth_version == 'v1':
                depth_src = self.all_tracks_static_depth[f][track_sel]

            points_3d_src = (src_ray_homo * depth_src.unsqueeze(-1))   # N x 3

            points_3d_world = (R @ points_3d_src.T + t).T
            
            control_pts_init[track_sel] += (points_3d_world * conf[:,None])

        control_pts_init = control_pts_init / (torch.sum(all_conf+1e-16, axis=0).unsqueeze(-1))

        self.controlpoints_static.set_translation(torch.arange(self.num_points_static), control_pts_init.float())
        
        return control_pts_init
    

    def filter_cotrackerv2(self):
        """
        Filter points according to visibility and oob values
        Fully vectorized implementation matching the behavior of the original function
        """
        h, w = self.height, self.width

        # Check if points are within image boundaries

        within_bound1 = np.logical_and(self.all_tracks_full[:,:,1] >= 0, self.all_tracks_full[:,:,1] < h)
        within_bound1 = np.logical_and(within_bound1, self.all_tracks_full[:,:,0] >= 0)
        within_bound1 = np.logical_and(within_bound1, self.all_tracks_full[:,:,0] < w)
        
        # Clip coordinates to image boundaries
        clip_all_tracks = self.all_tracks_full.copy()
        clip_all_tracks[:,:,0] = np.clip(clip_all_tracks[:,:,0], 0, w-1)
        clip_all_tracks[:,:,1] = np.clip(clip_all_tracks[:,:,1], 0, h-1)
        if self.depth_version == 'v2':
            self.all_tracks_full[:,:,0] = np.clip(self.all_tracks_full[:,:,0], 0, w-1)
            self.all_tracks_full[:,:,1] = np.clip(self.all_tracks_full[:,:,1], 0, h-1)

        # Update visibility based on boundary check
        self.all_vis_full = np.logical_and(self.all_vis_full, within_bound1)
        # Get masks
        static_mask = self.dyn_masks_filters == 0
        dyn_mask = self.dyn_masks_filters == 1
        invalid_mask = self.dyn_masks_filters == 2

        frames, N, _ = self.all_tracks_full.shape
        
        # Create indices for all frames and tracks
        frame_indices = np.arange(frames).reshape(frames, 1).repeat(N, axis=1)
        
        # Get and clamp track coordinates
        x_coords = clip_all_tracks[..., 0].astype(np.int64)
        y_coords = clip_all_tracks[..., 1].astype(np.int64)
        
        # Create base visibility mask (visible AND not invalid)
        base_visibility = np.copy(self.all_vis_full)
        
        # Check which points are in invalid regions
        is_invalid = invalid_mask[frame_indices, y_coords, x_coords]
        
        # Combine visibility with invalid check - THIS IS THE KEY DIFFERENCE
        # In the original function, invalid points are excluded from visibility check during scoring
        effective_visibility = np.logical_and(base_visibility, ~is_invalid)
        
        # Get static/dynamic scores for each track
        is_static = static_mask[frame_indices, y_coords, x_coords]
        is_dynamic = dyn_mask[frame_indices, y_coords, x_coords]
        
        # Only count points that are effectively visible
        visible_static = np.logical_and(is_static, effective_visibility)
        visible_dynamic = np.logical_and(is_dynamic, effective_visibility)

        
        # Calculate totals per track
        total_visible = np.sum(effective_visibility, axis=0)
        total_static = np.sum(visible_static, axis=0)
        total_dynamic = np.sum(visible_dynamic, axis=0)
        
        # Initialize all labels as outliers (3)
        all_labels = np.full(N, 3)
        
        # Identify tracks by type matching the logic in the original function
        #is_fully_static = total_static >= total_visible
        is_fully_dynamic = total_dynamic == total_visible
        if self.dataset == 'tum' or self.dataset == 'tumd':
          is_fully_static = total_static >= total_visible
        else:
          is_fully_static = (total_static >= 0.5 * total_visible) & is_static[self.track_init_frames_full.astype(np.int32()), np.arange(is_static.shape[1])]
        is_long_enough = total_visible >= 1#(0.1 * frames)
        
        # Apply classification rules
        all_labels[np.logical_and(is_fully_static, is_long_enough)] = 0  # Long static tracks
        all_labels[np.logical_and(is_fully_static, ~is_long_enough)] = 2  # Short static tracks
        all_labels[is_fully_dynamic] = 1  # Dynamic tracks
        # Anything not matching these conditions remains as outlier (3)
        #assert (all_labels[:218] == 3).sum() == 0
        self.all_labels_full = all_labels
        
        assert(len(self.all_labels_full) == self.all_tracks_full.shape[1])

    def set_lr(self, lr, optimizers):

        for o in self.active_optimizers:
            if o in optimizers:
                for param_group in self.optimizers[o].param_groups:
                    param_group['lr'] = lr 

    def cosine_schedule(self, t, lr_start, lr_end):
        assert 0 <= t <= 1
        return lr_end + (lr_start - lr_end) * (1+np.cos(t * np.pi))/2
    
    def rot_to_angle(self, rots):

        batch_trace = torch.vmap(torch.trace)(rots)

        batch_trace = torch.clamp((batch_trace - 1) / 2, -1+1e-4, 1-1e-4) # setting it to 1 leads to gradient issues

        batch_trace = torch.acos(batch_trace)

        return batch_trace

    def calc_pose_smooth_loss(self, ids=None):

        if ids == None:
            ids = [x for x in range(self.num_frames)]

        ids = sorted(ids)

        assert((max(ids) - min(ids)) == (len(ids) - 1)) # ids should be consecutive
        
        Rs, ts = self.poses(ids)

        v1 = ts[1:-1,:,:] - ts[:-2,:,:]
        v2 = ts[2:,:,:] - ts[1:-1,:,:]

        v1_norm = torch.norm(v1, dim=1, p=2)
        v2_norm = torch.norm(v2, dim=1, p=2)
        v_diff = v2 - v1

        v_avg_norm = (v1_norm + v2_norm) / 2
        v_diff_norm = torch.norm(v_diff, dim=1, p=2)

        reg_t = torch.mean(v_diff_norm / (v_avg_norm + 1e-12))   # exp to stretch and -1 to make it to 0

        R1 = torch.bmm(Rs[1:-1, :, :], torch.transpose(Rs[:-2, :, :], 1, 2))
        R2 = torch.bmm(Rs[2:, :, :], torch.transpose(Rs[1:-1, :, :], 1, 2))

        R_diff = torch.bmm(R2, torch.transpose(R1, 1, 2))

        ang_avg = (self.rot_to_angle(R1) + self.rot_to_angle(R2)) / 2
        ang_diff = self.rot_to_angle(R_diff)

        ang_diff[ang_diff < 0.02] = 0          # due to clamping issues, we just round to 0 if ang diff is really small

        reg_R = torch.mean((ang_diff / (ang_avg + 1e-12))) # exp to stretch and -1 to make it to 0

        return reg_t, reg_R

    def optimize_BA(self):

        """
        Performs bundle alignment over all frames. Note this is not frame pair wise loss
        """

        self.reset_optimizer(lr=self.opt.ba_lr, patience=20)

        self.init_BA()

        logging.info("Starting optimization for BA")

        self.active_optimizers = ["rotation_pose", "translation_pose", "intrinsics", "points_stat_t", "motion_stat_t"]
        #self.active_optimizers = ["points_stat_t", "motion_stat_t"]    
        if not self.opt.opt_intrinsics:
            self.active_optimizers.remove("intrinsics")


        logging.info("Optimized variables are: ")
        logging.info(self.active_optimizers)

        earlystopper = EarlyStopper(patience=100)

        if len(self.active_optimizers) !=  0:
        
            for epoch in range(self.opt.num_BA_epochs):
                self.epoch = epoch
                total = torch.tensor(0.0).cuda()
                for o in self.active_optimizers:
                    self.optimizers[o].zero_grad()
                if epoch < self.opt.num_motion_start_epochs:
                    total_reproj_loss = self.BA_fast()

                    total_reproj_loss *= self.opt.reproj_weight

                    total += total_reproj_loss

                    total_depth_loss = self.Temp_depth_fast()
                else:
                    total_reproj_loss = self.BA_fast(with_motion=True)

                    total_reproj_loss *= self.opt.reproj_weight

                    total += total_reproj_loss

                    total_depth_loss = self.Temp_depth_fast(with_motion=True)

                total_depth_loss *= self.opt.reproj_weight

                total += total_depth_loss
                

                if self.opt.pose_smooth_weight_t > 0:
                    loss_pose_smooth_t, loss_poss_smooth_R = self.calc_pose_smooth_loss()
                    loss_pose_smooth_t = loss_pose_smooth_t * self.opt.pose_smooth_weight_t
                    loss_pose_smooth_R = loss_poss_smooth_R * self.opt.pose_smooth_weight_r
                else:
                    loss_pose_smooth_t = torch.tensor(0)
                    loss_pose_smooth_R = torch.tensor(0)

                total += (loss_pose_smooth_t + loss_pose_smooth_R)
                total.backward(retain_graph=False)

                if earlystopper.early_stop(total):
                    break

                for o in self.active_optimizers:
                    self.optimizers[o].step()
                    if o in self.schedulers:
                        self.schedulers[o].step(total)

                if epoch % self.opt.print_every == 0:
                    logging.info(f"Currently at epoch {epoch} out of {self.opt.num_BA_epochs} with pose smooth loss t: {loss_pose_smooth_t.item()}, pose smooth loss R: {loss_pose_smooth_R.item()}, reproj loss: {total_reproj_loss.item()}, depth loss: {total_depth_loss.item()}")
                    logging.info(f"Total loss: {total.item()} with lr: {self.schedulers['translation_pose'].get_last_lr()}")
                    # logging.info(f"Total loss: {total.item()}")

                torch.cuda.empty_cache()

    def laplacian_reg(self, outliers=False):

        """
        Apply laplacian rigidity regularization
        We simply want the distance to neighbors to be consistent
        """


        control_pts, _ = self.controlpoints_dyn.forward()
        vis_dyn = self.all_vis_dyn.T
        neighbors_idx = self.all_neighbors 

        neighbors_loc = control_pts[neighbors_idx]
        neighbors_vis = vis_dyn[neighbors_idx]                               # N x neighbors x F

        control_pts  = control_pts.unsqueeze(1)                              # N x 1 x F x 3

        diff = control_pts  - neighbors_loc

        neighbors_norm = torch.norm(diff, dim=-1, p=2)

        neighbors_norm_diff = torch.abs(neighbors_norm[:,:,1:] - neighbors_norm[:,:,:-1]) # N x neighbors x F-1

        vis_mask_neigh = torch.logical_and(neighbors_vis[:,:,1:], neighbors_vis[:,:,:-1])
        vis_mask_curr = torch.logical_and(vis_dyn[:,1:], vis_dyn[:, :-1])
        vis_mask = torch.logical_and(vis_mask_curr.unsqueeze(1), vis_mask_neigh)
        

        if outliers:
            outlier = torch.sum(neighbors_norm_diff * vis_mask, dim=1) / (torch.sum(vis_mask, dim=1)+1e-12)
            return outlier                      # return outliers instead
        

        laplacian_loss = torch.mean(neighbors_norm_diff[vis_mask])

        return laplacian_loss

    def laplacian_reg_upsample(self, outliers=False):

        """
        Apply laplacian rigidity regularization
        We simply want the distance to neighbors to be consistent
        """


        control_pts = self.controlpoints_dynamic_full
        vis_dyn = self.all_vis_full_dyn.T
        neighbors_idx = self.all_neighbors_full

        neighbors_loc = control_pts[neighbors_idx]
        neighbors_vis = vis_dyn[neighbors_idx]                               # N x neighbors x F

        control_pts  = control_pts.unsqueeze(1)                              # N x 1 x F x 3

        diff = control_pts  - neighbors_loc

        neighbors_norm = torch.norm(diff, dim=-1, p=2)

        neighbors_norm_diff = torch.abs(neighbors_norm[:,:,1:] - neighbors_norm[:,:,:-1]) # N x neighbors x F-1

        vis_mask_neigh = torch.logical_and(neighbors_vis[:,:,1:], neighbors_vis[:,:,:-1])
        vis_mask_curr = torch.logical_and(vis_dyn[:,1:], vis_dyn[:, :-1])
        vis_mask = torch.logical_and(vis_mask_curr.unsqueeze(1), vis_mask_neigh)
        

        if outliers:
            outlier = torch.sum(neighbors_norm_diff * vis_mask, dim=1) / (torch.sum(vis_mask, dim=1)+1e-12)
            return outlier                      # return outliers instead
        

        laplacian_loss = torch.mean(neighbors_norm_diff[vis_mask])

        return laplacian_loss

    @torch.no_grad()
    def get_neighbors_h(self):
        """
        Get neighbors of each point in a batched manner.
        """
        neighbors = 4
        self.all_neighbors = torch.zeros((self.num_points_dyn, neighbors), dtype=torch.long, device=self.device)
        all_idx = torch.arange(self.num_points_dyn, dtype=torch.long, device=self.device)

        # Group points by initial frame
        frame_to_points = {}
        for i in range(self.num_points_dyn):
            f = self.track_init_frames_dyn[i].item()
            if f not in frame_to_points:
                frame_to_points[f] = []
            frame_to_points[f].append(i)

        # Process each frame
        for f, point_indices in frame_to_points.items():
            point_indices = torch.tensor(point_indices, dtype=torch.long, device=self.device)

            # Get tracks and visibility for the frame
            track = self.all_tracks_dyn[f]
            vis = self.all_vis_dyn[f]
            track_sel = torch.where(vis)[0]  # Indices of visible tracks

            # Compute 3D points
            K = self.get_intrinsics_K([f])[0]
            control_pts_2d = track[track_sel]
            control_pts_2d_homo = torch.cat(
                (control_pts_2d, torch.ones((control_pts_2d.shape[0], 1), device=control_pts_2d.device)), dim=1
            )  # N x 3

            src_ray = (torch.linalg.inv(K) @ control_pts_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:, -1].unsqueeze(-1) + 1e-16)

            # Compute depth
            if self.depth_version == 'v2':
                depth = self.get_depth([f]) # Batch depth fetching
                depth_src = self.interpolate(depth.unsqueeze(-1), control_pts_2d)
            else:  # v1
                depth_src = self.all_tracks_dyn_depth[f][track_sel, None]

            points_3d_src = src_ray_homo * depth_src  # 3D points for visible tracks

            # Subset for points in this frame
            point_mask = torch.isin(all_idx[track_sel], point_indices)
            query_points_3d = points_3d_src[point_mask]  # 3D points for query points
            query_indices = all_idx[track_sel][point_mask]  # Indices in all_idx

            # Compute KNN for all query points in this frame
            knn_result = knn_points(
                query_points_3d.unsqueeze(0),  # (1, M, 3)
                points_3d_src.unsqueeze(0),   # (1, N, 3)
                K=neighbors + 1
            )
            idx = knn_result.idx[0, :, 1:]  # (M, neighbors), skip self

            # Map back to global indices
            self.all_neighbors[query_indices] = all_idx[track_sel][idx]

        return self.all_neighbors

    @torch.no_grad()
    def get_neighbors_upsample_h(self):
        """
        Get neighbors of each point in a batched manner.
        """
        neighbors = 4
        self.all_neighbors_full = torch.zeros((self.num_points_dyn_full, neighbors), dtype=torch.long, device=self.device)
        all_idx = torch.arange(self.num_points_dyn_full, dtype=torch.long, device=self.device)

        # Group points by initial frame
        frame_to_points = {}
        for i in range(self.num_points_dyn_full):
            f = self.track_init_frames_full_dyn[i].item()
            if f not in frame_to_points:
                frame_to_points[f] = []
            frame_to_points[f].append(i)

        # Process each frame
        for f, point_indices in frame_to_points.items():
            point_indices = torch.tensor(point_indices, dtype=torch.long, device=self.device)

            # Get tracks and visibility for the frame
            track = self.all_tracks_dyn_full[f]
            vis = self.all_vis_full_dyn[f]
            track_sel = torch.where(vis)[0]  # Indices of visible tracks

            # Compute 3D points
            K = self.get_intrinsics_K([f])[0]
            control_pts_2d = track[track_sel]
            control_pts_2d_homo = torch.cat(
                (control_pts_2d, torch.ones((control_pts_2d.shape[0], 1), device=control_pts_2d.device)), dim=1
            )  # N x 3

            src_ray = (torch.linalg.inv(K) @ control_pts_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:, -1].unsqueeze(-1) + 1e-16)

            # Compute depth
            if self.depth_version == 'v2':
                depth = self.get_depth([f]) # Batch depth fetching
                depth_src = self.interpolate(depth.unsqueeze(-1), control_pts_2d)
            else:  # v1
                depth_src = self.all_tracks_depth_full_dyn[f][track_sel]

            points_3d_src = src_ray_homo * depth_src  # 3D points for visible tracks

            # Subset for points in this frame
            point_mask = torch.isin(all_idx[track_sel], point_indices)
            query_points_3d = points_3d_src[point_mask]  # 3D points for query points
            query_indices = all_idx[track_sel][point_mask]  # Indices in all_idx

            # Compute KNN for all query points in this frame
            knn_result = knn_points(
                query_points_3d.unsqueeze(0),  # (1, M, 3)
                points_3d_src.unsqueeze(0),   # (1, N, 3)
                K=neighbors + 1
            )
            idx = knn_result.idx[0, :, 1:]  # (M, neighbors), skip self

            # Map back to global indices
            self.all_neighbors_full[query_indices] = all_idx[track_sel][idx]

        return self.all_neighbors_full

    def get_3d_points(self, f, control_pts_2d, depth1):
        # Compute 3D points
        K = self.get_intrinsics_K([f])[0]
        #control_pts_2d = track[track_sel]
        control_pts_2d_homo = torch.cat(
            (control_pts_2d, torch.ones((control_pts_2d.shape[0], 1), device=control_pts_2d.device)), dim=1
        )  # N x 3

        src_ray = (torch.linalg.inv(K) @ control_pts_2d_homo.T).T
        src_ray_homo = src_ray / (src_ray[:, -1].unsqueeze(-1) + 1e-16)

        # Compute depth
        if self.depth_version == 'v2':
            depth = self.get_depth([f]) # Batch depth fetching
            depth_src = self.interpolate(depth.unsqueeze(-1), control_pts_2d)
        else:  # v1
            depth_src = depth1

        points_3d_src = src_ray_homo * depth_src  # 3D points for visible tracks
        return points_3d_src
    
    @torch.no_grad()
    def upsample_static2(self):
        """
        Get neighbors of each point in a batched manner.
        """

        control_pts_static = self.controlpoints_static.forward()
        motion_world, _ = self.motion_static()
        static_filter =  motion_world.abs().max(1)[0].max(-1)[0] < 0.5
        motion_world[static_filter] = motion_world[static_filter].mean(1)[:, None].repeat(1, motion_world.shape[1], 1)
        dyn_filter_vis = self.all_vis_static[:, self.prev_idx][:, ~static_filter]
        update_motion = propagate_motion(motion_world[~static_filter], dyn_filter_vis)
        motion_world[~static_filter] = update_motion
        control_pts_static = control_pts_static.unsqueeze(1).repeat(1, self.num_frames, 1)
        control_pts_static[self.prev_idx] = control_pts_static[self.prev_idx] + motion_world

        idx_all = torch.arange(control_pts_static.shape[0]).cuda()
        mask = ~torch.isin(idx_all, self.prev_idx)
        idx_filtered = idx_all[mask]
        self.prev_idx = self.prev_idx[static_filter]
        static_idx = torch.cat([self.prev_idx, idx_filtered], dim=0)
        control_pts_static = control_pts_static[static_idx]
        self.all_tracks_static = self.all_tracks_static[:, static_idx]
        self.all_tracks_static_depth =self.all_tracks_static_depth[:, static_idx]
        self.all_vis_static = self.all_vis_static[:, static_idx]
        self.all_tracks_static_init = self.all_tracks_static_init[static_idx]
        self.track_init_frames_static = self.track_init_frames_static[static_idx]
        self.static_confidences = self.static_confidences[:, static_idx]
        self.num_points_static = self.all_tracks_static.shape[1] 

        dyn_idx = idx_all[~torch.isin(idx_all, static_idx)]
        dyn_idx1 = torch.nonzero(self.all_static_filter_final, as_tuple=False).squeeze(1)[dyn_idx]
        self.all_labels[dyn_idx1] = 1
        dyn_mask = self.all_labels == 1
        self.dyn_mask_final = dyn_mask
        self.track_init_frames_dyn = self.track_init_frames[dyn_mask]
        self.all_tracks_dyn = self.all_tracks[:, dyn_mask, :2]
        self.all_vis_dyn = self.all_vis[:, dyn_mask]
        self.dyn_confidences = self.confidences[:, dyn_mask]
        self.num_points_dyn = self.all_tracks_dyn.shape[1]
        self.controlpoints_dyn = ControlPointsDynamic(number_of_points=self.num_points_dyn, number_of_frames=self.num_frames, with_norm=False).to(self.device) 
        translation_dyn_params, _ = self.controlpoints_dyn.get_params()
        self.t_dyn_optimizer = torch.optim.Adam([translation_dyn_params], lr=self.opt.cp_translation_dyn_lr, weight_decay=0, amsgrad=True)
        self.optimizers['points_dyn_t'] = self.t_dyn_optimizer
        if self.depth_version == 'v2':
            self.get_track_depths()
        elif self.depth_version == 'v1':
            self.all_tracks_dyn_depth = self.all_tracks[:, dyn_mask, 2]
        for i in dyn_idx1:
            indices = torch.tensor(self.merged_dict[int(i)]).cuda()
            mask = (self.all_labels_full[indices] == 0) | (self.all_labels_full[indices] == 2)
            self.all_labels_full[indices[mask]] = 1

        all_static_filter = torch.logical_or(self.all_labels_full == 0, self.all_labels_full == 2)
        vis_filter = torch.sum(self.all_vis_full, axis=0) > 0                                                                # things must be visible at least once
        all_static_filter = torch.logical_and(all_static_filter, vis_filter)
        self.all_static_filter_final_full = all_static_filter
        self.all_tracks_static_full = self.all_tracks_full[:, all_static_filter]
        self.track_init_frames_full_static = self.track_init_frames_full[all_static_filter]
        self.all_vis_full_static = self.all_vis_full[:, all_static_filter]
        self.all_tracks_depth_full_static = self.all_tracks_depth_full[:, all_static_filter]
        self.num_points_static_full = self.all_tracks_static_full.shape[1]

        neighbors = 3
        self.controlpoints_static_full = torch.zeros((self.num_points_static_full, self.num_frames, 3), dtype=torch.float, device=self.device)
        all_idx = torch.arange(self.num_points_static_full, dtype=torch.long, device=self.device)
        valid_item = torch.zeros(self.num_points_static_full, dtype=torch.bool, device=self.device)
        # Group points by initial frame
        frame_to_points = {}
        for i in range(self.num_points_static_full):
            f = self.track_init_frames_full_static[i].item()
            if f not in frame_to_points:
                frame_to_points[f] = []
            frame_to_points[f].append(i)
        # Process each frame
        for f, point_indices in frame_to_points.items():
            point_indices = torch.tensor(point_indices, dtype=torch.long, device=self.device)

            # Get tracks and visibility for the frame
            track = self.all_tracks_static_full[f]
            vis = self.all_vis_full_static[f]
            depth = self.all_tracks_depth_full_static[f]
            track_sel = torch.where(vis)[0]  # Indices of visible tracks
            points_3d_src = self.get_3d_points(f, track[track_sel], depth[track_sel])
            # Subset for points in this frame
            point_mask = torch.isin(all_idx[track_sel], point_indices)
            query_points_3d = points_3d_src[point_mask]  # 3D points for query points
            query_indices = all_idx[track_sel][point_mask]  # Indices in all_idx

            # Get downsample tracks and visibility for the frame
            track_ds = self.all_tracks_static[f]
            vis_ds = self.all_vis_static[f]
            depth_ds = self.all_tracks_static_depth[f]
            track_sel_ds = torch.where(vis_ds)[0]  # Indices of visible tracks
            points_3d_src_ds = self.get_3d_points(f, track_ds[track_sel_ds], depth_ds[track_sel_ds][:, None])

            # Compute KNN for all query points in this frame
            knn_result = knn_points(
                query_points_3d.unsqueeze(0),  # (1, M, 3)
                points_3d_src_ds.unsqueeze(0),   # (1, N, 3)
                K=neighbors + 1
            )
            idx = knn_result.idx[0]  # Shape: (num_points1, k)
            dists = knn_result.dists[0]
            weights = 1 / (dists + 1e-8) 
            weights_sum = weights.sum(dim=-1, keepdim=True)
            probabilities = weights / (weights_sum)
            # weights = torch.exp(-dists/4)  # 或者 -dists / σ
            # weights_sum = weights.sum(dim=-1, keepdim=True)
            # probabilities = weights / weights_sum
            # scale = control_points_depth / unidepth_vals
            vis_controlpoints_static = control_pts_static[track_sel_ds, f]#.clone()
            K = self.get_intrinsics_K([f])[0]
            Rs, ts = self.get_poses([f])
            R = Rs[0]
            t = ts[0]
            control_pts_2D, control_points_depth = self.project(R, t, K, vis_controlpoints_static)
            unidepth_vals = depth_ds[track_sel_ds]
            scale = control_points_depth / unidepth_vals
            scales = torch.sum(scale[idx] * probabilities, axis=1)
            #idx = knn_result.idx[0, :, 1:]  # (M, neighbors), skip self
            depthmap_res =  depth[track_sel][point_mask][:, 0] * scales
            coords_2d_homo = torch.concatenate((track[track_sel][point_mask], torch.ones((len(track[track_sel][point_mask]),1)).float().to(self.device)), dim=1)
            src_ray =  (torch.linalg.inv(K) @ coords_2d_homo.T).T
            src_ray_homo = src_ray / (src_ray[:,-1].unsqueeze(-1)+1e-16)
            points_3d_src = src_ray_homo * depthmap_res.flatten().unsqueeze(-1)
            points_3d_world = (R @ points_3d_src.T + t).T
            # Map back to global indices
            valid_item[query_indices] = True
            self.controlpoints_static_full[query_indices] = points_3d_world[:, None].repeat(1, self.num_frames, 1)#.clone()

        self.controlpoints_static_full = self.controlpoints_static_full[valid_item]
        self.all_static_filter_final_full[self.all_static_filter_final_full.clone()] = valid_item
        self.track_init_frames_full_static = self.track_init_frames_full_static[valid_item]
        self.all_tracks_depth_full_static = self.all_tracks_depth_full_static[:, valid_item]
        self.all_vis_full_static = self.all_vis_full_static[:, valid_item]
        self.num_points_static_full = self.controlpoints_static_full.shape[0]
    

    @torch.no_grad()
    def upsample_dyn(self):
        """
        Get neighbors of each point in a batched manner.
        """
        
        all_dyn_filter = self.all_labels_full == 1
        self.dyn_mask_final_full = all_dyn_filter

        self.all_tracks_dyn_full = self.all_tracks_full[:, all_dyn_filter]
        self.track_init_frames_full_dyn = self.track_init_frames_full[all_dyn_filter]
        self.all_vis_full_dyn = self.all_vis_full[:, all_dyn_filter]
        self.num_points_dyn_full = self.all_tracks_dyn_full.shape[1]
        self.all_tracks_depth_full_dyn = self.all_tracks_depth_full[:, all_dyn_filter]
        neighbors = 3
        self.controlpoints_dynamic_full = torch.zeros((self.num_points_dyn_full, self.num_frames, 3), dtype=torch.float, device=self.device)
        all_idx = torch.arange(self.num_points_dyn_full, dtype=torch.long, device=self.device)
        valid_item = torch.zeros(self.num_points_dyn_full, dtype=torch.bool, device=self.device)
        # Group points by initial frame
        frame_to_points = {}
        for i in range(self.num_points_dyn_full):
            f = self.track_init_frames_full_dyn[i].item()
            if f not in frame_to_points:
                frame_to_points[f] = []
            frame_to_points[f].append(i)

        control_pts_dyn, _ = self.controlpoints_dyn.forward()


        # Process each frame
        for f, point_indices in frame_to_points.items():
            point_indices = torch.tensor(point_indices, dtype=torch.long, device=self.device)

            # Get tracks and visibility for the frame
            track = self.all_tracks_dyn_full[f]
            vis = self.all_vis_full_dyn[f]
            depth = self.all_tracks_depth_full_dyn[f]
            track_sel = torch.where(vis)[0]  # Indices of visible tracks
            points_3d_src = self.get_3d_points(f, track[track_sel], depth[track_sel])
            # Subset for points in this frame
            point_mask = torch.isin(all_idx[track_sel], point_indices)
            query_points_3d = points_3d_src[point_mask]  # 3D points for query points
            query_indices = all_idx[track_sel][point_mask]  # Indices in all_idx

            # Get downsample tracks and visibility for the frame
            track_ds = self.all_tracks_dyn[f]
            vis_ds = self.all_vis_dyn[f]
            depth = self.all_tracks_dyn_depth[f]
            track_sel_ds = torch.where(vis_ds)[0]  # Indices of visible tracks
            points_3d_src_ds = self.get_3d_points(f, track_ds[track_sel_ds], depth[track_sel_ds][:, None])

            # Compute KNN for all query points in this frame
            knn_result = knn_points(
                query_points_3d[..., :2].unsqueeze(0),  # (1, M, 3)
                points_3d_src_ds[..., :2].unsqueeze(0),   # (1, N, 3)
                K=neighbors + 1
            )
            idx = knn_result.idx[0]  # Shape: (num_points1, k)
            dists = knn_result.dists[0]
            weights = 1 / (dists + 1e-8) 
            weights_sum = weights.sum(dim=-1, keepdim=True)
            probabilities = weights / (weights_sum)
            # weights = torch.exp(-dists)  # 或者 -dists / σ
            # weights_sum = weights.sum(dim=-1, keepdim=True)
            # probabilities = weights / weights_sum
            vis_controlpoints_dyn = control_pts_dyn[track_sel_ds]#.clone()
          
            #idx = knn_result.idx[0, :, 1:]  # (M, neighbors), skip self

            # Map back to global indices
            valid_item[query_indices] = True
            self.controlpoints_dynamic_full[query_indices] = torch.sum(vis_controlpoints_dyn[idx] * probabilities[..., None, None], axis=1)#.clone()

        self.controlpoints_dynamic_full = self.controlpoints_dynamic_full[valid_item]
        self.dyn_mask_final_full[self.dyn_mask_final_full.clone()] = valid_item
        self.track_init_frames_full_dyn = self.track_init_frames_full_dyn[valid_item]
        self.all_tracks_depth_full_dyn = self.all_tracks_depth_full_dyn[:, valid_item]
        self.all_vis_full_dyn = self.all_vis_full_dyn[:, valid_item]
        self.num_points_dyn_full = self.controlpoints_dynamic_full.shape[0]

    @torch.no_grad()
    def filter_dyn(self):

        if self.num_points_dyn == 0:
            return

        with torch.no_grad():
            self.get_neighbors_h()
            _, _, outliers_smooth = self.smooth_reg(filter_outlier=True)
            outliers_laplacian = self.laplacian_reg(outliers=True)
            outlier_score = self.opt.dyn_smooth_weight_t * outliers_smooth +  self.opt.dyn_laplacian_weight_t * outliers_laplacian
            outliers = torch.concat((outlier_score , outlier_score [:,-1:]), dim=-1)
            points_energy = torch.max(outliers, dim=1)[0]

            thrs = torch.quantile(points_energy, 0.98) + 1.0
            outliers = points_energy > thrs
            self.outliers_dyn = outliers

            self.all_vis_dyn[:, self.outliers_dyn] = False

    @torch.no_grad()
    def filter_upsample_dyn(self):

        if self.num_points_dyn_full == 0:
            return

        with torch.no_grad():
            self.get_neighbors_upsample_h()
            _, _, outliers_smooth = self.smooth_reg_upsample(filter_outlier=True)
            outliers_laplacian = self.laplacian_reg_upsample(outliers=True)
            outlier_score = self.opt.dyn_smooth_weight_t * outliers_smooth +  self.opt.dyn_laplacian_weight_t * outliers_laplacian
            outliers = torch.concat((outlier_score , outlier_score [:,-1:]), dim=-1)
            points_energy = torch.max(outliers, dim=1)[0]

            thrs = torch.quantile(points_energy, 0.98) + 1.0
            outliers = points_energy > thrs
            self.outliers_dyn_full = outliers

            self.all_vis_full_dyn[:, self.outliers_dyn_full] = False

    def optimize_dyn(self):

        """ 
        Optimization
        Loss Implemented:
        L_flow -> 2D reprojection loss of 3D points
        L_depth -> depth loss compared to pred depth
        """

        self.reset_optimizer()

        self.get_neighbors_h()

        logging.info("Starting stage 3 dynamic optimization")

        self.active_optimizers = ["points_dyn_t"]

        logging.info("Optimized variables are: ")
        logging.info(self.active_optimizers)

        earlystopper = EarlyStopper(patience=100)

        if len(self.active_optimizers) != 0:

            for epoch in range(self.opt.num_dyn_epochs):

                for o in self.active_optimizers:
                    self.optimizers[o].zero_grad()
                
                loss_log = {}

                loss_smooth_t, _, _ = self.smooth_reg()
                loss_laplacian = self.laplacian_reg()

                total_reproj_loss = self.BA_fast(is_dyn=True)
                total_reproj_loss *= self.opt.reproj_weight

                total_depth_loss = self.Temp_depth_fast(is_dyn=True)
                total_depth_loss *= self.opt.reproj_weight

                total_loss =  total_reproj_loss + self.opt.dyn_smooth_weight_t * loss_smooth_t + self.opt.dyn_laplacian_weight_t * loss_laplacian + total_depth_loss 
                


                loss_log["reproj_loss"] = total_reproj_loss.item()
                loss_log["depth_loss"] = total_depth_loss.item()
                loss_log["smooth_t"] = self.opt.dyn_smooth_weight_t * loss_smooth_t.item()
                loss_log["laplacian_loss"] = self.opt.dyn_laplacian_weight_t * loss_laplacian.item()

                total_loss.backward()

                if earlystopper.early_stop(total_loss):
                    break

                for o in self.active_optimizers:
                    self.optimizers[o].step()
                    if o in self.schedulers:
                        self.schedulers[o].step(total_loss)

                if epoch % self.opt.print_every == 0:
                    logging.info(f"Finished epoch {epoch} with {loss_log}")
                    logging.info(f"Total loss: {total_loss.item()} with lr: {self.schedulers['points_dyn_t'].get_last_lr()}")

        logging.info("Finished optimization")

    def unfreeze_frames(self):

        for f in range(self.num_frames):
            pos_param = getattr(self.poses, f'delta_rotation_{f}')
            pos_param.requires_grad = True
            trans_param = getattr(
            self.poses, f'delta_translation_{f}')
            trans_param.requires_grad = True

    def ordered_ratio(self, disp_a, disp_b):
        ratio_a = torch.maximum(disp_a, disp_b) / \
            (torch.minimum(disp_a, disp_b)+1e-5)
        return ratio_a - 1
    
    def optimize_clip(self, clip_range):
        """
        Process a single clip and return optimized poses for that clip
        """
        start_frame, end_frame = clip_range
        
        # Initialize keyframe buffer for this clip
        self.keyframe_buffer = KeyFrameBuffer(buffer_size=5)
        
        # Add first two frames of the clip
        self.keyframe_buffer.add_keyframe(start_frame)
        if start_frame + 1 < end_frame:
            self.keyframe_buffer.add_keyframe(start_frame + 1)
            
        # Optimize first two frames if they exist
        if len(self.keyframe_buffer.buffer) > 1:
            self.optimize_over_keyframe_buffer()
            self.freeze_frame(self.keyframe_buffer.buffer[0])
        
        # Process remaining frames in the clip
        for x in range(start_frame + 2, end_frame):
            self.keyframe_buffer.add_keyframe(x)
            self.init_pose(x)  # just use the previous frame as init
            self.optimize_over_keyframe_buffer()
            self.freeze_frame(self.keyframe_buffer.buffer[0])
        
        # Get poses for this clip
        self.unfreeze_frames()
        clip_poses = self.poses.forward_raw_value(torch.arange(start_frame, end_frame))

        return (clip_poses[0].detach(), clip_poses[1].detach(), torch.arange(start_frame, end_frame))

    def optimize_init_sliding(self):
        """
        Performs first stage of sliding window optimization to obtain a good initial pose estimate
        Assumes depth is GT for this stage. Performs pair wise warping
        Processes frames in clips of 5 frames each in parallel and concatenates results
        """

        logging.info("Starting init optimization")

        self.active_optimizers = ["rotation_pose", "translation_pose", "intrinsics"]

        if not self.opt.opt_intrinsics:
            self.active_optimizers.remove("intrinsics")
        else:
            logging.info("Initial intrinsic matrix is: ")
            logging.info(self.get_intrinsics_K())

        logging.info("Optimized variables are: ")
        logging.info(self.active_optimizers)

        # Calculate number of clips
        clip_size = 5
        num_clips = (self.num_frames + clip_size - 1) // clip_size
        
        # Generate clip ranges
        clip_ranges = [
            (i * clip_size, min((i + 1) * clip_size, self.num_frames))
            for i in range(num_clips)
        ]
        
        # Create a pool of workers with 'spawn'
        with mp.Pool(processes=min(mp.cpu_count(), num_clips)) as pool:
            optimize_clip_partial = partial(self.optimize_clip)
            results = pool.map(optimize_clip_partial, clip_ranges)
        
        clip_ranges_offset = [
            (i * clip_size -1, min(i * clip_size+1, self.num_frames))
            for i in range(1, num_clips)
        ]
        # Create a pool of workers with 'spawn'
        with mp.Pool(processes=min(mp.cpu_count(), num_clips-1)) as pool:
            optimize_clip_partial = partial(self.optimize_clip)
            results_offset = pool.map(optimize_clip_partial, clip_ranges_offset)
            
        # Concatenate all poses
        for idx, clip_result in enumerate(results):
            if idx > 0:
                offset_rotation = (results_offset[idx-1][0][1]-results_offset[idx-1][0][0]) - (clip_result[0][0] - self.poses.get_raw_value(results_offset[idx-1][2][0])[0][0]).clone()
                offset_translation = (results_offset[idx-1][1][1]-results_offset[idx-1][1][0]) - (clip_result[1][0] - self.poses.get_raw_value(results_offset[idx-1][2][0])[1][0]).clone()
            else:
                offset_rotation = torch.zeros_like(clip_result[0][0])
                offset_translation = torch.zeros_like(clip_result[1][0])
            for i in range(clip_result[2].shape[0]):
                self.poses.set_rotation_and_translation(clip_result[2][i], (clip_result[0][i] + offset_rotation)[None], (clip_result[1][i] + offset_translation)[None])

        logging.info(f"Done with init optimization")

    def optimize_init(self):

        """
        Performs first stage of sliding window optimization to obtain a good initial pose estimate
        Assumes depth is GT for this stage. Performs pair wise warping
        """

        logging.info("Starting init optimization")

        self.active_optimizers = ["rotation_pose", "translation_pose", "intrinsics"]

        if not self.opt.opt_intrinsics:
            self.active_optimizers.remove("intrinsics")
        else:
            logging.info("Initial intrinsic matrix is: ")
            logging.info(self.get_intrinsics_K())

        logging.info("Optimized variables are: ")
        logging.info(self.active_optimizers)


        self.keyframe_buffer = KeyFrameBuffer(buffer_size=5)
        self.keyframe_buffer.add_keyframe(0)
        self.keyframe_buffer.add_keyframe(1)

        self.optimize_over_keyframe_buffer()
        self.freeze_frame(self.keyframe_buffer.buffer[0])

        for x in range(2, self.num_frames):
            self.keyframe_buffer.add_keyframe(x)

            self.init_pose(x)  # just use the previous frame as init
            self.optimize_over_keyframe_buffer()
            self.freeze_frame(self.keyframe_buffer.buffer[0])
            
        self.unfreeze_frames()
        
        logging.info(f"Done with init optimization")

    def optimize_over_pairs(self, pairs):

        all_pairs = torch.tensor(pairs)
        

        Rs_from, ts_from = self.get_poses(all_pairs[:, 0])         # N x 3 x 3,     N x 3 x 1
        Rs_to, ts_to = self.get_poses(all_pairs[:, 1])

        K_from = self.get_intrinsics_K(all_pairs[:, 0])
        K_to = self.get_intrinsics_K(all_pairs[:, 1])

        static_tracks = self.all_tracks_static_downsample[all_pairs[:, 0]]
        B, N, _ = static_tracks.shape
        static_homo_from = torch.concatenate((static_tracks, torch.ones((B, N, 1)).to(self.device)), axis=-1)
        
        static_cam_from = torch.einsum("bji,bni->bnj", torch.linalg.inv(K_from), static_homo_from)
        static_cam_from =  static_cam_from / (static_cam_from[:,:,2].unsqueeze(-1)+1e-16)
        static_cam_from = static_cam_from * self.all_tracks_static_depth_downsample[all_pairs[:,0]].unsqueeze(-1)      # project into 3D

        static_world_from = torch.einsum("bni,bmi->bmn", Rs_from, static_cam_from) + ts_from.permute(0,2,1)    # F x N x 3

        static_cam_to = torch.einsum("bin,bmi->bmn", Rs_to, static_world_from - ts_to.permute(0,2,1))         # F x N x 3

        static_coord = torch.einsum("bji,bni->bnj", K_to, static_cam_to)
        static_coord = static_coord / (static_coord[:,:,2].unsqueeze(-1)+1e-16)
        static_coord = static_coord[:,:,:2]

        static_tracks_vis_from = self.all_vis_static_downsample[all_pairs[:, 0]]
        static_tracks_vis_to = self.all_vis_static_downsample[all_pairs[:, 1]]
        mask = torch.logical_and(static_tracks_vis_from, static_tracks_vis_to).float()
        
        if self.dataset != 'tum' and self.dataset != 'tumd':
            depth_mask_from = self.all_tracks_static_depth_downsample[all_pairs[:,0]] < 50
            depth_mask_to = self.all_tracks_static_depth_downsample[all_pairs[:,1]] < 50
            mask = torch.logical_and(torch.logical_and(mask, depth_mask_from), depth_mask_to)
        static_tracks_to = self.all_tracks_static_downsample[all_pairs[:, 1]]

        # flow_error = self.loss_fn(gt_coord - pred_coord, self.flow_norm_l1)
        flow_error = flow_norm(static_tracks_to - static_coord, self.flow_norm_l1)
        flow_error = torch.sum(flow_error * mask, dim=1) / (torch.sum(mask, dim=1)+1e-16)
        flow_error = torch.mean(flow_error)
        

        return flow_error

    def optimize_over_pairs_ensemble(self, pairs):

        all_pairs = torch.tensor(pairs)
        

        Rs_from, ts_from = self.get_poses(all_pairs[:, 0])         # N x 3 x 3,     N x 3 x 1
        Rs_to, ts_to = self.get_poses(all_pairs[:, 1])

        K_from = self.get_intrinsics_K(all_pairs[:, 0])
        K_to = self.get_intrinsics_K(all_pairs[:, 1])

        static_tracks = self.all_tracks_static_downsample[all_pairs[:, 0]]
        B, N, _ = static_tracks.shape
        static_homo_from_ = torch.concatenate((static_tracks, torch.ones((B, N, 1)).to(self.device)), axis=-1)
        
        static_homo_from_ = torch.einsum("bji,bni->bnj", torch.linalg.inv(K_from), static_homo_from_)
        static_cam_from =  static_homo_from_ / (static_homo_from_[:,:,2].unsqueeze(-1)+1e-16)
        static_cam_from = static_cam_from * self.all_tracks_static_depth_downsample[all_pairs[:,0]].unsqueeze(-1)      # project into 3D

        static_world_from = torch.einsum("bni,bmi->bmn", Rs_from, static_cam_from) + ts_from.permute(0,2,1)    # F x N x 3

        static_cam_to = torch.einsum("bin,bmi->bmn", Rs_to, static_world_from - ts_to.permute(0,2,1))         # F x N x 3

        static_coord = torch.einsum("bji,bni->bnj", K_to, static_cam_to)
        static_coord = static_coord / (static_coord[:,:,2].unsqueeze(-1)+1e-16)
        static_coord = static_coord[:,:,:2]

        static_tracks_vis_from = self.all_vis_static_downsample[all_pairs[:, 0]]
        static_tracks_vis_to = self.all_vis_static_downsample[all_pairs[:, 1]]
        mask = torch.logical_and(static_tracks_vis_from, static_tracks_vis_to).float()
        
        depth_mask_from = self.all_tracks_static_depth_downsample[all_pairs[:,0]] < 50
        depth_mask_to = self.all_tracks_static_depth_downsample[all_pairs[:,1]] < 50
        mask = torch.logical_and(torch.logical_and(mask, depth_mask_from), depth_mask_to)

        static_tracks_to = self.all_tracks_static_downsample[all_pairs[:, 1]]

        # flow_error = self.loss_fn(gt_coord - pred_coord, self.flow_norm_l1)
        flow_error = flow_norm(static_tracks_to - static_coord, self.flow_norm_l1)
        flow_error = torch.sum(flow_error * mask, dim=1) / (torch.sum(mask, dim=1) + 1e-16)
        flow_error = torch.mean(flow_error)
        
        static_homo_to_ = torch.concatenate((static_tracks_to, torch.ones((B, N, 1)).to(self.device)), axis=-1)
        static_homo_to_ = torch.einsum("bji,bni->bnj", torch.linalg.inv(K_to), static_homo_to_)
        E = compute_essential_matrix_batch(Rs_from, ts_from, Rs_to, ts_to)
        sampson_error = sampson_error_batch(static_homo_from_, static_homo_to_, E)
        sampson_error = torch.sum(sampson_error * mask, dim=1) / (torch.sum(mask, dim=1) + 1e-16)
        sampson_error = torch.mean(sampson_error)
        return flow_error + sampson_error

    def optimize_over_pairs_cycle(self, pairs):

        all_pairs = torch.tensor(pairs)
        

        Rs_from, ts_from = self.get_poses(all_pairs[:, 0])         # N x 3 x 3,     N x 3 x 1
        Rs_to, ts_to = self.get_poses(all_pairs[:, 1])

        K_from = self.get_intrinsics_K(all_pairs[:, 0])
        K_to = self.get_intrinsics_K(all_pairs[:, 1])

        static_tracks = self.all_tracks_static_downsample[all_pairs[:, 0]]
        B, N, _ = static_tracks.shape
        static_homo_from_ = torch.concatenate((static_tracks, torch.ones((B, N, 1)).to(self.device)), axis=-1)

        static_homo_from_ = torch.einsum("bji,bni->bnj", torch.linalg.inv(K_from), static_homo_from_)
        static_cam_from =  static_homo_from_ / (static_homo_from_[:,:,2].unsqueeze(-1)+1e-16)
        static_cam_from = static_cam_from * self.all_tracks_static_depth_downsample[all_pairs[:,0]].unsqueeze(-1)      # project into 3D

        static_world_from = torch.einsum("bni,bmi->bmn", Rs_from, static_cam_from) + ts_from.permute(0,2,1)    # F x N x 3

        static_cam_to = torch.einsum("bin,bmi->bmn", Rs_to, static_world_from - ts_to.permute(0,2,1))         # F x N x 3

        static_coord = torch.einsum("bji,bni->bnj", K_to, static_cam_to)
        static_coord = static_coord / (static_coord[:,:,2].unsqueeze(-1)+1e-16)
        static_coord = static_coord[:,:,:2]

        static_tracks_vis_from = self.all_vis_static_downsample[all_pairs[:, 0]]
        static_tracks_vis_to = self.all_vis_static_downsample[all_pairs[:, 1]]
        mask = torch.logical_and(static_tracks_vis_from, static_tracks_vis_to).float()
        
        if self.dataset != 'tum' and self.dataset != 'tumd':
            depth_mask_from = self.all_tracks_static_depth_downsample[all_pairs[:,0]] < 50
            depth_mask_to = self.all_tracks_static_depth_downsample[all_pairs[:,1]] < 50
            mask = torch.logical_and(torch.logical_and(mask, depth_mask_from), depth_mask_to)

        static_tracks_to = self.all_tracks_static_downsample[all_pairs[:, 1]]
        flow_error = flow_norm(static_tracks_to - static_coord, self.flow_norm_l1)
        static_homo_to_cycle = torch.concatenate((static_coord, torch.ones((B, N, 1)).to(self.device)), axis=-1)
        
        static_homo_to_cycle = torch.einsum("bji,bni->bnj", torch.linalg.inv(K_to), static_homo_to_cycle)
        static_homo_to_cycle =  static_homo_to_cycle / (static_homo_to_cycle[:,:,2].unsqueeze(-1)+1e-16)
        static_homo_to_cycle = static_homo_to_cycle * self.all_tracks_static_depth_downsample[all_pairs[:,1]].unsqueeze(-1)      # project into 3D

        static_world_to_cycle = torch.einsum("bni,bmi->bmn", Rs_to, static_homo_to_cycle) + ts_to.permute(0,2,1)    # F x N x 3

        static_cam_from_cycle = torch.einsum("bin,bmi->bmn", Rs_from, static_world_to_cycle - ts_from.permute(0,2,1))         # F x N x 3

        static_coord_cycle = torch.einsum("bji,bni->bnj", K_from, static_cam_from_cycle)
        static_coord_cycle = static_coord_cycle / (static_coord_cycle[:,:,2].unsqueeze(-1)+1e-16)
        static_coord_cycle = static_coord_cycle[:,:,:2]
        flow_error_cycle = flow_norm(static_tracks - static_coord_cycle, self.flow_norm_l1)
        #print(flow_error.max())
        static_homo_to_ = torch.concatenate((static_tracks_to, torch.ones((B, N, 1)).to(self.device)), axis=-1)
        static_homo_to_ = torch.einsum("bji,bni->bnj", torch.linalg.inv(K_to), static_homo_to_)
        E = compute_essential_matrix_batch(Rs_from, ts_from, Rs_to, ts_to)
        sampson_error = sampson_error_batch(static_homo_from_, static_homo_to_, E)
        filter_large_error_mask = (flow_error < 0.1)  #& (sampson_error < 0.1)# &  (flow_error_cycle < 0.1)
        mask_final = torch.logical_and(mask, filter_large_error_mask.detach())
        flow_error = torch.sum(flow_error * mask_final, dim=1) / (torch.sum(mask_final, dim=1)+1e-16)
        flow_error = torch.mean(flow_error)

        return flow_error 

    def optimize_over_keyframe_buffer(self):

        """
        Optimize using 2D pair wise flow over exhaustive pair wise matching in keyframe buffers
        """

        all_pairs = list(itertools.permutations(self.keyframe_buffer.buffer, 2))

        earlystopper_stage1 = EarlyStopper(patience=20)
        earlystopper_stage2 = EarlyStopper(patience=20)
        stem = 0
        self.reset_optimizer()
        for epoch in range(self.opt.num_init_epochs):

            for o in self.active_optimizers:
                self.optimizers[o].zero_grad()

            if stem == 0:
                if self.opt.use_sampson:
                    total_reproj_loss = self.optimize_over_pairs_ensemble(all_pairs)
                else:
                    total_reproj_loss = self.optimize_over_pairs(all_pairs)
            elif stem == 1:
                total_reproj_loss = self.optimize_over_pairs_cycle(all_pairs)

            total = total_reproj_loss * self.opt.reproj_weight

            if self.opt.pose_smooth_weight_t > 0 and len(self.keyframe_buffer.buffer) >= 3:
                loss_pose_smooth_t, loss_poss_smoother_R = self.calc_pose_smooth_loss(self.keyframe_buffer.buffer)
                loss_pose_smooth_t = loss_pose_smooth_t * self.opt.pose_smooth_weight_t
                loss_pose_smooth_R = loss_poss_smoother_R * self.opt.pose_smooth_weight_r
            else:
                loss_pose_smooth_t = torch.tensor(0)
                loss_pose_smooth_R = torch.tensor(0)

            total += (loss_pose_smooth_t + loss_pose_smooth_R)
            total.backward()

            if stem == 0:
                if earlystopper_stage1.early_stop(total):
                    stem = 1
            elif stem == 1:
                if earlystopper_stage2.early_stop(total):
                    break
                

            for o in self.active_optimizers:
                self.optimizers[o].step()
                if o in self.schedulers:
                    self.schedulers[o].step(total)

            if epoch % self.opt.print_every == 0:
                logging.info(f"Currently at epoch {epoch} out of {self.opt.num_init_epochs} for frames {self.keyframe_buffer.buffer[0]}-{self.keyframe_buffer.buffer[-1]} with pose smooth loss t: {loss_pose_smooth_t.item()} * {self.opt.pose_smooth_weight_t}, pose smooth loss R: {loss_pose_smooth_R.item()} * {self.opt.pose_smooth_weight_r}, reproj loss: {total_reproj_loss.item()} * {self.opt.reproj_weight}")
                logging.info(f"Total loss: {total.item()} with lr: {self.schedulers['translation_pose'].get_last_lr()}")

        self.reset_optimizer()

def propagate_motion(motion_world: torch.Tensor, dyn_filter_vis: torch.Tensor) -> torch.Tensor:
    """
    Update motion_world based on dyn_filter_vis.

    For each point k, if dyn_filter_vis[n, k] == True and dyn_filter_vis[n+1, k] == False,
    the motion_world[k, n+1:, :] is set to motion_world[k, n, :] until the next True is encountered or the sequence ends.
    Essentially, it propagates the last observed motion vector forward when a point becomes temporarily invisible.
    
    Args:
        motion_world: torch.Tensor, shape [num_points, num_frames, 3], e.g., [25082, 75, 3].
                      The motion vectors (displacement) of points in world coordinates.
        dyn_filter_vis: torch.Tensor, shape [num_frames, num_points], e.g., [75, 25082], 
                        containing boolean values (True/False) indicating visibility/validity.
    
    Returns:
        motion_world: The updated motion_world tensor.
    """
    num_points, num_frames, _ = motion_world.shape
    motion_world = motion_world.clone()  # Avoid modifying the original tensor

    for k in range(num_points):  # Iterate through each point
        last_true_frame = -1  # Index of the last frame where the point was visible (True)
        for n in range(num_frames):  # Iterate through each frame
            if dyn_filter_vis[n, k]:  # Current frame is visible (True)
                last_true_frame = n  # Update the index of the last visible frame
            elif last_true_frame >= 0:  # Current frame is invisible (False) AND there was a previous visible frame
                # Set the motion of the current invisible frame (n) to the motion 
                # observed at the last visible frame (last_true_frame)
                motion_world[k, n, :] = motion_world[k, last_true_frame, :]

    return motion_world



def filter_integer_coords_v2(sparse_queried_coords, sparse_queried_frames):
    """
    Filter sparse_queried_coords by keeping only the unique floored [y, x] coordinates,
    and record the merged original indices for each unique coordinate.

    Args:
        sparse_queried_coords (np.ndarray): Shape [M, 3], coordinates with [y, x, ...].
        sparse_queried_frames (np.ndarray): Shape [M], frame indices.

    Returns:
        tuple: (selected_indices, merged_dict)
            - selected_indices: Shape [K], indices in original array for each unique coordinate.
            - merged_dict: dict[int, list[int]], maps unique index (0..K-1) to original indices merged.
    """
    coords = np.around(sparse_queried_coords[..., :2]).astype(np.int64)  # [M, 2]

    # Optionally check this line if needed:
    coords = coords[sparse_queried_frames.astype(np.int64), np.arange(coords.shape[1])]  # seems unnecessary

    frames = sparse_queried_frames[..., np.newaxis]  # [M, 1]
    combined = np.concatenate([coords, frames], axis=-1)  # [M, 3]

    # Find unique (y, x, frame) combinations
    unique_combined, inverse_indices = np.unique(combined, axis=0, return_inverse=True)

    selected_indices = np.zeros(unique_combined.shape[0], dtype=np.int64)
    merged_dict = defaultdict(list)

    for i in range(unique_combined.shape[0]):
        matched = np.nonzero(inverse_indices == i)[0]
        selected_indices[i] = matched[0]
        merged_dict[i] = matched.tolist()

    return selected_indices, merged_dict

def compute_essential_matrix_batch(R_from, t_from, R_to, t_to):
    """
    Compute the Essential matrix E from the relative rotation and translation between two camera poses, in batch.
    
    Args:
        R_from (torch.Tensor): Rotation matrix of the first camera, shape (N, 3, 3).
        t_from (torch.Tensor): Translation vector of the first camera, shape (N, 3, 1).
        R_to (torch.Tensor): Rotation matrix of the second camera, shape (N, 3, 3).
        t_to (torch.Tensor): Translation vector of the second camera, shape (N, 3, 1).
    
    Returns:
        torch.Tensor: Essential matrix, shape (N, 3, 3).
    """
    # Compute relative rotation: R_relative = R_to * R_from^T
    R_relative = torch.matmul(R_to, R_from.transpose(1, 2))  # Shape: (N, 3, 3)
    
    # Compute relative translation: t_relative = t_to - R_relative * t_from
    t_relative = t_to - torch.matmul(R_relative, t_from)  # Shape: (N, 3, 1)
    
    # Compute the skew-symmetric matrix of t_relative: [t_relative]_x
    t_skew = torch.zeros((t_relative.shape[0], 3, 3), dtype=torch.float32, device=t_relative.device)
    t_skew[:, 0, 1] = -t_relative[:, 2, 0]
    t_skew[:, 0, 2] = t_relative[:, 1, 0]
    t_skew[:, 1, 0] = t_relative[:, 2, 0]
    t_skew[:, 1, 2] = -t_relative[:, 0, 0]
    t_skew[:, 2, 0] = -t_relative[:, 1, 0]
    t_skew[:, 2, 1] = t_relative[:, 0, 0]
    
    # Compute the Essential matrix: E = [t_relative]_x * R_relative
    E = torch.matmul(t_skew, R_relative)  # Shape: (N, 3, 3)
    
    return E

def sampson_error_batch(x1, x2, E_batch):
    """
    Compute the Sampson error for a batch of corresponding points and the Essential matrices.
    
    Args:
        x1 (torch.Tensor): Corresponding points in the first image, shape (B, N, 3).
        x2 (torch.Tensor): Corresponding points in the second image, shape (B, N, 3).
        E_batch (torch.Tensor): Essential matrix for each batch, shape (B, 3, 3).
    
    Returns:
        torch.Tensor: Sampson error, shape (B, N).
    """
    B, N, _ = x1.shape  # B: batch size, N: number of points

    # Cross product of points in homogeneous coordinates: x1_cross_E * x2
    # For batch processing, we need to use broadcasting
    x1_cross_E = torch.cross(x1, torch.matmul(E_batch, x2.transpose(1, 2)).transpose(1, 2), dim=2)
    x2_cross_E = torch.cross(x2, torch.matmul(E_batch.transpose(1, 2), x1.transpose(1, 2)).transpose(1, 2), dim=2)
    
    # Compute the Sampson error using the formula:
    # e_s = || x' x2 ||^2 / (|| A * x2 ||^2 + || B * x1 ||^2)
    num = torch.sum(x1_cross_E ** 2, dim=2)  # Numerator: || x' x2 ||^2 (B, N)
    denom = torch.sum(torch.matmul(x1, E_batch) ** 2, dim=2) + torch.sum(torch.matmul(x2, E_batch.transpose(1, 2)) ** 2, dim=2)  # Denominator: (B, N)
    
    sampson_err = num / (denom + 1e-6)  # Sampson error (B, N)
    return sampson_err
