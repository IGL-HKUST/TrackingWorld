# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import imageio
import numpy as np
import torch
from densetrack3d.datasets.utils import DeltaData, add_noise_depth, aug_depth
from densetrack3d.models.model_utils import get_grid, get_points_on_a_grid
from PIL import Image
from torchvision.transforms import ColorJitter, GaussianBlur
import torch.nn.functional as F
try:
    from densetrack3d.datasets.s3_utils import create_client, get_client_stream, read_s3_json
    has_s3 = True
except:
    has_s3 = False

class BasicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
    ):
        super(BasicDataset, self).__init__()
        np.random.seed(0)
        torch.manual_seed(0)
        self.data_root = data_root
        self.seq_len = seq_len
        self.traj_per_sample = traj_per_sample
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.use_augs = use_augs
        self.crop_size = crop_size

        # photometric augmentation
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))

        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.pad_bounds = [0, 100]
        self.crop_size = crop_size
        self.resize_lim = [0.25, 2.0]  # sample resizes from here
        self.resize_delta = 0.2
        self.max_crop_offset = 50

        self.do_flip = True
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.5


    def getitem_helper(self, index):
        return NotImplementedError

    def __getitem__(self, index):
        gotit = False


        while not gotit:
            sample, gotit = self.getitem_helper(index)

            if gotit:
                return sample, True
            
            index = (index +1) % self.__len__()

        # sample, gotit = self.getitem_helper(index)
        # if not gotit:
        #     print("warning: sampling failed")
        #     # fake sample, so we can still collate
        #     sample = CoTrackerData(
        #         video=torch.zeros((self.seq_len, 3, self.crop_size[0], self.crop_size[1])),
        #         trajectory=torch.zeros((self.seq_len, self.traj_per_sample, 2)),
        #         visibility=torch.zeros((self.seq_len, self.traj_per_sample)),
        #         valid=torch.zeros((self.seq_len, self.traj_per_sample)),
        #     )

        return sample, gotit

    def add_photometric_augs(self, rgbs, trajs, visibles, sparse_trajs, sparse_visibles, eraser=True, replace=True):
        # T, , _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        # assert S == T

        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.eraser_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = np.random.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        mean_color = np.mean(rgbs[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                        rgbs[i][y0:y1, x0:x1, :] = mean_color

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, :, 0] >= x0, trajs[i, :, :, 0] < x1),
                            np.logical_and(trajs[i, :, :, 1] >= y0, trajs[i, :, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0

                        sparse_occ_inds = np.logical_and(
                            np.logical_and(sparse_trajs[i, :, 0] >= x0, sparse_trajs[i, :, 0] < x1),
                            np.logical_and(sparse_trajs[i, :, 1] >= y0, sparse_trajs[i, :, 1] < y1),
                        )
                        sparse_visibles[i, sparse_occ_inds] = 0


            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:
            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs
            ]
            rgbs_alt = [
                np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt
            ]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(
                        np.random.randint(1, self.replace_max + 1)
                    ):  # number of times to occlude
                        xc = np.random.randint(0, W)
                        yc = np.random.randint(0, H)
                        dx = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = np.random.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = np.random.randint(0, H - hei)
                        x00 = np.random.randint(0, W - wid)
                        fr = np.random.randint(0, S)
                        rep = rgbs_alt[fr][y00 : y00 + hei, x00 : x00 + wid, :]
                        rgbs[i][y0:y1, x0:x1, :] = rep

                        occ_inds = np.logical_and(
                            np.logical_and(trajs[i, :, :, 0] >= x0, trajs[i, :, :, 0] < x1),
                            np.logical_and(trajs[i, :, :, 1] >= y0, trajs[i, :, :, 1] < y1),
                        )
                        visibles[i, occ_inds] = 0

                        sparse_occ_inds = np.logical_and(
                            np.logical_and(sparse_trajs[i, :, 0] >= x0, sparse_trajs[i, :, 0] < x1),
                            np.logical_and(sparse_trajs[i, :, 1] >= y0, sparse_trajs[i, :, 1] < y1),
                        )
                        sparse_visibles[i, sparse_occ_inds] = 0

            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, trajs, visibles, sparse_trajs, sparse_visibles

    def add_spatial_augs(self, rgbs, depths, pred_depths, trajs, trajs_depth, visibles, flows, flow_depths, sparse_trajs, sparse_visibles, dino_features):
        # T, N, __ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        # assert S == T

        ok_inds = visibles[0] > 0

        rgbs = [rgb.astype(np.float32) for rgb in rgbs]
        depths = [depth.astype(np.float32) for depth in depths]
        dino_features = [dino_feature.astype(np.float32) for dino_feature in dino_features]
        pred_depths = [pred_depth.astype(np.float32) for pred_depth in pred_depths]
        flows = [flow.astype(np.float32) for flow in flows]
        flow_depths = [flow_depth.astype(np.float32) for flow_depth in flow_depths]
        visibles = [visible.astype(bool) for visible in visibles]
        trajs = [traj.astype(np.float32) for traj in trajs]
        trajs_depth = [traj_d.astype(np.float32) for traj_d in trajs_depth]

        ############ spatial transform ############

                # # padding
                # pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
                # pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
                # pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
                # pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

                # rgbs = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs]
                # trajs[:, :, :, 0] += pad_x0
                # trajs[:, :, :, 1] += pad_y0
        H, W = rgbs[0].shape[:2]

                # # scaling + stretching
                # scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
                # scale_x = scale
                # scale_y = scale
                # H_new = H
                # W_new = W

                # scale_delta_x = 0.0
                # scale_delta_y = 0.0

                # rgbs_scaled = []
                # flows_scaled, visibles_scaled = [], []
                # for s in range(S):
                #     if s == 1:
                #         scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                #         scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
                #     elif s > 1:
                #         scale_delta_x = (
                #             scale_delta_x * 0.8
                #             + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                #         )
                #         scale_delta_y = (
                #             scale_delta_y * 0.8
                #             + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                #         )
                #     scale_x = scale_x + scale_delta_x
                #     scale_y = scale_y + scale_delta_y

                #     # bring h/w closer
                #     scale_xy = (scale_x + scale_y) * 0.5
                #     scale_x = scale_x * 0.5 + scale_xy * 0.5
                #     scale_y = scale_y * 0.5 + scale_xy * 0.5

                #     # don't get too crazy
                #     scale_x = np.clip(scale_x, 0.2, 2.0)
                #     scale_y = np.clip(scale_y, 0.2, 2.0)

                #     H_new = int(H * scale_y)
                #     W_new = int(W * scale_x)

                #     # make it at least slightly bigger than the crop area,
                #     # so that the random cropping can add diversity
                #     H_new = np.clip(H_new, self.crop_size[0] + 10, None)
                #     W_new = np.clip(W_new, self.crop_size[1] + 10, None)
                #     # recompute scale in case we clipped
                #     scale_x = (W_new - 1) / float(W - 1)
                #     scale_y = (H_new - 1) / float(H - 1)

                #     rgbs_scaled.append(cv2.resize(rgbs[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR))

                #     flow_scaled = cv2.resize(flows[s], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
                #     flow_scaled[..., 0] *= scale_x
                #     flow_scaled[..., 1] *= scale_y
                #     flows_scaled.append(flow_scaled)

                #     visibles_scaled.append(cv2.resize(visibles[s].astype(np.uint8), (W_new, H_new), interpolation=cv2.INTER_NEAREST).astype(bool))
                    
                #     trajs[s, :, :, 0] *= scale_x
                #     trajs[s, :, :, 1] *= scale_y

                # rgbs = rgbs_scaled
                # flows = flows_scaled
                # visibles = visibles_scaled

        
                # vis_trajs = trajs[:, ok_inds]  # S,?,2

                # if ok_inds.sum() > 0:
                #     mid_x = np.mean(vis_trajs[0, :,  0])
                #     mid_y = np.mean(vis_trajs[0, :,  1])
                # else:
                #     mid_y = self.crop_size[0]
                #     mid_x = self.crop_size[1]

                # x0 = int(mid_x - self.crop_size[1] // 2)
                # y0 = int(mid_y - self.crop_size[0] // 2)

                # offset_x = 0
                # offset_y = 0

                # for s in range(S):
                #     # on each frame, shift a bit more
                #     if s == 1:
                #         offset_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                #         offset_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset)
                #     elif s > 1:
                #         offset_x = int(
                #             offset_x * 0.8
                #             + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                #         )
                #         offset_y = int(
                #             offset_y * 0.8
                #             + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1) * 0.2
                #         )
                #     x0 = x0 + offset_x
                #     y0 = y0 + offset_y

                #     H_new, W_new = rgbs[s].shape[:2]
                #     if H_new == self.crop_size[0]:
                #         y0 = 0
                #     else:
                #         y0 = min(max(0, y0), H_new - self.crop_size[0] - 1)

                #     if W_new == self.crop_size[1]:
                #         x0 = 0
                #     else:
                #         x0 = min(max(0, x0), W_new - self.crop_size[1] - 1)

                #     rgbs[s] = rgbs[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
                #     flows[s] = flows[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
                #     visibles[s] = visibles[s][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

                #     trajs[s, :, :, 0] -= x0
                #     trajs[s, :, :, 1] -= y0

        y0 = 0 if self.crop_size[0] >= H else np.random.randint(0, H - self.crop_size[0])
        x0 = 0 if self.crop_size[1] >= W else np.random.randint(0, W - self.crop_size[1])

        rgbs = [rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for rgb in rgbs]
        depths = [depth[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for depth in depths]
        dino_features = [dino_feature[:, y0//8 : y0//8 + self.crop_size[0]//8, x0//8 : x0//8 + self.crop_size[1]//8] for dino_feature in dino_features]
        pred_depths = [pred_depth[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for pred_depth in pred_depths]
        flows = [flow[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for flow in flows]
        flow_depths = [flow_depth[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for flow_depth in flow_depths]
        visibles = [visible[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for visible in visibles]

        for i in range(len(trajs)):
            trajs[i][:, :, 0] -= x0
            trajs[i][:, :, 1] -= y0
            trajs[i] = trajs[i][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        
        trajs_depth = [traj_d[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for traj_d in trajs_depth]


        sparse_trajs[:, :, 0] -= x0
        sparse_trajs[:, :, 1] -= y0

        H_new = self.crop_size[0]
        W_new = self.crop_size[1]

        # flip
        h_flipped = False
        v_flipped = False
        if self.do_flip:
            # h flip
            if np.random.rand() < self.h_flip_prob:
                h_flipped = True
                rgbs = [rgb[:, ::-1] for rgb in rgbs]
                depths = [depth[:, ::-1] for depth in depths]
                dino_features = [dino_feature[:, :, ::-1] for dino_feature in dino_features]
                pred_depths = [pred_depth[:, ::-1] for pred_depth in pred_depths]
            # v flip
            if np.random.rand() < self.v_flip_prob:
                v_flipped = True
                rgbs = [rgb[::-1] for rgb in rgbs]
                depths = [depth[::-1] for depth in depths]
                dino_features = [dino_feature[:, ::-1] for dino_feature in dino_features]
                pred_depths = [pred_depth[::-1] for pred_depth in pred_depths]

        if h_flipped:
            # trajs = [W_new - traj[:, :, 0] for traj in trajs]
            for i in range(len(trajs)):
                trajs[i][:, :, 0] = W_new - trajs[i][:, :, 0]
                trajs[i] = trajs[i][:, ::-1]
            trajs_depth = [traj_d[:, ::-1] for traj_d in trajs_depth]
            # trajs[:, :, :, 0] = W_new - trajs[:, :, :, 0]
            flows = [flow[:, ::-1] * [-1.0, 1.0] for flow in flows]
            flow_depths = [flow_d[:, ::-1] for flow_d in flow_depths]
            visibles = [visible[:, ::-1] for visible in visibles]

            sparse_trajs[:, :, 0] = W_new - sparse_trajs[:, :, 0]

        if v_flipped:
            for i in range(len(trajs)):
                trajs[i][:, :, 1] = H_new - trajs[i][:, :, 1]
                trajs[i] = trajs[i][::-1]
            trajs_depth = [traj_d[::-1] for traj_d in trajs_depth]
            # trajs = [H_new - traj[:, :, 1] for traj in trajs]
            # trajs[:, :, :, 1] = H_new - trajs[:, :, :, 1]
            flows = [flow[::-1] * [1.0, -1.0] for flow in flows]
            flow_depths = [flow_d[::-1] for flow_d in flow_depths]
            visibles = [visible[::-1] for visible in visibles]

            sparse_trajs[:, :, 1] = H_new - sparse_trajs[:, :, 1]

        rgbs = np.stack(rgbs, axis=0)
        depths = np.stack(depths, axis=0)
        dino_features = np.stack(dino_features, axis=0)
        pred_depths = np.stack(pred_depths, axis=0)
        flows = np.stack(flows, axis=0)
        flow_depths = np.stack(flow_depths, axis=0)
        visibles = np.stack(visibles, axis=0)
        trajs = np.stack(trajs, axis=0)
        trajs_depth = np.stack(trajs_depth, axis=0)

            # new_grid = get_grid(self.crop_size[0], self.crop_size[1], dtype="numpy", normalize=False)[None] # 1 H W 2
            # new_traj = flows + new_grid
            # out_of_bound_traj = (new_traj[...,0] < 0) | (new_traj[...,0] > self.crop_size[1] - 1) | (new_traj[...,1] < 0) | (new_traj[...,1] > self.crop_size[0] - 1)
        out_of_bound_traj = (trajs[...,0] < 0) | (trajs[...,0] > self.crop_size[1] - 1) | (trajs[...,1] < 0) | (trajs[...,1] > self.crop_size[0] - 1)
        visibles[out_of_bound_traj] = 0

        out_of_bound_sparse_traj = (sparse_trajs[...,0] < 0) | (sparse_trajs[...,0] > self.crop_size[1] - 1) | (sparse_trajs[...,1] < 0) | (sparse_trajs[...,1] > self.crop_size[0] - 1)
        sparse_visibles[out_of_bound_sparse_traj] = 0

        return rgbs, depths, pred_depths, trajs, trajs_depth, visibles, flows, flow_depths, sparse_trajs, sparse_visibles, dino_features

    def crop_rgb_and_flow(self, rgbs, depths, pred_depths, trajs, trajs_depth, visibles, flows, flow_depths, sparse_trajs, sparse_visibles, dino_features):

        # rgbs, depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility
        # T, N, _ = trajs.shape

        # S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        # assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if self.crop_size[0] >= H_new else np.random.randint(0, H_new - self.crop_size[0])
        x0 = 0 if self.crop_size[1] >= W_new else np.random.randint(0, W_new - self.crop_size[1])

        rgbs = [rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for rgb in rgbs]
        dino_features = [dino_feature[:, y0//8 : y0//8 + self.crop_size[0]//8, x0//8 : x0//8 + self.crop_size[1]//8] for dino_feature in dino_features]
        depths = [depth[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for depth in depths]
        pred_depths = [pred_depths[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for pred_depths in pred_depths]
        flows = [flow[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for flow in flows]
        flow_depths = [flow_depth[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for flow_depth in flow_depths]
        visibles = [visible[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for visible in visibles]

        # new_trajs = []
        trajs = [traj.astype(np.float32) for traj in trajs]
        for i in range(len(trajs)):
            trajs[i][:, :, 0] -= x0
            trajs[i][:, :, 1] -= y0
            trajs[i] = trajs[i][y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        
        trajs_depth = [traj_d[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for traj_d in trajs_depth]

        sparse_trajs[:, :, 0] -= x0
        sparse_trajs[:, :, 1] -= y0

        rgbs = np.stack(rgbs, axis=0)
        depths = np.stack(depths, axis=0)
        dino_features = np.stack(dino_features, axis=0)
        pred_depths = np.stack(pred_depths, axis=0)
        flows = np.stack(flows, axis=0)
        flow_depths = np.stack(flow_depths, axis=0)
        visibles = np.stack(visibles, axis=0)
        trajs = np.stack(trajs, axis=0)
        trajs_depth = np.stack(trajs_depth, axis=0)

        out_of_bound_traj = (trajs[...,0] < 0) | (trajs[...,0] > self.crop_size[1] - 1) | (trajs[...,1] < 0) | (trajs[...,1] > self.crop_size[0] - 1)
        visibles[out_of_bound_traj] = 0

        out_of_bound_sparse_traj = (sparse_trajs[...,0] < 0) | (sparse_trajs[...,0] > self.crop_size[1] - 1) | (sparse_trajs[...,1] < 0) | (sparse_trajs[...,1] > self.crop_size[0] - 1)
        sparse_visibles[out_of_bound_sparse_traj] = 0

        # rgbs = [rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for rgb in rgbs]
        # flows = [flow[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for flow in flows]
        # visi_maps = [visi_map[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for visi_map in visi_maps]
        # trajs[:, :, 0] -= x0
        # trajs[:, :, 1] -= y0

        # rgbs = np.stack(rgbs, axis=0)
        # flows = np.stack(flows, axis=0)
        # visi_maps = np.stack(visi_maps, axis=0)

        # new_grid = get_grid(self.crop_size[0], self.crop_size[1], dtype="numpy", normalize=False)[None] # 1 H W 2
        # new_traj = flows + new_grid
        # out_of_bound_traj = (new_traj[...,0] < 0) | (new_traj[...,0] > self.crop_size[1] - 1) | (new_traj[...,1] < 0) | (new_traj[...,1] > self.crop_size[0] - 1)
        # visi_maps[out_of_bound_traj] = 0

        return rgbs, depths, pred_depths, trajs, trajs_depth, visibles, flows, flow_depths, sparse_trajs, sparse_visibles, dino_features


class KubricDataset(BasicDataset):
    def __init__(
        self,
        data_root,
        crop_size=(384, 512),
        seq_len=24,
        traj_per_sample=768,
        sample_vis_1st_frame=False,
        use_augs=False,
        use_extra=False,
        len_data=-1,
        is_val=False,
        use_gt_depth=True,
        add_noise_depth=False,
        read_from_s3=False,
    ):
        super(KubricDataset, self).__init__(
            data_root=data_root,
            crop_size=crop_size,
            seq_len=seq_len,
            traj_per_sample=traj_per_sample,
            sample_vis_1st_frame=sample_vis_1st_frame,
            use_augs=use_augs,
        )

        self.read_from_s3 = read_from_s3
        self.s3_client = None

        self.pad_bounds = [0, 25]
        self.resize_lim = [0.75, 1.25]  # sample resizes from here
        self.resize_delta = 0.05
        self.max_crop_offset = 15


        # self.seq_names = [
        #     fname
        #     for fname in os.listdir(data_root)
        #     if os.path.isdir(os.path.join(data_root, fname))
        # ]
        #self.seq_names = self.seq_names[:3000]
        self.seq_names = [f"{fid:04d}" for fid in range(0, 4900)]

        self.use_gt_depth = use_gt_depth
        self.add_noise_depth = add_noise_depth

        self.is_val = is_val
        
        if len_data > -1:
            stride = len(self.seq_names) // len_data
            self.seq_names = self.seq_names[::stride]
            

        # if is_val:
        #     self.seq_names = sorted(self.seq_names)[::100]
        print("found %d unique videos in %s" % (len(self.seq_names), self.data_root))

    def _initialize_s3(self):
        assert has_s3
        if self.s3_client is None:
            self.s3_client = create_client()
        return self.s3_client

    def __getitem__(self, index):

        try:
            if self.is_val:
                sample, gotit = self.getitem_helper(index)
                return sample
            
            gotit = False


            while not gotit:
                sample, gotit = self.getitem_helper(index)

                if gotit:
                    return sample, True
                
                index = (index +1) % self.__len__()
        except:
            if self.read_from_s3 and self.s3_client is not None:
                del self.s3_client
                self.s3_client = None

            index = (index + 1) % self.__len__()
            return self.__getitem__(index)

    def lsq_depth(self, depth, pred_depth):
        T, H, W = depth.shape[:3] # T H W 1

        
        x = pred_depth.copy().flatten()
        y = depth.copy().flatten()
        A = np.vstack([x, np.ones(len(x))]).T

        s, t = np.linalg.lstsq(A, y, rcond=None)[0]

        aligned_pred_depth = pred_depth * s + t


        return aligned_pred_depth.reshape(T, H, W, 1), s, t 

    def getitem_helper(self, index):
        if self.read_from_s3:
            self._initialize_s3()

        gotit = True
        seq_name = self.seq_names[index]
        data_root_ = self.data_root
        img_ids = [f"{i:03d}" for i in range(24)]

        rgb_path = os.path.join(data_root_, seq_name, "frames")
        

        rgbs = []
        for i in img_ids:
            img_path = os.path.join(rgb_path, f"{i}.png")
            if self.read_from_s3:
                img = imageio.v2.imread(get_client_stream(self.s3_client, img_path))
            else:
                img = imageio.v2.imread(img_path)
            rgbs.append(img)
        rgbs = np.stack(rgbs)

        depth_dir = os.path.join(data_root_, seq_name, 'depths')
        depths = []
        for i in img_ids:
            depth_path = os.path.join(depth_dir, f"{i}.png")
            if self.read_from_s3:
                depth = read_s3_img_cv2(self.s3_client, depth_path, is_depth=True).astype(np.float32)
            else:
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH).astype(np.float32)
            
            depths.append(depth)
        depths = np.stack(depths)[..., None]   # T, H, W, 1

        # pred_depth_path = os.path.join(data_root_, seq_name, "unidepth_depths.npy")
        # # assert os.path.exists(pred_depth_path)
        # if self.read_from_s3:
        #     pred_depths = np.load(get_client_stream(self.s3_client, pred_depth_path))
        # else:
        #     pred_depths = np.load(pred_depth_path)
        # pred_depths = pred_depths.astype(np.float32)[..., None]
        pred_depths = np.zeros_like(depths)

        dino_dir = os.path.join(data_root_, seq_name, 'dino_embeddings', 'dino_embed_video.pt')
        dino_features = torch.load(dino_dir, map_location=torch.device('cpu'))
        dino_features = F.interpolate(dino_features, size=(rgbs.shape[1]//8, rgbs.shape[2]//8), mode='bilinear', align_corners=False).numpy()
        if np.random.rand() < 0.5 or self.is_val:
            is_reverse = False
            npy_dense_path = os.path.join(data_root_, seq_name, seq_name + "_dense.npy")
            if self.read_from_s3:
                dense_annot_dict = np.load(get_client_stream(self.s3_client, npy_dense_path), allow_pickle=True).item()
            else:
                dense_annot_dict = np.load(npy_dense_path, allow_pickle=True).item()
            dense_traj_2d = dense_annot_dict["coords"].astype(np.float32)
            dense_traj_depth = dense_annot_dict["reproj_depth"].astype(np.float32)[..., None]
            dense_visibility = dense_annot_dict["visibility"].astype(bool)
            dense_queries = dense_annot_dict["queries"].astype(np.float32) # N, 3
        else:
            is_reverse = True
            npy_dense_path = os.path.join(data_root_, seq_name, seq_name + "_dense_reverse.npy")
            if self.read_from_s3:
                dense_annot_dict = np.load(get_client_stream(self.s3_client, npy_dense_path), allow_pickle=True).item()
            else:
                dense_annot_dict = np.load(npy_dense_path, allow_pickle=True).item()
            dense_traj_2d = dense_annot_dict["coords"].astype(np.float32)
            dense_traj_depth = dense_annot_dict["reproj_depth"].astype(np.float32)[..., None]
            dense_visibility = dense_annot_dict["visibility"].astype(bool)
            dense_queries = dense_annot_dict["queries"].astype(np.float32) # N, 3

        npy_path = os.path.join(data_root_, seq_name, seq_name + ".npy")
        if self.read_from_s3:
            annot_dict = np.load(get_client_stream(self.s3_client, npy_path), allow_pickle=True).item()
        else:
            annot_dict = np.load(npy_path, allow_pickle=True).item()

        sparse_traj_2d = annot_dict["coords"].astype(np.float32)
        sparse_traj_depth = annot_dict["reproj_depth"].astype(np.float32)[..., None]
        sparse_visibility = annot_dict["visibility"].astype(bool)
        depth_range = annot_dict["depth_range"].astype(float)
        # sparse_queries = annot_dict["sparse_queries"].astype(np.float32) # N, 3

        # if self.use_gt_depth:
        depth_min = depth_range[0]
        depth_max = depth_range[1]
        depth_f32 = depths.astype(float)
        
        depths = depth_min + depth_f32 * (depth_max-depth_min) / 65535.0

        # aligned_pred_depths = self.lsq_depth(depths, pred_depths)[0]
        # aligned_pred_depths[aligned_pred_depths<depth_min] = depth_min
        # aligned_pred_depths[aligned_pred_depths>depth_max] = depth_max

        # pred_depths = aligned_pred_depths

        # if not self.use_gt_depth:
        #     if np.random.rand() < 0.5:
                
        #         depths = aligned_pred_depths
        #         if np.any(np.isnan(depths)):
        #             print("warning: depth contains nan")
        #             gotit = False
        #             return None, gotit
                
        #         if np.any(np.isinf(np.absolute(depths))):
        #             print("warning: depth contains inf")
        #             gotit = False
        #             return None, gotit

        #         depths[depths<depth_min] = depth_min
        #         depths[depths>depth_max] = depth_max



        if is_reverse:
            rgbs = rgbs[::-1]
            dino_features = dino_features[::-1]
            depths = depths[::-1]
            pred_depths = pred_depths[::-1]
            sparse_traj_2d = sparse_traj_2d[:, ::-1]
            sparse_traj_depth = sparse_traj_depth[:, ::-1]
            sparse_visibility = sparse_visibility[:, ::-1]

            dense_traj_2d = dense_traj_2d[:, ::-1]
            dense_traj_depth = dense_traj_depth[:, ::-1]
            dense_visibility = dense_visibility[:, ::-1]
        

        if np.any(np.isinf(sparse_traj_2d)) or np.any(np.isinf(dense_traj_2d)):
            print("warning: traj contains inf")
            gotit = False
            return None, gotit

        

        # random crop
        assert self.seq_len <= len(rgbs)
        if self.seq_len < len(rgbs) and not self.is_val:
            # start_ind = np.random.choice(len(rgbs) - self.seq_len, 1)[0]
            start_ind = 0

            rgbs = rgbs[start_ind : start_ind + self.seq_len]
            dino_features = dino_features[start_ind : start_ind + self.seq_len]
            depths = depths[start_ind : start_ind + self.seq_len]
            pred_depths = pred_depths[start_ind:start_ind + self.seq_len]
            sparse_traj_2d = sparse_traj_2d[:, start_ind : start_ind + self.seq_len]
            sparse_traj_depth = sparse_traj_depth[:, start_ind : start_ind + self.seq_len]
            sparse_visibility = sparse_visibility[:, start_ind : start_ind + self.seq_len]


            dense_traj_2d = dense_traj_2d[:, start_ind : start_ind + self.seq_len]
            dense_traj_depth = dense_traj_depth[:, start_ind : start_ind + self.seq_len]
            dense_visibility = dense_visibility[:, start_ind : start_ind + self.seq_len]

        sparse_traj_2d = np.transpose(sparse_traj_2d, (1, 0, 2)) # T N 2
        sparse_traj_depth = np.transpose(sparse_traj_depth, (1, 0, 2))
        sparse_visibility = np.transpose(np.logical_not(sparse_visibility), (1, 0)) # T N 

        dense_traj_2d = np.transpose(dense_traj_2d, (1, 0, 2)) # T N 2
        dense_traj_depth = np.transpose(dense_traj_depth, (1, 0, 2))
        dense_visibility = np.transpose(np.logical_not(dense_visibility), (1, 0)) # T N 

        dense_queries = (dense_queries[:, 1:] - 0.5).astype(int)
        dense_queries = np.stack([dense_queries[:, 1], dense_queries[:, 0]], axis=1) # yx -> xy

        dense_queries_indices =  512 * dense_queries[:, 1] + dense_queries[:, 0]
        sort_indices = np.argsort(dense_queries_indices)

        dense_traj_grid = dense_traj_2d[:, sort_indices].reshape(self.seq_len,512,512,2)
        dense_traj_depth_grid = dense_traj_depth[:, sort_indices].reshape(self.seq_len,512,512,1)
        dense_visi_maps = dense_visibility[:, sort_indices].reshape(self.seq_len,512,512)

        ori_grid = get_grid(512, 512, dtype="numpy", normalize=False, align_corners=False)[None] # 1 H W 2
        dense_flows = dense_traj_grid - ori_grid
        dense_flow_depths = dense_traj_depth_grid - dense_traj_depth_grid[0:1]


        # print(visibility[0].sum())
        
            
        # else:
        if not self.is_val:
            if self.use_augs:
                rgbs, dense_traj_grid, dense_visi_maps, sparse_traj_2d, sparse_visibility = \
                    self.add_photometric_augs(rgbs, dense_traj_grid, dense_visi_maps, sparse_traj_2d, sparse_visibility)
                rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility, dino_features = \
                    self.add_spatial_augs(rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility, dino_features)
            else:
                rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility, dino_features = self.crop_rgb_and_flow(rgbs, depths, pred_depths, dense_traj_grid, dense_traj_depth_grid, dense_visi_maps, dense_flows, dense_flow_depths, sparse_traj_2d, sparse_visibility, dino_features)

        # if not self.is_val:
        #     rgbs, flows, visi_maps = self.crop_rgb_and_flow(rgbs, flows, visi_maps)

                # visibility[traj_2d[:, :, 0] > self.crop_size[1] - 1] = False
                # visibility[traj_2d[:, :, 0] < 0] = False
                # visibility[traj_2d[:, :, 1] > self.crop_size[0] - 1] = False
                # visibility[traj_2d[:, :, 1] < 0] = False

        

        rgbs = torch.from_numpy(np.ascontiguousarray(rgbs)).permute(0, 3, 1, 2).float()
        depths = torch.from_numpy(np.ascontiguousarray(depths)).permute(0, 3, 1, 2).float()
        dino_features = torch.from_numpy(np.ascontiguousarray(dino_features)).float()
        pred_depths = torch.from_numpy(np.ascontiguousarray(pred_depths)).permute(0, 3, 1, 2).float()
        dense_flows = torch.from_numpy(np.ascontiguousarray(dense_flows)).permute(0, 3, 1, 2).float()
        dense_flow_depths = torch.from_numpy(np.ascontiguousarray(dense_flow_depths)).permute(0, 3, 1, 2).float()
        dense_visi_maps = torch.from_numpy(np.ascontiguousarray(dense_visi_maps)).float()

        dense_traj_grid = torch.from_numpy(np.ascontiguousarray(dense_traj_grid)).permute(0, 3, 1, 2).float()
        dense_traj_depth_grid = torch.from_numpy(np.ascontiguousarray(dense_traj_depth_grid)).permute(0, 3, 1, 2).float()

        sparse_traj_2d = torch.from_numpy(np.ascontiguousarray(sparse_traj_2d)).float()
        sparse_traj_depth = torch.from_numpy(np.ascontiguousarray(sparse_traj_depth)).float()
        sparse_visibility = torch.from_numpy(np.ascontiguousarray(sparse_visibility)).float()

        if not self.is_val:
            visibile_pts_first_frame_inds = (sparse_visibility[0]).nonzero(as_tuple=False)[:, 0]
            if self.sample_vis_1st_frame:
                visibile_pts_inds = visibile_pts_first_frame_inds
            else:
                visibile_pts_mid_frame_inds = (sparse_visibility[self.seq_len // 2]).nonzero(as_tuple=False)[
                    :, 0
                ]
                visibile_pts_inds = torch.cat(
                    (visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0
                )

            if len(visibile_pts_inds) >= self.traj_per_sample:
                point_inds = torch.randperm(len(visibile_pts_inds))[: self.traj_per_sample]
            else:
                point_inds = np.random.choice(len(visibile_pts_inds), self.traj_per_sample, replace=True)
            visible_inds_sampled = visibile_pts_inds[point_inds]

            sparse_traj_2d = sparse_traj_2d[:, visible_inds_sampled].float()
            sparse_traj_depth = sparse_traj_depth[:, visible_inds_sampled].float()
            sparse_visibility = sparse_visibility[:, visible_inds_sampled]
        sparse_valids = torch.ones_like(sparse_visibility)

        # print("videodepth", depths.max(), depths.min())

        depth_init = depths[0].clone()
        if not self.is_val and self.use_augs:
            depths = aug_depth(depths,
                    grid=(8, 8),
                    scale=(0.85, 1.15),
                    shift=(-0.05, 0.05),
                    gn_kernel=(7, 7),
                    gn_sigma=(2, 2),
                    mask_depth=(depths >= 0.01)
            )

            if self.add_noise_depth:
                depths = add_noise_depth(
                    depths, 
                    gn_sigma=0.3, 
                    mask_depth=(depths >= 0.01)
                )

        
        dense_valid = torch.ones_like(dense_visi_maps)


        # if self.is_val:
        #     rgbs = rgbs[:8]
        #     depths = depths[:8]
        #     dense_flows = dense_flows[6].permute(1, 2, 0)
        #     dense_visi_maps = dense_visi_maps[6]

        # T, _, H, W = rgbs.shape
        sample = DeltaData(
            video=rgbs,
            videodepth=depths,
            videodepth_pred=pred_depths,
            dino_features=dino_features,
            depth_init=depth_init,
            trajectory=sparse_traj_2d,
            trajectory_d=sparse_traj_depth,
            visibility=sparse_visibility,
            valid=sparse_valids,
            dense_trajectory=dense_traj_grid,
            dense_trajectory_d=dense_traj_depth_grid,
            dense_valid=dense_valid,
            flow=dense_flows,
            flow_depth=dense_flow_depths,
            flow_alpha=dense_visi_maps,
            seq_name=seq_name,
            dataset_name="kubric_dense",
            depth_min=torch.tensor(depth_range[0]),
            depth_max=torch.tensor(depth_range[1]),
        )

        # print(sample)
        return sample, gotit

    def __len__(self):
        return len(self.seq_names)

train_dataset = KubricDataset(
            data_root="/zhdd/dataset/ljh/movi_f/kubric_processed_mix_3d/",
            crop_size=(384, 512),
            seq_len=16,
            traj_per_sample=256,
            sample_vis_1st_frame=True,
            use_augs=True,
            use_gt_depth=True,
        )
train_dataset.getitem_helper(0)