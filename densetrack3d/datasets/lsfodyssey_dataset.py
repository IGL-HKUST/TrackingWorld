import glob
import io
import os
import pickle
from typing import Mapping, Tuple, Union

import cv2
import mediapy as media
import numpy as np
import torch
from densetrack3d.datasets.lsf_augmentor import LongTermSceneFlowAugmentor
from densetrack3d.datasets.utils import DeltaData
from PIL import Image


#

DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]


class LSFOdyssey(torch.utils.data.Dataset):
    def __init__(
        self,
        aug_params=None,
        root="none",
        seq_len=40,
        track_point_num=128,
        split="train",
        train_mode=False,
        track_mode="frame_first",
    ):
        super(LSFOdyssey, self).__init__()

        self.track_mode = track_mode
        self.train_mode = train_mode

        self.augmentor = None
        if aug_params is not None:
            self.augmentor = LongTermSceneFlowAugmentor(train_mode, **aug_params)

        self.seq_len = seq_len
        self.track_point_num = track_point_num
        self.split = split

        self.meta_list = []

        data_root = "data/demo" if split == "demo" else f"{root}/{split}"

        seq_path_list = []
        for seq_path in glob.glob(os.path.join(data_root, "*")):
            seq_path = seq_path.replace("\\", "/")
            if os.path.isdir(seq_path):
                seq_path_list.append(seq_path)
        seq_path_list = sorted(seq_path_list)

        for seq_path in seq_path_list:
            for sample_path in glob.glob(os.path.join(seq_path, "*")):
                sample_path = sample_path.replace("\\", "/")

                # print(sample_path)
                mp4_name_path = f"{sample_path}/rgb.mp4"
                deps_name_path = f"{sample_path}/deps.npz"
                track_name_path = f"{sample_path}/track.npz"
                intris_name_path = f"{sample_path}/intris.npz"
                self.meta_list += [
                    {
                        "name": sample_path,
                        "mp4_name_path": mp4_name_path,
                        "deps_name_path": deps_name_path,
                        "track_name_path": track_name_path,
                        "intris_name_path": intris_name_path if split != "train" else None,
                    }
                ]

    def __getitem__(self, index):
        if self.train_mode:
            while True:
                sample = self.get_data_unit(index)
                if sample is None:
                    index = (index + 1) % len(self.meta_list)
                else:
                    # if self.augmentor is not None:
                    #     sample = self.augmentor(sample)
                    return sample, True
        else:
            return self.get_data_unit(index)
        # return self.get_data_unit(index)
        # data_invalid = True
        # while data_invalid:

        #     index = index % len(self.meta_list)
        #     du = self.get_data_unit(index)

        #     if self.augmentor is not None:
        #         du = self.augmentor(du)

        #     outputs, data_invalid, index = self.prepare_output(du)

        # return outputs

    def __len__(self):
        return len(self.meta_list)

    def __rmul__(self, v):
        self.meta_list = v * self.meta_list
        return self

    def read_mp4(self, name_path):
        vidcap = cv2.VideoCapture(name_path)
        frames = []
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if ret == False:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        vidcap.release()

        return frames

    def get_data_unit(self, index):

        # du = {}

        mp4_name_path = self.meta_list[index]["mp4_name_path"]
        deps_name_path = self.meta_list[index]["deps_name_path"]
        track_name_path = self.meta_list[index]["track_name_path"]
        intris_name_path = self.meta_list[index]["intris_name_path"]

        video_name = self.meta_list[index]["name"].split("/")[-2] + "_" + self.meta_list[index]["name"].split("/")[-1]

        rgbs = self.read_mp4(mp4_name_path)  # list, each shape: (H, W, 3), cv2 type
        rgbs = np.stack(rgbs)  # shape: (T, H, W, 3)

        d = dict(np.load(deps_name_path, allow_pickle=True))
        if "deps" in d:
            depths = d["deps"].astype(np.float32)  # shape: (T, 1, H, W)
        elif "track_g" in d:
            depths = d["track_g"].astype(np.float32)  # shape: (T, 1, H, W)

        depth_range = (depths.min(), depths.max())

        track = dict(np.load(track_name_path, allow_pickle=True))[
            "track_g"
        ]  # shape: (T, N, 5), include: trajs, trajs_z, visibs, valids

        H, W = rgbs.shape[1:3]
        intris = None
        if intris_name_path != None:
            d = dict(np.load(intris_name_path, allow_pickle=True))
            intris = d["intris"][0]
            extris = d["extris"]
        else:
            intris = np.array([[W, 0.0, W // 2], [0.0, H, H // 2], [0.0, 0.0, 1.0]])[None, ...].repeat(
                len(rgbs), axis=0
            )

        trajs_uv = track[..., 0:2]  # shape: (T, N, 2)
        trajs_z = track[..., 2:3]  # shape: (T, N, 1)

        query_points_uv = trajs_uv[0]  # shape: (N, 2)
        query_points_z = trajs_z[0]  # shape: (N, 1)
        query_points_t = np.zeros_like(query_points_z)  # shape: (N, 1)
        query_points = np.concatenate([query_points_t, query_points_uv, query_points_z], axis=-1)  # shape: (N, 4)

        if self.track_point_num == -1:
            track_point_num = query_points.shape[0]
        else:
            track_point_num = self.track_point_num

        if self.seq_len == -1:
            seq_len = rgbs.shape[0]
        else:
            seq_len = self.seq_len

        if self.augmentor is not None:
            self.augmentor.video_augs(rgbs)

        rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()  # shape: (T, 3, H, W)
        depths = torch.from_numpy(depths).float()  # shape: (T, H, W)
        visibs = torch.from_numpy(track[..., 3])  # shape: (T, N_total)
        valids = torch.from_numpy(track[..., 4])  # shape: (T, N_total)
        trajs_uv = torch.from_numpy(trajs_uv)  # shape: (T, N_total, 2)
        trajs_z = torch.from_numpy(trajs_z)  # shape: (T, N_total, 1)
        query_points = torch.from_numpy(query_points)  # shape: (N, 4)
        # if intris is not None:
        intris = torch.from_numpy(intris)

        # data_invalid = False
        index = None

        visibile_pts_first_frame_inds = (visibs[0]).nonzero(as_tuple=False)[:, 0]
        visibile_pts_inds = visibile_pts_first_frame_inds

        if self.train_mode:
            point_inds = torch.randperm(len(visibile_pts_inds))[:track_point_num]

            if len(point_inds) < track_point_num:
                return None
        else:
            step = len(visibile_pts_inds) // track_point_num
            point_inds = list(range(0, len(visibile_pts_inds), step))[:track_point_num]

        visible_inds_sampled = visibile_pts_inds[point_inds]
        trajs_uv = trajs_uv[:, visible_inds_sampled].float()  # shape: (T, N, 2)
        trajs_z = trajs_z[:, visible_inds_sampled].float()  # shape: (T, N, 1)
        visibs = visibs[:, visible_inds_sampled]  # shape: (T, N)
        valids = valids[:, visible_inds_sampled]  # shape: (T, N)

        query_points = query_points[visible_inds_sampled, :]

        segs = torch.ones_like(depths)

        depth_init = depths[0].clone()

        # depths[depths < 1] = 1.0
        # depths[depths > 45] = 45

        # NOTE dummy dense trajectory
        T, _, H, W = rgbs.shape
        dense_trajectory = torch.zeros((T, 2, H, W)).float()
        dense_trajectory_d = torch.zeros((T, 1, H, W)).float()
        flow = torch.zeros((T, 2, H, W)).float()
        flow_depth = torch.zeros((T, 1, H, W)).float()
        flow_alpha = torch.zeros((T, 1, H, W)).float()
        dense_valid = torch.zeros((T, 1, H, W)).float()

        # print("intris", intris.shape)

        if self.train_mode:
            # return DeltaData(
            #     video=rgbs,
            #     videodepth=depths,
            #     segmentation=segs,
            #     trajectory=torch.cat([trajs_uv, trajs_z], dim=-1),
            #     visibility=visibs,
            #     valid=valids,
            #     seq_name=video_name,
            #     # query_points=query_points,
            #     # trajectory3d=torch.cat([trajs_uv, trajs_z], dim=-1)
            # )
            return DeltaData(
                video=rgbs,
                videodepth=depths,
                depth_init=depth_init,
                segmentation=segs,
                trajectory=trajs_uv,
                trajectory_d=trajs_z,
                visibility=visibs,
                valid=valids,
                dense_trajectory=dense_trajectory,
                dense_trajectory_d=dense_trajectory_d,
                dense_valid=dense_valid,
                flow=flow,
                flow_depth=flow_depth,
                flow_alpha=flow_alpha,
                seq_name=video_name,
                dataset_name="lsfodyssey",
                depth_min=torch.tensor(depth_range[0]),
                depth_max=torch.tensor(depth_range[1]),
                # query_points=query_points,
                # trajectory3d=torch.cat([trajs_uv, trajs_z], dim=-1)
            )

        else:
            return DeltaData(
                video=rgbs,
                videodepth=depths,
                segmentation=segs,
                trajectory=trajs_uv,
                visibility=visibs,
                valid=valids,
                seq_name=video_name,
                query_points=query_points,
                trajectory3d=torch.cat([trajs_uv, trajs_z], dim=-1),
                intrs=intris,
            )
