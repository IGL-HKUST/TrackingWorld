import glob
import io
import os
import pickle
from typing import Mapping, Tuple, Union

import cv2
import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from densetrack3d.datasets.predefined_path import TARTANAIR_DIR
from densetrack3d.datasets.utils import DeltaData
from densetrack3d.models.geometry_utils import least_square_align
from PIL import Image
from scipy.spatial.transform import Rotation


# TAG_FLOAT = 202021.25
# TAG_CHAR = "PIEH"


class TartanAirDataset(torch.utils.data.Dataset):
    def __init__(self, data_root=TARTANAIR_DIR, use_metric_depth=False, rgb_folder="image_0"):
        super(TartanAirDataset, self).__init__()

        self.use_metric_depth = use_metric_depth
        self.data_root = data_root
        self.rgb_folder = rgb_folder

        self.scene_names = sorted([s for s in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, s))])

        if os.path.exists(os.path.join(self.data_root, "depthcrafter_depth.pkl")):
            with open(os.path.join(self.data_root, "depthcrafter_depth.pkl"), "rb") as handle:
                self.depthcrafter_depth = pickle.load(handle)
        else:
            self.depthcrafter_depth = None

        if os.path.exists(os.path.join(self.data_root, "unidepth_depth.pkl")):
            with open(os.path.join(self.data_root, "unidepth_depth.pkl"), "rb") as handle:
                self.unidepth_depth = pickle.load(handle)
        else:
            self.unidepth_depth = None

    def __getitem__(self, index):
        scene_name = self.scene_names[index]

        rgb_path = os.path.join(self.data_root, scene_name, self.rgb_folder)
        # depth_path = os.path.join(self.data_root, scene_name, self.depth_folder)

        img_names = sorted([n for n in os.listdir(rgb_path) if n.endswith(".png") or n.endswith(".jpg")])
        # depth_names = sorted([n for n in os.listdir(depth_path) if n.endswith(".png")])

        video = []

        for i, img_name in enumerate(img_names):
            image = cv2.imread(os.path.join(rgb_path, img_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video.append(image)

        video = np.stack(video)
        videodepth = self.unidepth_depth[scene_name].astype(np.float32)

        max_depth_valid = videodepth[videodepth < 100].max()
        min_depth_valid = videodepth[videodepth > 0].min()

        videodepth[videodepth > max_depth_valid] = max_depth_valid
        videodepth[videodepth < min_depth_valid] = min_depth_valid

        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        videodepth = torch.from_numpy(videodepth).float()

        max_depth_valid = torch.tensor(max_depth_valid)
        min_depth_valid = torch.tensor(min_depth_valid)

        if self.use_metric_depth:
            videodepth = videodepth.unsqueeze(1)
        else:
            videodisp = torch.from_numpy(self.depthcrafter_depth[scene_name]).float()
            videodepth = least_square_align(videodepth, videodisp, return_align_scalar=False, query_frame=0)

            videodepth[videodepth > max_depth_valid] = max_depth_valid
            videodepth[videodepth < min_depth_valid] = min_depth_valid

            videodepth = videodepth.unsqueeze(1)

        data = DeltaData(
            video=video,
            videodepth=videodepth,
            seq_name=scene_name,
        )

        return data

    def __len__(self):
        return len(self.scene_names)
