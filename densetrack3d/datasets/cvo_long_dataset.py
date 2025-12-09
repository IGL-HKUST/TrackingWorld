import os
import os.path as osp
import pickle as pkl
import random
from collections import OrderedDict

import lmdb
import numpy as np
import pyarrow as pa
import torch
from densetrack3d.datasets.utils import DeltaData
from densetrack3d.models.model_utils import get_grid
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ColorJitter, GaussianBlur


def get_alpha_consistency(bflow, fflow, thresh_1=0.01, thresh_2=0.5, thresh_mul=1):
    norm = lambda x: x.pow(2).sum(dim=-1).sqrt()
    B, H, W, C = bflow.shape

    mag = norm(fflow) + norm(bflow)
    grid = get_grid(H, W, shape=[B], device=fflow.device)
    grid[..., 0] = grid[..., 0] + bflow[..., 0] / (W - 1)
    grid[..., 1] = grid[..., 1] + bflow[..., 1] / (H - 1)
    grid = grid * 2 - 1
    fflow_warped = torch.nn.functional.grid_sample(
        fflow.permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=True
    )
    flow_diff = bflow + fflow_warped.permute(0, 2, 3, 1)
    occ_thresh = thresh_1 * mag + thresh_2
    occ_thresh = occ_thresh * thresh_mul
    alpha = norm(flow_diff) < occ_thresh
    alpha = alpha.float()
    return alpha


def sample_frame_inds(n_ori=8, n_frame=24):
    inds = [0, 1, 2, 3, 4, 5, 6, 7]

    cur_step = 7
    for i in range(8, n_frame):
        if cur_step == n_ori - 1:
            if random.random() > 0.1:
                cur_step -= 1
        elif cur_step == 0:
            if random.random() > 0.1:
                cur_step += 1
        else:
            rand = random.random()
            if rand > 0.6:
                cur_step += 1
            elif rand > 0.2:
                cur_step -= 1

        inds.append(cur_step)

    return inds


class CVO_sampler_lmdb:
    """Data sampling"""

    all_keys = ["imgs", "imgs_blur", "fflows", "bflows", "delta_fflows", "delta_bflows"]

    def __init__(self, data_root, keys=None, split=None):
        if split == "extended":
            self.db_path = osp.join(data_root, "cvo_test_extended.lmdb")
        elif split == "test":
            self.db_path = osp.join(data_root, "cvo_test.lmdb")
        else:
            self.db_path = osp.join(data_root, "cvo_train.lmdb")
        self.split = split

        # print(self.db_path)
        self.deserialize_func = pa.deserialize if "train" in self.split else pkl.loads

        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.samples = self.deserialize_func(txn.get(b"__samples__"))
            self.length = len(self.samples)

        self.keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(self.keys)

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return self.length

    def sample(self, index):
        sample = OrderedDict()
        with self.env.begin(write=False) as txn:
            for k in self.keys:
                key = "{:05d}_{:s}".format(index, k)
                value = self.deserialize_func(txn.get(key.encode()))
                if "flow" in key and self.split in ["clean", "final", "train"]:  # Convert Int to Floating
                    value = value.astype(np.float32)
                    value = (value - 2**15) / 128.0
                if "imgs" in k:
                    k = "imgs"
                sample[k] = value
        return sample


class CVOLong(Dataset):
    all_keys = ["fflows", "bflows", "delta_fflows", "delta_bflows"]

    def __init__(self, data_root, keys=None, split="train", crop_size=(384, 512), traj_per_sample=768, seq_len=24):
        keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(keys)
        if split == "final":
            keys.append("imgs_blur")
        else:
            keys.append("imgs")
        self.split = split

        self.data_root = data_root
        self.sampler = CVO_sampler_lmdb(data_root, keys, split)

        print(f"Found {self.sampler.length} samples for CVO {split}")

        self.crop_size = crop_size
        self.traj_per_sample = traj_per_sample
        self.seq_len = seq_len

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

    def add_photometric_augs(self, rgbs, alphas, eraser=True, replace=True):
        # T, N, _ = trajs.shape

        # S = len(rgbs)
        S, H, W, _ = rgbs.shape
        # assert S == T

        if eraser:
            ############ eraser transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            for i in range(1, S):
                if np.random.rand() < self.eraser_aug_prob:
                    for _ in range(np.random.randint(1, self.eraser_max + 1)):  # number of times to occlude
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

                        # occ_inds = np.logical_and(
                        #     np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                        #     np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        # )

                        alphas[i][y0:y1, x0:x1] = 0
                        # visibles[i, occ_inds] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        if replace:
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]
            rgbs_alt = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs_alt]

            ############ replace transform (per image after the first) ############
            rgbs = [rgb.astype(np.float32) for rgb in rgbs]
            rgbs_alt = [rgb.astype(np.float32) for rgb in rgbs_alt]
            for i in range(1, S):
                if np.random.rand() < self.replace_aug_prob:
                    for _ in range(np.random.randint(1, self.replace_max + 1)):  # number of times to occlude
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

                        # occ_inds = np.logical_and(
                        #     np.logical_and(trajs[i, :, 0] >= x0, trajs[i, :, 0] < x1),
                        #     np.logical_and(trajs[i, :, 1] >= y0, trajs[i, :, 1] < y1),
                        # )
                        # visibles[i, occ_inds] = 0

                        alphas[i][y0:y1, x0:x1] = 0
            rgbs = [rgb.astype(np.uint8) for rgb in rgbs]

        ############ photometric augmentation ############
        if np.random.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            rgbs = [np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        if np.random.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            rgbs = [np.array(self.blur_aug(Image.fromarray(rgb)), dtype=np.uint8) for rgb in rgbs]

        return rgbs, alphas

    def __getitem__(self, index):
        sample = self.sampler.sample(index)

        # depth_path = os.path.join(self.data_root, "cvo_test_depth", f"{index:05d}.npy")
        # videodepth = np.load(depth_path)
        # videodepth = torch.from_numpy(videodepth).float()
        # videodepth = rearrange(videodepth, 't h w -> t () h w')

        video = sample["imgs"].copy()
        # video = video / 255.0
        video = rearrange(video, "h w (t c) -> t h w c", c=3)

        # NOTE concat and flip video
        # video = torch.flip(video, dims=[0]) # flip temporal
        video = np.concatenate([video, video[-1][None]], axis=0)  # 8 C H W

        # videodepth = torch.flip(videodepth, dims=[0]) # flip temporal
        # videodepth = torch.cat([videodepth, videodepth[-1].unsqueeze(0)], dim=0) # 8 C H W

        # breakpoint()
        T, H, W, _ = video.shape

        zero_flow = np.zeros((H, W, 2)).astype(float)

        fflow = np.ascontiguousarray(sample["fflows"].copy()).astype(float)
        fflow = rearrange(fflow, "h w (t c) -> t h w c", c=2)  # 0->2, ..., 0->6
        delta_fflows = np.ascontiguousarray(sample["delta_fflows"].copy()).astype(float)
        delta_fflows = rearrange(delta_fflows, "h w (t c) -> t h w c", c=2)  # T C H W
        forward_flows = np.concatenate(
            [zero_flow[None], delta_fflows[0][None], fflow, fflow[-1][None]], axis=0
        )  # 0->0, 0->1, 0->2, ..., 0->6, 7 h w 2

        bflow = np.ascontiguousarray(sample["bflows"].copy()).astype(float)
        bflow = rearrange(bflow, "h w (t c) -> t h w c", c=2)  # 6->0
        delta_bflows = np.ascontiguousarray(sample["delta_bflows"].copy()).astype(float)
        delta_bflows = rearrange(delta_bflows, "h w (t c) -> t h w c", c=2)
        backward_flows = np.concatenate(
            [zero_flow[None], delta_bflows[0][None], bflow, bflow[-1][None]], axis=0
        )  # 0->0, 1->0, 2->0, ..., 6->0

        # breakpoint()

        if self.split in ["clean", "final", "train"]:
            thresh_1 = 0.01
            thresh_2 = 0.5
        elif self.split == "extended":
            thresh_1 = 0.1
            thresh_2 = 0.5
        else:
            raise ValueError(f"Unknown split {self.split}")

        alpha = get_alpha_consistency(
            torch.from_numpy(forward_flows).float(),  # NOTE swap here
            torch.from_numpy(backward_flows).float(),
            thresh_1=thresh_1,
            thresh_2=thresh_2,
        )  # 7 c h w
        alpha = alpha.numpy()

        # video, alpha = self.add_photometric_augs(video, alpha)
        # video = torch.from_numpy(np.stack(video, axis=0)).float()

        video = torch.from_numpy(video).float()
        video = rearrange(video, "t h w c -> t c h w")

        alpha = torch.from_numpy(alpha).float()
        # alpha = rearrange(alpha, 't h w > t c h w')

        forward_flows = torch.from_numpy(forward_flows).float()
        forward_flows = rearrange(forward_flows, "t h w c -> t c h w")

        # print("forward_flows", forward_flows.max(), forward_flows.min())

        # print(video.shape)

        y0 = 0 if self.crop_size[0] >= H else np.random.randint(0, H - self.crop_size[0])
        x0 = 0 if self.crop_size[1] >= W else np.random.randint(0, W - self.crop_size[1])
        # video = [rgb[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]] for rgb in rgbs]
        video = video[:, :, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]  # T C H W
        forward_flows = forward_flows[:, :, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        # backward_flows = backward_flows[:, :, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]
        alpha = alpha[:, y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1]]

        repeat_inds = sample_frame_inds(T, self.seq_len)
        video = video[repeat_inds]
        forward_flows = forward_flows[repeat_inds]
        alpha = alpha[repeat_inds]

        # backward_flows = video[repeat_inds]
        # Replicate data
        # n_repeat = 3 # 8 x 3 = 24 same as KUBRIC
        # video = video.repeat_interleave(repeats=n_repeat, dim=0)
        # forward_flows = forward_flows.repeat_interleave(repeats=n_repeat, dim=0)
        # backward_flows = backward_flows.repeat_interleave(repeats=n_repeat, dim=0)
        # alpha = alpha.repeat_interleave(repeats=n_repeat, dim=0)

        T = video.shape[0]
        assert T == self.seq_len

        # segs = torch.ones(T, 1, H, W).float()

        # trajectory = torch.zeros(T, 1, 2).float()
        # visibility = torch.zeros(T, 1).float()

        # print(forward_flows.shape, alpha.shape)
        # print("post", video.shape)
        data = DeltaData(
            video=video,
            # videodepth=videodepth,
            trajectory=torch.zeros((T, self.traj_per_sample, 2)),
            visibility=torch.zeros((T, self.traj_per_sample)),
            valid=torch.zeros((T, self.traj_per_sample)),
            flow=forward_flows,
            flow_alpha=alpha,
            seq_name=f"{index:05d}",
            dataset_name="cvo",
            # data_type=torch.ones((1))
        )

        # data = {
        #     "video": video,
        #     "alpha": alpha,
        #     "flow": bflow
        # }

        return data, True

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return len(self.sampler)
