# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import io
import os
import pickle
from typing import Mapping, Tuple, Union

import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from densetrack3d.datasets.utils import DeltaData
from densetrack3d.utils.io import read_pickle

try:
    from densetrack3d.datasets.s3_utils import create_client, get_client_stream, read_s3_json
    has_s3 = True
except:
    has_s3 = False


DatasetElement = Mapping[str, Mapping[str, Union[np.ndarray, str]]]

UINT16_MAX = 65535


def resize_video(video: np.ndarray, output_size: Tuple[int, int]) -> np.ndarray:
    """Resize a video to output_size."""
    # If you have a GPU, consider replacing this with a GPU-enabled resize op,
    # such as a jitted jax.image.resize.  It will make things faster.
    return media.resize_video(video, output_size)


def sample_queries_first(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.
    Given a set of frames and tracks with no query points, use the first
    visible point in each track as the query.
    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3]
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1]
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1]
    """
    valid = np.sum(~target_occluded, axis=1) > 0
    target_points = target_points[valid, :]
    target_occluded = target_occluded[valid, :]

    query_points = []
    for i in range(target_points.shape[0]):
        index = np.where(target_occluded[i] == 0)[0][0]
        x, y = target_points[i, index, 0], target_points[i, index, 1]
        query_points.append(np.array([index, y, x]))  # [t, y, x]
    query_points = np.stack(query_points, axis=0)

    return {
        "video": frames[np.newaxis, ...],
        "query_points": query_points[np.newaxis, ...],
        "target_points": target_points[np.newaxis, ...],
        "occluded": target_occluded[np.newaxis, ...],
    }


def sample_queries_strided(
    target_occluded: np.ndarray,
    target_points: np.ndarray,
    frames: np.ndarray,
    query_stride: int = 5,
) -> Mapping[str, np.ndarray]:
    """Package a set of frames and tracks for use in TAPNet evaluations.

    Given a set of frames and tracks with no query points, sample queries
    strided every query_stride frames, ignoring points that are not visible
    at the selected frames.

    Args:
      target_occluded: Boolean occlusion flag, of shape [n_tracks, n_frames],
        where True indicates occluded.
      target_points: Position, of shape [n_tracks, n_frames, 2], where each point
        is [x,y] scaled between 0 and 1.
      frames: Video tensor, of shape [n_frames, height, width, 3].  Scaled between
        -1 and 1.
      query_stride: When sampling query points, search for un-occluded points
        every query_stride frames and convert each one into a query.

    Returns:
      A dict with the keys:
        video: Video tensor of shape [1, n_frames, height, width, 3].  The video
          has floats scaled to the range [-1, 1].
        query_points: Query points of shape [1, n_queries, 3] where
          each point is [t, y, x] scaled to the range [-1, 1].
        target_points: Target points of shape [1, n_queries, n_frames, 2] where
          each point is [x, y] scaled to the range [-1, 1].
        trackgroup: Index of the original track that each query point was
          sampled from.  This is useful for visualization.
    """
    tracks = []
    occs = []
    queries = []
    trackgroups = []
    total = 0
    trackgroup = np.arange(target_occluded.shape[0])
    for i in range(0, target_occluded.shape[1], query_stride):
        mask = target_occluded[:, i] == 0
        query = np.stack(
            [
                i * np.ones(target_occluded.shape[0:1]),
                target_points[:, i, 1],
                target_points[:, i, 0],
            ],
            axis=-1,
        )
        queries.append(query[mask])
        tracks.append(target_points[mask])
        occs.append(target_occluded[mask])
        trackgroups.append(trackgroup[mask])
        total += np.array(np.sum(target_occluded[:, i] == 0))

    return {
        "video": frames[np.newaxis, ...],
        "query_points": np.concatenate(queries, axis=0)[np.newaxis, ...],
        "target_points": np.concatenate(tracks, axis=0)[np.newaxis, ...],
        "occluded": np.concatenate(occs, axis=0)[np.newaxis, ...],
        "trackgroup": np.concatenate(trackgroups, axis=0)[np.newaxis, ...],
    }


def get_chunk_index(len_chunks, list_index):
    cumulative_length = 0
    for chunk_index, len_chunk in enumerate(len_chunks):
        # prev_cumulative_length = cumulative_length
        # cumulative_length += len_chunk
        if list_index < cumulative_length + len_chunk:
            local_index = list_index - cumulative_length
            return chunk_index, local_index
        else:
            cumulative_length = cumulative_length + len_chunk
    return -1, -1


class TapVid2DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        dataset_type="davis",
        resize_to_256=True,
        resize_to_cotracker=False,
        queried_first=True,
        read_from_s3=False,
    ):
        self.data_root = data_root
        self.dataset_type = dataset_type
        self.resize_to_256 = resize_to_256
        self.resize_to_cotracker = resize_to_cotracker
        self.queried_first = queried_first

        self.read_from_s3 = read_from_s3

        if self.read_from_s3:
            assert has_s3
            self.client = create_client()

        if self.dataset_type == "kinetics":
            self.supsample_rate = 1
            len_chunks = []
            all_paths = sorted(
                glob.glob(os.path.join(data_root, f"tapvid_{dataset_type}", "*_of_0010_with_depth.pkl"))
            )
            points_dataset = []
            for data_path in all_paths:

                data = (
                    pickle.load(get_client_stream(self.client, data_path))
                    if self.read_from_s3
                    else read_pickle(data_path)
                )

                # FIXME subsample here
                data = data[:: self.supsample_rate]
                points_dataset = points_dataset + data

                len_chunks.append(len(data))
            self.points_dataset = points_dataset
            self.len_chunks = len_chunks
        else:
            data_path = os.path.join(data_root, f"tapvid_{dataset_type}", f"tapvid_{dataset_type}_with_depth_zoedepth.pkl")
            print(data_path, read_from_s3)
            self.points_dataset = pickle.load(get_client_stream(self.client, data_path)) if self.read_from_s3 else read_pickle(data_path)
            
            #print(self.points_dataset)
            if self.dataset_type == "davis":
                self.video_names = [i for i in range(len(self.points_dataset))]#list(self.points_dataset.keys())

        print("found %d unique videos in %s" % (len(self.points_dataset), data_root))

    def __getitem__(self, index):
        if self.dataset_type == "davis":
            video_name = self.video_names[index]
        else:
            video_name = index
        sample = self.points_dataset[video_name].copy()
        frames = sample["video"]

        if isinstance(frames[0], bytes):
            # TAP-Vid is stored and JPEG bytes rather than `np.ndarray`s.
            def decode(frame):
                byteio = io.BytesIO(frame)
                img = Image.open(byteio)
                return np.array(img)

            frames = np.array([decode(frame) for frame in frames])

        target_points = sample["points"]

        if "depth_preds_zoe" in sample.keys() and sample["depth_preds_zoe"] is not None:
            depths = sample["depth_preds_zoe"][..., None]
        else:
            chunk_id, local_id = get_chunk_index(self.len_chunks, index)
            local_id = local_id * self.supsample_rate
            depth_path = os.path.join(
                self.data_root,
                f"tapvid_{self.dataset_type}",
                "depths",
                f"{chunk_id:04d}_of_0010",
                f"{local_id:04d}.npz",
            )
            depths = np.load(depth_path)["depth_preds_zoe"][..., None]

            # depth_min, depth_max = sample['depth_min'], sample['depth_max']
            # depths = depths * (depth_max - depth_min) / UINT16_MAX + depth_min
            # print("depths", depths.shape, depths.min(), depths.max(), frames.shape)
            # print(index, chunk_id, local_id)

            len_depth = len(depths)
            if len_depth != len(frames):
                return self.__getitem__((index + 1) % len(self))
        print(frames.shape)
        if self.resize_to_cotracker:
            frames = resize_video(frames, [384, 512])
            target_points *= np.array([512, 384])
        elif self.resize_to_256:
            frames = resize_video(frames, [256, 256])
            target_points *= np.array([256, 256])
        else:
            target_points *= np.array([frames.shape[2], frames.shape[1]])

        T, H, W, C = frames.shape
        N, T, D = target_points.shape

        target_occ = sample["occluded"]
        if self.queried_first:
            converted = sample_queries_first(target_occ, target_points, frames)
        else:
            converted = sample_queries_strided(target_occ, target_points, frames)
        assert converted["target_points"].shape[1] == converted["query_points"].shape[1]

        trajs = torch.from_numpy(converted["target_points"])[0].permute(1, 0, 2).float()  # T, N, D

        rgbs = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
        depths = torch.from_numpy(depths).permute(0, 3, 1, 2).float()
        
        if self.resize_to_cotracker:
            depths = F.interpolate(depths, size=(384, 512), mode="bilinear")
        elif self.resize_to_256:
            depths = F.interpolate(depths, size=(256, 256), mode="bilinear")

        segs = torch.ones(T, 1, H, W).float()
        visibles = torch.logical_not(torch.from_numpy(converted["occluded"]))[0].permute(1, 0)  # T, N
        query_points = torch.from_numpy(converted["query_points"])[0]  # T, N

        # # NOTE debug
        # rgbs = rgbs[:90]
        # depths = depths[:90]
        # segs = segs[:90]
        # mask_valid = query_points[:, 0] < 90
        # query_points = query_points[mask_valid]
        # trajs = trajs[:90, mask_valid]
        # visibles = visibles[:90, mask_valid]

        data = DeltaData(
            video=rgbs,
            videodepth=depths,
            segmentation=segs,
            trajectory=trajs,
            visibility=visibles,
            seq_name=str(video_name),
            query_points=query_points,
            dataset_name=self.dataset_type,
        )

        # print("debug", data.dataset_name, data.seq_name)
        return data

    def __len__(self):
        return len(self.points_dataset)
