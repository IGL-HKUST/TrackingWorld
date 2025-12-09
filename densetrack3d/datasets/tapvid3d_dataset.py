import io
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from einops import rearrange
from PIL import Image

from densetrack3d.datasets.utils import DeltaData
from densetrack3d.models.geometry_utils import least_square_align
from densetrack3d.models.model_utils import depth_to_disparity, get_grid, sample_features5d

try:
    from densetrack3d.datasets.s3_utils import create_client, get_client_stream, read_s3_json
    has_s3 = True
except:
    has_s3 = False


UINT16_MAX = 65535
TAPVID3D_ROOT = None


def get_jpeg_byte_hw(jpeg_bytes: bytes):
    with io.BytesIO(jpeg_bytes) as img_bytes:
        img = Image.open(img_bytes)
        img = img.convert("RGB")
    return np.array(img).shape[:2]


def get_new_hw_with_given_smallest_side_length(*, orig_height: int, orig_width: int, smallest_side_length: int = 256):
    orig_shape = np.array([orig_height, orig_width])
    scaling_factor = smallest_side_length / np.min(orig_shape)
    resized_shape = np.round(orig_shape * scaling_factor)
    return (int(resized_shape[0]), int(resized_shape[1])), scaling_factor


def project_points_to_video_frame(camera_pov_points3d, camera_intrinsics, height, width):
    """Project 3d points to 2d image plane."""
    u_d = camera_pov_points3d[..., 0] / (camera_pov_points3d[..., 2] + 1e-8)
    v_d = camera_pov_points3d[..., 1] / (camera_pov_points3d[..., 2] + 1e-8)

    f_u, f_v, c_u, c_v = camera_intrinsics

    u_d = u_d * f_u + c_u
    v_d = v_d * f_v + c_v

    # Mask of points that are in front of the camera and within image boundary
    masks = camera_pov_points3d[..., 2] >= 1
    masks = masks & (u_d >= 0) & (u_d < width) & (v_d >= 0) & (v_d < height)
    return np.stack([u_d, v_d], axis=-1), masks


class TapVid3DDataset(Dataset):

    def __init__(
        self,
        data_root,
        datatype="pstudio",
        crop_size=256,
        debug=False,
        use_metric_depth=True,
        split="minival",
        read_from_s3=False,
    ):

        # if split == "all":
        #     datatype = datatype + "_all"

        self.read_from_s3 = read_from_s3
        self.datatype = datatype
        self.data_root = os.path.join(data_root, datatype)

        if self.read_from_s3:
            assert has_s3
            self.client = create_client()

            tapvid3d_metadata_path = os.path.join(TAPVID3D_ROOT, "meta_data.json")
            self.data_root = os.path.join(TAPVID3D_ROOT, datatype)

            meta_data = read_s3_json(self.client, tapvid3d_metadata_path)
            self.video_names = meta_data[datatype]
        else:
            self.video_names = sorted([f.split(".")[0] for f in os.listdir(self.data_root) if f.endswith(".npz")])[::100][1:]

        self.debug = debug
        self.crop_size = crop_size
        self.use_metric_depth = use_metric_depth

        print(f"Found {len(self.video_names)} samples for TapVid3D {datatype}")

    def __len__(self):
        if self.debug:
            return 200
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]

        gt_path = os.path.join(self.data_root, f"{video_name}.npz")

        # with open(gt_path, 'rb') as in_f:
        #     in_npz = np.load(in_f, allow_pickle=True)

        if self.read_from_s3:
            in_npz = np.load(get_client_stream(self.client, gt_path), allow_pickle=True)
        else:
            in_npz = np.load(gt_path, allow_pickle=True)

        images_jpeg_bytes = in_npz["images_jpeg_bytes"]
        video = []
        for frame_bytes in images_jpeg_bytes:
            arr = np.frombuffer(frame_bytes, np.uint8)
            image_bgr = cv2.imdecode(arr, flags=cv2.IMREAD_UNCHANGED)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            video.append(image_rgb)
        video = np.stack(video, axis=0)

        metric_videodepth = in_npz["depth_preds"]  # NOTE UniDepth
        # metric_videodepth = in_npz['depth_preds_zoe'] # NOTE ZoeDepth

        if self.use_metric_depth:
            videodepth = metric_videodepth
        else:
            videodisp = in_npz["depth_preds_depthcrafter"]
            videodisp = videodisp.astype(np.float32) / UINT16_MAX
            videodepth = least_square_align(metric_videodepth, videodisp, return_align_scalar=False)

        queries_xyt = in_npz["queries_xyt"]
        tracks_xyz = in_npz["tracks_XYZ"]
        visibles = in_npz["visibility"]
        intrinsics_params = in_npz["fx_fy_cx_cy"]
        if self.datatype == "drivetrack":
          extrinsics_w2c = in_npz["extrinsics_w2c"]
          extrinsics_w2c = torch.from_numpy(extrinsics_w2c).float()
        else:
          extrinsics_w2c = torch.ones((video.shape[0], 4, 4))

        tracks_uv, _ = project_points_to_video_frame(tracks_xyz, intrinsics_params, video.shape[1], video.shape[2])

        scaling_factor = 1.0
        intrinsics_params_resized = intrinsics_params * scaling_factor
        intrinsic_mat = np.array(
            [
                [intrinsics_params_resized[0], 0, intrinsics_params_resized[2]],
                [0, intrinsics_params_resized[1], intrinsics_params_resized[3]],
                [0, 0, 1],
            ]
        )
        intrinsic_mat = torch.from_numpy(intrinsic_mat).float()
        intrinsic_mat = intrinsic_mat[None].repeat(video.shape[0], 1, 1)

        video = torch.from_numpy(video).permute(0, 3, 1, 2).float()
        videodepth = torch.from_numpy(videodepth).float().unsqueeze(1)
        segs = torch.ones_like(videodepth)

        trajectory_3d = torch.from_numpy(tracks_xyz).float()  # T N D
        trajectory_2d = torch.from_numpy(tracks_uv).float()  # T N 2
        visibility = torch.from_numpy(visibles)
        query_points = torch.from_numpy(queries_xyt).float()

        sample_coords = torch.cat([query_points[:, 2:3], query_points[:, :2]], dim=-1)[None, None, ...]  # 1 1 N 3

        rgb_h, rgb_w = video.shape[2], video.shape[3]
        depth_h, depth_w = videodepth.shape[2], videodepth.shape[3]
        if rgb_h != depth_h or rgb_w != depth_w:
            sample_coords[..., 1] = sample_coords[..., 1] * depth_w / rgb_w
            sample_coords[..., 2] = sample_coords[..., 2] * depth_h / rgb_h

        query_points_depth = sample_features5d(videodepth[None], sample_coords, mode="nearest")
        query_points_depth = query_points_depth.squeeze(0, 1)
        query_points_3d = torch.cat(
            [query_points[:, 2:3], query_points[:, :2], query_points_depth], dim=-1
        )  # NOTE by default, query is N 3: xyt but we use N 3: txy

        data = DeltaData(
            video=video,
            videodepth=videodepth,
            segmentation=segs,
            trajectory=trajectory_2d,
            trajectory3d=trajectory_3d,
            visibility=visibility,
            seq_name=video_name,
            query_points=query_points_3d,
            intrs=intrinsic_mat,
            extrs=extrinsics_w2c
        )

        return data
