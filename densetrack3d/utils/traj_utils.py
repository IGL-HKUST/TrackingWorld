
import os
import numpy as np

# from cotracker.datasets.utils import DeltaData
from densetrack3d.datasets.mpi_sintel import cam_read_sintel
from evo.tools import file_interface, plot
from scipy.spatial.transform import Rotation


# from PIL import Image
# from typing import Mapping, Tuple, Union


TAG_FLOAT = 202021.25
TAG_CHAR = "PIEH"


def load_sintel_traj(gt_file):
    # Refer to ParticleSfM
    gt_pose_lists = sorted(os.listdir(gt_file))
    gt_pose_lists = [os.path.join(gt_file, x) for x in gt_pose_lists]
    tstamps = [float(x.split("/")[-1][:-4].split("_")[-1]) for x in gt_pose_lists]
    gt_poses = [cam_read_sintel(f)[1] for f in gt_pose_lists]
    xyzs, wxyzs = [], []
    tum_gt_poses = []
    for gt_pose in gt_poses:
        gt_pose = np.concatenate([gt_pose, np.array([[0, 0, 0, 1]])], 0)
        gt_pose_inv = np.linalg.inv(gt_pose)  # world2cam -> cam2world
        xyz = gt_pose_inv[:3, -1]
        xyzs.append(xyz)
        R = Rotation.from_matrix(gt_pose_inv[:3, :3])
        xyzw = R.as_quat()  # scalar-last for scipy
        wxyz = np.array([xyzw[-1], xyzw[0], xyzw[1], xyzw[2]])
        wxyzs.append(wxyz)
        tum_gt_pose = np.concatenate([xyz, wxyz], 0)
        tum_gt_poses.append(tum_gt_pose)

    tum_gt_poses = np.stack(tum_gt_poses, 0)
    tum_gt_poses[:, :3] = tum_gt_poses[:, :3] - np.mean(tum_gt_poses[:, :3], 0, keepdims=True)
    tt = np.expand_dims(np.stack(tstamps, 0), -1)
    return tum_gt_poses, tt


def load_traj(gt_traj_file, traj_format="sintel", skip=0, stride=2):
    """Read trajectory format. Return in TUM-RGBD format.
    Returns:
        traj_tum (N, 7): camera to world poses in (x,y,z,qx,qy,qz,qw)
        timestamps_mat (N, 1): timestamps
    """
    if traj_format == "replica":
        traj_tum, timestamps_mat = load_replica_traj(gt_traj_file)
    elif traj_format == "sintel":
        traj_tum, timestamps_mat = load_sintel_traj(gt_traj_file)
    elif traj_format == "tartanair":
        traj = file_interface.read_tum_trajectory_file(gt_traj_file)
        xyz = traj.positions_xyz
        xyz = xyz[:, [1, 2, 0]]
        quat = traj.orientations_quat_wxyz
        quat = quat[:, [0, 2, 3, 1]]
        timestamps_mat = traj.timestamps
        traj_tum = np.column_stack((xyz, quat))
    elif traj_format == "tum":
        traj = file_interface.read_tum_trajectory_file(gt_traj_file)
        xyz = traj.positions_xyz
        # shift -1 column -> w in back column
        # quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
        quat = traj.orientations_quat_wxyz

        timestamps_mat = traj.timestamps
        traj_tum = np.column_stack((xyz, quat))
    else:
        raise NotImplementedError

    traj_tum = traj_tum[skip::stride]
    timestamps_mat = timestamps_mat[skip::stride]
    return traj_tum, timestamps_mat
