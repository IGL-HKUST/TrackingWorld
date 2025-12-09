from matplotlib.pyplot import grid
import torch
from torch import nn
from torch.nn import functional as F
import math
from pytorch3d.transforms import so3_exp_map, so3_log_map

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

@torch.jit.script
def hat(v: torch.Tensor) -> torch.Tensor:
    """
    Compute the Hat operator [1] of a batch of 3D vectors.

    Args:
        v: Batch of vectors of shape `(minibatch , 3)`.

    Returns:
        Batch of skew-symmetric matrices of shape
        `(minibatch, 3 , 3)` where each matrix is of the form:
            `[    0  -v_z   v_y ]
             [  v_z     0  -v_x ]
             [ -v_y   v_x     0 ]`

    Raises:
        ValueError if `v` is of incorrect shape.

    [1] https://en.wikipedia.org/wiki/Hat_operator
    """

    N, dim = v.shape
    # if dim != 3:
    #    raise ValueError("Input vectors have to be 3-dimensional.")

    h = torch.zeros((N, 3, 3), dtype=v.dtype, device=v.device)

    x, y, z = v.unbind(1)

    h[:, 0, 1] = -z
    h[:, 0, 2] = y
    h[:, 1, 0] = z
    h[:, 1, 2] = -x
    h[:, 2, 0] = -y
    h[:, 2, 1] = x

    return h

@torch.jit.script
def _so3_log_map(R: torch.Tensor, eps: float = 1e-4):
    """
    Compute the logarithm map of a rotation matrix.
    Args:
        R: Rotation matrices of shape (N, 3, 3)
        eps: Small constant for numerical stability
    Returns:
        log_rot: Rotation vectors of shape (N, 3)
    """
    cos_theta = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1.0) / 2.0
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # avoid NaNs due to numerical errors
    theta = torch.acos(cos_theta)  # rotation angle
    theta = torch.clamp(theta, min=eps)

    # Compute skew-symmetric part
    log_rot_hat = 0.5 * (R - R.transpose(1, 2))

    # Recover rotation axis
    rx = log_rot_hat[:, 2, 1]
    ry = log_rot_hat[:, 0, 2]
    rz = log_rot_hat[:, 1, 0]
    axis = torch.stack([rx, ry, rz], dim=1)

    # Normalize and scale by theta
    axis_norm = torch.norm(axis, dim=1, keepdim=True).clamp(min=eps)
    log_rot = axis / axis_norm * theta[:, None]

    return log_rot

@torch.jit.script
def _so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001):
    """
    A helper function that computes the so3 exponential map and,
    apart from the rotation matrix, also returns intermediate variables
    that can be re-used in other functions.
    """
    _, dim = log_rot.shape
    # if dim != 3:
    #    raise ValueError("Input tensor shape has to be Nx3.")

    nrms = (log_rot * log_rot).sum(1)
    # phis ... rotation angles
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)

    R = (
        # pyre-fixme[16]: `float` has no attribute `__getitem__`.
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    return R, rot_angles, skews, skews_square

class CameraIntrinsics(nn.Module):
    def __init__(self, init_focal_lengthx=1, init_focal_lengthy=1):
        super().__init__()
        self.focal_lengthx = nn.Parameter(torch.tensor(init_focal_lengthx))
        self.focal_lengthy = nn.Parameter(torch.tensor(init_focal_lengthy))

    def register_shape(self, orig_shape) -> None:
        self.orig_shape = orig_shape
        H, W = orig_shape
        intrinsics_mat_buffer = torch.zeros(3, 3)
        intrinsics_mat_buffer[0, -1] = (W - 1) / 2
        intrinsics_mat_buffer[1, -1] = (H - 1) / 2
        intrinsics_mat_buffer[2, -1] = 1
        self.register_buffer("intrinsics_mat", intrinsics_mat_buffer)

    def get_K(self, with_batch_dim=False) -> torch.Tensor:
        intrinsics_mat = self.intrinsics_mat.clone()
        intrinsics_mat[0, 0] = self.focal_lengthx
        intrinsics_mat[1, 1] = self.focal_lengthy

        if with_batch_dim:
            return intrinsics_mat[None, ...]
        else:
            return intrinsics_mat

def signed_expm1(x):
    sign = torch.sign(x)
    return sign * torch.expm1(torch.abs(x))
        

class DepthShiftCollection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter(
            f"shift", nn.Parameter(torch.FloatTensor([0.0]))
        )

    def forward(self):
        return torch.exp(self.shift)

    
        
class DepthScaleShiftCollection(torch.nn.Module):
    def __init__(self, n_points=10, use_inverse=False, grid_size=1):
        super().__init__()
        self.grid_size = grid_size
        for n in range(n_points):
            self.register_parameter(
                f"shift_{n}", nn.Parameter(torch.FloatTensor([0.0]))
            )
            self.register_parameter(
                f"scale_{n}", nn.Parameter(
                    torch.ones([1, 1, grid_size, grid_size]))
            )

        self.use_inverse = use_inverse
        self.output_shape = None

    def set_outputshape(self, output_shape):
        self.output_shape = output_shape

    def forward(self, index):
        shift = getattr(self, f"shift_{index}")
        scale = getattr(self, f"scale_{index}")
        if self.use_inverse:
            scale = torch.exp(scale)  # 1 / (scale ** 4)
        if self.grid_size != 1:
            scale = F.interpolate(scale, self.output_shape,
                                  mode='bilinear', align_corners=True)
        return scale, shift

    def set_scale(self, index, scale):
        scale_param = getattr(self, f"scale_{index}")
        if self.use_inverse:
            scale = math.log(scale)  # (1 / scale) ** 0.25
        scale_param.data.fill_(scale)

    def get_scale_data(self, index):
        scale = getattr(self, f"scale_{index}").data
        if self.use_inverse:
            scale = torch.exp(scale)  # 1 / (scale ** 4)
        if self.grid_size != 1:
            scale = F.interpolate(scale, self.output_shape,
                                  mode='bilinear', align_corners=True)
        return scale
    
class UncertaintyCollectionTracks(torch.nn.Module):

    """
    Direct optimization of weights for point clouds (includes both static and dynamic components)
    We take ELU so that weights are always positive
    This is a variant that assigns uncertainty to entire track
    """

    def __init__(self, number_of_points=10) -> None:
        super().__init__()
        uncertainty = torch.zeros([number_of_points, 1])
        self.register_parameter("uncertainty", nn.Parameter(uncertainty))

        self.number_of_points = number_of_points

        self.ELU = nn.ELU()

    def get_params(self):
        return self.uncertainty
    
    def forward(self, points=None):

        if points is None:
            points = torch.arange(self.number_of_points)

        return self.ELU(self.uncertainty[points]) + 1 # add 1 for it to be positive
    
class UncertaintyCollection(torch.nn.Module):

    """
    Direct optimization of weights for point clouds (includes both static and dynamic components)
    We take ELU so that weights are always positive
    """

    def __init__(self, number_of_points=10, number_of_frames=10) -> None:
        super().__init__()
        uncertainty = torch.zeros([number_of_frames, number_of_points])
        self.register_parameter("uncertainty", nn.Parameter(uncertainty))

        self.number_of_points = number_of_points
        self.number_of_frames = number_of_frames

        self.ELU = nn.ELU()

    def get_params(self):
        return self.uncertainty

    def set_confidence(self, confidence):
        self.uncertainty.data = confidence.detach().clone()

    def forward(self, frames=None, points=None):

        if points is None:
            points = torch.arange(self.number_of_points)

        if frames is None:
            frames = torch.arange(self.number_of_frames)

        return self.ELU(self.uncertainty[frames][:, points])# + 1 # add 1 for it to be positive
    
class ControlPointsDynamic(torch.nn.Module):

    """
    Optimized pose for control points. Note that transformation is relative to an undefined canonical space.
    Seems like more efficient to initialize as a grid? n x f x 7
    """

    def __init__(self, with_norm=False, number_of_points=10, number_of_frames=10) -> None:
        super().__init__()
        zero_translation = torch.zeros([number_of_points, number_of_frames, 3]) + 1e-6 

        self.register_parameter("delta_translation", nn.Parameter(zero_translation)) # N x F x 3
        
        self.number_of_points = number_of_points
        self.number_of_frames = number_of_frames
        self.with_norm = with_norm

        if with_norm:
            zero_normal = torch.zeros([number_of_points, number_of_frames, 3]) + 1e-4
            self.register_parameter("delta_normal", nn.Parameter(zero_normal)) # N x 3

    def get_params(self):

        if self.with_norm:
            norm = self.delta_normal
        else:
            norm = None
        
        return self.delta_translation, norm

    def set_translation(self, frame, translation):
        self.delta_translation.data[:, frame] = translation.detach().clone()

    def set_normal(self, frame, normal):

        self.delta_normal.data[:, frame] = normal.detach().clone()

    def get_raw_value(self, point_idx, frame):

        if frame is None:
            frame = torch.arange(self.number_of_frames)
        if point_idx is None:
            point_idx = torch.arange(self.number_of_points)

        translation = self.delta_translation[point_idx][:,frame, :]

        if self.with_norm:
            norm = self.delta_normal[point_idx][:, frame, :]
            norm = torch.nn.functional.normalize(norm, p=2, dim=-1)
        else:
            norm = None

        return translation, norm
    
    def get_translation(self, frame):

        """
        Returns the translation for all points for the given frame.
        """
        
        return self.delta_translation[:,frame,:]

    def forward(self, frames=None, points=None):

        """
        Returns all the rotation matrices and translations for the given frames and points.
        returns:
        R_out: (len(points), len(frames), 3, 3)
        t_out: (len(points), len(frames), 3, 1)
        """
        if frames is None:
            frames = torch.arange(self.number_of_frames)
        if points is None:
            points = torch.arange(self.number_of_points)
        if isinstance(frames, int):
            frames = [frames]

        t_out, norm = self.get_raw_value(points, frames)    

        return t_out, norm

class ControlPointsMotion(torch.nn.Module):

    """
    Optimized pose for control points. Note that transformation is relative to an undefined canonical space.
    Seems like more efficient to initialize as a grid? n x f x 7
    """

    def __init__(self, number_of_points=10) -> None:
        super().__init__()
        zero_translation = torch.zeros([number_of_points, 9]) + 1e-4 

        self.register_parameter("delta_translation", nn.Parameter(zero_translation)) # N x 9
        
        self.number_of_points = number_of_points

    def set_translation(self, point_idx, translation):
        assert(len(point_idx) == len(translation))
        self.delta_translation.data[point_idx] = translation.detach().clone()

    def get_param(self):

        return self.delta_translation
    
    def forward(self, points=None):

        """
        Returns all the translations for the given points.
        returns:
        t_out: (len(points), 3)
        """

        if points is None:
            points = torch.arange(self.number_of_points)
  
        t_out = self.delta_translation[points]

        return  t_out

class ControlPoints(torch.nn.Module):

    """
    Optimized pose for control points. Note that transformation is relative to an undefined canonical space.
    Seems like more efficient to initialize as a grid? n x f x 7
    """

    def __init__(self, number_of_points=10) -> None:
        super().__init__()
        zero_translation = torch.zeros([number_of_points, 3]) + 1e-4 

        self.register_parameter("delta_translation", nn.Parameter(zero_translation)) # N x 3
        
        self.number_of_points = number_of_points

    def set_translation(self, point_idx, translation):
        assert(len(point_idx) == len(translation))
        self.delta_translation.data[point_idx] = translation.detach().clone()

    def get_param(self):

        return self.delta_translation
    
    def forward(self, points=None):

        """
        Returns all the translations for the given points.
        returns:
        t_out: (len(points), 3)
        """

        if points is None:
            points = torch.arange(self.number_of_points)
  
        t_out = self.delta_translation[points]

        return  t_out

class CameraPoseDeltaCollectionv2(torch.nn.Module):
    def __init__(self, number_of_points=10) -> None:
        super().__init__()
        zero_rotation = torch.ones([1, 3]) * 1e-5
        zero_translation = torch.zeros([1, 3, 1]) + 1e-4
        for n in range(number_of_points):
            self.register_parameter(
                f"delta_rotation_{n}", nn.Parameter(zero_rotation))
            self.register_parameter(
                f"delta_translation_{n}", nn.Parameter(zero_translation)
            )
            self.register_buffer(
                f"actual_rotation_{n}", torch.eye(3)[None, ...]
            )

        self.register_buffer("zero_rotation", zero_rotation)
        self.register_buffer("zero_translation", torch.zeros([1, 3, 1]))
        self.traced_so3_exp_map = None
        self.number_of_points = number_of_points

    def update_rotation(self, idxes):

        """
        Update buffer actual rotation and set delta rotation to 0
        """

        with torch.no_grad():
            for idx in idxes:
                delta_rotation, _ = self.get_raw_value(idx)
                delta_rotation_mtx = _so3_exp_map(delta_rotation)[0]
                act_rot = getattr(self, f"actual_rotation_{idx}")
                new_rot = delta_rotation_mtx @ act_rot
                # Normalize rotation matrix
                new_rot = 1.5 * new_rot - 0.5 * new_rot @ new_rot.transpose(-2, -1) @ new_rot
                act_rot.data = new_rot
                random_sign = torch.tensor(1. if torch.rand(1) > 0.5 else -1.)
                delta_rotation.data = random_sign * self.zero_rotation.detach().clone()

    def get_rotation_and_translation_params(self):
        rotation_params = []
        translation_params = []
        for n in range(self.number_of_points):
            rotation_params.append(getattr(self, f"delta_rotation_{n}"))
            translation_params.append(getattr(self, f"delta_translation_{n}"))
        return rotation_params, translation_params

    def set_rotation_and_translation(self, index, rotation_so3, translation):
        delta_rotation = getattr(self, f"delta_rotation_{index}")
        delta_translation = getattr(self, f"delta_translation_{index}")
        delta_rotation.data = rotation_so3.detach().clone()
        delta_translation.data = translation.detach().clone()

    def set_first_frame_pose(self, R, t):
        self.zero_rotation.data = R.detach().clone().reshape([1, 3, 3])
        self.zero_translation.data = t.detach().clone().reshape([1, 3, 1])

    def get_raw_value(self, index):
        so3 = getattr(self, f"delta_rotation_{index}")
        translation = getattr(self, f"delta_translation_{index}")
        return so3, translation

    def forward(self, list_of_index):
        se_3 = []
        t_out = []
        act_rot = []
        for idx in list_of_index:
            delta_rotation, delta_translation = self.get_raw_value(idx)
            se_3.append(delta_rotation)
            t_out.append(delta_translation)
            act_rot.append(getattr(self, f"actual_rotation_{idx}"))
        se_3 = torch.cat(se_3, dim=0)
        t_out = torch.cat(t_out, dim=0)
        act_rot = torch.cat(act_rot, dim=0)
        if self.traced_so3_exp_map is None:
            self.traced_so3_exp_map = torch.jit.trace(
                _so3_exp_map, (se_3,))
        R_out = _so3_exp_map(se_3)[0] # returns rotation matrix
        R_out = torch.bmm(R_out, act_rot)    # add delta to original

        return R_out, t_out
    
    def forward_raw_value(self, list_of_index):
        se_3 = []
        t_out = []
        act_rot = []
        for idx in list_of_index:
            delta_rotation, delta_translation = self.get_raw_value(idx)
            se_3.append(delta_rotation)
            t_out.append(delta_translation)
            act_rot.append(getattr(self, f"actual_rotation_{idx}"))
        se_3 = torch.cat(se_3, dim=0)
        t_out = torch.cat(t_out, dim=0)
        # act_rot = torch.cat(act_rot, dim=0)
        # if self.traced_so3_exp_map is None:
        #     self.traced_so3_exp_map = torch.jit.trace(
        #         _so3_exp_map, (se_3,))
        # R_out = _so3_exp_map(se_3)[0] # returns rotation matrix
        # R_out = torch.bmm(R_out, act_rot)    # add delta to original

        return se_3, t_out

    def forward_index(self, index):
        # if index == 0:
        #     return self.zero_rotation, self.zero_translation
        # else:
        delta_rotation, delta_translation = self.get_raw_value(index)
        if self.traced_so3_exp_map is None:
            self.traced_so3_exp_map = torch.jit.trace(
                _so3_exp_map, (delta_rotation,))
        R = _so3_exp_map(delta_rotation)[0]
        return R, delta_translation

class CameraPoseDeltaCollection(torch.nn.Module):
    def __init__(self, number_of_points=10) -> None:
        super().__init__()
        zero_rotation = torch.ones([1, 3]) * 1e-3
        zero_translation = torch.zeros([1, 3, 1]) + 1e-4
        for n in range(number_of_points):
            self.register_parameter(
                f"delta_rotation_{n}", nn.Parameter(zero_rotation))
            self.register_parameter(
                f"delta_translation_{n}", nn.Parameter(zero_translation)
            )
        self.register_buffer("zero_rotation", torch.eye(3)[None, ...])
        self.register_buffer("zero_translation", torch.zeros([1, 3, 1]))
        self.traced_so3_exp_map = None
        self.number_of_points = number_of_points

    def get_rotation_and_translation_params(self):
        rotation_params = []
        translation_params = []
        for n in range(self.number_of_points):
            rotation_params.append(getattr(self, f"delta_rotation_{n}"))
            translation_params.append(getattr(self, f"delta_translation_{n}"))
        return rotation_params, translation_params

    def set_rotation_and_translation(self, index, rotation_so3, translation):
        delta_rotation = getattr(self, f"delta_rotation_{index}")
        delta_translation = getattr(self, f"delta_translation_{index}")
        delta_rotation.data = rotation_so3.detach().clone()
        delta_translation.data = translation.detach().clone()

    def set_first_frame_pose(self, R, t):
        self.zero_rotation.data = R.detach().clone().reshape([1, 3, 3])
        self.zero_translation.data = t.detach().clone().reshape([1, 3, 1])

    def get_raw_value(self, index):
        so3 = getattr(self, f"delta_rotation_{index}")
        translation = getattr(self, f"delta_translation_{index}")
        return so3, translation

    def forward_raw_value(self, list_of_index):
        se_3 = []
        t_out = []
        for idx in list_of_index:
            delta_rotation, delta_translation = self.get_raw_value(idx)
            se_3.append(delta_rotation)
            t_out.append(delta_translation)
        se_3 = torch.cat(se_3, dim=0)
        t_out = torch.cat(t_out, dim=0)
        return se_3, t_out

    def forward(self, list_of_index):
        se_3 = []
        t_out = []
        for idx in list_of_index:
            delta_rotation, delta_translation = self.get_raw_value(idx)
            se_3.append(delta_rotation)
            t_out.append(delta_translation)
        se_3 = torch.cat(se_3, dim=0)
        t_out = torch.cat(t_out, dim=0)
        if self.traced_so3_exp_map is None:
            self.traced_so3_exp_map = torch.jit.trace(
                _so3_exp_map, (se_3,))
        R_out = _so3_exp_map(se_3)[0] # returns rotation matrix

        return R_out, t_out

    def forward_index(self, index):
        # if index == 0:
        #     return self.zero_rotation, self.zero_translation
        # else:
        delta_rotation, delta_translation = self.get_raw_value(index)
        if self.traced_so3_exp_map is None:
            self.traced_so3_exp_map = torch.jit.trace(
                _so3_exp_map, (delta_rotation,))
        R = _so3_exp_map(delta_rotation)[0]
        return R, delta_translation

