import cv2
import os
import numpy as np
import torch
import math
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import knn_points
from scipy.spatial import cKDTree

def depth_vis(depth_map):

    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map_vis = cv2.applyColorMap(((depth_map)*255).astype(np.uint8), cv2.COLORMAP_HOT)
    
    return depth_map_vis

def flow_norm(flow, is_l1):

    """
    Normalize flow vectors
    """

    if is_l1:
        norm = torch.norm(flow, p=1, dim=-1)
    else:
        norm = torch.norm(flow, p=2, dim=-1)

    return norm

def fill_depth_nearest(depth_map, rgb, mask):
    # Get the coordinates of the masked (invalid) and unmasked (valid) pixels
    masked_coords = np.argwhere(mask)
    unmasked_coords = np.argwhere(~mask)
    
    # Extract depth values of unmasked pixels
    unmasked_depths = depth_map[~mask]
    
    # Build a KDTree with the unmasked coordinates
    tree = cKDTree(unmasked_coords)
    
    # Query the nearest unmasked neighbors for each masked location
    distances, indices = tree.query(masked_coords, k=9)

    rgb_curr = rgb[mask][:,None,:]
    rgb_tgt = rgb[~mask][indices]

    rgb_diff = np.linalg.norm(rgb_curr - rgb_tgt, axis=-1)
    idx = np.argmin(rgb_diff, axis=1)

    indices = indices[np.arange(idx.shape[0]), idx]
    
    # Replace the depth values in the mask locations with the nearest neighbors
    depth_map[mask] = unmasked_depths[indices]
    
    return depth_map

def clean_depth_outliers(depth, rgb, thrs=3.0):

    grad_x = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)  # Gradient along x-axis
    grad_y = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)  # Gradient along y-axis

    # Optionally, compute the gradient magnitude (Euclidean distance)
    grad_magnitude = cv2.magnitude(grad_x, grad_y)

    mask = grad_magnitude > thrs

    clean_depth = fill_depth_nearest(depth.copy(), rgb, mask)

    return clean_depth

def l1_loss_with_uncertainty(err, c=None, norm=False):

    if c is None:
        c = torch.ones_like(err)

    loss = torch.sum(err*c)
    weight = torch.sum(c) + 1e-8
    return loss / (weight+1e-8), weight

def remove_outlier(points, threshold=2.0):

    """
    Remove points that are far from the mean
    points: torch tensor of shape Nx3
    """
    # Use knn_points to find neighbors and compute distances
    K = 4  # Number of neighbors
    knn_result = knn_points(points[None, :, :], points[None, :, :], K=K)

    # Extract distances and indices
    distances = knn_result.dists[..., 1:]  # Skip the first distance (distance to itself)

    # Compute mean distance and standard deviation for each point
    mean_distances = distances.mean(dim=-1)
    std_distances = distances.std(dim=-1)

    # Compute mean of mean distances across the batch
    mean_of_mean_distances = mean_distances.mean()

    # Identify outliers
    outliers = mean_distances > mean_of_mean_distances + threshold * std_distances

    return (~outliers).squeeze()
  

class KeyFrameBuffer:

    """
    Buffer that holds maximum of buffer_size keyframes.
    """

    def __init__(self, buffer_size=5):
        self.buffer = []
        self.buffer_size = buffer_size

    def add_keyframe(self, index):
        if index in self.buffer:
            return
        if len(self.buffer) == self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(index)

    def clear(self):
        del self.buffer
        self.buffer = []

def filter_cotracker(masks, shape, all_tracks, all_vis):

    """
    Filter points according to visibility and oob values
    """

    filtered_tracks = []
    filtered_visibilities = []

    h, w = shape

    all_labels = []

    within_bound = np.logical_and(all_tracks[:,:,1] >= 0, all_tracks[:,:,1] < h)
    within_bound = np.logical_and(within_bound, all_tracks[:,:,0] >= 0)
    within_bound = np.logical_and(within_bound, all_tracks[:,:,0] < w)
    within_bound = np.logical_and(within_bound, all_vis == 1)

    all_tracks[:,:,0] = np.clip(all_tracks[:,:,0], 0, w-1)
    all_tracks[:,:,1] = np.clip(all_tracks[:,:,1], 0, h-1)

    all_vis = np.logical_and(all_vis, within_bound)        # tracklets that goes out of bounds are not visible

    for i in range(all_tracks.shape[1]):

        print(f"Done with {i} of tracks")

        track = all_tracks[:,i,:]
        visibility = all_vis[:,i]

        static_mask = masks==0
        dyn_mask = masks==1
        invalid_mask = masks==2

        visibility= np.logical_and(visibility, ~invalid_mask[track[:,1], track[:,0]])   # turn point to invisible for invalid points

        total = sum(visibility)

        if total < 5:    # if point is not visible for more than 5 frames
            all_labels.append(2)
            continue

        static_score = static_mask[track[visibility][:,1], track[visibility][:,0]]   # amount in static
        dyn_score = dyn_mask[track[visibility][:,1], track[visibility][:,0]]   # amount in dynamic
        
        if static_score > 0.8 * total:
            all_labels.append(0)
        elif dyn_score > 0.8 * total:
            all_labels.append(1)
        else:
            all_labels.append(2)

def draw_frame(frame, track_frame, vis_frame):

    """
    Visualize predicted points and gt points
    """

    for i, point in enumerate(track_frame):
        if vis_frame[i]:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 255, 0), -1)  # Red circle
        else:
            cv2.circle(frame, (int(point[0]), int(point[1])), 2, (0, 0, 255), -1)  # Green circle

    return frame

def vis_training(rgbs, tracks, vis, path):

    """
    Visualize point tracks
    """
    num_frames, _, _ = tracks.shape

    os.makedirs(path, exist_ok=True)

    for f in range(num_frames):

        frame = rgbs[f]
        track_frame = tracks[f, :, :] # gt labels
        vis_frame = vis[f, :]

        frame = draw_frame(frame.copy(), track_frame, vis_frame)

        cv2.imwrite(os.path.join(f"{path}/", str(f).zfill(4)+".png"), frame)

class BadPixelMetric:
    def __init__(self, threshold=1.5, depth_cap=80):
        self.__threshold = threshold
        self.__depth_cap = depth_cap


    def compute_scale_and_shift(self, prediction, target, mask, entire_video=False):

        """
        Performs least squares scaling using per image / video depending on arg entire_video
        """

        if not entire_video:
            # solves for least squarers solution of Ax = b
            # system matrix: A = [[a_00, a_01], [a_10, a_11]]
            a_00 = np.sum(mask * prediction * prediction, (1, 2))
            a_01 = np.sum(mask * prediction, (1, 2))
            a_11 = np.sum(mask, (1, 2))

            # right hand side: b = [b_0, b_1]
            b_0 = np.sum(mask * prediction * target, (1, 2))
            b_1 = np.sum(mask * target, (1, 2))

        else:
            a_00 = np.sum(mask * prediction * prediction)
            a_01 = np.sum(mask * prediction)
            a_11 = np.sum(mask)

            # right hand side: b = [b_0, b_1]
            b_0 = np.sum(mask * prediction * target)
            b_1 = np.sum(mask * target)

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        x_0 = np.zeros_like(b_0)
        x_1 = np.zeros_like(b_1)

        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

        return x_0, x_1

    def get_median_scale(self, pred, tgt, mask, entire_video=False):
        
        scales = np.ones((len(mask)))
        for i in range(len(mask)):
            if np.sum(mask)==0:
                print("EMPTY MASK")
                continue
            scale = np.median(tgt[i][mask[i]] / (pred[i][mask[i]]+1e-16))
            scales[i] = scale

        if entire_video:
            scale = np.mean(scales)

        return scales

    def __call__(self, prediction, target, mask, entire_video=False):

        scale, shift = self.compute_scale_and_shift(prediction, target, mask, entire_video)
        if not entire_video:
            prediction_aligned = scale.reshape((-1, 1, 1)) * prediction + shift.reshape((-1, 1, 1))
        if entire_video:
            prediction_aligned = scale * prediction + shift

        prediction_aligned[prediction_aligned > self.__depth_cap] = self.__depth_cap

        if np.mean(scale) == 0:
            return prediction_aligned

        return prediction_aligned
    
class EarlyStopper:
    """
    EarlyStopper class to stop training if validation loss does not improve
    by at least min_delta for patience epochs.
    """
    
    def __init__(self, patience=1, min_delta=0.0001):
        """
        Args:
            patience (int): Number of epochs to wait before early stopping
            min_delta (float): Minimum decrease in loss to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        """
        Returns True if training should stop (loss hasn't improved for patience epochs)
        
        Args:
            validation_loss (float): Current epoch's validation loss
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if validation_loss < self.min_validation_loss:
            improvement = self.min_validation_loss - validation_loss
            if improvement >= self.min_delta:
                self.min_validation_loss = validation_loss
                self.counter = 0
                return False
            
        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False