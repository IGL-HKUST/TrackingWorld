import os

import cv2
import matplotlib.cm as cm
import numpy as np
import torch


def read_video(video, max_res):

    original_height, original_width = video.shape[1:3]

    height = round(original_height / 64) * 64
    width = round(original_width / 64) * 64

    # resize the video if the height or width is larger than max_res
    if max(height, width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale / 64) * 64
        width = round(original_width * scale / 64) * 64

    frames = []

    for frame in video:
        frame = cv2.resize(frame, (width, height))
        frames.append(frame.astype("float32") / 255.0)

    frames = np.array(frames)
    return frames, original_height, original_width
