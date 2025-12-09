# @File: util.py
# @Project: SceneTracker
# @Author : wangbo
# @Time : 2024.07.04

from functools import partial

import albumentations as A
import numpy as np
import torch


def augment_video(augmenter, **kwargs):
    assert isinstance(augmenter, A.ReplayCompose)
    keys = kwargs.keys()
    for i in range(len(next(iter(kwargs.values())))):
        data = augmenter(
            **{key: kwargs[key][i] if key not in ["bboxes", "keypoints"] else [kwargs[key][i]] for key in keys}
        )
        if i == 0:
            augmenter = partial(A.ReplayCompose.replay, data["replay"])
        for key in keys:
            if key == "bboxes":
                kwargs[key][i] = np.array(data[key]).reshape(4)
            elif key == "keypoints":
                kwargs[key][i] = np.array(data[key]).reshape(2)
            else:
                kwargs[key][i] = data[key]


class LongTermSceneFlowAugmentor:
    def __init__(self, train_mode, crop_size=(384, 512)):

        self.train_mode = train_mode

        self.color_augmenter = A.ReplayCompose(
            [
                # A.GaussNoise(p=0.2),
                A.OneOf(
                    [
                        A.MotionBlur(p=0.2),
                        A.MedianBlur(blur_limit=3, p=0.1),
                        A.Blur(blur_limit=3, p=0.1),
                    ],
                    p=0.2,
                ),
                A.OneOf(
                    [
                        A.CLAHE(clip_limit=2),
                        A.Sharpen(),
                        A.Emboss(),
                        A.RandomBrightnessContrast(),
                    ],
                    p=0.2,
                ),
                A.RGBShift(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.HueSaturationValue(p=0.3),
                A.ImageCompression(quality_lower=50, quality_upper=100, p=0.3),
            ],
            p=0.8,
        )

    def color_transform(self, du):

        rgbs = du.video

        if isinstance(rgbs, torch.Tensor):
            rgbs = rgbs.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # shape: (T, H, W, 3)

        augment_video(self.color_augmenter, image=rgbs)

        du.video = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()  # shape: (T, 3, H, W)

        return du

    def __call__(self, du):

        if self.train_mode:
            du = self.color_transform(du)

        return du

    def video_augs(self, video):
        augment_video(self.color_augmenter, image=video)
