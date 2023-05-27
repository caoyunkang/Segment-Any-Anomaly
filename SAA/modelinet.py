import timm
from copy import deepcopy
from typing import Tuple

import numpy as np
import timm
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore


class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return F.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
            self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
            self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class ModelINet(torch.nn.Module):
    # hrnet_w32, wide_resnet50_2
    def __init__(self, device, backbone_name='wide_resnet50_2', out_indices=(1, 2, 3), checkpoint_path='',
                 pool_last=False):
        super().__init__()
        # Determine if to output features.
        kwargs = {'features_only': True if out_indices else False}
        if out_indices:
            kwargs.update({'out_indices': out_indices})
        print(backbone_name)

        self.device = device
        self.backbone = timm.create_model(model_name=backbone_name, pretrained=True, checkpoint_path=checkpoint_path,
                                          **kwargs)
        self.backbone.eval()
        self.backbone = self.backbone.to(self.device)

        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1)) if pool_last else None

        self.pixel_mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(-1, 1, 1).to(self.device)
        self.pixel_std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(-1, 1, 1).to(self.device)

        self.img_size = 1024
        self.transform_size = ResizeLongestSide(self.img_size)

    def set_img_size(self, img_size):
        self.img_size = img_size
        self.transform_size = ResizeLongestSide(self.img_size)

    def preprocess(self, image: np.ndarray):
        """Normalize pixel values and pad to a square input."""

        input_image = self.transform_size.apply_image(image)
        input_image_torch = torch.as_tensor(input_image, device=self.device)
        x = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        ratio_h = h / self.img_size
        ratio_w = w / self.img_size
        return x, ratio_h, ratio_w

    @torch.no_grad()
    def forward(self, x):
        x, ratio_h, ratio_w = self.preprocess(x)
        x = x.to(self.device)

        # Backbone forward pass.
        features = self.backbone(x)

        # Adaptive average pool over the last layer.
        if self.avg_pool:
            fmap = features[-1]
            fmap = self.avg_pool(fmap)
            fmap = torch.flatten(fmap, 1)
            features.append(fmap)

        size_0 = features[0].shape[2:]

        for i in range(1, len(features)):
            features[i] = F.interpolate(features[i], size_0)

        features = torch.cat(features, dim=1)
        features = F.normalize(features, dim=1)

        return features, ratio_h, ratio_w
