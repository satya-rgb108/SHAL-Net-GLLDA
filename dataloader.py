import os
import glob

import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms


class LoadDataset(data.Dataset):
    """SHAL-Net dataset loader with Pillow-version compatibility."""

    def __init__(self, data_dir, tr, img_size=None):
        self.data_dir = data_dir
        self.transform = tr
        self.img_size = img_size
        self.img_list = self.load_imglist()

        if not self.img_list:
            raise RuntimeError(
                f"No images found in '{data_dir}'. Expected images directly in "
                "that folder or one directory level below it."
            )

    def __getitem__(self, idx):
        img, img_clahe, name = self.load_img(self.img_list[idx])
        return img, img_clahe, name

    def __len__(self):
        return len(self.img_list)

    def load_img(self, img_path):
        # SHAL-Net expects RGB input with three channels.
        with Image.open(img_path) as im:
            img = im.convert("RGB")

        if self.img_size is not None:
            # PIL expects (width, height). SHAL-Net uses square sizes here.
            resampling = getattr(Image, "Resampling", Image)
            img = img.resize(
                (self.img_size[0], self.img_size[1]),
                resample=resampling.LANCZOS,
            )

        img_norm = self.transform(img) if self.transform is not None else transforms.ToTensor()(img)
        img_clahe_norm = self.img_HE(img_norm)
        img_name = os.path.basename(img_path)
        return img_norm, img_clahe_norm, img_name

    def load_imglist(self):
        # Support either class/subfolder/image.png or image.png layouts.
        image_patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
        img_list = []
        for pattern in image_patterns:
            img_list.extend(glob.glob(os.path.join(self.data_dir, "*", pattern)))
        if not img_list:
            for pattern in image_patterns:
                img_list.extend(glob.glob(os.path.join(self.data_dir, pattern)))
        return sorted(set(img_list))

    def img_HE(self, tensor):
        """Create the single-channel CLAHE/bilateral-filter image used by SHAL-Net."""
        def clahe(im):
            ycrcb = cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)
            channels = list(cv2.split(ycrcb))
            clahe_op = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(1, 1))
            channels[0] = clahe_op.apply(channels[0])
            ycrcb = cv2.merge(channels)
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB)

        # tensor is expected to be [3, H, W] with values in [0, 1].
        im = tensor.detach().mul(255).clamp(0, 255).byte()
        im = im.cpu().numpy().transpose(1, 2, 0).copy()

        im = clahe(im)
        im = cv2.bilateralFilter(im, d=5, sigmaColor=10, sigmaSpace=10)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        im = im.astype(np.uint8, copy=False)
        return transforms.ToTensor()(im)
