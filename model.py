"""SHAL-Net DecomNet with the proposed lightweight GLL-DA module.

Changes relative to the original SHAL-Net model.py:
1. Adds a Global-Local Feature Refinement (GLLDA) module.
2. Inserts that module after DecomNet's x6 feature and before conv7.
3. Preserves DecomNet's constructor and six-item forward return signature so the
   existing train.py and loss.py can remain unchanged for the first experiment.

Expected input shapes:
    x:   [B, 3, H, W]  low-light RGB image
    x_h: [B, 1, H, W]  histogram-equalized grayscale image (as expected by conv1)

The original SHAL-Net preprocessing/data loader should be checked to confirm x_h
is single-channel before training.
"""

import cv2
import torch
import torch.nn as nn
import math
import numpy as np
from torchvision import transforms


class Conv(nn.Module):
    """Standard 3x3 convolution used by the original SHAL-Net."""

    def __init__(self, in_channel, out_channel):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        return self.conv(x)


class DepScp_Conv(nn.Module):
    """Depthwise 3x3 convolution followed by pointwise 1x1 convolution."""

    def __init__(self, in_channel, out_channel):
        super(DepScp_Conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channel,
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
        )

    def forward(self, x):
        return self.point_conv(self.depth_conv(x))


class GLLDA(nn.Module):
    """Ablation-ready global/local feature refinement module.

    Modes:
        full   : global branch + local branch + fusion (default)
        global : global branch only
        local  : local branch only
        none   : identity; disables GLLDA

    Disabled branches are not instantiated, so parameter counts reflect
    the active architecture.
    """

    VALID_MODES = {"full", "global", "local", "none"}

    def __init__(self, channels=64, mode="full"):
        super(GLLDA, self).__init__()

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid GLLDA mode {mode!r}. Choose from {sorted(self.VALID_MODES)}."
            )

        self.mode = mode
        hidden = max(channels // 4, 8)

        if mode in {"full", "global"}:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.global_mlp = nn.Sequential(
                nn.Conv2d(channels, hidden, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, channels * 3, kernel_size=1),
            )

        if mode in {"full", "local"}:
            self.local_conv = DepScp_Conv(channels, channels)
            self.local_act = nn.LeakyReLU(negative_slope=0.01, inplace=True)
            self.spatial_attention = nn.Conv2d(
                in_channels=2, out_channels=1, kernel_size=7,
                stride=1, padding=3,
            )

        if mode == "full":
            self.fusion = nn.Conv2d(channels * 2, channels, kernel_size=1)

        if mode != "none":
            self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _global_forward(self, x):
        global_context = self.global_pool(x)
        global_params = self.global_mlp(global_context)
        channel_logits, delta_gamma, beta = torch.chunk(
            global_params, chunks=3, dim=1
        )
        channel_weights = torch.sigmoid(channel_logits)
        delta_gamma = 0.1 * torch.tanh(delta_gamma)
        beta = 0.1 * torch.tanh(beta)
        film_features = (1.0 + delta_gamma) * x + beta
        return channel_weights * film_features

    def _local_forward(self, x):
        local_features = self.local_act(self.local_conv(x))
        avg_map = torch.mean(local_features, dim=1, keepdim=True)
        max_map = torch.max(local_features, dim=1, keepdim=True)[0]
        spatial_input = torch.cat([avg_map, max_map], dim=1)
        spatial_weights = torch.sigmoid(self.spatial_attention(spatial_input))
        return local_features * spatial_weights

    def forward(self, x):
        if self.mode == "none":
            return x
        if self.mode == "global":
            return x + self.residual_scale * self._global_forward(x)
        if self.mode == "local":
            return x + self.residual_scale * self._local_forward(x)

        global_features = self._global_forward(x)
        local_features = self._local_forward(x)
        combined = torch.cat([global_features, local_features], dim=1)
        fused = self.fusion(combined)
        return x + self.residual_scale * fused


class SCR(nn.Module):
    """Original SHAL-Net learnable structure/texture recovery branch."""

    def __init__(self):
        super(SCR, self).__init__()
        filters = 16
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.ds_conv1 = DepScp_Conv(3, filters)
        self.ds_conv2 = DepScp_Conv(filters, filters)
        self.ds_conv3 = DepScp_Conv(filters, filters)
        self.ds_conv4 = DepScp_Conv(filters, 1)

    def forward(self, x, x_h):
        """Return SCR-adjusted original and histogram-equalized inputs."""
        x1 = self.relu(self.ds_conv1(x))
        x2 = self.relu(self.ds_conv2(x1))
        x3 = self.relu(self.ds_conv3(x2))
        alpha = self.sigmoid(self.ds_conv4(x3)) + 1

        gam_batch = self.cal_gam_val(x.detach())
        x_scr = torch.pow(x, gam_batch) + (alpha * x)
        x_h_scr = torch.pow(x_h, gam_batch) + (alpha * x_h)
        return x_scr, x_h_scr

    def cal_gam_val(self, im):
        gam_batch = torch.tensor([])
        for i in range(im.shape[0]):
            im_np = im[i].mul(255).byte()
            im_np = im_np.detach().cpu().numpy().transpose(1, 2, 0)
            mean_gray = np.mean(np.max(im_np, axis=2))

            if mean_gray < 255 / 2:
                if mean_gray == 0:
                    # Default value avoids division by zero for an all-black image.
                    gam_batch = torch.cat(
                        [gam_batch, torch.tensor([0.25])], dim=0
                    )
                else:
                    gam = math.log10(0.5) / math.log10(mean_gray / 255.0)
                    gam_batch = torch.cat(
                        [gam_batch, torch.tensor([min(1, gam)])], dim=0
                    )
            else:
                gam_batch = torch.cat([gam_batch, torch.tensor([1.0])], dim=0)

        gam_batch = gam_batch.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return gam_batch.to(im.device)


class DecomNet(nn.Module):
    """SHAL-Net decomposition network with GLLDA inserted before conv7.

    The constructor and forward return tuple remain compatible with the
    original SHAL-Net train.py and loss.py.
    """

    def __init__(self, gll_da_mode="full"):
        super(DecomNet, self).__init__()
        filters = 64
        self.SCR = SCR()
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.conv1 = Conv(4, filters)
        self.conv2 = Conv(filters, filters)
        self.conv3 = Conv(filters, filters)
        self.conv4 = Conv(filters, filters)
        self.conv5 = Conv(filters * 2, filters)
        self.conv6 = Conv(filters * 2, filters)

        # New module: refines x6 using global and local information.
        self.gll_da = GLLDA(channels=filters, mode=gll_da_mode)

        # Output layout remains 4 channels: 3 for reflectance R, 1 for illumination L.
        self.conv7 = Conv(filters * 2, 4)

    def forward(self, x, x_h):
        x_scr, x_h_scr = self.SCR(x, x_h)

        # Original SHAL-Net feature extraction and skip connections.
        x_out = torch.cat([x, x_h_scr], dim=1)
        x1 = self.relu(self.conv1(x_out))
        x2 = self.relu(self.conv2(x1))
        x3 = self.relu(self.conv3(x2))
        x4 = self.relu(self.conv4(x3))
        x5 = self.relu(self.conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.conv6(torch.cat([x2, x5], dim=1)))

        # Proposed change: global/local refinement before the output layer.
        x6_refined = self.gll_da(x6)
        x7 = self.sigmoid(self.conv7(torch.cat([x1, x6_refined], dim=1)))

        # Original SHAL-Net reflectance/illumination split.
        R, L = torch.split(x7, [3, 1], dim=1)

        # Original hue extraction used by the existing hue loss.
        R_hue = self.rgb2hsv(R)[:, 0, :, :] / 360.0
        R_hue = transforms.GaussianBlur(kernel_size=7, sigma=(3, 3))(R_hue)
        x_hue = self.rgb2hsv(x)[:, 0, :, :] / 360.0
        x_hue = transforms.GaussianBlur(kernel_size=7, sigma=(3, 3))(x_hue)

        return x_scr, x_h_scr, R, L, R_hue, x_hue

    def rgb2hsv(self, input, epsilon=1e-10):
        assert input.shape[1] == 3, "rgb2hsv expects a 3-channel RGB tensor"
        r, g, b = input[:, 0], input[:, 1], input[:, 2]
        max_rgb, argmax_rgb = input.max(1)
        min_rgb, argmin_rgb = input.min(1)
        max_min = max_rgb - min_rgb + epsilon

        h1 = 60.0 * (g - r) / max_min + 60.0
        h2 = 60.0 * (b - g) / max_min + 180.0
        h3 = 60.0 * (r - b) / max_min + 300.0

        h = torch.stack((h2, h3, h1), dim=0).gather(
            dim=0, index=argmin_rgb.unsqueeze(0)
        ).squeeze(0)
        s = max_min / (max_rgb + epsilon)
        v = max_rgb
        return torch.stack((h, s, v), dim=1)


class EnhStage:
    """Original SHAL-Net enhancement stage; unchanged by GLLDA."""

    def __init__(self):
        super().__init__()

    def enh(self, im, R, L, im_h_scr, scale=0.85):
        mean = torch.mean(im)
        gam = math.log10(scale) / math.log10(mean)
        out = self.guided_filter(R, im_h_scr) * torch.pow(L, gam)
        return out

    def guided_filter(self, im, guide):
        im_np = im.mul(255).squeeze(0).byte()
        im_np = im_np.detach().cpu().numpy().transpose(1, 2, 0)
        guide_np = guide.squeeze(0).mul(255).byte()
        guide_np = guide_np.detach().cpu().numpy().transpose(1, 2, 0)
        im_dns = cv2.ximgproc.guidedFilter(guide_np, im_np, 1, 50, -1)
        return transforms.ToTensor()(im_dns).to(im.device)