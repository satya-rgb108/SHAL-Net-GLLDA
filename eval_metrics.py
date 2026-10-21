import argparse
import os
import warnings

import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from skimage.color import deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tabulate import tabulate
from tqdm import tqdm

warnings.filterwarnings("ignore")

to_tensor = transforms.ToTensor()
lpips_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net="alex", verbose=False).to(lpips_device).eval()


def _rgb_array(image):
    """Convert a PIL image or array into an 8-bit RGB NumPy array."""
    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        array = np.asarray(image)
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = cv2.cvtColor(array, cv2.COLOR_GRAY2RGB)
        elif array.ndim == 3 and array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_RGBA2RGB)
        elif array.ndim == 3 and array.shape[2] == 3:
            # Arrays are assumed to be RGB, not BGR.
            pass
        else:
            raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _paired_arrays(im1, im2):
    a = _rgb_array(im1)
    b = _rgb_array(im2)
    if a.shape != b.shape:
        raise ValueError(f"Image shape mismatch: reference {a.shape}, output {b.shape}")
    return a, b


def PSNR(im1, im2):
    a, b = _paired_arrays(im1, im2)
    return peak_signal_noise_ratio(a, b, data_range=255)


def SSIM(im1, im2):
    a, b = _paired_arrays(im1, im2)
    return structural_similarity(a, b, channel_axis=2, data_range=255)


def LPIPS(im1, im2):
    a, b = _paired_arrays(im1, im2)
    # LPIPS expects RGB tensors in [-1, 1] when normalize=False.
    ta = torch.from_numpy(a.copy()).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    tb = torch.from_numpy(b.copy()).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
    ta = ta.to(lpips_device)
    tb = tb.to(lpips_device)
    with torch.no_grad():
        return float(lpips_model(ta, tb).item())


def MAE(im1, im2):
    a, b = _paired_arrays(im1, im2)
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean() / 255.0)


def DeltaE(im1, im2):
    a, b = _paired_arrays(im1, im2)
    lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2LAB).astype(np.float32)
    return float(deltaE_ciede2000(lab_a, lab_b).mean())


def cal_score(GT_dir, img_dir):
    if not os.path.isdir(GT_dir):
        raise FileNotFoundError(f"Reference folder not found: {GT_dir}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Enhanced-image folder not found: {img_dir}")

    # Match files by basename, rather than assuming the sorted lists align.
    allowed_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def image_map(folder):
        mapping = {}
        for filename in os.listdir(folder):
            path = os.path.join(folder, filename)
            if os.path.isfile(path) and os.path.splitext(filename)[1].lower() in allowed_ext:
                mapping[filename] = path
        return mapping

    gt_map = image_map(GT_dir)
    output_map = image_map(img_dir)

    gt_names = set(gt_map)
    output_names = set(output_map)
    missing_outputs = sorted(gt_names - output_names)
    extra_outputs = sorted(output_names - gt_names)

    if missing_outputs or extra_outputs:
        message = []
        if missing_outputs:
            message.append(f"Missing enhanced images ({len(missing_outputs)}): {missing_outputs[:5]}")
        if extra_outputs:
            message.append(f"Extra enhanced images ({len(extra_outputs)}): {extra_outputs[:5]}")
        raise ValueError("Reference/output filenames do not match. " + " | ".join(message))

    names = sorted(gt_names)
    if not names:
        raise RuntimeError(f"No supported images found in {GT_dir}")

    scores = {"PSNR": [], "SSIM": [], "LPIPS": [], "MAE": [], "DeltaE": []}

    for name in tqdm(names, desc="Evaluating"):
        with Image.open(gt_map[name]) as gt_image:
            gt_image = gt_image.convert("RGB")
            with Image.open(output_map[name]) as output_image:
                output_image = output_image.convert("RGB")
                if gt_image.size != output_image.size:
                    raise ValueError(
                        f"Size mismatch for {name}: reference {gt_image.size}, "
                        f"output {output_image.size}. Resize consistently before evaluation."
                    )
                scores["PSNR"].append(PSNR(gt_image, output_image))
                scores["SSIM"].append(SSIM(gt_image, output_image))
                scores["LPIPS"].append(LPIPS(gt_image, output_image))
                scores["MAE"].append(MAE(gt_image, output_image))
                scores["DeltaE"].append(DeltaE(gt_image, output_image))

    headers = ["Metric", "Mean score", "Direction"]
    rows = [
        ["PSNR (dB)", f"{np.mean(scores['PSNR']):.4f}", "Higher is better"],
        ["SSIM", f"{np.mean(scores['SSIM']):.4f}", "Higher is better"],
        ["LPIPS", f"{np.mean(scores['LPIPS']):.4f}", "Lower is better"],
        ["MAE (0–1)", f"{np.mean(scores['MAE']):.4f}", "Lower is better"],
        ["DeltaE CIEDE2000", f"{np.mean(scores['DeltaE']):.4f}", "Lower is better"],
    ]
    print(f"\nEvaluated {len(names)} matched image pairs.")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    return scores


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate enhanced images against references.")
    parser.add_argument("--gt_dir", type=str, default="datasets/test_data/reference/LOL_15")
    parser.add_argument("--img_dir", type=str, default="results/LOL_15_GLLDA/Enh")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cal_score(args.gt_dir, args.img_dir)
