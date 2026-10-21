import argparse
import os
import time
import warnings

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import DecomNet, EnhStage
from dataloader import LoadDataset
import util

warnings.filterwarnings("ignore")


def load_model(weights_path, device):
    """Load a checkpoint created by the same DecomNet architecture."""
    model = DecomNet().to(device)

    try:
        checkpoint = torch.load(
            weights_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(weights_path, map_location=device)

    # Support either a raw state_dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    # Remove DataParallel prefix if present.
    if isinstance(checkpoint, dict) and checkpoint:
        if all(key.startswith("module.") for key in checkpoint):
            checkpoint = {
                key[len("module."):]: value
                for key, value in checkpoint.items()
            }

    # Strict loading intentionally catches architecture/checkpoint mismatches.
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def detect(opt, device, test_loader, model):
    model.eval()
    enh_module = EnhStage()
    total_model_time = 0.0
    num_images = 0

    enhanced_dir = os.path.join(opt.result_folder, "Enh")
    reflectance_dir = os.path.join(opt.result_folder, "R")
    illumination_dir = os.path.join(opt.result_folder, "L")
    for folder in (enhanced_dir, reflectance_dir, illumination_dir):
        os.makedirs(folder, exist_ok=True)

    with torch.no_grad():
        for input_img, input_h, names in test_loader:
            input_img = input_img.to(device)
            input_h = input_h.to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()

            _, input_h_scr, R, L, _, _ = model(input_img, input_h)
            output = enh_module.enh(input_img, R, L, input_h_scr)

            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            total_model_time += elapsed
            num_images += input_img.size(0)
            for i, name in enumerate(names):
                print(f"Image: {name}\tInference time: {elapsed / len(names):.4f}s")
                util.save_img(output[i:i + 1], enhanced_dir, name)
                util.save_img(R[i:i + 1], reflectance_dir, name)
                util.save_img(L[i:i + 1], illumination_dir, name)

    if num_images:
        print(f"Average model + enhancement time: {total_model_time / num_images:.4f}s/image")
    print(f"Enhanced images saved to: {enhanced_dir}")


def main(opt):
    if not os.path.isdir(opt.test_data):
        raise FileNotFoundError(f"Test data folder does not exist: {opt.test_data}")
    if not os.path.isfile(opt.weights):
        raise FileNotFoundError(f"Checkpoint file does not exist: {opt.weights}")

    if torch.cuda.is_available() and opt.device.lower() != "cpu":
        device = torch.device(f"cuda:{opt.device}")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_set = LoadDataset(opt.test_data, test_transform)
    if len(test_set) == 0:
        raise RuntimeError(f"No images found in test folder: {opt.test_data}")

    test_loader = DataLoader(
        dataset=test_set,
        num_workers=opt.num_workers,
        batch_size=1,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )
    print(f"Number of test images: {len(test_set)}")

    model = load_model(opt.weights, device)
    detect(opt, device, test_loader, model)


def parse_opt():
    parser = argparse.ArgumentParser(description="Run SHAL-Net/GLL-DA low-light enhancement.")
    parser.add_argument("--test_data", type=str,
                        default="datasets/test_data/low-light/LOL_15")
    parser.add_argument("--weights", type=str, default="ckpts/gll_da/epoch_1.pth")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="0",
                        help="CUDA device index (e.g. 0) or 'cpu'")
    parser.add_argument("--result_folder", type=str, default="results/LOL_15_GLLDA")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_opt())
