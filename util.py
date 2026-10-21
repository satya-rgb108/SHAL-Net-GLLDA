
import os
import torchvision


def save_img(img, im_dir, im_name):
    """
    Save an image tensor to disk.

    Args:
        img: Image tensor, normally [1, C, H, W] or [C, H, W].
        im_dir: Destination directory.
        im_name: Image filename, optionally including subdirectories.
    """
    save_path = os.path.join(im_dir, im_name)

    folder_dir = os.path.dirname(save_path)
    if folder_dir:
        os.makedirs(folder_dir, exist_ok=True)

    torchvision.utils.save_image(img.detach().cpu(), save_path)
