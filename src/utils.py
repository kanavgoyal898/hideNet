import torch
import numpy as np
import matplotlib.pyplot as plt

from constants import *

def split_image(image):
    """
    Splits an image into two parts: one containing pixels at even row-odd column positions
    and the other containing pixels at odd row-even column positions.

    Args:
        image (numpy.ndarray or torch.Tensor): Input image of shape (B, C, H, W), (C, H, W) or (H, W).

    Returns:
        tuple: (e_img, o_img)
            - e_img (numpy.ndarray): Image with pixels from even rows and odd columns.
            - o_img (numpy.ndarray): Image with pixels from odd rows and even columns.
    """

    H, W = image.shape[-2], image.shape[-1]

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().to(torch.float32).numpy()

    e_img, o_img = None, None
    if image.ndim == 4:     # (B, C, H, W)
        e_img = image[:, :, 0::2, 1::2]
        o_img = image[:, :, 1::2, 0::2]
    elif image.ndim == 3:   # (C, H, W)
        e_img = image[:, 0::2, 1::2]
        o_img = image[:, 1::2, 0::2]
    elif image.ndim == 2:   # (H, W)
        e_img = image[0::2, 1::2]
        o_img = image[1::2, 0::2]
    else:
        raise ValueError("Unsupported image dimensions.")
    
    return e_img, o_img

def merge_images(e_img, o_img):
    """
    Merges two images back into a single image by interleaving their pixels.

    Args:
        e_img (numpy.ndarray or torch.Tensor): Image with pixels from even rows and odd columns.
        o_img (numpy.ndarray or torch.Tensor): Image with pixels from odd rows and even columns.

    Returns:
        numpy.ndarray: Merged image of shape (B, C, 2H, 2W), (C, 2H, 2W) or (2H, 2W).
    """

    H, W = e_img.shape[-2], e_img.shape[-1] 

    if isinstance(e_img, torch.Tensor):
        e_img = e_img.detach().cpu().to(torch.float32).numpy()
    if isinstance(o_img, torch.Tensor):
        o_img = o_img.detach().cpu().to(torch.float32).numpy()
    
    if e_img.ndim == 4:     # (B, C, H, W)
        if e_img.shape[0] != o_img.shape[0]:
            raise ValueError("Batch size mismatch.")
        if e_img.shape[1] != o_img.shape[1]:
            raise ValueError("Channel size mismatch.")
        merged_img = np.zeros((e_img.shape[0], e_img.shape[1], 2 * H, 2 * W))
        merged_img[:, :, 0::2, 1::2] = e_img
        merged_img[:, :, 1::2, 0::2] = o_img
    elif e_img.ndim == 3:   # (C, H, W)
        if e_img.shape[0] != o_img.shape[0]:
            raise ValueError("Batch size mismatch.")
        merged_img = np.zeros((e_img.shape[0], 2 * H, 2 * W))
        merged_img[:, 0::2, 1::2] = e_img
        merged_img[:, 1::2, 0::2] = o_img
    elif e_img.ndim == 2:   # (H, W)
        merged_img = np.zeros((2 * H, 2 * W))
        merged_img[0::2, 1::2] = e_img
        merged_img[1::2, 0::2] = o_img
    else:
        raise ValueError("Unsupported image dimensions.")

    return merged_img

def to_bw(image):
    """
    Converts an image to grayscale by averaging channel values.

    Args:
        image (numpy.ndarray or torch.Tensor): Input image of shape (B, C, H, W), (C, H, W), (H, W), or (B, H, W).

    Returns:
        numpy.ndarray: Grayscale image of shape (B, H, W) or (H, W).
    """

    if image.ndim == 4:     # (B, C, H, W)
        return image.mean(axis=1, keepdims=True)        # (B, C, H, W) -> (B, 1, H, W)
    elif image.ndim == 3:   # (B, H, W) or (C, H, W)
        if image.shape[0] != BATCH_SIZE:
            return image.mean(axis=0, keepdims=True)    # (C, H, W) -> (1, H, W)
        return image                                    # (B, H, W) -> (B, H, W)
    elif image.ndim == 2:   # (H, W)
        return image                                    # (H, W) -> (H, W)
    else:
        raise ValueError("Unsupported image dimensions.")

def show_image(img, title="Image Title"):
    """
    Displays a single grayscale image.

    Args:
        img (numpy.ndarray): Image of shape (H, W) or (C, H, W).
        title (str, optional): Title of the image. Defaults to "Image Title".

    Returns:
        None
    """

    if img.ndim == 3 or img.ndim == 2:
        img = img.transpose(1, 2, 0) if img.ndim == 3 else img
        plt.imshow(img, cmap="gray")
        plt.title(title)
        plt.axis("off")
        plt.show()
    else:
        raise ValueError("Input tensor should have 3 (C, H, W) or 2 (H, W) dimensions.")

def show_images(image1, image2, title1="Image Title 1", title2="Image Title 2"): 
    """
    Displays two grayscale images side by side.

    Args:
        image1 (numpy.ndarray): First image of shape (H, W) or (C, H, W).
        image2 (numpy.ndarray): Second image of shape (H, W) or (C, H, W).
        title1 (str, optional): Title of the first image. Defaults to "Image Title 1".
        title2 (str, optional): Title of the second image. Defaults to "Image Title 2".

    Returns:
        None
    """

    if (image1.ndim == 3 or image1.ndim == 2) and (image2.ndim == 3 or image2.ndim == 2):  
        image1 = image1.transpose(1, 2, 0) if image1.ndim == 3 else image1
        image2 = image2.transpose(1, 2, 0) if image2.ndim == 3 else image2
        plt.figure(figsize=(10, 2))
        plt.subplot(1, 2, 1)
        plt.imshow(image1, cmap="gray")
        plt.title(title1)
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(image2, cmap="gray")
        plt.title(title2)
        plt.axis("off")
        plt.show()
    else:
        raise ValueError("Input tensors should have 3 (C, H, W) or 2 (H, W) dimensions.")
    