from constants import *
from utils import *

import random
import torch
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

image = np.random.rand(BATCH_SIZE, NUM_CHANNELS, IMG_HEIGHT, IMG_WIDTH)

# Split Image
e_img, o_img = split_image(image)
print(f"Original Image Shape: \t\t{image.shape}")
print(f"Even Image Shape: \t\t{e_img.shape}")
print(f"Odd Image Shape: \t\t{o_img.shape}")

# Sample Image
show_images(e_img[0], o_img[0], "Even Image", "Odd Image")


# Merge Image
merged_img = merge_images(e_img, o_img)
print(f"Original Even Image Shape: \t{e_img.shape}")
print(f"Original Odd Image Shape: \t{o_img.shape}")
print(f"Merged Image Shape: \t\t{merged_img.shape}")

# Sample Image
show_image(merged_img[0], "Merged Image")
