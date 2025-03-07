import torch

EPOCHS = 5
BATCH_SIZE = 16

IMG_PATH = '../image.jpg'
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = 256
LEARNING_RATE = 1e-3

NUM_HEADS = 5
NUM_CHANNELS = 1
INTERMEDIATE_CHANNELS = 64

CNN_DEPTH = 1

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
