import torch

EPOCHS = 100
BATCH_SIZE = 16

IMG_PATH = '../image.jpg'
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_SIZE = 256
LEARNING_RATE = 1e-3

NUM_HEADS = 3
NUM_BLOCKS = 3
NUM_CHANNELS = 3

INTERMEDIATE_CHANNELS = 256

CNN_DEPTH = 3

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
