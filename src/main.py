from constants import *
from model import Model
from utils import *

import random
import torch
import torchvision
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.CenterCrop(IMG_SIZE),
    torchvision.transforms.ToTensor(),
])

dataset_train = torchvision.datasets.Flowers102(root='../data/', split='train', download=True, transform=transform)
dataset_val = torchvision.datasets.Flowers102(root='../data/', split='val', download=True, transform=transform)
dataset_train_val = torch.utils.data.ConcatDataset([dataset_train, dataset_val])

dataset_test = torchvision.datasets.Flowers102(root='../data/', split='test', download=True, transform=transform)

train_valid_loader = torch.utils.data.DataLoader(dataset_train_val, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

model = Model()
model.to(DEVICE)

print(f"Training model on {DEVICE} with {sum(p.numel() for p in model.parameters()):,} parameters.\n")

losses, simies = train_model(model, train_valid_loader, monitor=True)
plot_graph(losses, 'Loss')
plot_graph(simies, 'Structural Similarity')
