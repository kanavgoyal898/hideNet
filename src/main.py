from constants import *
from model import *
from utils import *

import os
import random
import torch
import torchvision
import numpy as np

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((IMG_SIZE, IMG_SIZE)),
    torchvision.transforms.CenterCrop(IMG_SIZE),
    torchvision.transforms.ToTensor(),
])

dataset_train = torchvision.datasets.OxfordIIITPet(root='../data/', split='trainval', download=True, transform=transforms)
dataset_test = torchvision.datasets.OxfordIIITPet(root='../data/', split='test', download=True, transform=transforms)

train_valid_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

model = UNetLikeLite()
model.to(DEVICE)

print(f"Training model on {DEVICE} with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters.\n")

losses, simies = train_model(model, train_valid_loader, monitor=True)
plot_graphs(losses, 5, 'Loss')
plot_graphs(simies, 5, 'Structural Similarity')

plot_graph(np.mean(losses, axis=1), 'Loss')
plot_graph(np.mean(simies, axis=1), 'Structural Similarity')
os.makedirs('./parameters', exist_ok=True)
model_path = f'./parameters/{model.__class__.__name__}.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}\n")
