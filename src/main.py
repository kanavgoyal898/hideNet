from constants import *
from model import Model
from utils import *

import os
import random
import sklearn
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

def train_model(model, train_loader, test_loader):
    model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    maes = []
    losses = []
    for epoch in range(EPOCHS):
        num_steps = len(train_loader)

        iter = 0
        loss_e = []
        for images, _ in train_loader:
            if NUM_CHANNELS == 1:
                images = to_bw(images)
            x, y = split_image(images)
            x = torch.tensor(x)
            y = torch.tensor(y)
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            out, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_e.append(loss.item())
            print(f"TRAINING MODE \t|| Epoch: {epoch+1:4d}/{EPOCHS:4d} | Iteration: {iter+1:4d}/{num_steps:4d} | Loss: {loss.item():.4f}")

            iter += 1
        
        mae = test_model(model, test_loader)

        maes.append(mae)
        losses.extend(loss_e)
        print(f"EVALUATION MODE || Epoch: {epoch+1:4d}/{EPOCHS:4d} | Loss: {np.mean(loss_e):.4f}")
        print(f"EVALUATION MODE || Epoch: {epoch+1:4d}/{EPOCHS:4d} | Mean Absolute Error: {mae:.4f}")
        print(f"-------------------------------------------------------------------------")

    return losses, maes

@torch.no_grad()
def test_model(model, test_loader):
    model.eval()

    X, Y, Y_ = [], [], []
    for images, _ in test_loader:
        if NUM_CHANNELS == 1:
            images = to_bw(images)
        x, y = split_image(images)
        x = torch.tensor(x)
        y = torch.tensor(y)
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        y_, _ = model(x, y)

        X.append(x.cpu().numpy())
        Y.append(y.cpu().numpy())
        Y_.append(y_.cpu().numpy())

    X = np.concatenate(X)
    Y = np.concatenate(Y)
    Y_ = np.concatenate(Y_)

    Y = Y.reshape(-1)
    Y_ = Y_.reshape(-1)

    return sklearn.metrics.mean_absolute_error(Y, Y_)    

def train_and_test():
    model = Model()
    model.to(DEVICE)

    print(f"Training model on {DEVICE} with {sum(p.numel() for p in model.parameters()):,} parameters.\n")

    losses, maes = train_model(model, train_valid_loader, test_loader)
    mae = test_model(model, test_loader)

    if os.path.exists("./models") == False:
        os.mkdir("./models")
    torch.save(model.state_dict(), "./models/model.pth")
    print(f"Model saved at ./models/model.pth")

train_and_test()
