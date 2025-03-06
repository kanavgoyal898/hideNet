from constants import *
from model import Model
from utils import *

import random
import skimage
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
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Training model on {DEVICE} with {sum(p.numel() for p in model.parameters()):,} parameters.\n")

def train_model():
    losses, simies = [], []

    for epoch in range(EPOCHS):   

        iter = 0
        num_steps = len(train_valid_loader)
    
        loss_e, simi_e = [], []
        for images, _ in train_valid_loader:
            if NUM_CHANNELS == 1:
                images = to_bw(images)
            images = images.to('cpu').detach().numpy()
            images_ = images

            d = 0
            # Split Images
            while d < CNN_DEPTH:
                e_img, o_img = split_image(images_)

                x = torch.from_numpy(e_img).to(torch.float32).to(DEVICE)
                y = torch.from_numpy(o_img).to(torch.float32).to(DEVICE)

                y_, loss = model(x, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                images_ = e_img
                d += 1


            d = 0
            # Merge Images
            while d < CNN_DEPTH:
                x = e_img
                x = torch.from_numpy(x).to(torch.float32).to(DEVICE)
                y, _ = model(x)

                e_img = x.to('cpu').detach().numpy()
                o_img = y.to('cpu').detach().numpy()

                images_ = merge_images(e_img, o_img)

                e_img = images_
                o_img = None
                d += 1

            images = images.transpose(0, 2, 3, 1)
            images_ = images_.transpose(0, 2, 3, 1)

            similarity = skimage.metrics.structural_similarity(images, images_, channel_axis=-1, data_range=images.max() - images.min())
            print(f"Epoch {epoch+1:4d}/{EPOCHS:4d} |\t Step {iter+1:4d}/{num_steps:4d} |\t Loss: {loss.item():.4f} |\t Structural Similarity: {similarity:.4f}")

            loss_e.append(loss.item())
            simi_e.append(similarity)

            iter += 1

        losses.append(np.mean(loss_e))
        simies.append(np.mean(simi_e))

    return losses, simies

def plot_graph(values, metric='Metric'):
    epochs = range(1, len(values)+1)
    plt.plot(epochs, values)
    plt.xlabel('Epochs')
    plt.ylabel(metric)

    plt.show();

losses, simies = train_model()
plot_graph(losses, 'Loss')
plot_graph(simies, 'Structural Similarity')

