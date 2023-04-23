import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

import os
import pandas as pd
import numpy as np
from torchvision.io import read_image

import random

class CustomNormalDataset(Dataset):
    def __init__(self, mean=0.0, stdev=1.0, save_dist=True, size=10000):
        self.mean = mean
        self.stdev = stdev
        self.size = size
        self.save_dist = save_dist
        if (save_dist):
            self.dist = self.generate_distribution()

    def __len__(self):
        return self.size
    
    def generate_distribution(self):
        return np.random.default_rng().normal(
            loc=self.mean,        # The mean of the distribution
            scale=self.stdev,      # The standard deviation 
            size=self.size       # The size or shape of your array
        ).astype(np.float32)

    def __getitem__(self, idx):
        if (self.save_dist):
            return self.dist[idx], 0
        else:
            return self.generate_distribution()[idx], 0
    
class CustomPoissonDataset(Dataset):
    def __init__(self, lam=5, size=10000, save_dist=True):
        self.lam = lam
        self.size = size
        self.save_dist = save_dist
        if (save_dist):
            self.dist = self.generate_distribution()

    def generate_distribution(self):
        return np.random.default_rng().poisson(
            lam=self.lam,
            size=self.size
        ).astype(np.float32)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if (self.save_dist):
            return self.dist[idx], 1
        else:
            return self.generate_distribution()[idx], 1
        
class MixedNormalPoissonDataset(Dataset):
    def __init__(self, mean=4.0, stdev=1.0, lam=5, size=10000, sample_size = 10, save_dist=True, noisy_poisson=True):
        self.mean = mean
        self.stdev = stdev
        self.lam = lam
        self.size = size
        self.save_dist = save_dist
        self.sample_size = sample_size
        self.noisy_poisson = noisy_poisson
        if (save_dist):
            self.dist = []
            self.dist.append(self.generate_normal_distribution())
            self.dist.append(self.generate_poisson_distribution())

    def generate_poisson_distribution(self):
        if self.noisy_poisson:
            noise = np.random.normal(0,1,self.size);
        else:
            noise = 0
        return (noise + np.random.default_rng().poisson(
            lam=self.lam,
            size=self.size,
        )).astype(np.float32)
        
    def generate_normal_distribution(self):
        return np.random.default_rng().normal(
            loc=self.mean,        # The mean of the distribution
            scale=self.stdev,      # The standard deviation 
            size=self.size,       # The size or shape of your array
        ).astype(np.float32)
    
    def return_random_indices(self, idx):
        # this is DEFINITELY super inefficient but hey.
        # pdist_size = self.size/100 if self.size % 100 == 0 else 1
        # We're sorta bullshitting it, but we just want to make sure we generate numbers within our range.
        # So we take 200 random values from a distribution then add the current index to it to ensure we're sampling from a gaussian around that point
        # If we're too far past the index, we ... well, we'll need to do something, but for now we just REALLY bullshit it.
        size = 200
        if (idx + size >= self.size):
            size = self.size-idx
        # p = np.random.default_rng().normal(
        #     loc=mean,        # The mean of the distribution
        #     scale=self.stdev,      # The standard deviation 
        #     size=self.size       # The size or shape of your array
        #     )
        # psum = p.sum()
        # # p /= np.linalg.norm(p)
        # p /= psum
        # return np.random.choice(self.size, self.sample_size, replace=True, p=p)
        # Pull a bunch of random numbers, add the mean to each to recenter around that point.
        # print(size, idx, self.size)
        return np.random.choice(size, self.sample_size, replace=True)+idx
    
    def __len__(self):
        return self.size # this is the size of any distribution you're sampling from, NOT the total number of distributions*size!

    def __getitem__(self, idx):
        distIndex = random.getrandbits(1)
        index = self.return_random_indices(idx)
        if (self.save_dist):
            return self.dist[distIndex][index], distIndex
        else:
            if distIndex == 0:
                return self.generate_normal_distribution()[index], distIndex
            else:
                return self.generate_poisson_distribution()[index], distIndex
            

train_dataloader = DataLoader(MixedNormalPoissonDataset(size=1048576), batch_size=64, shuffle=False)
test_dataloader = DataLoader(MixedNormalPoissonDataset(size=1024), batch_size=64, shuffle=False)

# this gets our features and labels
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Mine
# Shape of X [N, C, H, W]: torch.Size([64, 10])
# Shape of y: torch.Size([64]) torch.bool

# Original:
# Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
# Shape of y: torch.Size([64]) torch.int64

# Check and print if we're using CUDA.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.ReLU(),
            nn.Linear(10, 10)
        )

    def forward(self, x):
        # print("FORWARD", x.shape)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Loss function and optimizer; standard stuff here.
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

# Now to evaluate and see for ourselves!
model.eval()
evalSet = MixedNormalPoissonDataset(size=40960)
evalItems = np.zeros((80, 10)).astype(np.float32)
# evalItems = []
# shove into an array!
evalLabels = []
for i in range(0, 80):
    ei = evalSet[1000+i]
    evalItems[i, :] = ei[0]
    evalLabels.append(ei[1])

# print("EVALUATION SET: ", evalItems[0][0], evalItems[0][1])
# actual = evalItems[0][1]

# Turn it into a tensor, then shove it into the device.
evalItems = torch.tensor(evalItems)
evalItems = evalItems.to(device)


# print(evalItems.shape)

correctness = 0

with torch.no_grad():
    pred = model(evalItems)
    predicted = pred[0].argmax(0)
    actual = evalLabels[0]
    for i in range(0, 80):
        predicted = pred[i].argmax(0)
        actual = evalLabels[i]
        print(f'Predicted: "{predicted}", Actual: "{actual}", DISTRIBUTION: "{evalItems[i, :]}"')
        # print("DISTRIBUTION: ", evalItems[i, :])
        if (predicted == actual):
            correctness += 1

print(f'FINAL SCORE: {correctness}')
