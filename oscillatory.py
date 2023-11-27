# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as funct  # contains useful functions for machine learning
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
import uuid  # for generating random uuids
import json # for saving the outputs in json files

# Define a custom dataset that inherits from the Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data.requires_grad_()
        self.labels = labels

    def __len__(self):  # Returns the number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):  #  Loads and returns a sample from the dataset at the given index idx
        return self.data[idx], self.labels[idx]


class FeedForwardNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNet, self).__init__()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def forward(self, x):
        return self.linear_relu_stack(x)


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, grad):
        """

        :param pred: f(model outputs)
        :param target: f(training data)
        :param grad: gradient of the model output 
        :return:
        """
        # Calculate the (quadratic) loss
        loss = (target + pred * torch.abs(grad)) ** 2
        return 0.25 * loss.mean()


class OscillatoryFlows:
    def __init__(self, h, data_size, device):
        self.h = h
        self.data_size = data_size
        self.device = device

    def generate_samples(self, batch_size=0, scale=3):
        if batch_size == 0:
            batch_size = 0.01 * self.data_size
        data = (torch.rand(self.data_size) - 0.5) * 2 * scale
        train_size = int(self.data_size - self.data_size * 0.1)
        x_train = data[:train_size].to(self.device)
        x_test = data[train_size:].to(self.device)
        return x_train, x_test

    def compute_weight(self,
            x: torch.Tensor):
        """
        exponential part of the function
        :param x:
        :return:
        """
        return torch.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

    def compute_oscil(self,
            x: torch.Tensor,
            h: float):
        """
        Oscillatory (cosine) part of the function
        :param x:
        :param h: oscillation frequency
        :return:
        """
        return torch.cos(torch.tensor(h) * x)

    def f(self,
            x: torch.Tensor,
            h: float):
        return torch.flatten(self.compute_weight(x) * self.compute_oscil(x, h))

    def training(self,
            train_loader,
            model,
            optimizer,
            loss_criterion):

        model.train()  # set model to training mode (unnecessary here, but good practice)
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()  # Reset the gradients from the previous iteration
            outputs = model(inputs[:,None])
            grads, = grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)
            pred = self.f(outputs, self.h)
            loss = loss_criterion(pred, labels, grads)

            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # l1_norm = sum(p.abs().sum() for p in model.parameters())

            loss.backward()  # Calculate the gradient with respect to each parameter
            optimizer.step()  # Adjust the network parameters using the gradients calculated by .backward()
        print(f"training loss: {loss.item():>7f}")
        return model, loss.cpu().detach().numpy()

    def testing(self, test_loader, model, loss_criterion):
        model.eval()  # Set the model to evaluation mode (unnecessary here, but good practice)
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        test_loss, correct = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        # with torch.no_grad():
        for inputs, labels in test_loader:
            # inputs.requires_grad = True  # Set requires_grad property of the tensors, to use autograd on them later
            outputs = model(inputs[:, None])
            grads, = grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)
            pred = self.f(outputs, self.h)

            test_loss += loss_criterion(pred, labels, grads).item()

        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        return test_loss
