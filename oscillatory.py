# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as funct  # contains useful functions for machine learning
from torch.utils.data import Dataset, DataLoader
from torch.autograd import grad
import uuid  # for generating random uuids
import json # for saving the outputs in json files
import matplotlib.pyplot as plt
from matplotlib import cm
from tqdm import tqdm  # for displaying progress bars in the training

# ray for parameter optimization
# from ray import tune
# from ray.air import Checkpoint, session
# from ray.tune.schedulers import ASHAScheduler

# Define a custom dataset that inherits from the Dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels, weight_labels=[1]):
        self.data = data.requires_grad_()
        self.labels = labels
        self.weight_labels = torch.ones(len(labels)) if len(weight_labels)==1 else weight_labels

    def __len__(self):  # Returns the number of samples in the dataset
        return len(self.labels)

    def __getitem__(self, idx):  #  Loads and returns a sample from the dataset at the given index idx
        return self.data[idx], self.labels[idx], self.weight_labels[idx]


class FeedForwardNet(nn.Module):
    def __init__(self, layer_dimensions):
        super(FeedForwardNet, self).__init__()

        layers = []
        for i in range(len(layer_dimensions) - 1):
            layers.append(nn.Linear(in_features=layer_dimensions[i], out_features=layer_dimensions[i+1]))
            layers.append(nn.ReLU())

        self.linear_relu_stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.linear_relu_stack(x)


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, grad, weight=torch.tensor(1)):
        """

        :param pred: f(model outputs)
        :param target: f(training data)
        :param grad: gradient of the model output 
        :return:
        """
        # Calculate the (quadratic) loss
        loss = (target + weight * pred * torch.abs(grad)) ** 2
        return 0.25 * loss.mean()


class OscillatoryFlows:
    def __init__(self, h, device, id, normal_dist=False):
        self.h = h
        self.device = device
        self.id = id  # custom uuid to identify the trained network (esp. for saving)
        self.normal_dist = normal_dist
        self.seed = torch.seed()

    def generate_samples(self, sample_size, scale=3):
        """
        generates 2 uniformly distributed samples in [-scale, scale]
        :param scale: size of the interval which is sampled from
        :return:
        """
        if self.normal_dist:
            x = torch.randn(sample_size, device=self.device)  # generate normal distributed function
        else:
            x = (torch.rand(sample_size, device=self.device) - 0.5) * 2 * scale  # generate uniform distribution
        return x

    def compute_weight(self,
                       x: torch.Tensor,
                       inverse=False):
        """
        exponential part of the function
        :param x:
        :return:
        """
        weight = 0.5 * x**2 if inverse else torch.exp(-0.5 * x ** 2)
        return weight

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
        return (2 * torch.pi)**-0.5 * torch.flatten(self.compute_weight(x) * self.compute_oscil(x, h))


    def compute_integral_analytic(self,
            h: float):
        """
        Compute the analytical intergal value of f: I = e^(-h^2 / 2)
        :param h:
        :return:
        """
        return np.exp(-h**2 / 2)


    def training(self,
            train_loader,
            model,
            optimizer,
            loss_criterion):

        model.train()  # set model to training mode (unnecessary here, but good practice)
        for i, (inputs, labels, weight_labels) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()  # Reset the gradients from the previous iteration
            # with torch.autocast(device_type=self.device):
            outputs = model(inputs[:,None])
            grads, = grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)
            if self.normal_dist:
                pred = self.compute_oscil(outputs, self.h)
                weight = torch.exp(0.5 * inputs**2 - 0.5 * outputs**2) #* self.compute_weight(outputs)
                loss = loss_criterion(pred, labels, grads, weight=weight)
            else:
                pred = self.f(outputs, self.h)
                loss = loss_criterion(pred, labels, grads)


            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # l1_norm = sum(p.abs().sum() for p in model.parameters())

            loss.backward()  # Calculate the gradient with respect to each parameter
            optimizer.step()  # Adjust the network parameters using the gradients calculated by .backward()
        print(f"training loss: {loss.detach():>7f}")
        return model, loss.cpu().detach().numpy()


    def testing(self, test_loader, model, loss_criterion, scale):
        model.eval()  # Set the model to evaluation mode (unnecessary here, but good practice)
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        test_loss, integral_estimate = 0, 0

        # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
        # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
        # with torch.no_grad():
        for inputs, labels, weight_labels in tqdm(test_loader):
            # inputs.requires_grad = True  # Set requires_grad property of the tensors, to use autograd on them later
            outputs = model(inputs[:, None])
            grads, = grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)
            if self.normal_dist:
                pred = self.compute_oscil(outputs, self.h)
                weight = torch.exp(0.5 * inputs ** 2 - 0.5 * outputs ** 2)
                test_loss += loss_criterion(pred, labels, grads, weight=weight)
                prediction = (8 * torch.pi)**-0.5 * (self.f(inputs, self.h) +
                                        torch.flatten(self.f(outputs.detach(), self.h)) * torch.abs(grads))
            else:
                pred = self.f(outputs, self.h)
                test_loss += loss_criterion(pred, labels, grads).item()
                # x_test_sort = test_loader[0][0].sort()[0]
                prediction = 0.5 * (self.f(inputs.sort()[0], self.h) +
                                torch.flatten(self.f(outputs.sort()[0].detach(), self.h)) * torch.abs(grads.sort()[0]))
                integral_estimate += 2 * scale * torch.mean(prediction.detach().cpu())

        integral_estimate /= num_batches
        test_loss /= num_batches
        print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
        return float(test_loss), prediction, float(integral_estimate)

    def save_model(self,
           path: str,
           loss: list,
           model,
           learning_rate: float,
           dimensions: list,
           plot_results: list,
           test_size: int,
           samples: list,
           epochs: int,
           time: str,
           integral_estimates: list,
           id=None) -> dict:
        """

        :param samples: original sample distribution in range (-scale, scale)
        :param path: relative path to the data folder
        :param loss:
        :param model:
        :param learning_rate:
        :param dimensions: dimension vector of the layers (including input and output layer)
        :param plot_results: array with some intermediate function samples.
            The first entry is the initial sample distribution
        :param scale: length of the sampled domain
        :return:
        """
        if id == None:
            id = self.id

        torch.save(model, path + '/' + id)
        ana_int = self.compute_integral_analytic(self.h)
        calc_int = float(integral_estimates[-1])
        org_int = float(integral_estimates[0])
        save_dict = {
            'id': id,
            'loss': loss[-1],
            'integral result': round(calc_int, 6),
            'original integral approx': round(org_int, 6),
            'relative error': round((calc_int - ana_int) / ana_int, 2),
            'learning rate': learning_rate,
            # 'number of layers': len(dimensions) - 2,
            'layer dimensions': dimensions,
            'number of (test) samples': len(plot_results[0]),
            'distributing': "normal" if self.normal_dist else "uniform",
            'epochs': epochs,
            'time': time,
            'seed': str(self.seed),

            'integral history': list(integral_estimates),
            'loss history': loss,
            'learned graph': list(plot_results),
            'samples': list(samples)}
        with open(path + '/' + id + '.json', 'w', encoding='utf-8') as file:
            json.dump(save_dict, file)
        return save_dict

    def plot_results(self, path, id=None):
        if id == None:
            id = self.id

        # load data from json file
        with open(path + '/' + id + '.json', 'r', encoding='utf-8') as file:
            data_dict = json.load(file)

        cmap = cm.get_cmap('inferno', int(round(len(data_dict['learned graph'])*1.3)))
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        ax1.plot(data_dict['loss history'])
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('training loss')

        # ax2.plot(data_dict[0], self.f(data_dict[0], self.h), label=str(i))
        for i, graph in enumerate(data_dict['learned graph']):
            ax2.plot(data_dict['samples'], graph, label=f"epoch {i * 10}", c=cmap(i))
        # ax2.legend()

        ax3.plot(data_dict['integral history'])
        ax3.hlines(self.compute_integral_analytic(self.h), 0, len(data_dict['integral history']), color='grey')

        return fig