# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as funct  # contains useful functions for machine learning
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import grad

def compute_weight(
        x: torch.Tensor):
    """
    exponential part of the function
    :param x:
    :return:
    """
    return torch.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def compute_oscil(
        x: torch.Tensor,
        h: float):
    """
    Oscillatory (cosine) part of the function
    :param x:
    :param h: oscillation frequency
    :return:
    """
    return torch.cos(torch.tensor(h) * x)

def f(x: torch.Tensor, h: float):
    return torch.flatten(compute_weight(x) * compute_oscil(x, h))

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
        # self.input = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        # self.hidden_1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim*2)
        # self.hidden_2 = nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim*2)
        # self.hidden_3 = nn.Linear(in_features=hidden_dim*2, out_features=hidden_dim)
        # self.output = nn.Linear(in_features=hidden_dim, out_features=output_dim)

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(in_features=hidden_dim, out_features=output_dim),
        )

    def forward(self, x):
        # x = funct.relu(self.input(x))
        # x = funct.relu(self.hidden_1(x))
        # x = funct.relu(self.hidden_2(x))
        # x = funct.relu(self.hidden_3(x))
        # return self.output(x)
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


def train_loop(train_loader, model, optimizer,loss_criterion):
    #loss_hist = []
    size = len(train_loader.dataset)
    model.train()  # set model to training mode (unnecessary here, but good practice)
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()  # Reset the gradients from the previous iteration
        # inputs.requires_grad = True  # Set requires_grad property of the tensors, to use autograd on them later

        outputs = model(inputs[:,None])
        grads, = grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)
        pred = f(outputs, _h)
        loss = loss_criterion(pred, labels, grads)

        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # l1_norm = sum(p.abs().sum() for p in model.parameters())

        loss.backward()  # Calculate the gradient with respect to each parameter
        optimizer.step()  # Adjust the network parameters using the gradients calculated by .backward()

        #loss_hist.append(loss.detach().numpy())

        #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        if i % 100 == 0:
          current = (i + 1) * len(inputs)
          print(f"loss: {loss.item():>7f}  [{current:>5d}/{size:>5d}]")
    return loss.detach().numpy()

def test_loop(test_loader, model, loss_criterion):
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
        pred = f(outputs, _h)

        test_loss += loss_criterion(pred, labels, grads).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss

_h = 4
scale = 3

data_size = 10 ** 5  # Replace with the desired number of data
data = (torch.rand(data_size) -0.5) * 2 * scale  # Generate uniformly distributed data scaled to [-scale, scale]

# Split data into a training and a test data set
train_size = int(data_size - data_size * 0.1)
x_train = data[:train_size]
x_test = data[train_size:]

# Define your custom dataset using the data and labels
train_dataset = CustomDataset(x_train, f(x_train, _h))
test_dataset = CustomDataset(x_test, f(x_test, _h))

# Define the batch size
batch_size = 10**3

# Create the train_loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=int(batch_size * 0.1), shuffle=True)

# Run the code on GPU or MPS, if possible, otherwise us CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



input_dimension = 1
hidden_dimension = 128
output_dimension = 1
num_epochs = 10
learning_rate = 0.02

model = FeedForwardNet(input_dimension, hidden_dimension, output_dimension).to(device)
# print(model)
loss_criterion = CustomLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Use Stochastic Gradient descend to train

test_hist = []
train_hist = []
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_hist.append(train_loop(train_loader, model, optimizer, loss_criterion))
    test_hist.append(test_loop(test_loader, model, loss_criterion))
print("Done!")




plt.plot(np.array(train_hist), label="training loss", color='green')
plt.plot(np.array(test_hist), label="testing loss", color='b')
plt.xlabel('step')
plt.ylabel('loss')
plt.legend()
plt.show()

x_test_sort = x_test[:batch_size].sort()[0]  # returns sorted samples (.sort() returns a tuple (values, indices))
# inputs = test_loader[0][0]
outputs = model(x_test_sort[:,None])
grads, = grad(outputs, x_test_sort, grad_outputs=torch.ones_like(outputs), create_graph=True)
prediction = 0.5 *  (f(x_test_sort, _h) + torch.flatten(f(outputs.detach(), _h)) * torch.abs(grads))

print(2 * scale * torch.mean(f(model(x_test_sort[:,None]), _h)))
print(2 * scale * torch.mean(model(x_test_sort[:,None])))
print(2 * scale * torch.mean(prediction))
print(2 * scale * torch.mean(f(x_test_sort, _h)))

x_test_sort.detach_()
plt.plot(x_test_sort, f(x_test_sort, _h), label="f(x)", color='b')
plt.plot(x_test_sort, grads.detach(), label="g'(x)")
plt.plot(x_test_sort, f(model(outputs).detach(), _h), label="f(g(x))", color='r')
plt.plot(x_test_sort, model(x_test_sort[:,None]).detach(), label="g(x)", color='orange')
plt.plot(x_test_sort, prediction.detach(), label="1/2(f(x)+f(g(x))*|g'(x)|)")
plt.legend()
plt.grid()
plt.show()


# plt.plot(x_test_sort, f(model(x_test_sort[:,None]).detach(), _h), color='r')
# plt.show()
