import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import grad

from oscillatory import OscillatoryFlows
from oscillatory import CustomDataset
from oscillatory import FeedForwardNet
from oscillatory import CustomLoss

_h = 4
scale = 3

data_size = 10 ** 5  # Replace with the desired number of data
batch_size = 10 ** 3  # Define the batch size

dimensions = [1, 100, 1]

num_epochs = 10
learning_rate = 0.025

# Run the code on GPU or MPS, if possible, otherwise us CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

net = OscillatoryFlows(_h, data_size, device)


# Define your custom dataset using the data and labels
x_train, x_test = net.generate_samples()
train_dataset = CustomDataset(x_train, net.f(x_train, _h))
test_dataset = CustomDataset(x_test, net.f(x_test, _h))
# Create the train_loader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=int(batch_size * 0.1), shuffle=True)


model = FeedForwardNet(dimensions).to(device)
# print(model)
loss_criterion = CustomLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Use Stochastic Gradient descend to train


# fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

x_test_sort = x_test[:batch_size].sort()[0]  # returns sorted samples (.sort() returns a tuple (values, indices))

test_hist = torch.empty(num_epochs).to(device)
train_hist = torch.empty(num_epochs).to(device)
results = []  # array containing all integral estimates
learned_graph =[x_test_sort.detach().cpu().tolist()]  # array containing all trained sample data

# ax2.plot(x_test_sort.detach().cpu(), net.f(x_test_sort, _h).detach().cpu(), label="f(x)")

start_time = time.time()
for t in range(num_epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    model, loss = net.training(train_loader, model, optimizer, loss_criterion)
    train_hist[t] = torch.tensor(loss)
    test_hist[t] = torch.tensor(net.testing(test_loader, model, loss_criterion))
    if t % 10 == 0:
        outputs = model(x_test_sort[:, None])
        grads, = grad(outputs, x_test_sort, grad_outputs=torch.ones_like(outputs), create_graph=True)
        prediction = 0.5 * (net.f(x_test_sort, _h) + torch.flatten(net.f(outputs.detach(), _h)) * torch.abs(grads))
        learned_graph.append(prediction.detach().cpu().numpy().tolist())
        # ax2.plot(x_test_sort.detach().cpu(), prediction.detach().cpu(), label=f"epoch {t}")
        results.append((2 * scale * torch.mean(prediction)).detach().cpu())

print(f"time for training: {time.time() - start_time:.0f}s")

saved = net.save_model('data', test_hist.tolist(), model, learning_rate, dimensions, learned_graph)
print(saved['id'])
fig = net.plot_results('data', saved['id'])
plt.show()

# ax2.legend()
#
# ax1.plot(np.array(train_hist.cpu()), label="training loss", color='green')
# ax1.plot(np.array(test_hist.cpu()), label="testing loss", color='b')
# ax1.set_xlabel('step')
# ax1.set_ylabel('loss')
# ax1.legend()
#
# print(results)
# ax3.plot(np.arange(len(results)) * 10, results)
# ax3.grid()
# plt.show()
#
# x_test_sort = x_test[:batch_size].sort()[0]  # returns sorted samples (.sort() returns a tuple (values, indices))
# # inputs = test_loader[0][0]
# outputs = model(x_test_sort[:,None])
# grads, = grad(outputs, x_test_sort, grad_outputs=torch.ones_like(outputs), create_graph=True)
# prediction = 0.5 * (net.f(x_test_sort, _h) + torch.flatten(net.f(outputs.detach(), _h)) * torch.abs(grads))
#
# print(2 * scale * torch.mean(net.f(model(x_test_sort[:,None]), _h)))
# print(2 * scale * torch.mean(model(x_test_sort[:,None])))
# print(2 * scale * torch.mean(prediction))
# print(2 * scale * torch.mean(net.f(x_test_sort, _h)))
#
# x_test_sort = x_test_sort.cpu().detach()
#
# plt.plot(x_test_sort, net.f(x_test_sort, _h), label="f(x)", color='b')
# plt.plot(x_test_sort, grads.detach().cpu(), label="g'(x)")
# plt.plot(x_test_sort, net.f(model(outputs).detach().cpu(), _h), label="f(g(x))", color='r')
# plt.plot(x_test_sort, model(x_test_sort[:,None].to(device)).cpu().detach(), label="g(x)", color='orange')
# plt.plot(x_test_sort, prediction.cpu().detach(), label="1/2(f(x)+f(g(x))*|g'(x)|)")
# plt.legend()
# plt.grid()
# plt.show()