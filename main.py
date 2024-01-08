import numpy as np
import time
import torch
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.autograd import grad
import uuid  # for generating random uuids

# ray for parameter optimization
# from ray import tune
# from ray.air import Checkpoint, session
# from ray.tune.schedulers import ASHAScheduler

from oscillatory import OscillatoryFlows
from oscillatory import CustomDataset
from oscillatory import FeedForwardNet
from oscillatory import CustomLoss

# torch.set_float32_matmul_precision("medium")  # decrease precision of matrix multiplications to increase speed

_h = 4
scale = 3

data_size = 10 ** 5  # Replace with the desired number of data
batch_size = 10 ** 3  # Define the batch size
test_size = data_size

test_batch_size = int(batch_size * (test_size / data_size))

dimensions = [1, 30, 1]

num_epochs = 11
learning_rate = 0.1

data_folder = 'data/trash'

normal_dist = False


# Run the code on GPU or MPS, if possible, otherwise us CPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

total_loss_hist = []
total_plot_hist = []
total_integral_hist = []
total_id_list = []

for i in range(2):
    if len(sys.argv) > 1:
        id = sys.argv[1]
    else:
        id = str(uuid.uuid1())
    print(id)

    # torch.seed()
    net = OscillatoryFlows(_h, device, id, normal_dist)

    # Define your custom dataset using the data and labels
    x_train = net.generate_samples(data_size)
    x_test = net.generate_samples(test_size)
    x_test_large = net.generate_samples(10**8)

    if normal_dist:
        train_dataset = CustomDataset(x_train, net.compute_oscil(x_train, _h),
                                      weight_labels=net.compute_weight(x_train, inverse=True))
        test_dataset = CustomDataset(x_test, net.compute_oscil(x_test, _h),
                                     weight_labels=net.compute_weight(x_test, inverse=True))
    else:
        train_dataset = CustomDataset(x_train, net.f(x_train, _h))
        test_dataset = CustomDataset(x_test, net.f(x_test, _h))
        large_test_dataset = CustomDataset(x_test_large, net.f(x_test, _h))

    # Create the train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    large_test_loader = DataLoader(large_test_dataset, batch_size=10**5, shuffle=True)

    model = FeedForwardNet(dimensions).to(device)
    loss_criterion = CustomLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # Use Stochastic Gradient descend to train
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7,
                                                              threshold=0.001, threshold_mode='rel', min_lr=1e-6)

    model, save_dict = net.run(train_loader, test_loader, model, optimizer, loss_criterion, lr_scheduler, num_epochs, scale, data_folder)
    fig = net.plot_results(data_folder)
    plt.show()

    total_id_list.append(id)
    total_integral_hist.append(float(save_dict['integral history'][-1]))
    total_plot_hist.append(list(save_dict['learned graph'][-1]))
    total_loss_hist.append(float(save_dict['loss history'][-1]))

net.save_model(data_folder, list(total_loss_hist), model, learning_rate, dimensions, list(total_plot_hist), test_size,
               save_dict['samples'], num_epochs,
               f"10 models", list(total_integral_hist), 'total train 1')

fig = net.plot_results(data_folder, id='total train 1')
plt.show()
