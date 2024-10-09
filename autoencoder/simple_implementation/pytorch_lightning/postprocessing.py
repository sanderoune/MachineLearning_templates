import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np    

model_version = 'version_0'
# data
dataset = MNIST('../MNIST/', train=True, download=True, transform=transforms.ToTensor()) # transforms.ToTensor() -> convert images in the range [0,1]
mnist_train, mnist_val = random_split(dataset, [50000, 10000])

# visualize dataset samples
testing_indices = np.random.randint(len(mnist_train), size=3)
img, label = mnist_train[testing_indices[0]]

def plot_fig(x, title = ''):
    plt.figure(1)
    plt.imshow(x.squeeze(), cmap='gray')
    plt.title(title)
    plt.savefig('../results/' + model_version, dpi = 300)
    plt.close()