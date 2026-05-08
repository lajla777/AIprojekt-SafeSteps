import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#transforMACIJE
train_transform = None
validation_transform = None

#DATASETS
train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
valid_dataset = datasets.ImageFolder(root='data/valid', transform=validation_transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=validation_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#MODEL
model = None #verjetno iz mobilenetv2

#tu se za optimizacijo

#TRAINING AND VALIDATION

#mogoc se grafi da se vidi ucenje oz kk je natancen model, kaka je izguba

#tu dodaj un matrix

#pa se neki najverjetneje bo treba dodat

