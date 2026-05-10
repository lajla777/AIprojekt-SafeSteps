import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#transformacije
train_transform = transforms.Compose([
    transforms.RandomResize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

#DATASETS
train_dataset = datasets.ImageFolder(root='data/train', transform=train_transform)
valid_dataset = datasets.ImageFolder(root='data/valid', transform=validation_transform)
test_dataset = datasets.ImageFolder(root='data/test', transform=validation_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = train_dataset.classes

#MODEL
model = models.mobilenet_v2(weight='IMAGENET1K_V1')

#tu se za optimizacijo

#TRAINING AND VALIDATION

#mogoc se grafi da se vidi ucenje oz kk je natancen model, kaka je izguba

#tu dodaj un matrix

#pa se neki najverjetneje bo treba dodat

