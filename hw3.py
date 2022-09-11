# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         hw3
# Description:
# Author:       xinzhuang
# Date:         2022/3/13
# Function:
# Version：
# Notice: 图像分类的卷积神经网络
# -------------------------------------------------------------------------------
# !/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, Subset, DataLoader
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

batch_size = 128
train_set = DatasetFolder(r"data/ml2020spring-hw3/training", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(r"data/ml2020spring-hw3/validation", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
test_set = DatasetFolder(r"data/ml2020spring-hw3/testing", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

train_loader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
valid_loader = DataLoader(valid_set, shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_set, shuffle=False, batch_size=batch_size)


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4, 4, 0)
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.flatten(1)
        x = self.fc_layers(x)
        return x


def get_pseudo_labels(dataset, model, threshold=0.65):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    softmax = nn.Softmax(dim=-1)

    for batch in tqdm(data_loader):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)

    model.train()
    return dataset


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Classifier().to(device)
model.device = device

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.00005)
n_epochs = 80
do_semi = False

for epoch in range(n_epochs):
    if do_semi:
        pseudo_set = get_pseudo_labels(valid_loader, model)
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)

    model.train()
    train_loss = []
    train_acc = []

    for batch in tqdm(train_loader):
        imags, labels = batch

        logits = model(imags.to(device))
        loss = criterion(logits, labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        train_loss.append(loss.item())
        train_acc.append(acc)

    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_acc) / len(train_acc)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()
    valid_loss = []
    valid_accs = []

    for batch in tqdm(valid_loader):
        imgs, labels = batch
        with torch.no_grad:
            logits = model(imgs.to(device))
        loss = criterion(logits, labels.to(device))
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_accs = sum(valid_accs) / len(valid_accs)
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    model.eval()
    predictions = []
    for batch in tqdm(test_loader):
        imgs, labels = batch
        with torch.no_grad:
            logits = model(imgs.to(device))
        predictions.extend(logits.argmax(dim=-1).cpu().numpy.tolist())

    with open("output/ml2021sprint-hw3/predict.csv", "w") as f:
        f.write("Id,Category\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")