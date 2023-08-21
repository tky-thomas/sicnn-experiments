'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn
import random

import math
import sys
from time import sleep

from matplotlib import pyplot as plt


def train_xent(model, optimizer, loader, device=torch.device('cuda'), batch_size=32):
    model.train()
    accuracy = 0
    criterion = nn.CrossEntropyLoss().to(device)
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        pred = output.argmax(1, keepdim=True)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        accuracy += pred.eq(target.view_as(pred)).sum().item()

        i_bars = round(batch_idx * 20 / (len(loader.dataset)/batch_size))
        sys.stdout.write('\r')
        # Progress Bar
        sys.stdout.write("Epoch Progress: [%-20s] %d%% %d | %d Accuracy: %.2f"
                         % ('=' * i_bars,
                            5 * i_bars,
                            batch_idx,
                            int(len(loader.dataset)/batch_size),
                            accuracy / ((batch_idx + 1) * batch_size)))
        sys.stdout.flush()
        sleep(0.001)
    
    print("\n")
    accuracy /= len(loader.dataset)
    return accuracy


def train_xent_segmentation(model, optimizer, loader, device=torch.device('cuda'), batch_size=8):
    model.train()
    total_loss = 0
    criterion = nn.BCEWithLogitsLoss().to(device)

    for batch_idx, (data, target, _) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss

        i_bars = round(batch_idx * 20 / (len(loader.dataset) / batch_size))
        sys.stdout.write('\r')
        # Progress Bar
        sys.stdout.write("Epoch Progress: [%-20s] %d%% %d | %d Avg Loss: %.6f"
                         % ('=' * i_bars,
                            5 * i_bars,
                            batch_idx,
                            int(len(loader.dataset) / batch_size),
                            total_loss / ((batch_idx + 1) * batch_size)))
        sys.stdout.flush()
        sleep(0.001)

    print("\n")
    total_loss /= len(loader.dataset)
    return total_loss


def test_acc(model, loader, device=torch.device('cuda')):
    model.eval()
    accuracy = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(1, keepdim=True)
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    accuracy /= len(loader.dataset)
    return accuracy


def test_acc_segmentation(model, loader, device=torch.device('cuda')):
    model.eval()
    criterion = nn.BCEWithLogitsLoss().to(device)
    total_loss = 0
    with torch.no_grad():
        for data, target, _ in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss

    # Prints out a picture of the dataset and displays it
    image, label, img_path = loader.dataset[random.randint(0, len(loader.dataset))]
    image, label = image.to(device), label.to(device)
    image = torch.unsqueeze(image, 0)
    label = torch.unsqueeze(label, 0)
    prediction = model(image)

    plt.figure()
    plt.imshow(image[0].permute(1, 2, 0).detach().cpu())
    plt.figure()
    plt.imshow(prediction[0].permute(1, 2, 0).detach().cpu())
    plt.figure()
    plt.imshow(label[0].permute(1, 2, 0).detach().cpu())
    plt.show()
    print(img_path)

    total_loss /= len(loader.dataset)
    return total_loss
