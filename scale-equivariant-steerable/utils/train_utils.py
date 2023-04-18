'''MIT License. Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja'''
import torch
import torch.nn as nn

import math
import sys
from time import sleep


def train_xent(model, optimizer, loader, device=torch.device('cuda')):
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

        i_bars = math.floor(batch_idx * 21 / (len(loader.dataset)/64))
        sys.stdout.write('\r')
        # Progress Bar
        sys.stdout.write("Epoch Progress: [%-20s] %d%% %d | %d Accuracy: %.2f"
                         % ('=' * i_bars,
                            5 * i_bars,
                            batch_idx,
                            int(len(loader.dataset)/64),
                            accuracy / ((batch_idx + 1) * 64)))
        sys.stdout.flush()
        sleep(0.001)

    print("\n")
    accuracy /= len(loader.dataset)
    return accuracy


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
