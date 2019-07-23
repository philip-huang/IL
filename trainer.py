import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import argparse
import os

import mnist
import utils

from torchvision import transforms
from model.baseline import *

def train(args, model, device, train_loader, optimizer, epoch, loss_fn, verbose=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if verbose and (batch_idx % args.log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, loss_fn, val=False):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = float(correct) / len(test_loader.dataset)
    print('{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        'Val' if val else 'Test',
        test_loss, correct, len(test_loader.dataset), acc * 100.))

    return test_loss, acc


def get_loader(dataset, args, device, split):
    use_cuda = ("cuda" in device.type)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    if split == "train":
        batch_size = args.batch_size
    elif split == 'test' or split == 'val':
        batch_size = args.test_batch_size
    elif split == 'coreset':
        batch_size = None
    
    if batch_size is None:
        batch_size = len(dataset)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, **kwargs)

    return loader

def get_model(args, device, MLE=False):
    if args.model == "dnn":
        model = Dnn()
    if args.model == "cnn":
        model = ConvNet()
    if args.model == "vcl":
        model = MFVI_DNN(MLE=MLE)
    model_name = 'mnist_{}.pt'.format(args.model)

    return model.to(device), model_name

def get_loss_fn(args, device, model, old_model=None):
    if args.model == "dnn":
        return F.nll_loss
    if args.model == "cnn":
        return F.nll_loss
    if args.model == "vcl":
        return VCL_loss(model, old_model).to(device)

def get_model_path(labels, model_name):
    dirname = "{}-{}".format(labels[0], labels[-1])
    dirpath = os.path.join(os.getcwd(),'ckpts', dirname)
    if not (os.path.exists(dirpath)):
        os.mkdir(dirpath)
    path = os.path.join(dirpath, model_name)

    return path