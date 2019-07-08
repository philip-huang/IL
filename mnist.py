from torchvision import datasets

import numpy as np
import torch
import torch.utils.data as data

def getVal(trainval):
    val_size = 1000
    train_size = len(trainval) - val_size
    val_indices = range(train_size, train_size + val_size)
    val = data.Subset(trainval, val_indices)
    return val

def getTrain(trainval):
    val_size = 1000
    train_size = len(trainval) - val_size
    train_indices = range(train_size)
    train = data.Subset(trainval, train_indices)
    return train

class MNIST(datasets.MNIST):

    def __init__(self, root, numbers, **kwargs):
        super(MNIST, self).__init__(root, **kwargs)
        self.index = []
        self.numbers = numbers
        self.select()

    def select(self):
        if not hasattr(self.numbers, "__len__"):
            self.numbers = [self.numbers]
        index = []
        for i, num in enumerate(self.targets):
            if num.item() in self.numbers:
                index.append(i)
        self.targets = self.targets[index]
        self.data = self.data[index]

def test0():
    Mnist = MNIST("data", [0, 1], download=True)
    print(Mnist.targets.size())

def test1():
    Mnist = MNIST("data", [0, 1], download=True)
    train_dataset = getTrain(Mnist)
    val_dataset = getVal(Mnist)
    print('train size:', len(train_dataset))
    print('val size:', len(val_dataset))
    im = val_dataset[0][0]
    im.show()

if __name__ == "__main__":
    test0()
    test1()