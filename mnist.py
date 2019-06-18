from torchvision import datasets

import numpy as np
import torch
import torch.utils.data as data


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
   

if __name__ == "__main__":
    Mnist = MNIST(".data/", [0, 1], download=True)
    print(Mnist.targets.size())