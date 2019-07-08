from torchvision import datasets
from torchvision import transforms

import numpy as np
import torch
import torch.utils.data as data

IMAGE_SIZE = 28 * 28
PERMUTATIONS = [None] + [np.random.permutation(IMAGE_SIZE) for _ in range(5)]

def getVal(trainval):
    val_size = len(trainval) // 6
    train_size = len(trainval) - val_size
    val_indices = range(train_size, train_size + val_size)
    val = data.Subset(trainval, val_indices)
    return val

def getTrain(trainval):
    val_size = len(trainval) // 6
    train_size = len(trainval) - val_size
    train_indices = range(train_size)
    train = data.Subset(trainval, train_indices)
    return train

def _permutate_image_pixels(image, permutation):
    '''Permutate the pixels of an image according to [permutation].
    [image]         3D-tensor containing the image
    [permutation]   <ndarray> of pixel-indeces in their new order'''

    if permutation is None:
        return image
    else:
        c, h, w = image.size()
        image = image.view(c, -1)
        image = image[:, permutation]  #--> same permutation for each channel
        image = image.view(c, h, w)
        return image

class PermutedMNIST(datasets.MNIST):
    permutations = [None] + [np.random.permutation(IMAGE_SIZE) for _ in range(100)]

    def __init__(self, root, task_id, **kwargs):
        super(PermutedMNIST, self).__init__(root, **kwargs)
        self.task_id = task_id
        self.permu = PermutedMNIST.permutations[task_id]
        self.permu_transform = transforms.Lambda(lambda x: _permutate_image_pixels(x, self.permu))

        if self.transform is not None:
            self.transform = transforms.Compose([self.transform, self.permu_transform])
        else:
            self.transform = self.permu_transform

class SplitMNIST(datasets.MNIST):
    numbers_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def __init__(self, root, task_id, **kwargs):
        super(SplitMNIST, self).__init__(root, **kwargs)
        self.index = []
        self.task_id = task_id
        self.numbers = SplitMNIST.numbers_list[task_id]
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
    Mnist = SplitMNIST("data", 0, download=True)
    print(Mnist.targets.size())


def test1():
    Mnist = SplitMNIST("data", 0, download=True)
    train_dataset = getTrain(Mnist)
    val_dataset = getVal(Mnist)
    print('train size:', len(train_dataset))
    print('val size:', len(val_dataset))
    im = val_dataset[0][0]
    im.show()

def test2():
    trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    toPIL = transforms.ToPILImage()
    Mnist = PermutedMNIST("data", 1, download=True, transform=trans) 
    im = toPIL(Mnist[0][0])
    im.show()

    Mnist2 = PermutedMNIST("data", 1, download=True, transform=trans, train=False)
    im = toPIL(Mnist[0][0])
    im.show()

if __name__ == "__main__":
    test0()
    test1()
    test2()