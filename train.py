import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import mnist
import argparse
import os

from torchvision import transforms
from model.baseline import *

def train(args, model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader, loss_fn):
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
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        acc * 100.))

    return test_loss, acc

def get_loader(numbers, args, use_cuda, train):
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trans = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    batch_size = args.batch_size if train else args.test_batch_size
    loader = torch.utils.data.DataLoader(
        mnist.MNIST('data', numbers, train=train, download=True, transform=trans),
        batch_size=batch_size, shuffle=True, **kwargs)

    return loader

def get_test_loaders(task_id, args, use_cuda):
    loaders = []
    for num in range(task_id + 1):
        loader = get_loader([num * 2, num * 2 + 1], args, use_cuda, False)
        loaders.append(loader)
    
    return loaders

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 2)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, default='vcl',
                        help="model type (dnn or cnn or vcl)")
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize Model
    model_dict = {"dnn": Dnn, "cnn": ConvNet, "vcl": MFVI_DNN}

    Model = model_dict[args.model]
    model = Model().to(device)
    model_name = 'mnist_{}.pt'.format(args.model)

    # Loss Function and Optimizer
    loss_dict = {"dnn": F.nll_loss, "cnn": F.nll_loss, "vcl": VCL_loss(model)}
    loss_fn = loss_dict[args.model]
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Initialize Training Stuff
    numbers_list = np.split(np.arange(10), 5)
    task_accs = np.zeros((5, 5))
    dir_list = ['0-1', '2-3', '4-5', '6-7', '8-9']
    best_model = model.state_dict()

    # train all tasks incrementally
    for i, numbers in enumerate(numbers_list):
        print ("===========TASK {}: {}=============".format(i + 1, dir_list[i]))
        train_loader = get_loader(numbers, args, use_cuda, True)
        test_loaders = get_test_loaders(i, args, use_cuda)

        # Restore from before
        if i >= 1:
            model = Model(old_model = model)
            path = os.path.join(os.getcwd(), 'ckpts', dir_list[i-1], model_name)
            model.load_state_dict(torch.load(path), strict=False)
            print ("Restored from {}".format(path))

        for epoch in range(1, args.epochs + 1):
            accs = np.zeros(5)
            train(args, model, device, train_loader, optimizer, epoch, loss_fn)
            # Evaluate all tasks
            for j, loader in enumerate(test_loaders):
                _, accs[j] = test(args, model, device, loader, loss_fn)
            if accs[i] > task_accs[i, i]:
                task_accs[i, :] = accs
                best_model = model.state_dict()

        # save model
        dirpath = os.path.join(os.getcwd(),'ckpts', dir_list[i])
        if not (os.path.exists(dirpath)):
            os.mkdir(dirpath)
        path = os.path.join(dirpath, model_name)
        torch.save(best_model, path)

if __name__ == '__main__':
    main()
