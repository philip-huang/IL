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

def fit(args, model, device, optimizer, loss_fn, numbers_list, task_id):
    # Dataset Loader
    train_loader = get_loader(numbers_list[task_id], args, device, True)
    # Log Best Accuracy
    best_accs = np.zeros(len(numbers_list))

    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Prepare model for current task
        model.set_range(numbers_list[task_id])
        train(args, model, device, train_loader, optimizer, epoch, loss_fn)
        # Evaluate all tasks
        accs = np.zeros(len(numbers_list))
        for j in range(task_id + 1):
            # Prepare model and dataset for the task j
            loader = get_loader(numbers_list[j], args, device, False)
            model.set_range(numbers_list[j])

            _, accs[j] = test(args, model, device, loader, loss_fn)
            if accs[task_id] > best_accs[task_id]:
                best_accs = accs
                best_state = model.state_dict()
    
    return best_state, best_accs

def get_loader(numbers, args, device, train):
    use_cuda = ("cuda" in device.type)
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    trans = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
    dataset = mnist.MNIST('data', numbers, train=train, download=True, transform=trans)
    batch_size = args.batch_size if train else args.test_batch_size
    if batch_size == None:
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

def get_model_path(numbers, model_name):
    dirname = "{}-{}".format(numbers[0], numbers[-1])
    dirpath = os.path.join(os.getcwd(),'ckpts', dirname)
    if not (os.path.exists(dirpath)):
        os.mkdir(dirpath)
    path = os.path.join(dirpath, model_name)

    return path

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=None, metavar='N',
                        help='input batch size for training (default: 6000)')
    parser.add_argument('--test-batch-size', type=int, default=None, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
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
    np.random.seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize Training Stuff
    numbers_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    task_accs = np.zeros((5, 5))

    # Pretraining
    print ("=============Pretraining ==================")
    model, name = get_model(args, device, MLE=True)
    loss_fn = get_loss_fn(args, device, model)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    fit(args, model, device, optimizer, loss_fn, numbers_list, 0)

    # train all tasks incrementally
    for task_id, numbers in enumerate(numbers_list):
        print ("===========TASK {}: {}=============".format(task_id + 1, numbers))
        # Model
        old_model = model
        model, model_name = get_model(args, device)
        model.load_state_dict(old_model.state_dict())
        # Loss Function and Optimizer
        loss_fn = get_loss_fn(args, device, model, old_model)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        # Fit
        best_state, task_accs[task_id, :]=fit(args, model, device, optimizer, loss_fn, numbers_list, task_id)

        # save model
        path = get_model_path(numbers, model_name)
        torch.save(best_state, path)

    utils.plot(task_accs)
    avg_acc = np.mean(task_accs[-1, :])
    print ("Final Average Accuracy: {}".format(avg_acc))

if __name__ == '__main__':
    main()
