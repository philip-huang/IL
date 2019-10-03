import torch
import numpy as np
import argparse

import vcl
import utils

def get_train_args():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test-batch-size', type=int, default=None, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--early-stop-after', type=int, default=3, metavar='N',
                        help='Early Stopping After # epochs (default: 3')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--model', type=str, default='vcl',
                        help="model type (dnn or cnn or vcl)")
    parser.add_argument('--dataset', type=str, default='fashionMNIST',
                        help="dataset type (splitMNIST or permutedMNIST or fashionMNIST")
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--coreset-size', type=int, default=40, metavar='N',
                        help='size of coreset (default 40)')
    parser.add_argument('--coreset-epochs', type=int, default=40, metavar='N',
                        help='number of epochs to train for coreset (default 5)')
    args = parser.parse_args()
    return args


def main():
    args = get_train_args()
    utils.set_seed(args.seed)
    device = utils.get_device(args)

    # Initialize Dataset for each tasks
    assert args.dataset in ["splitMNIST", "permutedMNIST", 'fashionMNIST']
    if args.dataset == "splitMNIST" or args.dataset == 'fashionMNIST':
        labels_list = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    elif args.dataset == 'permutedMNIST':
        labels_list = [list(range(10))] * 5
    
    # Run VCL
    task_final_accs, all_accs = vcl.run_vcl(args, device, labels_list)
    
    # Plots
    config_str = '_{}_coreset_{}'.format(args.dataset, args.coreset_size)
    utils.plot_small(task_final_accs, config_str)
    utils.plot_all(all_accs, config_str)
    
    avg_acc = np.mean(all_accs[-1])
    print ("Final Average Accuracy: {}".format(avg_acc))

if __name__ == '__main__':
    main()