import numpy as np
import os.path as osp
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import torch
import torchvision.utils

def plot_all(all_accs, config_str, show=False):
    fig, ax = plt.subplots()
    epochs = np.arange(len(all_accs))
    all_accs = np.array(all_accs)
    num_tasks = all_accs.shape[1]
    
    for i in range(num_tasks):
        ax.plot(epochs, all_accs[:, i], label='Task {}'.format(i + 1))
    
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Epochs')
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig('acc_all_{}.png'.format(config_str))
    if show:
        plt.show()

def plot_small(task_accs, config_str, show=False):
    number = task_accs.shape[0]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)

    x = np.arange(1, 1 + number)
    axes[0].set_ylabel('Accuracy')
    for i, ax in enumerate(axes):
        ax.plot(x, task_accs[:, i])
        ax.set_xlabel('Tasks')
        ax.set_title('Task {}'.format(i+1))

    fig.savefig('acc_{}.png'.format(config_str))
    if show:
        plt.show()

def save_generated_ims(tensor, config_str, train_id, test_id):
    fname = "{}_task{}_{}".format(config_str, train_id, test_id)
    torchvision.utils.save_image(tensor, fname, nrows=10)

def test_plotsmall():
    accs = np.zeros((5, 5))
    accs[:, 0] = 1.0
    accs[1:, 1] = 0.8
    accs[2:, 2] = 0.9
    accs[3:, 3] = 0.92
    accs[4:, 4] = 0.93
    plot_small(accs, 'splitMNIST')

def test_plotall():
    accs = []
    accs.append([0.8, 0.5, 0.5, 0.5, 0.5])
    accs.append([0.9, 0.5, 0.5, 0.5, 0.5])
    accs.append([0.9, 0.8, 0.55, 0.5, 0.45])
    plot_all(accs, 'splitMNIST',True)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_device(args):
    use_cuda = (not args.no_cuda and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")

if __name__ == "__main__":
    test_plotall()
