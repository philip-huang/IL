import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os.path as osp

def plot_all(all_accs, show=False):
    labels = ['0-1', '2-3', '4-5', '6-7', '8-9']
    fig, ax = plt.subplots()
    epochs = np.arange(len(all_accs))
    all_accs = np.array(all_accs)
    num_tasks = all_accs.shape[1]
    
    for i in range(num_tasks):
        ax.plot(epochs, all_accs[:, i], label='Task {}: {}'.format(i, labels[i]))
    
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('Epochs')
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig('acc_all_splitMNIST.png')
    if show:
        plt.show()

def plot_small(task_accs, show=False):
    number = task_accs.shape[0]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)

    x = np.arange(1, 1 + number)
    axes[0].set_ylabel('Accuracy')
    for i, ax in enumerate(axes):
        ax.plot(x, task_accs[:, i])
        ax.set_xlabel('Tasks')
        ax.set_title('Task {}'.format(i+1))

    fig.savefig('acc_splitMNIST.png')
    if show:
        plt.show()

def test_plotsmall():
    accs = np.zeros((5, 5))
    accs[:, 0] = 1.0
    accs[1:, 1] = 0.8
    accs[2:, 2] = 0.9
    accs[3:, 3] = 0.92
    accs[4:, 4] = 0.93
    plot_small(accs)

def test_plotall():
    accs = []
    accs.append([0.8, 0.5, 0.5, 0.5, 0.5])
    accs.append([0.9, 0.5, 0.5, 0.5, 0.5])
    accs.append([0.9, 0.8, 0.55, 0.5, 0.45])
    plot_all(accs, True)

if __name__ == "__main__":
    test_plotall()
