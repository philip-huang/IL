import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import os.path as osp

def plot(task_accs, show=False):
    number = task_accs.shape[0]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=True)

    x = np.arange(1, 1 + number)
    axes[0].set_ylabel('Accuracy')
    for i, ax in enumerate(axes):
        ax.plot(x, task_accs[:, i])
        ax.set_xlabel('Tasks')
        ax.set_title('Task {}'.format(i+1))

    fig.savefig('accuracy_splitMNIST.png')
    if show:
        plt.show()

def test_plot():
    accs = np.zeros((5, 5))
    accs[:, 0] = 1.0
    accs[1:, 1] = 0.8
    accs[2:, 2] = 0.9
    accs[3:, 3] = 0.92
    accs[4:, 4] = 0.93
    plot(accs)

if __name__ == "__main__":
    test_plot()
