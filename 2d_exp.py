import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)


# Optimal Function
# -0.5x^2 + pi/2 * x + 1 - pi^2/8

a = np.random.randn(3) * 0.1
lr = 0.01
momentum = 0.9
iterations = 20000
batch_size = 32

def f(x):
    return np.power(x, 2) * a[0] + x * a[1] + a[2]

def y(x):
    return np.sin(x)

def loss_f(pred, gt):
    return 0.5 * np.mean(np.power(pred - gt, 2))

def loss_true(low, high, num=5000):
    x = np.linspace(low, high, num=num)
    return 0.5 * np.mean(np.power(y(x) - f(x), 2))

def expected_loss(param, low, high):
    def antideriv(x):
        result = 0
        a = param[0]
        b = param[1]
        c = param[2]
        result += 15 * (np.cos(x) + 8 * a * x + 4 * b) * np.sin(x)
        result -= 60 * (x * (a * x + b ) + c - 2 * a) * np.cos(x)
        result -= 6 * a * a * np.power(x, 5)
        result -= 15 * a * b * np.power(x, 4)
        result -= 10 * (2 * a * c + b * b) * np.power(x, 3)
        result -= 30 * b * c * x * x
        result -= 15 * (2 * c * c + 1) * x
        result = result / (-30)
        return result
    
    loss = antideriv(high) - antideriv(low)
    loss /= (high - low) # probability
    loss /= 2 # 
    return loss

def loss_surface():
    fig = plt.figure(figsize=(300, 1000))
    def add(title, subplot, low, high):
        #A, B, C = np.mgrid[-1:1:10j, -1:1:10j, -1:1:10j]
        A, B, C = np.mgrid[-0.75:0.25:10j, -0.25:1.5:10j, -0.5:0.5:10j] # zoomed in version
        x = np.linspace(low, high, num=5000)
        losses = []
        for (a, b, c) in zip(A.flatten(), B.flatten(), C.flatten()):
            # Monte Carlo calculation of Loss
            #pred = np.power(x, 2) * a + x * b + c 
            #gt = np.sin(x)
            #l = 0.5 * np.mean(np.power(pred - gt, 2))
            l_expected = expected_loss([a, b, c], low, high)
            losses.append(l_expected)

        ax = fig.add_subplot(subplot, projection='3d')
        scat = ax.scatter(A, B, C, c=losses)
        fig.colorbar(scat, shrink=0.5, aspect=5)
        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_zlabel("c")
        ax.set_title(title)

        # Min Loss
        index = np.argmin(losses)
        ax.scatter(A.flatten()[index], B.flatten()[index], C.flatten()[index], marker='x', s=100, c='red')
    
    add("task1: 0-pi/2", 131, 0, np.pi/2)
    add("task1: pi/2-pi", 132, np.pi/2, np.pi)
    add("gt: 0-pi", 133, 0, np.pi)
    plt.show()

def plot_train(history_1, history_2):
    fig = plt.figure()
    def add(title, subplot, history, low, high):
        ax = fig.add_subplot(subplot, projection='3d')
        history = np.array(history)
        loss_true = [expected_loss(p, low, high) for p in history]
        scat = ax.scatter(history[:, 0], history[:, 1], history[:, 2], c=loss_true)
        fig.colorbar(scat, shrink=0.5, aspect=5)

        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_zlabel("c")
        ax.set_xlim3d(-0.75, 0.25)
        ax.set_ylim3d(-0.25, 1.5)
        ax.set_zlim3d(-0.5, 0.5)
        ax.set_title(title)


        # Start and End
        ax.scatter(history[0, 0], history[0, 1], history[0, 2], marker='x', s=100)
        ax.scatter(history[-1, 0], history[-1, 1], history[-1, 2], marker='o', s=100)

    add("task1: task1_loss", 221, history_1, 0, np.pi/2)
    add("task1: true_loss", 222, history_1, 0, np.pi)
    add("task2: task2_loss", 223, history_2, np.pi/2, np.pi)
    add("task2: true_loss", 224, history_2, 0, np.pi)
    plt.show()


def plot(low, high):
    x = np.arange(low, high, 0.01)
    pred = f(x)
    gt = y(x)
    plt.plot(x, gt, label='ground truth')
    plt.plot(x, pred, label="prediction")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

loss_surface()

v = 0
history_1 = []
for i in range(iterations):
    x = np.random.uniform(low=0.0, high= np.pi/2, size=batch_size)
    pred = f(x)
    gt = y(x)
    train_loss = loss_f(pred, gt)
    test_loss = loss_true(0, np.pi)
    # 
    grad_a = np.mean((pred -gt) * np.power(x, 2))
    grad_b = np.mean((pred -gt) * x)
    grad_c = np.mean(pred - gt)
    grad = np.array([grad_a, grad_b, grad_c])
    
    #
    v = momentum * v + (1-momentum) * grad
    a = a - lr * v
    history_1.append(a)

print(train_loss, test_loss)
print(a)
plot(0, np.pi)

v = 0
history_2 = []
for i in range(iterations):
    x = np.random.uniform(low=np.pi/2, high= np.pi, size=batch_size)
    pred = f(x)
    gt = y(x)
    train_loss = loss_f(pred, gt)
    test_loss = loss_true(np.pi/2, np.pi)
    # 
    grad_a = np.mean((pred -gt) * np.power(x, 2))
    grad_b = np.mean((pred -gt) * x)
    grad_c = np.mean(pred - gt)
    grad = np.array([grad_a, grad_b, grad_c])
    
    #
    v = momentum * v + (1-momentum) * grad
    a = a - lr * v
    history_2.append(a)

print(train_loss, test_loss)
print(a)
plot(0, np.pi)
plot_train(history_1, history_2)
