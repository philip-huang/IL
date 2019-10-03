import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
np.random.seed(1)


# Optimal Function
# -0.5x^2 + pi/2 * x + 1 - pi^2/8

mode = "online" # online
a = np.random.randn(3) * 0.1
lr = 0.01
momentum = 0.9
iterations = 10000 # NORAMLLIY 20000 FOR GRAIDNET DESCENT
batch_size = 32
damping = 0.001
si_c = 1
rls_lamda = 0.99999
omega2 = 0.5
num_of_mc_iters = 4
bgd_eta = 1
bgd_std_init = 0.05

def f(x):
    return np.power(x, 2) * a[0] + x * a[1] + a[2]

def y(x, omega=1):
    return np.sin(omega * x)

def loss_f(pred, gt):
    return 0.5 * np.mean(np.power(pred - gt, 2))

def task1_batch(batch_size):
    if mode == "continual":
        x = np.random.uniform(low=0.0, high= np.pi/2, size=batch_size)
        gt = y(x)
    elif mode == "online":
        x = np.random.uniform(low=0.0, high=np.pi, size=batch_size)
        gt = y(x, omega=1)
    return x, gt

def task2_batch(batch_size):
    if mode == "continual":
        x = np.random.uniform(low=np.pi/2, high= np.pi, size=batch_size)
        gt = y(x)
    elif mode == "online":
        x = np.random.uniform(low=0.0, high=np.pi, size=batch_size)
        gt = y(x, omega=omega2)
    return x, gt

def loss_true(low, high, num=5000, omega=1):
    x = np.linspace(low, high, num=num)
    return 0.5 * np.mean(np.power(y(x, omega) - f(x), 2))

def expected_loss(param, low, high, omega=1, d=1):
    def antideriv(x):
        result = 0
        a = param[0]
        b = param[1]
        c = param[2]
        w = omega
        result -= d * (d * w * np.cos(w * x) + 8 * a * x + 4 * b) * np.sin(w * x) / (2 * w * w)
        result += 2 * d * (w * w * (x * (a * x + b) + c) - 2 * a) * np.cos(w * x) / (w * w * w)
        result += (6 * a * a * np.power(x, 5) + 5 * (3 * a * b * np.power(x, 4) \
            + 2 * (2 * a * c + b * b) * np.power(x, 3) + 3 * (d * d + 2 * c * c) * x)) / 30
        result += b * c * x * x
        # result += 15 * (np.cos(x) + 8 * a * x + 4 * b) * np.sin(x)
        # result -= 60 * (x * (a * x + b ) + c - 2 * a) * np.cos(x)
        # result -= 6 * a * a * np.power(x, 5)
        # result -= 15 * a * b * np.power(x, 4)
        # result -= 10 * (2 * a * c + b * b) * np.power(x, 3)
        # result -= 30 * b * c * x * x
        # result -= 15 * (2 * c * c + 1) * x
        # result = result / (-30)
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
    def add(title, subplot, history, low=0, high=np.pi, omega=1):
        ax = fig.add_subplot(subplot, projection='3d')
        history = np.array(history)
        loss_true = [expected_loss(p, low, high, omega=omega) for p in history]
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

    if mode == "continual":
        add("task1: task1_loss", 231, history_1, low=0, high=np.pi/2)
        add("task1: task2_loss", 232, history_1, low=np.pi/2, high=np.pi)
        add("task1: true_loss", 233, history_1, low=0, high=np.pi)
        add("task2: task2_loss", 234, history_2, low=np.pi/2, high=np.pi)
        add("task2: task1_loss", 235, history_2, low=0, high=np.pi/2)
        add("task2: true_loss", 236, history_2, low=0, high=np.pi)
    else:
        add("task1: task1_loss", 221, history_1, omega=1)
        add("task1: task2_loss", 222, history_1, omega=omega2)
        add("task2: task2_loss", 223, history_2, omega=omega2)
        add("task2: task1_loss", 224, history_2, omega=1)
    plt.show()


def plot(low, high, omega=1):
    x = np.arange(low, high, 0.01)
    pred = f(x)
    gt = y(x, omega)
    plt.plot(x, gt, label='ground truth')
    plt.plot(x, pred, label="prediction")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

def exp_sgd():
    global a
    #loss_surface()

    v = 0
    history_1 = []
    for i in range(iterations):
        x, gt = task1_batch(batch_size) 
        pred = f(x)
        train_loss = loss_f(pred, gt)
        grad_a = np.mean((pred -gt) * np.power(x, 2))
        grad_b = np.mean((pred -gt) * x)
        grad_c = np.mean(pred - gt)
        grad = np.array([grad_a, grad_b, grad_c])
        
        #
        v = momentum * v + (1-momentum) * grad
        a = a - lr * v
        history_1.append(a)

    print(train_loss)
    print(a)
    plot(0, np.pi, omega=1)

    v = 0
    history_2 = []
    for i in range(iterations):
        x, gt = task2_batch(batch_size)
        pred = f(x)
        train_loss = loss_f(pred, gt)
        grad_a = np.mean((pred -gt) * np.power(x, 2))
        grad_b = np.mean((pred -gt) * x)
        grad_c = np.mean(pred - gt)
        grad = np.array([grad_a, grad_b, grad_c])
        
        #
        v = momentum * v + (1-momentum) * grad
        a = a - lr * v
        history_2.append(a)

    print(train_loss)
    print(a)
    omega=1 if mode=="continual" else omega2
    plot(0, np.pi, omega=omega)
    plot_train(history_1, history_2)

def exp_si():
    global a

    w_task1 = np.zeros(3)
    a_begin = a
    v = 0
    history_1 = []
    for i in range(iterations):
        x, gt = task1_batch(batch_size)
        pred = f(x)
        train_loss = loss_f(pred, gt)

        # Grad
        grad_a = np.mean((pred -gt) * np.power(x, 2))
        grad_b = np.mean((pred -gt) * x)
        grad_c = np.mean(pred - gt)
        grad = np.array([grad_a, grad_b, grad_c])

        # Update
        v = momentum * v + (1 - momentum) * grad
        a_delta = lr * v
        a = a - a_delta
        history_1.append(a)
        w_task1 += grad * a_delta
    
    delta_task1 = a - a_begin
    reg_1 = np.divide(w_task1, np.power(delta_task1, 2) + damping)

    print(train_loss)
    print(a)
    plot(0, np.pi)

    w_task2 = np.zeros(3)
    a_begin = a
    v = 0
    history_2 = []
    for i in range(iterations):
        x, gt = task2_batch(batch_size)
        pred = f(x)
        train_loss = loss_f(pred, gt)

        # Grad from loss
        grad_a = np.mean((pred -gt) * np.power(x, 2))
        grad_b = np.mean((pred -gt) * x)
        grad_c = np.mean(pred - gt)
        grad = np.array([grad_a, grad_b, grad_c])

        # Grad from SI regularizer
        grad += 0.5 * si_c * reg_1 * (a - a_begin)

        # Update
        v = momentum * v + (1 - momentum) * grad
        a_delta = lr * v
        a = a - a_delta
        history_2.append(a)
        w_task2 += grad - a_delta
    
    delta_task2 = a - a_begin
    reg_2 = np.divide(w_task2, np.power(delta_task2, 2) + damping)

    print(train_loss)
    print(a)
    omega=1 if mode=="continual" else omega2
    plot(0, np.pi, omega=omega)
    plot_train(history_1, history_2)

def get_init_cov(low, high, size):
    x = np.random.uniform(low=low, high=high, size=size)
    basis = np.array([np.power(x, 2), x, np.ones_like(x)]) # 3 by N
    P_inv = np.matmul(basis, basis.T)
    P0 = np.linalg.inv(P_inv)
    print(P0)
    return P0

def exp_rls():
    global a

    P0 = get_init_cov(0, np.pi, 32)
    P_prev = P0
    history_1 = []
    for i in range(iterations):
        x, gt = task1_batch(1)
        basis = np.array([np.power(x, 2), x, np.ones_like(x)]) # 3 x 1
        K = P_prev @ basis @ np.linalg.inv(rls_lamda + basis.T @ P_prev @ basis)
        P = (np.eye(3) - K @ basis.T) @ P_prev / rls_lamda
        a = a + K @ (gt - basis.T @ a)
        P_prev = P
        history_1.append(a)

    print(a)
    plot(0, np.pi)

    history_2 = []
    for i in range(iterations):
        x, gt = task2_batch(1)
        basis = np.array([np.power(x, 2), x, np.ones_like(x)]) # 3 x 1
        K = P_prev @ basis @ np.linalg.inv(rls_lamda + basis.T @ P_prev @ basis)
        P = (np.eye(3) - K @ basis.T) @ P_prev / rls_lamda
        a = a + K @ (gt - basis.T @ a)
        history_2.append(a)
        P_prev = P

    print(a)
    omega=1 if mode=="continual" else omega2
    plot(0, np.pi, omega=omega)
    plot_train(history_1, history_2)


def exp_bgd():
    global a
    import torch
    from torch.nn import Parameter
    from bgd_optim.bgd_optimizer import BGD
    w = Parameter(torch.tensor(a, dtype=torch.float32))
    
    def gen():
        yield w
    params = gen()
    optim = BGD(params, bgd_std_init, mean_eta=bgd_eta, mc_iters=num_of_mc_iters)
    loss_fn = torch.nn.functional.mse_loss

    def train_task(datagen, omega):
        history = [w.detach().numpy()]
        for i in range(iterations):
            x, gt = datagen(batch_size)
            x = torch.tensor(x, dtype=torch.float32); gt = torch.tensor(gt, dtype=torch.float32)
            for k in range(0, num_of_mc_iters):
                torch.autograd.set_grad_enabled(True)
                optim.randomize_weights()

                # Forward:
                outputs = w[0] * x**2 + w[1] * x + w[2]
                loss = loss_fn(outputs, gt)
                print(loss)
                print(w)
                # Backprop 
                # Zero the gradient
                optim.zero_grad()
                loss.backward()
                # Accumulate gradients
                optim.aggregate_grads(batch_size=batch_size)
            optim.step()
            history.append(w.detach().numpy())
        print(loss)
        plot(0, np.pi, omega=omega)
        return history
    
    history1 = train_task(task1_batch, omega=1)
    history2 = train_task(task2_batch, omega=omega2 if mode=='online' else 1)
    plot_train(history1, history2)


exp_si()