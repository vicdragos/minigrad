from neural_net import *
import random

m = 4
b = 5  # parameters of the linear function


def f(x): return m * x + b


# Define the lower and upper bounds for the input range
min_x = 3
max_x = 5

min_y = min(f(min_x), f(max_x))
max_y = max(f(min_x), f(max_x))

# Generate the data within the range
xs = [[random.uniform(min_x, max_x)] for _ in range(4)]
ys = [f(x[0]) for x in xs]

# Normalize the data sets to [-0.5, 0.5] where tanh is quasi-linear

xs_norm = [[(x[0] - min_x) / (max_x - min_x) - 0.5] for x in xs]
ys_norm = [(y - min_y) / (max_y - min_y) - 0.5 for y in ys]

mlp = MLP(1, [1])

h = 0.05
for k in range(1000 +1):
    # forward pass
    ypred_norm = [mlp(x) for x in xs]  # predicted values
    loss = mse_loss(ys_norm, ypred_norm)

    # backward pass
    loss.backward()

    # update
    for p in mlp.parameters():
        p.data += -h * p.grad

    ypred = [(y + 0.5) * (max_y - min_y) + min_y for y in ypred_norm]

    if k % 100 == 0:
        print("\n\n{}th epoch     Loss =  {} \nPredicted: {} \nResult:    {} ".format(k, loss.data, ypred, ys))
