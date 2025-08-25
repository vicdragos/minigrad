from neural_net import *
from engine import *
from optimizer import *

xs = [(0, 0), (0, 1), (1, 0), (1, 1)]
ys = [0, 1, 1, 0] # expected values

perceptron = MLP(2, [4, 1])
optimizer = SGD(perceptron.parameters(), lr=0.05)

for k in range(3000):
    # forward pass
    ypred = [perceptron(x) for x in xs] # predicted values
    loss = mse_loss(ys, ypred)

    optimizer.zero_grad()

    # backward pass
    loss.backward()

    optimizer.step()

final_res = [1 if perceptron(x).data > 0.5 else 0 for x in xs]
print("Predictions:", final_res)
