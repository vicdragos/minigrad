import random
from engine import Value


class Neuron:
    def __init__(self, nin, act):
        # The bias and weights are randomly assigned a value between -1 and 1 when created
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.act = act

    def __call__(self, x):
        z = sum((wt * x for wt, x in zip(self.w, x)), self.b) # pre-activation
        if self.act == 'tanh':
            return z.tanh()
        elif self.act == 'relu' :
            return z.relu()
        else:
            return z

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout, act):
        self.neurons = [Neuron(nin, act) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]


class MLP:
    def __init__(self, nin, nouts, act = 'tanh'):  # nouts is a list containing the no. of outputs for each layer
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1], act if i != len(nouts)-1 else 'linear') for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)  # the input of the next layer is the output of the current layer
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

def sse_loss(ys, ypred):
    return sum((y1 - y2)**2 for y1, y2 in zip(ypred, ys))

def mse_loss(ys, ypred):
    return sum((y1 - y2)**2 for y1, y2 in zip(ypred, ys)) / len(ys)


