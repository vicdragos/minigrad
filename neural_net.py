import random
from engine import Value

class Neuron:
    def __init__(self, nin):
        #The bias and weights are randomly assigned a value between -1 and 1 when created
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # = sum( w * x) + b
        return sum( (wt * x for wt, x in zip(self.w, x)), self.b)

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1  else out

class MLP:
    def __init__(self, nin, nouts): # nouts is a list containing the no. of outputs for each layer
        sizes = [nin] + nouts
        self.layers = [ Layer(sizes[i], sizes[i+1]) for i in range(len(sizes))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x) # the input of the next layer is the ouput of the current layer
        return x




