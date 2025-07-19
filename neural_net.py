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


