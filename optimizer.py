class SGD:
    def __init__(self, parameters, lr=0.01):
        self.lr = lr
        self.parameters = parameters

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0