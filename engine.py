from math import log, exp


class Value:

    def __init__(self, data, _children=()):
        self.data = data
        self._prev = set(_children)

        self.grad = 0
        self._backward = lambda: None

    def __repr__(self):
        return f"data = {self.data}"

    # Operation definitions
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        out = Value(-self.data, (self,))

        def _backward():
            self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,))

        def _backward():
            self.grad += other * self.data ** (other - 1) * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def exp(self):
        out = Value(exp(self.data), (self,))

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(log(self.data), (self,))

        def _backward():
            self.grad += (self.data ** -1) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = (self * 2).exp()
        out = (x - 1) / (x + 1)
        return out

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return self - other

    # backward method
    def backward(self):
        nodes = []  # Will contain each node in topological order
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for u in v._prev: build(u)
                nodes.append(v)

        build(self)

        for v in nodes:
            v.grad = 0.0

        self.grad = 1.0
        for v in reversed(nodes):
            v._backward()
