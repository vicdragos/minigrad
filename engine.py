from math import log, exp

class Value:

    def __init__(self, data, _children = ()):
        self.data = data
        self._prev = set(_children)

        self.grad = 0
        self._backward = lambda : None

    def __repr__(self):
        return f"data = {self.data}"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        self._backward = _backward
        return out

    def __neg__(self):
        out = Value(-self.data, (self,))
        def _backward():
            self.grad += -out.grad
        self._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other))
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        self._backward = _backward
        return out

    def __pow__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data**other.data, (self, other))
        def _backward():
            self.grad += other.data * self.data ** (other.data - 1) * out.grad
            other.grad += self.data ** other.data * log(other.data) * out.grad
        self._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def exp(self):
        out = Value(exp(self.data), (self,))
        def _backward():
            self.grad += out * out.grad
        self._backward = _backward
        return out

    def log(self):
        out = Value(log(self.data), (self,))
        def _backward():
            self.grad += (self.data ** -1) * out.grad
        self._backward = _backward
        return out



    # TODO backward() method
