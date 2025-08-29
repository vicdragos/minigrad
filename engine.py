from math import log, exp, tanh


class Value:
    """
    This class is the core component of the autograd engine.Stores a single scalar value,
    its gradient with respect to a final output, and the backward pass logic.

    Each mathematical operation (`+`, `*`, `tanh`, etc.) creates a new `Value` object.
    This new object tracks its precedent "child" nodes and holds a `_backward` closure.
    This closure knows how to compute the local derivatives and propagate the gradient
    from the output `Value` back to its children.
    """

    def __init__(self, data, _children=()):
        # The scalar data of the Value.
        self.data = data
        # The gradient of the final output (loss) with respect to this Value.
        self.grad = 0.0

        # The local function that performs the chain rule for this operation. None by default for leaf nodes (inputs).
        self._backward = lambda: None
        # The set of child Value objects that produced this one.
        self._prev = set(_children)

    def __repr__(self):
        return f"Value(data={self.data})"


    # --- Operators ---

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

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)


    # --- Reverse Operators ---

    def __rtruediv__(self, other):
        return other * self ** -1

    def __rmul__(self, other):
        return other * self

    def __radd__(self, other):
        return other + self

    def __rsub__(self, other):
        return other - self


    # --- Mathematical Functions ---

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
        t = tanh(self.data)
        out = Value(t, (self,))

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,))
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out


    # --- Backpropagation ---

    def backward(self):
        """
        Performs backpropagation starting from this Value,
        propagating gradients for the entire computational graph.
        """
        # 1: List all the nodes in topological order, meaning that for
        # every operation, its inputs will appear before the outputs
        # (i.e. the order in which the computation is performed)
        nodes = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for u in v._prev:
                    build(u)
                nodes.append(v)
        build(self)

        # 2: Initialize the gradient to 1. The start of the chain rule
        self.grad = 1.0

        # 3: Apply the chain rule in reverse topological order.
        for v in reversed(nodes):
            v._backward()
