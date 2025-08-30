# minigrad

A simple, step-by-step implementation of a scalar-based autograd engine in Python, meant to solidify my understanding of the core concepts of deep learning. Built upon and heavily inspired by Andrej Karpathy's own autograd engine, `micrograd`.

### Motivation & Philosophy

The philosophy behind this project was to treat every foundational concept as a hypothesis to be tested and validated through implementation. Nothing was copied or taken for granted; every component was built and experimented with to develop a deep intuition. In essence, `minigrad` is a rigorous investigation into every concept that was not completely obvious to me from the start.

## Core Architectural Concepts

#### The `Value` Object
The foundational data structure of the engine. A `Value` object is a scalar that acts as a node in the computational graph. It holds its own floating-point `data`, its corresponding `grad` (gradient), and maintains pointers to the children `Value` objects from which it was derived. All mathematical operations are overloaded to return new `Value` objects, automatically building the graph.

#### Backpropagation
The `backward` method, when called on the final scalar loss `Value`, performs a full reverse-mode automatic differentiation. It first performs a topological sort of all nodes in the graph and then recursively applies the chain rule from the final node all the way back to the input parameters. This process correctly calculates and accumulates the gradients for every `Value` object in the graph.

## Implemented Features

The framework contains all the necessary components to train a simple neural network.

*   **Scalar Autograd Engine (`engine.py`):** A complete implementation of the `Value` object and its backward pass.
*   **Neural Network Module (`neural_net.py`):** 
    *   Provides `Neuron`, `Layer`, and `MLP` classes for building fully-connected networks.
    *   Supports swappable activation functions like `ReLU` and `Tanh`.
    *   Includes implementations for common loss functions like `mse_loss` and a numerically stable `cross_entropy_loss`.
*   **Optimizer Module (`optimizer.py`):** A formal `SGD` optimizer class that decouples the optimization logic from the model.

---

## Demonstration: Training on the `scikit-learn` Digits Dataset

To provide a concrete example of the engine in action, a complete training and evaluation pipeline has been implemented using the small `scikit-learn` digits dataset (8x8 images). This allows for rapid testing and demonstrates the end-to-end functionality of the framework.

The code for this demonstration can be found in the [**mini-MNIST.ipynb**](./mini-MSINT.ipynb) notebook.

Below is a snippet from the notebook, illustrating the core logic:

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

# Import the minigrad engine
from engine import Value
from neural_net import MLP, cross_entropy_loss
from optimizer import SGD

# 1. Load and prepare the dataset
digits = datasets.load_digits()
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1)) # Make the features into one-dim array
n_features = data.shape[1]
n_classes = 10
X = digits.data / 16.0 # Normalize pixel values to [0, 1]
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 2. Initialize the Model and Optimizer
model = MLP(n_features, [32, 16, n_classes], act = 'relu')
optim = SGD(model.parameters(), lr=0.05)

# 3. Training Loop
epochs = 5
for epoch in range(epochs):
    for i in range(len(X_train)):
        # Convert raw data to Value objects
        inputs = [Value(x) for x in X_train[i]]

        # Forward pass
        logits = model(inputs)
        loss = cross_entropy_loss(logits, [int(y_train[i])])

        # Backward pass and optimization step
        optim.zero_grad()
        loss.backward()
        optim.step()

    n_correct = 0
    for i in range(len(X_test)):
        inputs = [Value(x) for x in X_test[i]]

        ypred = model(inputs)
        index = np.argmax([y.data for y in ypred])

        n_correct += index == y_test[i]
    accuracy = n_correct / len(X_test)
    print(f"Epoch {epoch+1}, Accuracy: {accuracy*100:.2f}%")
```
The runtime for this training is notably slow. This is a deliberate and important outcome of the project's design. The engine is intentionally **scalar-based**, with every operation performed on individual Python objects. This benchmark serves as a direct demonstration of the immense performance gains achieved by the **vectorized, matrix-based operations** used in production libraries like PyTorch and NumPy. Thus, the slowness serves educational feature that provides a physical intuition for why those libraries are architected the way they are.
  
