"""
One-hidden-layer neural network (NumPy implementation).
Binary classification on a synthetic two-class concentric-circle dataset.

Architecture: 2 -> 8 (tanh) -> 1 (sigmoid)
Loss: binary cross-entropy
Optimizer: full-batch gradient descent

Usage: python nn_one_hidden_clean.py
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(1)


def load_data(m=400):
    """Generate two concentric circles: label 0 inside (r <= 1.5), label 1 outside."""
    theta = np.random.rand(m) * 2 * np.pi
    r = np.sqrt(np.random.rand(m)) * 3
    X = np.vstack((r * np.cos(theta), r * np.sin(theta)))
    Y = (r > 1.5).astype(float).reshape(1, m)
    return X, Y


def initialize_parameters(n_x, n_h, n_y):
    """Small random weights for W, zeros for b."""
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert W1.shape == (n_h, n_x)
    assert b1.shape == (n_h, 1)
    assert W2.shape == (n_y, n_h)
    assert b2.shape == (n_y, 1)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


def forward_propagation(X, params):
    """Z1 = W1X + b1, A1 = tanh(Z1), Z2 = W2A1 + b2, A2 = sigmoid(Z2)."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    Z1 = W1 @ X + b1
    A1 = np.tanh(Z1)
    Z2 = W2 @ A1 + b2
    A2 = 1 / (1 + np.exp(-Z2))

    assert A2.shape == (1, X.shape[1])
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


def compute_cost(A2, Y):
    """Binary cross-entropy."""
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8)) / m
    return float(np.squeeze(cost))


def backward_propagation(params, cache, X, Y):
    """Backpropagation; returns gradients with the same shapes as the parameters."""
    m = X.shape[1]
    W2 = params["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = (W2.T @ dZ2) * (1 - A1 ** 2)
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    assert dW1.shape == params["W1"].shape
    assert dW2.shape == params["W2"].shape
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def update_parameters(params, grads, learning_rate):
    """Vanilla gradient descent update."""
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


def nn_model(X, Y, n_h=8, num_iterations=10000, learning_rate=1.0):
    """Train the network and return the parameters and the cost history."""
    n_x = X.shape[0]
    n_y = Y.shape[0]
    params = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, params)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(params, cache, X, Y)
        params = update_parameters(params, grads, learning_rate)

        if i % 2000 == 0:
            print(f"iteration {i:5d}    cost = {cost:.6f}")
        costs.append(cost)

    return params, costs


def predict(params, X):
    """Threshold the predicted probability at 0.5."""
    A2, _ = forward_propagation(X, params)
    return (A2 > 0.5).astype(int)


def main():
    X, Y = load_data(400)
    print(f"X: {X.shape}   Y: {Y.shape}\n")

    params, costs = nn_model(X, Y, n_h=8, num_iterations=10000, learning_rate=1.0)

    acc = np.mean(predict(params, X) == Y) * 100
    print(f"\ntrain accuracy: {acc:.2f}%")

    # decision boundary
    xx, yy = np.meshgrid(np.linspace(-3.2, 3.2, 300), np.linspace(-3.2, 3.2, 300))
    grid = np.vstack((xx.ravel(), yy.ravel()))
    Z = predict(params, grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="RdBu")
    plt.scatter(X[0, Y[0] == 0], X[1, Y[0] == 0], s=12, label="class 0")
    plt.scatter(X[0, Y[0] == 1], X[1, Y[0] == 1], s=12, label="class 1")
    plt.legend(loc="upper right", fontsize=9)
    plt.title(f"1 hidden layer (n_h=8), acc={acc:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "nn_boundary.png"), dpi=130)

    # cost curve
    plt.figure(figsize=(6, 4))
    plt.plot(costs)
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("training cost")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "nn_cost.png"), dpi=130)

    print("saved: nn_boundary.png, nn_cost.png")


if __name__ == "__main__":
    main()
