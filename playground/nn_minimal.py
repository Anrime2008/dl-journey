"""
Minimal one-hidden-layer neural network in pure NumPy.

2 -> 8 (tanh) -> 1 (sigmoid), binary cross-entropy, full-batch gradient descent,
trained on a synthetic two-class concentric-circle dataset.

Run: python nn_minimal.py
"""

import numpy as np

np.random.seed(1)


def make_data(m=400):
    theta = np.random.rand(m) * 2 * np.pi
    r = np.sqrt(np.random.rand(m)) * 3
    X = np.vstack((r * np.cos(theta), r * np.sin(theta)))
    Y = (r > 1.5).astype(float).reshape(1, m)
    return X, Y


def init_params(n_x, n_h, n_y):
    return {
        "W1": np.random.randn(n_h, n_x) * 0.01,
        "b1": np.zeros((n_h, 1)),
        "W2": np.random.randn(n_y, n_h) * 0.01,
        "b2": np.zeros((n_y, 1)),
    }


def forward(X, p):
    Z1 = p["W1"] @ X + p["b1"]
    A1 = np.tanh(Z1)
    Z2 = p["W2"] @ A1 + p["b2"]
    A2 = 1 / (1 + np.exp(-Z2))
    return A2, {"A1": A1, "A2": A2}


def cost(A2, Y):
    m = Y.shape[1]
    return float(-np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8)) / m)


def backward(p, cache, X, Y):
    m = X.shape[1]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y
    dW2 = dZ2 @ A1.T / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m

    dZ1 = (p["W2"].T @ dZ2) * (1 - A1 ** 2)
    dW1 = dZ1 @ X.T / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


def step(p, g, lr):
    for k in ("W1", "b1", "W2", "b2"):
        p[k] -= lr * g["d" + k]
    return p


def main():
    X, Y = make_data()
    p = init_params(X.shape[0], 8, 1)

    for i in range(10000):
        A2, cache = forward(X, p)
        if i % 2000 == 0:
            print(f"iteration {i:5d}    cost = {cost(A2, Y):.6f}")
        p = step(p, backward(p, cache, X, Y), 1.0)

    A2, _ = forward(X, p)
    acc = np.mean((A2 > 0.5) == Y) * 100
    print(f"train accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()
