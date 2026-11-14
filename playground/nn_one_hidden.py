"""
一个隐藏层的神经网络（纯 numpy，对应吴恩达第一门课第三周作业）
=====================================================================
网络结构：  输入 2 维  →  隐藏层 n_h 个神经元  →  输出 1 个（0/1 二分类）

    X (2, m)  ──W1,b1──▶  Z1 (n_h, m)  ──tanh──▶  A1  ──W2,b2──▶  Z2 (1, m)  ──sigmoid──▶  A2 (1, m)

约定（和课件一致）：
    n_x = 输入维度  = 2
    n_h = 隐藏层单元数 = 8
    n_y = 输出维度  = 1
    m   = 样本个数
    W1 形状 (n_h, n_x)   b1 形状 (n_h, 1)
    W2 形状 (n_y, n_h)   b2 形状 (n_y, 1)

运行：python playground\\nn_one_hidden.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")          # 不弹窗，直接存图
import matplotlib.pyplot as plt

# 让图里的中文正常显示（默认字体没有汉字，会变成方块）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))

np.random.seed(1)              # 固定随机种子，结果可复现


# =====================================================================
# 第 1 步：造数据
# =====================================================================
def load_data(m=400):
    """
    生成一个「同心圆」数据集：内圈是类别 0，外圈是类别 1。
    这种数据用一条直线分不开，必须靠隐藏层把它「掰弯」——正是我们想验证的事。

    返回：
        X: (2, m)  每一列是一个样本的两个特征
        Y: (1, m)  每一列是对应标签（0 或 1）
    """
    theta = np.random.rand(m) * 2 * np.pi          # 随机角度
    r = np.sqrt(np.random.rand(m)) * 3             # 随机半径（开根号保证均匀分布）
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    X = np.vstack((x1, x2))                        # (2, m)

    Y = (r > 1.5).astype(float).reshape(1, m)      # 半径 > 1.5 算外圈 → 标签 1
    return X, Y


# =====================================================================
# 第 2 步：初始化参数
# =====================================================================
def initialize_parameters(n_x, n_h, n_y):
    """
    随机初始化 W（不能全 0，否则同一层的神经元学到的完全一样），b 初始化为 0。

    返回字典：
        W1 (n_h, n_x)   b1 (n_h, 1)
        W2 (n_y, n_h)   b2 (n_y, 1)
    """
    W1 = np.random.randn(n_h, n_x) * 0.01          # 很小的随机数
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    # 形状自检：形状错了后面会算出一堆莫名其妙的结果，这里先拦住
    assert W1.shape == (n_h, n_x)
    assert b1.shape == (n_h, 1)
    assert W2.shape == (n_y, n_h)
    assert b2.shape == (n_y, 1)

    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}


# =====================================================================
# 第 3 步：前向传播（从 X 一路算到预测值 A2）
# =====================================================================
def forward_propagation(X, params):
    """
    输入：
        X      (n_x, m)
        params 上一步的字典

    计算过程（四行，就是网络的全部「前向」）：
        Z1 = W1·X + b1       线性组合
        A1 = tanh(Z1)        激活（隐藏层用 tanh，输出在 -1~1）
        Z2 = W2·A1 + b2      再线性组合
        A2 = sigmoid(Z2)     压到 0~1，当作「是类别 1 的概率」

    返回：
        A2     (n_y, m)  预测概率
        cache  中间结果，反向传播要用（不然得重算一遍）
    """
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]

    Z1 = W1 @ X + b1                    # (n_h, m)
    A1 = np.tanh(Z1)                    # (n_h, m)
    Z2 = W2 @ A1 + b2                   # (n_y, m)
    A2 = 1 / (1 + np.exp(-Z2))          # sigmoid，直接写公式，不调库

    assert A2.shape == (1, X.shape[1])
    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache


# =====================================================================
# 第 4 步：算代价（预测得有多差，一个数字）
# =====================================================================
def compute_cost(A2, Y):
    """
    交叉熵：J = -(1/m) Σ [ y·log(â) + (1-y)·log(1-â) ]

    直觉：
        真值是 1，你预测 0.99 → log(0.99)≈-0.01，代价很小
        真值是 1，你预测 0.01 → log(0.01)≈-4.6，代价很大
        所以「猜错且自信」罚得最重。
    """
    m = Y.shape[1]
    # 加 1e-8 防止 log(0) 变成 -inf（数值稳定的小技巧，课件里也这么写）
    cost = -np.sum(Y * np.log(A2 + 1e-8) + (1 - Y) * np.log(1 - A2 + 1e-8)) / m
    return float(np.squeeze(cost))


# =====================================================================
# 第 5 步：反向传播（从代价倒着算每层参数该改多少）
# =====================================================================
def backward_propagation(params, cache, X, Y):
    """
    链式法则，六行公式（这是整个神经网络最核心的地方）：

        dZ2 = A2 - Y                      输出层误差，形状 (n_y, m)
        dW2 = (1/m) dZ2·A1ᵀ               W2 该改多少，形状 (n_y, n_h)
        db2 = (1/m) Σ dZ2                 b2 该改多少，形状 (n_y, 1)
        dZ1 = (W2ᵀ·dZ2) * (1 - A1²)      误差传回隐藏层，tanh 的导数就是 1-A1²
        dW1 = (1/m) dZ1·Xᵀ                W1 该改多少，形状 (n_h, n_x)
        db1 = (1/m) Σ dZ1                 b1 该改多少，形状 (n_h, 1)

    返回 grads 字典，和 params 一一对应。
    """
    m = X.shape[1]
    W2 = params["W2"]
    A1, A2 = cache["A1"], cache["A2"]

    dZ2 = A2 - Y                                  # (n_y, m)
    dW2 = dZ2 @ A1.T / m                          # (n_y, n_h)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m  # (n_y, 1)

    dZ1 = (W2.T @ dZ2) * (1 - A1 ** 2)            # (n_h, m)，tanh' = 1 - tanh²
    dW1 = dZ1 @ X.T / m                           # (n_h, n_x)
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m  # (n_h, 1)

    assert dW1.shape == params["W1"].shape
    assert dW2.shape == params["W2"].shape
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}


# =====================================================================
# 第 6 步：梯度下降（按梯度反方向挪一小步）
# =====================================================================
def update_parameters(params, grads, learning_rate):
    """
    W = W - 学习率 × dW
    b = b - 学习率 × db

    为什么要减：梯度指向上坡，减就是往下坡走，cost 才会变小。
    """
    params["W1"] -= learning_rate * grads["dW1"]
    params["b1"] -= learning_rate * grads["db1"]
    params["W2"] -= learning_rate * grads["dW2"]
    params["b2"] -= learning_rate * grads["db2"]
    return params


# =====================================================================
# 第 7 步：把上面六步串起来训练
# =====================================================================
def nn_model(X, Y, n_h=8, num_iterations=10000, learning_rate=1.0, print_cost=True):
    """
    每一轮做的事（顺序永远是这个）：
        前向 → 算代价 → 反向 → 更新
    """
    n_x = X.shape[0]
    n_y = Y.shape[0]
    params = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_iterations):
        A2, cache = forward_propagation(X, params)      # ① 前向：猜
        cost = compute_cost(A2, Y)                     # ② 算代价：错多少
        grads = backward_propagation(params, cache, X, Y)  # ③ 反向：谁该改、改多少
        params = update_parameters(params, grads, learning_rate)  # ④ 更新

        if print_cost and i % 2000 == 0:
            print(f"第 {i:5d} 轮   cost = {cost:.6f}")
        costs.append(cost)

    return params, costs


# =====================================================================
# 第 8 步：预测 + 看准确率
# =====================================================================
def predict(params, X):
    """概率 > 0.5 判为 1，否则 0。"""
    A2, _ = forward_propagation(X, params)
    return (A2 > 0.5).astype(int)


def main():
    X, Y = load_data(400)
    print("数据形状 X:", X.shape, " Y:", Y.shape, "\n")

    params, costs = nn_model(X, Y, n_h=8, num_iterations=10000, learning_rate=1.0)

    pred = predict(params, X)
    acc = np.mean(pred == Y) * 100
    print(f"\n训练集准确率：{acc:.2f}%")

    # ---- 画决策边界：看看隐藏层把空间掰成什么样了 ----
    xx, yy = np.meshgrid(np.linspace(-3.2, 3.2, 300), np.linspace(-3.2, 3.2, 300))
    grid = np.vstack((xx.ravel(), yy.ravel()))
    Z = predict(params, grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="RdBu")
    plt.scatter(X[0, Y[0] == 0], X[1, Y[0] == 0], s=12, c="#2f4a5c", label="class 0 (内圈)")
    plt.scatter(X[0, Y[0] == 1], X[1, Y[0] == 1], s=12, c="#a8342a", label="class 1 (外圈)")
    plt.legend(loc="upper right", fontsize=9)
    plt.title(f"1 hidden layer (n_h=8), acc={acc:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "nn_boundary.png"), dpi=130)
    print("决策边界已保存：playground/nn_boundary.png")

    # ---- 画 cost 曲线 ----
    plt.figure(figsize=(6, 4))
    plt.plot(costs, color="#a8342a")
    plt.xlabel("iteration")
    plt.ylabel("cost")
    plt.title("cost 下降曲线")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "nn_cost.png"), dpi=130)
    print("cost 曲线已保存：playground/nn_cost.png")


if __name__ == "__main__":
    main()
