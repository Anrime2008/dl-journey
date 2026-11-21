"""
任意层数的神经网络（把第一门课第四周的内容用上）
==========================================================
第三周：只会写"固定两层"（1 个隐藏层 + 1 个输出层），
        前向 4 行、反向 6 行，都是手写的，层数一变就得重写。

第四周：把它推广到 L 层。
        不再手写 Z1/A1/Z2/A2，而是用 for 循环，一层一层往前走；
        反向也是一层一层往回走。
        于是"几层"变成一个可以随便改的超参数。

这个文件就做这件事：网络层数、每层宽度都由一个列表决定，
比如 [2, 16, 16, 16, 1] 就是 输入2 → 16 → 16 → 16 → 输出1（4 层）。

最后还会做一次对比：浅网络 vs 深网络，在同一份螺旋数据上比准确率，
用来说明"为什么需要深"。

运行：python nn_deep.py
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
np.random.seed(0)


# =====================================================================
# 第 1 步：造数据（两条互相缠绕的螺旋）
# =====================================================================
def make_spirals(n_per_class=300, noise=0.12):
    """
    造两条螺旋线，每条 300 个点，互相交错。
    这种数据"一条直线绝对分不开"，而且浅网络只能画很粗糙的分界，
    正好用来对比浅和深。

    返回 X: (2, m)   Y: (1, m)
    """
    m_each = n_per_class

    # np.linspace(a, b, n)：在 a 到 b 之间均匀取 n 个数（含两端）
    t = np.linspace(0.2, 1.0, m_each) * 4 * np.pi
    # 上面这行：取 0.2 到 1.0 的比例，乘 2π，得到角度 0.4π ~ 2π

    # 半径跟着角度一起长，才形成螺旋
    r = t / (4 * np.pi)

    # 两条螺旋相差 180 度（就是加 π）
    ang0 = t                 # 第 0 类：角度 t
    ang1 = t + np.pi         # 第 1 类：角度 t + π

    # 坐标 + 一点随机噪声（np.random.randn 是标准正态噪声，乘 noise 控制大小）
    x0 = r * np.cos(ang0) + np.random.randn(m_each) * noise
    y0 = r * np.sin(ang0) + np.random.randn(m_each) * noise
    x1 = r * np.cos(ang1) + np.random.randn(m_each) * noise
    y1 = r * np.sin(ang1) + np.random.randn(m_each) * noise

    # np.hstack：把两个数组左右拼起来（(300,) 和 (300,) 拼成 (600,)）
    X = np.vstack((np.hstack((x0, x1)), np.hstack((y0, y1))))
    # np.zeros / np.ones：造全 0、全 1 数组；再 hstack 拼成标签行
    Y = np.hstack((np.zeros(m_each), np.ones(m_each))).reshape(1, -1)

    return X, Y


# =====================================================================
# 第 2 步：初始化参数（层数由列表决定）
# =====================================================================
def init_params(layer_dims):
    """
    layer_dims 是一个列表，描述每层有多少个神经元，例如 [2, 16, 1]。
    相邻两个数之间就有一组 W、b。

    这里用"列表套字典"来存：
        params[0] = 第 1 层的 {"W":..., "b":...}
        params[1] = 第 2 层的 {"W":..., "b":...}
    注意 Python 的列表下标从 0 开始，所以 params[0] 是"第 1 层"。

    为什么不用 params["W1"] 这种写法？
    因为层数事先不知道，没法写死键名；用列表就能 for 循环遍历。
    """
    params = []

    # range(len(layer_dims) - 1)：有几组相邻层，(2,16,1) 长度 3，就有 2 组
    for l in range(len(layer_dims) - 1):
        n_in = layer_dims[l]        # 这一层有几个输入
        n_out = layer_dims[l + 1]   # 这一层有几个神经元

        # He 初始化：乘 sqrt(2 / n_in)。
        # 为什么要缩放：层数一多，如果权重太大，信号会被放大到爆炸；
        # 太小又会被压到 0。乘这个系数能让每层输出的方差保持稳定。
        # 这是"为什么深网络能训起来"的关键细节之一。
        W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)
        b = np.zeros((n_out, 1))

        # 每层装一个字典，append 进列表
        params.append({"W": W, "b": b})

    return params


# =====================================================================
# 第 3 步：前向传播（用 for 循环走完所有层）
# =====================================================================
def forward(X, params):
    """
    一行公式，循环 L 次：
        Z[l] = W[l] · A[l-1] + b[l]
        A[l] = ReLU(Z[l])        （中间层）
        A[L] = sigmoid(Z[L])     （最后一层，输出概率）

    返回：
        A_last : 最终输出 (1, m)
        caches : 每一层的 (A_prev, Z)，反向传播要用
    """
    caches = []
    A = X                       # 第 0 层的"激活"就是输入本身
    L = len(params)             # 一共有几层（带参数的层）

    for l in range(L):
        A_prev = A              # 记住这一层的输入，反向要用
        W = params[l]["W"]
        b = params[l]["b"]

        Z = W @ A_prev + b      # 线性部分，形状 (n_out, m)

        # 最后一层用 sigmoid（输出 0~1 的概率），其余用 ReLU
        if l == L - 1:
            A = 1 / (1 + np.exp(-Z))
        else:
            A = np.maximum(0, Z)   # ReLU：负数全变成 0，正数不变

        caches.append((A_prev, Z))

    return A, caches


# =====================================================================
# 第 4 步：算代价（交叉熵，和第三周一样）
# =====================================================================
def compute_cost(A_last, Y):
    m = Y.shape[1]
    cost = -np.sum(Y * np.log(A_last + 1e-8) + (1 - Y) * np.log(1 - A_last + 1e-8)) / m
    return float(np.squeeze(cost))


# =====================================================================
# 第 5 步：反向传播（这是第四周最核心的部分）
# =====================================================================
def backward(params, caches, Y):
    """
    思路：从最后一层往前，一层一层算 dz、dW、db，并把"误差"传给前一层。

    对第 l 层，永远是三件事：
        dz[l] = da[l] ⊙ g'(Z[l])      ← 激活函数求导
        dW[l] = dz[l] · A[l-1]ᵀ / m
        db[l] = sum(dz[l]) / m
        da[l-1] = W[l]ᵀ · dz[l]        ← 把误差传回去

    上一轮你问"da 是干什么的"：它就在这里，是每一层往回传的那个量。
    """
    grads = [None] * len(params)     # 先占好位置，后面按下标填
    L = len(params)
    m = Y.shape[1]

    # ---------- 最后一层 ----------
    A_prev, Z_last = caches[-1]
    A_last = 1 / (1 + np.exp(-Z_last))

    # da：代价对输出 A 的导数。交叉熵求导得到这个分式。
    dA = -(Y / (A_last + 1e-8)) + (1 - Y) / (1 - A_last + 1e-8)

    # sigmoid 的导数：a(1-a)。两者相乘会正好化简成 A - Y，
    # 这里写完整版是为了让你看见 da 和 g'，读懂之后日常直接用 A-Y 就行。
    dZ = dA * (A_last * (1 - A_last))

    grads[L - 1] = {
        "dW": dZ @ A_prev.T / m,
        "db": np.sum(dZ, axis=1, keepdims=True) / m,
    }

    # 传给前一层
    dA = params[L - 1]["W"].T @ dZ

    # ---------- 中间层，从后往前 ----------
    # reversed(range(L-1)) 让 l 从 L-2 一直减到 0
    for l in reversed(range(L - 1)):
        A_prev, Z = caches[l]

        # ReLU 的导数：z > 0 时是 1，否则 0。
        # (Z > 0) 得到一堆 True/False，NumPy 里可以直接当 1/0 参与乘法。
        dZ = dA * (Z > 0)

        grads[l] = {
            "dW": dZ @ A_prev.T / m,
            "db": np.sum(dZ, axis=1, keepdims=True) / m,
        }

        # 把误差继续往回传（第 1 层之后没有更前的层了，可以不算）
        if l > 0:
            dA = params[l]["W"].T @ dZ

    return grads


# =====================================================================
# 第 6 步：参数更新（梯度下降）
# =====================================================================
def update(params, grads, lr):
    for l in range(len(params)):
        params[l]["W"] -= lr * grads[l]["dW"]
        params[l]["b"] -= lr * grads[l]["db"]
    return params


# =====================================================================
# 第 7 步：训练（把上面串起来）
# =====================================================================
def train(X, Y, layer_dims, epochs=12000, lr=0.5, print_every=2000):
    params = init_params(layer_dims)
    history = []

    for epoch in range(epochs):
        A_last, caches = forward(X, params)
        cost = compute_cost(A_last, Y)
        grads = backward(params, caches, Y)
        params = update(params, grads, lr)

        history.append(cost)
        if epoch % print_every == 0:
            print(f"  epoch {epoch:5d}   cost = {cost:.4f}")

    return params, history


# =====================================================================
# 第 8 步：预测
# =====================================================================
def predict(params, X):
    A_last, _ = forward(X, params)
    return (A_last > 0.5).astype(int)


# =====================================================================
# 附加：维度核对（第四周专门讲的一步）
# =====================================================================
def print_shapes(layer_dims):
    """
    第四周反复强调：写深层网络之前，先把每层的形状列出来。
    因为形状错了不会报错，只会算出莫名其妙的结果——
    "维度核对"是最快的 debug 手段。

    规则：
        W[l] : (n[l], n[l-1])
        b[l] : (n[l], 1)
        dW[l] 和 W[l] 同形状，db[l] 和 b[l] 同形状
    """
    print("  层   W 形状       b 形状      本层参数量")
    total = 0
    for l in range(len(layer_dims) - 1):
        n_in, n_out = layer_dims[l], layer_dims[l + 1]
        cnt = n_in * n_out + n_out
        total += cnt
        print("   {:<4d}({:<2d}, {:<2d})    ({}, 1)      {}".format(l + 1, n_out, n_in, n_out, cnt))
    print("   合计参数量：{}".format(total))


# =====================================================================
# 第 9 步：主程序 —— 浅网络 vs 深网络 对比
# =====================================================================
def main():
    X, Y = make_spirals()
    print(f"数据：X {X.shape}   Y {Y.shape}\n")

    # 维度核对：先把要用的网络形状列出来（第四周的习惯）
    print("最深的那个网络的维度核对：")
    print_shapes([2, 16, 16, 16, 1])
    print()

    # 三个结构：同样都是"2 进 1 出"，中间层数不同
    configs = [
        ("浅：2-4-1（1 个隐藏层）",        [2, 4, 1]),
        ("中：2-16-16-1（2 个隐藏层）",    [2, 16, 16, 1]),
        ("深：2-16-16-16-1（3 个隐藏层）", [2, 16, 16, 16, 1]),
    ]

    results = []
    for name, dims in configs:
        print(f"\n训练 {name}")
        np.random.seed(0)                      # 每次重新固定随机种子，保证可比
        params, hist = train(X, Y, dims, epochs=12000, lr=0.5)

        acc = np.mean(predict(params, X) == Y) * 100
        n_params = sum(p["W"].size + p["b"].size for p in params)
        print(f"  → 准确率 {acc:.2f}%   参数量 {n_params}")
        results.append((name, dims, params, hist, acc, n_params))

    # ---------- 画图：三条 cost 曲线 ----------
    plt.figure(figsize=(6, 4))
    for name, dims, params, hist, acc, n_params in results:
        plt.plot(hist, label=f"{'-'.join(map(str, dims))}  acc={acc:.1f}%")
    plt.xlabel("epoch")
    plt.ylabel("cost")
    plt.legend(fontsize=8)
    plt.title("浅 vs 深：训练代价曲线")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "deep_cost_curves.png"), dpi=130)

    # ---------- 画图：最深那个网络的决策边界 ----------
    name, dims, params, hist, acc, n_params = results[-1]
    xx, yy = np.meshgrid(np.linspace(-1.3, 1.3, 300), np.linspace(-1.3, 1.3, 300))
    grid = np.vstack((xx.ravel(), yy.ravel()))
    Z = predict(params, grid).reshape(xx.shape)

    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="RdBu")
    plt.scatter(X[0, Y[0] == 0], X[1, Y[0] == 0], s=8, label="class 0")
    plt.scatter(X[0, Y[0] == 1], X[1, Y[0] == 1], s=8, label="class 1")
    plt.legend(loc="upper right", fontsize=9)
    plt.title(f"最深网络 {'-'.join(map(str, dims))}  acc={acc:.1f}%")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "deep_boundary.png"), dpi=130)

    print("\n图片已保存：deep_cost_curves.png, deep_boundary.png")


# 固定写法：只有直接运行这个文件时才执行 main()
if __name__ == "__main__":
    main()
