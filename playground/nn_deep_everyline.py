"""
任意层数的神经网络 —— 每一行都注释的版本（第四周作业）
=========================================================
这个文件是给你这种"Python 语法还没学完"的人看的：
每一行代码，旁边或上方都有一句解释。
解释分两种：
    【语法】这行 Python 在干什么
    【数学】这行在算什么

运行方式：
    cd /d D:\dl-journey
    python playground\nn_deep_everyline.py

跑完会打印三种层数结构的准确率，并生成两张图。
"""


# =============================================================================
# 第 0 节：Python 语法小抄 —— 先花 10 分钟看这个，再看代码
# =============================================================================
#
# 1) import 是什么？
#    import numpy as np
#    相当于 C++ 的 #include，把别人的库拿进来用；as np 是起外号。
#
# 2) 列表（list）是什么？
#    a = [1, 2, 3]
#    相当于 C++ 的 vector，可以装任意东西，长度可变。
#    a[0] 是第一个元素（下标从 0 开始，和 C++ 一样）。
#    a.append(x) 往末尾加一个元素。
#
# 3) 字典（dict）是什么？
#    d = {"W": 1, "b": 2}
#    相当于 C++ 的 map<string, 值>，用"键"取"值"：d["W"] 得到 1。
#
# 4) 列表套字典是什么？
#    params = [{"W": ..., "b": ...}, {"W": ..., "b": ...}]
#    params[0] 是第 1 层的字典，params[0]["W"] 就是第 1 层的 W。
#
# 5) for 循环怎么写？
#    for i in range(3):     → i 依次等于 0, 1, 2（range(3) 不含 3）
#    注意 Python 用"缩进"表示范围，不用花括号。
#
# 6) 倒着循环怎么写？
#    for i in reversed(range(3)):   → i 依次等于 2, 1, 0
#
# 7) 函数怎么写？
#    def f(a, b=1):   定义函数 f，参数 a 必填，b 默认 1
#        return a + b
#
# 8) 占位列表：[None] * 3  → 造一个长度为 3、每个元素都是 None 的列表
#
# 9) f-string 是什么？
#    f"准确率 {acc:.2f}%"  → 把变量 acc 填进去，:.2f 表示保留两位小数
#
# 10) 用大括号取值的写法：params[l]["W"]  → 第 l+1 层的 W（因为下标从 0 开始）
#
# 11) @ 是什么？矩阵乘法，等价于 np.dot。
# 12) .T 是什么？转置。
# 13) np.xxx 是什么？numpy 里的函数，例如 np.sum 求和、np.maximum 取较大值。
# 14) (Z > 0) 是什么？对数组逐个判断，得到一堆 True/False；
#     在 numpy 里 True/False 可以直接当 1/0 参与乘法。
# 15) if __name__ == "__main__": 固定写法，表示"直接运行这个文件时才执行下面"。
#
# 看懂这 15 条，下面的代码逐行读就没有障碍了。
# =============================================================================


# ---------- 第 1 节：导入要用的库 ----------
import os                     # 【语法】导入 os，用来拼文件路径

import numpy as np            # 【语法】导入 numpy，别名 np，做矩阵运算
import matplotlib             # 【语法】导入 matplotlib，画图用
matplotlib.use("Agg")         # 【语法】设置成"不弹窗、直接存图"的模式
import matplotlib.pyplot as plt  # 【语法】导入画图接口，别名 plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]  # 【语法】指定中文字体，否则中文变方块
plt.rcParams["axes.unicode_minus"] = False                       # 【语法】让负号正常显示

HERE = os.path.dirname(os.path.abspath(__file__))  # 【语法】__file__ 是本文件路径；abspath 转绝对路径；dirname 取所在文件夹
np.random.seed(0)                                  # 【数学】固定随机种子，保证每次跑出来的结果一样


# =============================================================================
# 第 2 节：造数据（两条互相缠绕的螺旋）
# =============================================================================
def make_spirals(n_per_class=300, noise=0.12):
    # 【语法】def 定义函数；n_per_class=300 是默认参数，调用时不写就用 300
    """
    每一类造 n_per_class 个点，两类点互相缠绕成螺旋。
    返回 X（形状 2×600）和 Y（形状 1×600）。
    """
    m_each = n_per_class                                      # 【语法】把参数存进变量，后面少打字

    t = np.linspace(0.2, 1.0, m_each) * 4 * np.pi             # 【数学】在 0.2~1.0 之间取 300 个数，乘 4π 变成角度
    r = t / (4 * np.pi)                                       # 【数学】半径随角度线性增长，才形成螺旋

    ang0 = t                                                  # 【数学】第 0 类的角度
    ang1 = t + np.pi                                          # 【数学】第 1 类的角度，错开半圈（π）

    x0 = r * np.cos(ang0) + np.random.randn(m_each) * noise   # 【数学】极坐标转直角坐标 X，再加一点随机噪声
    y0 = r * np.sin(ang0) + np.random.randn(m_each) * noise   # 【数学】极坐标转直角坐标 Y，再加噪声
    x1 = r * np.cos(ang1) + np.random.randn(m_each) * noise   # 【数学】第 1 类的 X
    y1 = r * np.sin(ang1) + np.random.randn(m_each) * noise   # 【数学】第 1 类的 Y

    row_x = np.hstack((x0, x1))                               # 【语法】hstack 把两个一维数组左右拼成 600 个
    row_y = np.hstack((y0, y1))                               # 【语法】同上，拼 Y 坐标
    X = np.vstack((row_x, row_y))                             # 【语法】vstack 把两行上下叠成 2×600 的矩阵

    label0 = np.zeros(m_each)                                 # 【数学】前 300 个点的标签是 0
    label1 = np.ones(m_each)                                  # 【数学】后 300 个点的标签是 1
    Y = np.hstack((label0, label1))                           # 【语法】拼成 600 个标签
    Y = Y.reshape(1, -1)                                      # 【语法】reshape 改成 1 行 600 列；-1 表示"这维你自己算"

    return X, Y                                               # 【语法】一次返回两个值


# =============================================================================
# 第 3 节：初始化参数（层数由列表决定）
# =============================================================================
def init_params(layer_dims):
    # 【语法】layer_dims 是一个列表，例如 [2, 16, 1]
    """
    layer_dims 描述每层多少个神经元，相邻两个数之间有一组 W、b。
    返回 params 列表，params[0] 是第 1 层的参数。
    """
    params = []                                               # 【语法】先造一个空列表，等下往里 append

    for l in range(len(layer_dims) - 1):                      # 【语法】range(3) 得到 0,1,2；len 取列表长度
        n_in = layer_dims[l]                                  # 【语法】第 l 个元素 = 这一层的输入个数
        n_out = layer_dims[l + 1]                             # 【语法】第 l+1 个元素 = 这一层神经元个数

        W = np.random.randn(n_out, n_in) * np.sqrt(2.0 / n_in)  # 【数学】He 初始化：随机数乘 sqrt(2/n_in)，让每层输出方差稳定
        b = np.zeros((n_out, 1))                              # 【语法】造 n_out 行 1 列的全 0；注意是双括号
        params.append({"W": W, "b": b})                       # 【语法】把这一层的 W、b 装进字典，再 append 到列表

    return params                                             # 【语法】返回列表


# =============================================================================
# 第 4 节：前向传播（for 循环走完所有层）
# =============================================================================
def forward(X, params):
    """
    公式：Z[l] = W[l]·A[l-1] + b[l]，A[l] = 激活(Z[l])
    中间层用 ReLU，最后一层用 sigmoid。
    """
    caches = []                                               # 【语法】空列表，用来存每层的中间结果
    A = X                                                     # 【数学】第 0 层"激活"就是输入本身
    L = len(params)                                           # 【语法】len(params) = 一共几层

    for l in range(L):                                        # 【语法】l 从 0 到 L-1
        A_prev = A                                            # 【数学】先记下这一层的输入，反向传播要用
        W = params[l]["W"]                                    # 【语法】取第 l+1 层的 W（字典取值）
        b = params[l]["b"]                                    # 【语法】取第 l+1 层的 b

        Z = W @ A_prev + b                                    # 【数学】线性部分：矩阵乘 + 广播加
        is_last = (l == L - 1)                                # 【语法】判断是不是最后一层，== 是比较

        if is_last:                                           # 【语法】if 条件成立走下面
            A = 1 / (1 + np.exp(-Z))                          # 【数学】sigmoid，把数压到 0~1
        else:                                                 # 【语法】否则走这里
            A = np.maximum(0, Z)                              # 【数学】ReLU：复数逐个和 0 比，取大的

        caches.append((A_prev, Z))                            # 【语法】把 (A_prev, Z) 这个元组 append 进去

    return A, caches                                          # 【语法】返回最终输出和缓存


# =============================================================================
# 第 5 节：代价函数（交叉熵）
# =============================================================================
def compute_cost(A_last, Y):
    m = Y.shape[1]                                            # 【语法】shape 是形状元组，(1,600) 的 [1] 是 600

    y_log = Y * np.log(A_last + 1e-8)                         # 【数学】真值为 1 的样本，算 log(预测)
    one_minus = (1 - Y) * np.log(1 - A_last + 1e-8)           # 【数学】真值为 0 的样本，算 log(1-预测)；加 1e-8 防 log(0)
    cost = -np.sum(y_log + one_minus) / m                     # 【数学】全部加起来、除以样本数、取负号

    return float(np.squeeze(cost))                            # 【语法】squeeze 去掉多余维度，float 转成普通小数


# =============================================================================
# 第 6 节：反向传播（核心）
# =============================================================================
def backward(params, caches, Y):
    """
    每层三件事：
        dz[l] = da[l] ⊙ g'(Z[l])
        dW[l] = dz[l] · A[l-1]ᵀ / m
        db[l] = sum(dz[l]) / m
    然后把 da[l-1] = W[l]ᵀ · dz[l] 传给前一层。
    """
    L = len(params)                                           # 【语法】层数
    m = Y.shape[1]                                            # 【语法】样本数
    grads = [None] * L                                        # 【语法】造一个长度 L、全是 None 的列表占位

    A_prev_last, Z_last = caches[-1]                          # 【语法】caches[-1] 是最后一个元素（-1 表示倒数第一）
    A_last = 1 / (1 + np.exp(-Z_last))                        # 【数学】重新算一次最后一层输出（sigmoid）

    dA = -(Y / (A_last + 1e-8)) + (1 - Y) / (1 - A_last + 1e-8)  # 【数学】da：代价对输出 A 的导数
    dZ = dA * (A_last * (1 - A_last))                         # 【数学】sigmoid 的导数是 a(1-a)；两者相乘会化简成 A-Y

    W_last = params[L - 1]["W"]                               # 【语法】取最后一层的 W
    dW = dZ @ A_prev_last.T / m                               # 【数学】dW = dz · Aᵀ / m；.T 是转置
    db = np.sum(dZ, axis=1, keepdims=True) / m                # 【数学】按行求和（把 m 个样本汇总），保持 (n,1) 形状

    grads[L - 1] = {"dW": dW, "db": db}                       # 【语法】把最后一层的梯度存进列表对应位置

    dA = W_last.T @ dZ                                        # 【数学】把误差传回前一层：da[l-1] = W[l]ᵀ dz[l]

    for l in reversed(range(L - 1)):                          # 【语法】reversed 让 l 从 L-2 递减到 0
        A_prev, Z = caches[l]                                 # 【语法】取出这一层的缓存（元组解包）

        relu_mask = (Z > 0)                                   # 【语法】得到一堆 True/False
        dZ = dA * relu_mask                                   # 【数学】ReLU 的导数：z>0 是 1，否则 0；True/False 当 1/0 用

        dW = dZ @ A_prev.T / m                                # 【数学】dW = dz · Aᵀ / m
        db = np.sum(dZ, axis=1, keepdims=True) / m            # 【数学】db = Σdz / m
        grads[l] = {"dW": dW, "db": db}                       # 【语法】存进列表第 l 个位置

        if l > 0:                                             # 【语法】第 1 层之前没有层了，不用再传
            dA = params[l]["W"].T @ dZ                        # 【数学】继续往前传：da[l-1] = W[l]ᵀ dz[l]

    return grads                                              # 【语法】返回梯度列表


# =============================================================================
# 第 7 节：更新参数
# =============================================================================
def update(params, grads, lr):
    for l in range(len(params)):                              # 【语法】遍历每一层
        params[l]["W"] -= lr * grads[l]["dW"]                 # 【数学】W = W - 学习率 × dW；-= 是"减完存回"
        params[l]["b"] -= lr * grads[l]["db"]                 # 【数学】b = b - 学习率 × db
    return params                                             # 【语法】返回更新后的参数


# =============================================================================
# 第 8 节：训练
# =============================================================================
def train(X, Y, layer_dims, epochs=12000, lr=0.5, print_every=2000):
    params = init_params(layer_dims)                          # 【语法】调用上面的函数，拿到初始参数
    history = []                                              # 【语法】空列表，存每一轮的代价

    for epoch in range(epochs):                               # 【语法】循环 epochs 次
        A_last, caches = forward(X, params)                   # 【数学】① 前向：算预测
        cost = compute_cost(A_last, Y)                        # 【数学】② 算代价
        grads = backward(params, caches, Y)                   # 【数学】③ 反向：算梯度
        params = update(params, grads, lr)                    # 【数学】④ 更新参数
        history.append(cost)                                  # 【语法】把这次的代价 append 进历史

        if epoch % print_every == 0:                          # 【语法】% 取余；余数为 0 说明是每隔 print_every 轮
            print(f"  epoch {epoch:5d}   cost = {cost:.4f}")  # 【语法】f-string：{epoch:5d} 占 5 格整数，{cost:.4f} 保留 4 位小数

    return params, history                                    # 【语法】返回参数和代价历史


# =============================================================================
# 第 9 节：预测
# =============================================================================
def predict(params, X):
    A_last, _ = forward(X, params)                            # 【语法】_ 表示"这个返回值我不要，丢掉"
    return (A_last > 0.5).astype(int)                         # 【数学】概率 > 0.5 判 1，否则 0；astype(int) 把 True/False 转成 1/0


# =============================================================================
# 第 10 节：维度核对（第四周强调的习惯）
# =============================================================================
def print_shapes(layer_dims):
    """
    写深层网络前先列形状：
        W[l] 是 (n[l], n[l-1])
        b[l] 是 (n[l], 1)
    形状错了不会报错，只会算出莫名其妙的结果。
    """
    print("  层   W 形状        b 形状        本层参数量")     # 【语法】打印表头
    total = 0                                                 # 【语法】累加器，从 0 开始

    for l in range(len(layer_dims) - 1):                      # 【语法】有几组相邻层就循环几次
        n_in = layer_dims[l]                                  # 【语法】输入个数
        n_out = layer_dims[l + 1]                             # 【语法】输出个数
        cnt = n_in * n_out + n_out                            # 【数学】W 有 n_in×n_out 个，b 有 n_out 个
        total += cnt                                          # 【语法】累加
        print(f"   {l+1}    ({n_out}, {n_in})".ljust(18) + f"({n_out}, 1)".ljust(14) + str(cnt))  # 【语法】ljust 左对齐补空格

    print(f"   合计参数量：{total}")                           # 【语法】打印总数


# =============================================================================
# 第 11 节：主程序（浅 vs 深 对比）
# =============================================================================
def main():
    X, Y = make_spirals()                                     # 【语法】调用函数造数据，两个返回值分别接住
    print(f"数据：X {X.shape}   Y {Y.shape}\n")              # 【语法】\n 是换行符

    print("最深的那个网络的维度核对：")                        # 【语法】打印一行提示
    print_shapes([2, 16, 16, 16, 1])                          # 【语法】传一个列表进去
    print()                                                   # 【语法】空 print 就是输出一个换行

    configs = [                                               # 【语法】造一个列表，里面是三个元组（名字, 层结构）
        ("浅：2-4-1", [2, 4, 1]),
        ("中：2-16-16-1", [2, 16, 16, 1]),
        ("深：2-16-16-16-1", [2, 16, 16, 16, 1]),
    ]

    results = []                                              # 【语法】空列表，存每次实验的结果

    for name, dims in configs:                                # 【语法】遍历列表；每个元素是元组，自动拆成 name 和 dims
        print(f"\n训练 {name}")                               # 【语法】打印标题
        np.random.seed(0)                                     # 【数学】每次重新固定种子，保证三个网络起点可比
        params, hist = train(X, Y, dims)                      # 【语法】训练，拿到参数和代价历史

        pred = predict(params, X)                             # 【数学】在训练集上预测
        correct = (pred == Y)                                 # 【语法】逐个比较，得到 True/False 数组
        acc = np.mean(correct) * 100                          # 【数学】True 占的比例 ×100 = 准确率
        n_params = sum(p["W"].size + p["b"].size for p in params)  # 【语法】生成器表达式：把每层参数个数加起来

        print(f"  → 准确率 {acc:.2f}%   参数量 {n_params}")    # 【语法】f-string 打印
        results.append((name, dims, params, hist, acc))       # 【语法】把结果打包 append 进列表

    plt.figure(figsize=(6, 4))                                # 【语法】开一张 6×4 英寸的图
    for name, dims, params, hist, acc in results:             # 【语法】遍历结果
        plt.plot(hist, label=f"{'-'.join(map(str, dims))} acc={acc:.1f}%")  # 【语法】画曲线，label 是图例文字
    plt.xlabel("epoch")                                       # 【语法】横轴标题
    plt.ylabel("cost")                                        # 【语法】纵轴标题
    plt.legend(fontsize=8)                                    # 【语法】显示图例，字号 8
    plt.title("浅 vs 深：训练代价曲线")                        # 【语法】总标题
    plt.tight_layout()                                        # 【语法】自动调整边距，防止文字被切
    plt.savefig(os.path.join(HERE, "deep_cost_curves.png"), dpi=130)  # 【语法】存图，os.path.join 拼路径

    name, dims, params, hist, acc = results[-1]               # 【语法】取最后一个结果（最深那个网络）
    xs = np.linspace(-1.3, 1.3, 300)                          # 【数学】横向取 300 个坐标点
    ys = np.linspace(-1.3, 1.3, 300)                          # 【数学】纵向取 300 个坐标点
    xx, yy = np.meshgrid(xs, ys)                              # 【语法】生成 300×300 的网格坐标
    gx = xx.ravel()                                           # 【语法】ravel 把二维拉平成一维
    gy = yy.ravel()                                           # 【语法】同上
    grid = np.vstack((gx, gy))                                # 【语法】叠成 2×90000，作为 9 万个待预测点

    Z = predict(params, grid)                                 # 【数学】对 9 万个点逐个预测
    Z = Z.reshape(xx.shape)                                   # 【语法】再变回 300×300 的表格，方便画图

    plt.figure(figsize=(6, 5))                                # 【语法】新开一张图
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="RdBu")          # 【语法】画填充色块，alpha 是透明度
    mask0 = (Y[0] == 0)                                       # 【语法】挑出标签为 0 的那些位置
    plt.scatter(X[0, mask0], X[1, mask0], s=8, label="class 0")  # 【语法】画类别 0 的点，s 是点大小
    mask1 = (Y[0] == 1)                                       # 【语法】挑出标签为 1 的位置
    plt.scatter(X[0, mask1], X[1, mask1], s=8, label="class 1")  # 【语法】画类别 1 的点
    plt.legend(loc="upper right", fontsize=9)                 # 【语法】图例放右上角
    plt.title(f"最深网络 acc={acc:.1f}%")                      # 【语法】标题里带准确率
    plt.tight_layout()                                        # 【语法】调整边距
    plt.savefig(os.path.join(HERE, "deep_boundary.png"), dpi=130)  # 【语法】存第二张图

    print("\n图片已保存：deep_cost_curves.png, deep_boundary.png")  # 【语法】提示结束


if __name__ == "__main__":                                    # 【语法】只有"直接运行"这个文件时才执行 main()
    main()                                                    # 【语法】调用主函数
