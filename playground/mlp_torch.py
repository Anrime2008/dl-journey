"""
D5 · 两层神经网络跑 MNIST（PyTorch 版）
784 → 256 → 10，一个隐藏层。

运行：VS Code 里按 Ctrl+F5，或者命令行 python playground\\mlp_torch.py

跑完会打印每个 epoch 的测试 loss 和准确率，并在 playground/ 下存一张 loss_curve.png。
训练大概 2–4 分钟（CPU），风扇响是正常的。

看代码的顺序（别从第一行读到最后一行，按这个顺序看）：
  1. 先看「2. 模型」那三行 —— 网络到底长什么样
  2. 再看「3. 训练」里的 5 行 —— 就是不断重复：前向 → 算 loss → 反向 → 更新
  3. 最后回头看「1. 数据」 —— 只是把图片读进来、打包成一批一批

4 个练习（改完各跑一次，把结果记下来）：
  ① 把 EPOCHS 改成 1，看准确率掉到多少
  ② 把 LR 改成 0.1，看 loss 会不会乱跳
  ③ 把 BATCH 改成 8，看一个 epoch 慢多少
  ④ 把隐藏层 256 改成 32，看准确率还能不能到 95%
"""
import os

# Windows + conda 的常见坑：torch 自带一份 libomp，conda 的 numpy 用 mkl 自带 libiomp5md，
# 两个 OpenMP 运行时同时初始化会报 "OMP: Error #15"。加上下面这行让程序继续跑。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ---------- 超参数：先动这几个数就行 ----------
BATCH = 64        # 一次喂多少张图
EPOCHS = 5        # 整个训练集看几遍
LR = 0.001        # 学习率：每一步走多大

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ---------- 1. 数据 ----------
# ToTensor：把 0–255 的像素变成 0–1 的张量
# Normalize：减均值除标准差，让输入在 0 附近，训练更稳
tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_ds = datasets.MNIST(root=DATA, train=True, download=True, transform=tf)
test_ds = datasets.MNIST(root=DATA, train=False, download=True, transform=tf)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)

# ---------- 2. 模型 ----------
# Flatten：28x28 的图拉平成 784 个数
# Linear(784,256)：第一层，784 个输入 → 256 个隐藏神经元
# ReLU：激活函数，负数变 0（没有它，两层叠起来还是线性的）
# Linear(256,10)：第二层，256 → 10，对应 0–9 十个类别
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

loss_fn = nn.CrossEntropyLoss()                                  # 多分类的标准损失
opt = torch.optim.Adam(model.parameters(), lr=LR)                # 优化器：负责更新权重


def evaluate():
    """在整个测试集上算平均 loss 和准确率。"""
    model.eval()                       # 切到评估模式
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():              # 评估不需要算梯度，省内存
        for x, y in test_loader:
            out = model(x)
            loss_sum += loss_fn(out, y).item() * y.size(0)
            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, correct / total


# ---------- 3. 训练 ----------
print(f"训练集 {len(train_ds)} 张，测试集 {len(test_ds)} 张，共 {EPOCHS} 个 epoch\n")

history = []
for epoch in range(EPOCHS):
    model.train()                      # 切到训练模式
    for x, y in train_loader:
        out = model(x)                 # ① 前向：算预测
        loss = loss_fn(out, y)         # ② 算错得多离谱
        opt.zero_grad()                # ③ 梯度清零（不清会累加）
        loss.backward()                # ④ 反向：算出每个权重该往哪改
        opt.step()                     # ⑤ 更新权重

    test_loss, test_acc = evaluate()
    history.append((epoch + 1, test_loss, test_acc))
    print(f"epoch {epoch+1}/{EPOCHS}   测试 loss {test_loss:.4f}   准确率 {test_acc*100:.2f}%")

print("\n最终测试准确率：%.2f%%" % (history[-1][2] * 100))

# ---------- 4. 画曲线 ----------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h[0] for h in history]
    losses = [h[1] for h in history]
    accs = [h[2] * 100 for h in history]

    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(epochs, losses, "o-", color="#a8342a", label="test loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss", color="#a8342a")
    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, "s--", color="#2f4a5c", label="test acc %")
    ax2.set_ylabel("accuracy (%)", color="#2f4a5c")
    plt.title("MLP on MNIST")
    fig.tight_layout()
    out_png = os.path.join(HERE, "loss_curve.png")
    plt.savefig(out_png, dpi=130)
    print("曲线已保存：", out_png)
except Exception as e:  # 画图失败不影响训练结果
    print("画图跳过：", e)
