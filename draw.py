"""
画图
需要补充 3 个函数：draw_loss()、draw_learning_rate()、draw_decision_boundary()
"""
import numpy as np
import matplotlib.pyplot as plt

from logistic_regression import Logistic_Regression


def draw_loss(loss_list):
    """
    函数 5　画迭代轮数与损失函数的关系图
    功能：横轴为迭代轮数，纵轴为损失函数值，画出损失随迭代轮数变化的曲线
    输入：loss_list：Logistic_Regression() 返回的损失列表
    输出：无（只画图，不调用 plt.show()）
    """
    _, ax = plt.subplots(constrained_layout=True)
    ax.plot(range(1, len(loss_list) + 1), loss_list)
    ax.set(xlabel='迭代轮数', ylabel='对数损失', title='梯度下降损失曲线')
    ax.grid(alpha=0.25)


def draw_learning_rate(x, y, lr_list, num_iter):
    """
    函数 6　画不同学习率下损失函数的变化图
    功能：对 lr_list 中的每个学习率，调用 Logistic_Regression() 各训练 num_iter 轮
          （threshold 取 0，不提前停止），把各自的损失曲线画在同一张图上，用图例标出学习率
    输入：x、y：特征和标签
          lr_list：学习率列表
          num_iter：迭代轮数
    输出：无（只画图，不调用 plt.show()）
    """
    _, ax = plt.subplots(constrained_layout=True)
    for lr in lr_list:
        loss_list, _ = Logistic_Regression(x, y, lr, num_iter, threshold=0)
        ax.plot(range(1, len(loss_list) + 1), loss_list, label=f'学习率 {lr:g}')
    ax.set(xlabel='迭代轮数', ylabel='对数损失', title='不同学习率的损失曲线')
    ax.legend()
    ax.grid(alpha=0.25)


def draw_decision_boundary(x, y, w):
    """
    函数 7　画分类线
    功能：画出所有同学的散点图（挂科和没挂科用不同颜色、大小区分），
          再画出分类线 w0 + w1 × x1 + w2 × x2 = 0
    输入：x：特征，形状 (m, 2)，第 1 列学习时长，第 2 列刷手机时长
          y：标签，形状 (m,)
          w：Logistic_Regression() 返回的参数 [w0, w1, w2]
    输出：无（只画图，不调用 plt.show()）
    """
    _, ax = plt.subplots(constrained_layout=True)
    passed = y == 0
    failed = y == 1
    ax.scatter(x[passed, 0], x[passed, 1], s=18, color='tab:blue', alpha=0.65, label='没挂科')
    ax.scatter(x[failed, 0], x[failed, 1], s=32, color='tab:orange', alpha=0.65, label='挂科')

    w0, w1, w2 = w
    if abs(w2) >= abs(w1) and w2 != 0:
        study_hours = np.array(ax.get_xlim())
        phone_hours = -(w0 + w1 * study_hours) / w2
        ax.plot(study_hours, phone_hours, color='tab:red', label='分类线')
    elif w1 != 0:
        phone_hours = np.array(ax.get_ylim())
        study_hours = -(w0 + w2 * phone_hours) / w1
        ax.plot(study_hours, phone_hours, color='tab:red', label='分类线')

    ax.set(xlabel='每周课后学习时长（小时）', ylabel='每天刷手机时长（小时）',
           title='挂科预警分类线')
    ax.legend()
