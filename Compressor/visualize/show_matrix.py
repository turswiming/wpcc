import matplotlib.pyplot as plt
import numpy as np

def show_matrix(matrix):
    # 创建一个三维图表
    x = np.arange(matrix.shape[0])
    y = np.arange(matrix.shape[1])
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=60, azim=45)
    ax.dist = 0

    # 绘制三维图表
    ax.plot_surface(x, y, matrix.T, cmap='viridis')  # 转置矩阵以匹配 x 和 y 的形状
    plt.show()