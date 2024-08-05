import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
x = np.linspace(0, 2 * np.pi, 10)
y = np.sin(x)

# 快速傅里叶变换插值函数
def fft_interpolate(x, y, num_points):
    n = len(y)
    f = np.fft.fft(y)
    f = np.concatenate([f[:n//2], np.zeros(num_points - n), f[n//2:]])
    y_interp = np.fft.ifft(f) * num_points / n
    x_interp = np.linspace(x[0], x[-1], num_points)
    return x_interp, y_interp.real

# 插值
num_points = 20
x_interp, y_interp = fft_interpolate(x, y, num_points)

# 绘制原始数据和插值数据
plt.plot(x, y, 'o', label='Original data')
plt.plot(x_interp, y_interp, '-', label='FFT Interpolated data')
plt.legend()
plt.show()