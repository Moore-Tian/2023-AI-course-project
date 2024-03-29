from BP_net import back_propagation
import numpy as np
import matplotlib.pyplot as plt
import math
from utils import *


func = sin
interval = (-math.pi, math.pi)
train_num = 6000
test_num = 20
noise = 0.01
X_train, y_train = create_data(func=func, interval=interval, sample_num=train_num, noise=noise)
X_test, y_test = create_data(func=func, interval=interval, sample_num=test_num, noise=noise)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)


batch_size = 16
layers = [1, 16, 32, 1]
BP_net = back_propagation(layers=layers, batch_size=batch_size)
BP_net.train(X_train, y_train)
y_pred = BP_net.predict(X_test)
print("test error = ", mean_squared_error(y_pred, y_test))

# 生成 x 值的范围
x = np.linspace(-np.pi, np.pi, 300)  # 在 -π 到 π 的范围内生成 100 个均匀分布的点

# 计算对应的 y 值
y = np.squeeze(BP_net.predict(x[:, np.newaxis]))

vectorized_test = np.vectorize(test)

X_train = np.sort(X_train, axis=0)
y_train = np.sin(X_train)

# 绘制图像
plt.scatter(x, y, label='predict', s=2)
plt.plot(X_train, y_train, label='target')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graph of my_function')
plt.grid(True)
plt.show()
