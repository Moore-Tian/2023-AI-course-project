import os
from BP_net import back_propagation
from PIL import Image
import numpy as np
import pickle
import matplotlib.pyplot as plt
from utils import *


# 文件夹路径
folder_path = 'train_data/train'

# 初始化一个空列表，用于存储图像的一维数组
image_train = []
image_valid = []
num_label = 12

# 遍历每个子文件夹
for i in range(1, num_label+1):
    subfolder_path = os.path.join(folder_path, str(i))
    # 遍历每个子文件夹中的BMP图像
    for j in range(1, 521):
        image_path = os.path.join(subfolder_path, f'{j}.bmp')
        image = Image.open(image_path)
        gray_image = image.convert('L')
        array = np.where(np.array(gray_image).flatten() > 1, 0, 1)
        image_train.append(array)

        noised_image = np.where(add_noise(np.array(gray_image)).flatten() > 1, 0, 1)
        image_train.append(noised_image)

        translated_image = image_translation(gray_image)
        image_train.append(np.where(np.array(translated_image).flatten() > 1, 0, 1))

        croped_image = random_crop_and_pad(gray_image)
        image_train.append(np.where(np.array(croped_image).flatten() > 1, 0, 1))

    for j in range(521, 621):
        image_path = os.path.join(subfolder_path, f'{j}.bmp')
        image = Image.open(image_path)
        gray_image = image.convert('L')
        array = np.where(np.array(gray_image).flatten() > 1, 0, 1)
        image_valid.append(array)

# 将图像数组组合成一个矩阵
images_train = np.vstack(image_train)
images_valid = np.vstack(image_valid)

label_train = np.arange(num_label*520*4)//(520*4)
labels_train = np.eye(num_label)[label_train]

label_valid = np.arange(num_label*100)//100
labels_valid = label_valid

# 建立BP网络
batch_size = 16
dropout = 0.1
layers = [28*28, 256, 128, num_label]
BP_net = back_propagation(layers=layers, batch_size=batch_size, dropout=dropout, classifacation=True, X_valid=images_valid, y_valid=labels_valid)

# 开始训练
train_accuracy_scores, valid_accuracy_scores, train_loss = BP_net.train(images_train, labels_train)
labels_pred = np.argmax(np.array(BP_net.predict(images_valid)), axis=1)
print("accuracy score = ", accuracy_score(labels_valid, labels_pred))
x = np.arange(len(train_accuracy_scores))

# 训练效果展示
plt.figure(1)
plt.plot(x, train_accuracy_scores, color='red', label='train accuracy')
plt.plot(x, valid_accuracy_scores, color='blue', label='valid accuracy')
plt.xlabel('x')
plt.ylabel('y')
plt.title('accuracy')
plt.grid(True)
plt.show()

plt.figure(2)
plt.plot(x, train_loss, color='red', label='train loss')
plt.xlabel('x')
plt.ylabel('y')
plt.title('loss')
plt.show()

# 模型存储
i = 3
with open(f'{i}_W.pickle', 'wb') as f:
    pickle.dump(BP_net.Weights, f)

with open(f'{i}_b.pickle', 'wb') as f:
    pickle.dump(BP_net.Biases, f)
