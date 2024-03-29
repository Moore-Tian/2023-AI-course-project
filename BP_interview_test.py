import os
from BP_net import back_propagation
from PIL import Image
import numpy as np
import pickle


def accuracy_score(y_true, y_pred):
    equal = (y_true == y_pred)
    acc = np.sum(equal) / equal.size
    return acc


# 文件夹路径
folder_path = 'test_data/'

# 初始化一个空列表，用于存储图像的一维数组
image_test = []
num_label = 12

# 遍历每个子文件夹
for i in range(1, num_label+1):
    subfolder_path = os.path.join(folder_path, str(i))
    # 遍历每个子文件夹中的BMP图像
    for j in range(1, 241):
        image_path = os.path.join(subfolder_path, f'{j}.bmp')
        image = Image.open(image_path)
        gray_image = image.convert('L')
        array = np.where(np.array(gray_image).flatten() > 1, 0, 1)
        image_test.append(array)

# 将图像数组组合成一个矩阵
images_test = np.vstack(image_test)

labels_test = np.arange(num_label*240)//240

batch_size = 16
dropout = 0.0
layers = [28*28, 256, 128, num_label]
BP_net = back_propagation(layers=layers, batch_size=batch_size, classifacation=True)

i = 1
with open(f'{i}_W.pickle', 'rb') as fw:
    BP_net.Weights = pickle.load(fw)
with open(f'{i}_b.pickle', 'rb') as fb:
    BP_net.Biases = pickle.load(fb)

labels_pred = np.argmax(np.array(BP_net.predict(images_test)), axis=1)
print("accuracy score = ", accuracy_score(labels_test, labels_pred))
