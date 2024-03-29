import os
from my_CNN import CNN_
from PIL import Image
import numpy as np
import random


def accuracy_score(y_true, y_pred):
    equal = (y_true == y_pred)
    acc = np.sum(equal) / equal.size
    return acc


def add_noise(matrix):
    rows, cols = matrix.shape
    noisy_matrix = np.copy(matrix)
    indices = np.random.choice(rows * cols, 5, replace=False)
    noisy_matrix.flat[indices] = 0
    indices = np.random.choice(rows * cols, 5, replace=False)
    noisy_matrix.flat[indices] = 255
    return noisy_matrix


def image_translation(image):
    x_offset = np.random.randint(-2, 3)
    y_offset = np.random.randint(-2, 3)
    new_image = image.transform(image.size, Image.AFFINE, (1, 0, x_offset, 0, 1, y_offset))
    return new_image


def random_crop_and_pad(image):
    width, height = image.size
    left = random.randint(0, width - 22)
    top = random.randint(0, height - 22)
    right = left + 22
    bottom = top + 22
    cropped_img = image.crop((left, top, right, bottom))

    new_img = Image.new("L", (28, 28), color=255)
    pad_left = (28 - 22) // 2
    pad_top = (28 - 22) // 2
    new_img.paste(cropped_img, (pad_left, pad_top))

    return new_img


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
        array = np.array(gray_image)
        image_train.append([array/255])

        noised_image = add_noise(np.array(gray_image))
        image_train.append([noised_image/255])

        translated_image = image_translation(gray_image)
        image_train.append([np.array(translated_image)])

        croped_image = random_crop_and_pad(gray_image)
        image_train.append([np.array(croped_image)])

    for j in range(521, 621):
        image_path = os.path.join(subfolder_path, f'{j}.bmp')
        image = Image.open(image_path)
        gray_image = image.convert('L')
        array = np.array(gray_image)
        image_valid.append([array/255])

# 将图像数组组合成一个矩阵


label_train = np.arange(num_label*520*4)//(520*4)
labels_train = np.eye(num_label)[label_train]

label_valid = np.arange(num_label*100)//100
labels_valid = label_valid

batch_size = 16
dropout = 0
inputshape = [28, 28]
layers = [256, 128, num_label]
CNN_net = CNN_(input_shape=inputshape, DNN_layers=layers, batch_size=batch_size, dropout=dropout, X_valid=image_valid, y_valid=labels_valid)

CNN_net.train(image_train, labels_train)
labels_pred = np.argmax(np.array(CNN_net.predict(image_valid)), axis=1)
print("accuracy score = ", accuracy_score(labels_valid, labels_pred))
with open('contrast.txt', 'w') as file:
    file.write(str(labels_pred))
    file.write('\n')
    file.write(str(labels_valid))
