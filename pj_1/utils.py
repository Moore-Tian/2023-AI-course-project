import numpy as np
import random
from PIL import Image


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


# 生成参数
def sin(x):
    y = np.sin(x)
    return y


def create_data(func, interval, sample_num, noise=0.0, add_outlier=False, outlier_ratio=0.001):
    X = np.random.rand(sample_num, 1) * (interval[1]-interval[0]) + interval[0]
    y = func(X)

    epsilon = np.random.normal(0, noise, (sample_num, 1))
    y = y + epsilon

    if add_outlier:
        outlier_num = int(sample_num * outlier_ratio)
        if outlier_num != 0:
            outlier_idx = np.random.randint(sample_num, size=[outlier_num, 1])
            y[outlier_idx] = y[outlier_idx] * 5
    return X, y


def mean_squared_error(y_true, y_pred):
    error = -1
    error = np.mean(abs(y_true - y_pred))
    return error