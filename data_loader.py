import os
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Pre_processes():
    def image_normalize(image):
        return np.where(np.array(image) > 0, 0, 1)

    # 随机噪声
    def add_noise(image):
        image = np.array(image)
        rows, cols = image.shape
        noisy_image = np.copy(image)
        indices = np.random.choice(rows * cols, 5, replace=False)
        noisy_image.flat[indices] = 0
        indices = np.random.choice(rows * cols, 5, replace=False)
        noisy_image.flat[indices] = 255
        return np.where(noisy_image > 0, 0, 1)

    # 图像平移
    def image_translation(image):
        x_offset = np.random.randint(-2, 3)
        y_offset = np.random.randint(-2, 3)
        new_image = image.transform(image.size, Image.AFFINE, (1, 0, x_offset, 0, 1, y_offset))
        return np.where(np.array(new_image) > 0, 0, 1)

    # 图像裁剪
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

        return np.where(np.array(new_img) > 0, 0, 1)


class My_dataset(Dataset):
    # data augmentation
    methods = [Pre_processes.image_normalize, Pre_processes.image_translation, Pre_processes.random_crop_and_pad, Pre_processes.add_noise]

    def __init__(self, root_dir, labels, num_label=12):
        self.root_dir = root_dir
        self.labels = labels
        self.num_label = num_label

    def __len__(self):
        return 620 * len(My_dataset.methods) * self.num_label

    def __getitem__(self, index):
        method_class = index % len(My_dataset.methods)
        method = My_dataset.methods[method_class]

        folder_index = (index // len(My_dataset.methods)) // 620 + 1
        file_index = (index // len(My_dataset.methods)) % 620 + 1

        image_path = os.path.join(self.root_dir, str(folder_index), f"{file_index}.bmp")
        image = Image.open(image_path)
        size = image.size
        image = method(image)
        image = torch.tensor(image).reshape(1, size[0], size[1])

        label = self.labels[index // len(My_dataset.methods)]
        label_vector = torch.zeros(self.num_label)
        label_vector[label] = 1
        return image, label_vector
