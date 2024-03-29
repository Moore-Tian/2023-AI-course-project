import matplotlib.pyplot as plt
import torch
import numpy as np
from data_loader import My_dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import nn
from torch_CNN import CNN_dropout
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# get root_dir
root_dir = "./train_data/train/"
num_label = 12
labels = np.arange(num_label*620)//(620)

# 准备训练集和验证集
data_set = My_dataset(root_dir, labels)
num_image = len(data_set)

train_part = 0.9
num_train = int(num_image * train_part)
num_valid = num_image - num_train
train_set, valid_set = random_split(data_set, [num_train, num_valid])

train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=16, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

i = 2
model_path = f"./{i}CNN_model.pth"
# 准备模型
CNN_net = CNN_dropout().to(device)
loss_fn = nn.CrossEntropyLoss()
learning_rate = 0.001
optimizer = torch.optim.Adam(CNN_net.parameters(), lr=learning_rate, weight_decay=0.001)


epoch = 20
train_loss = []
valid_loss = []
train_accuracy = []
valid_accuracy = []

for i in range(epoch):
    print("epoch:", i)
    CNN_net.train()
    train_loss_value = 0.0
    # 开始分批次加载图片和标签
    for images, labels in train_loader:
        # 加载图片和标签到gpu
        images = images.to(device)
        images = images.to(torch.float)
        labels = labels.to(device)
        labels = labels.to(torch.float)

        # 将优化器中所有参数归零并完成前向传播
        optimizer.zero_grad()
        output = CNN_net(images)

        # 累积损失函数值并进行反向传播
        loss = loss_fn(output, labels)
        train_loss_value += loss.item()
        loss.backward()
        optimizer.step()

    train_loss_value /= len(train_loader)
    train_loss.append(train_loss_value)

    # 模型转为评估模式
    CNN_net.eval()
    valid_loss_value = 0
    train_loss_value = 0
    valid_correct = 0
    train_correct = 0
    # 评估模式下禁用梯度计算
    with torch.no_grad():
        # 分批次加载验证图片和标签
        for images, labels in valid_loader:
            images = images.to(device)
            images = images.to(torch.float)
            labels = labels.to(device)
            labels = labels.to(torch.float)

            probability_pred = CNN_net(images)
            valid_correct += (probability_pred.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
            valid_loss_value += loss_fn(probability_pred, labels).item()

    valid_loss_value /= len(valid_loader)
    valid_loss.append(valid_loss_value)

    valid_accuracy_value = valid_correct / len(valid_set)
    valid_accuracy.append(valid_accuracy_value)

x = np.arange(epoch)
plt.figure(1)
plt.plot(x, valid_accuracy, color='blue', label='valid accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('accuracy')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(2)
plt.plot(x, valid_loss, color='blue', label='valid loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss')
plt.grid(True)
plt.legend()
plt.show()

torch.save(CNN_net.state_dict(), model_path)
