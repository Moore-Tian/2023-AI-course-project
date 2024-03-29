from torch import nn


# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, num_label=12):
        super().__init__()
        self.net = nn.Sequential(
            # 两层卷积层，一层隐藏层
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_label)
        )

    def forward(self, x):
        x = self.net(x)
        return x


# 使用了残差连接的CNN
class CNN_res(nn.Module):
    def __init__(self, num_label=12):
        super().__init__()
        self.Conv = nn.Sequential(
            # 两层卷积层，一层隐藏层
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
        )

        # 采用平均池化
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.flatten = nn.Flatten()

        # 全连接层的序列模块
        self.DNN = nn.Sequential(

            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_label)
        )

        # 残差块
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(num_features=64)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.Conv(x)
        # 建立残差连接
        out += self.shortcut(x)
        out = self.relu(out)
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.DNN(out)
        return out


# 使用了批归一化后的CNN
class CNN_normalization(nn.Module):
    def __init__(self, num_label=12):
        super().__init__()
        self.net = nn.Sequential(
            # 在每层卷积层中添加一个批归一化层
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_label)
        )

    def forward(self, x):
        x = self.net(x)
        return x


# 使用了dropout的卷积神经网络
class CNN_dropout(nn.Module):
    def __init__(self, num_label=12):
        super().__init__()
        self.net = nn.Sequential(
            # 在隐藏层后添加一个dropout层
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(in_features=128, out_features=num_label)
        )

    def forward(self, x):
        x = self.net(x)
        return x


# 使用了dropout和批归一化的神经网络
class CNN_dropout_batchnorm(nn.Module):
    def __init__(self, num_label=12):
        super().__init__()
        self.net = nn.Sequential(
            # 在每层卷积层中添加一个批归一化层，并在全连接层中添加dropout层
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),


            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128, out_features=num_label)
        )

    def forward(self, x):
        x = self.net(x)
        return x
