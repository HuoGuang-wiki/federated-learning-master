#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# import torch
# from torch import nn
# import torch.nn.functional as F
#
#
# class MLP(nn.Module):
#     def __init__(self, dim_in, dim_hidden, dim_out):
#         super(MLP, self).__init__()
#         self.layer_input = nn.Linear(dim_in, dim_hidden)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.layer_hidden = nn.Linear(dim_hidden, dim_out)
#
#     def forward(self, x):
#         x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
#         x = self.layer_input(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.layer_hidden(x)
#         return x
#
#
# class CNNMnist(nn.Module):
#     def __init__(self, args):
#         super(CNNMnist, self).__init__()
#         self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, args.num_classes)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return x
#
#
# class CNNCifar(nn.Module):
#     def __init__(self, args):
#         super(CNNCifar, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, args.num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = x.view(-1, 16 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



# 这段代码定义了三个神经网络模型：MLP、CNNMnist和CNNCifar。
# 每个模型都继承自nn.Module，并实现了__init__和forward方法。
# __init__方法用于初始化模型的层，而forward方法定义了数据通过网络的前向传播过程。
# 这些模型可以用于不同的任务，例如分类MNIST或CIFAR数据集中的图像。


import torch
from torch import nn
import torch.nn.functional as F

# 定义一个多层感知机（MLP）模型
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        # 定义输入层到隐藏层的线性变换
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        # 定义Dropout层，用于防止过拟合
        self.dropout = nn.Dropout()
        # 定义隐藏层到输出层的线性变换
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        # 将输入x展平为一维向量
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        # 应用输入层到隐藏层的线性变换
        x = self.layer_input(x)
        # 应用Dropout
        x = self.dropout(x)
        # 应用ReLU激活函数
        x = self.relu(x)
        # 应用隐藏层到输出层的线性变换
        x = self.layer_hidden(x)
        return x

# 定义一个针对MNIST数据集优化的CNN模型
class CNNMnist(nn.Module):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        # 定义第一个卷积层，输入通道数为args.num_channels，输出通道数为10
        self.conv1 = nn.Conv2d(args.num_channels, 10, kernel_size=5)
        # 定义第二个卷积层，输入通道数为10，输出通道数为20
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # 定义Dropout2d层，用于防止过拟合
        self.conv2_drop = nn.Dropout2d()
        # 定义第一个全连接层，输入特征数为320，输出特征数为50
        self.fc1 = nn.Linear(320, 50)
        # 定义第二个全连接层，输入特征数为50，输出特征数为args.num_classes
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        # 应用第一个卷积层和ReLU激活函数，然后进行最大池化
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # 应用第二个卷积层，然后进行Dropout和ReLU激活，再进行最大池化
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # 将特征图展平为一维向量
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        # 应用第一个全连接层和ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用Dropout
        x = F.dropout(x, training=self.training)
        # 应用第二个全连接层
        x = self.fc2(x)
        return x

# 定义一个针对CIFAR数据集优化的CNN模型
class CNNCifar(nn.Module):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        # 定义第一个卷积层，输入通道数为3，输出通道数为6
        self.conv1 = nn.Conv2d(3, 6, 5)
        # 定义最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义第二个卷积层，输入通道数为6，输出通道数为16
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义第一个全连接层，输入特征数为16*5*5，输出特征数为120
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 定义第二个全连接层，输入特征数为120，输出特征数为84
        self.fc2 = nn.Linear(120, 84)
        # 定义第三个全连接层，输入特征数为84，输出特征数为args.num_classes
        self.fc3 = nn.Linear(84, args.num_classes)

    def forward(self, x):
        # 应用第一个卷积层和ReLU激活函数，然后进行最大池化
        x = self.pool(F.relu(self.conv1(x)))
        # 应用第二个卷积层和ReLU激活函数，然后进行最大池化
        x = self.pool(F.relu(self.conv2(x)))
        # 将特征图展平为一维向量
        x = x.view(-1, 16 * 5 * 5)
        # 应用第一个全连接层和ReLU激活函数
        x = F.relu(self.fc1(x))
        # 应用第二个全连接层和ReLU激活函数
        x = F.relu(self.fc2(x))
        # 应用第三个全连接层
        x = self.fc3(x)
        return x
