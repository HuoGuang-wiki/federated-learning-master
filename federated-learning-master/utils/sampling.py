#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


# import numpy as np
# from torchvision import datasets, transforms
#
# def mnist_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users
#
#
# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.targets.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
#     idxs = idxs_labels[0,:]
#
#     # divide and assign
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users
#
#
# def cifar_iid(dataset, num_users):
#     """
#     Sample I.I.D. client data from CIFAR10 dataset
#     :param dataset:
#     :param num_users:
#     :return: dict of image index
#     """
#     num_items = int(len(dataset)/num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users
#
#
# if __name__ == '__main__':
#     dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,), (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)

# 。


import numpy as np
from torchvision import datasets, transforms

# 定义一个函数，用于从MNIST数据集中随机采样I.I.D.（独立同分布）客户端数据
def mnist_iid(dataset, num_users):
    """
    从MNIST数据集中采样I.I.D.客户端数据
    :param dataset: 要采样的数据集
    :param num_users: 客户端数量
    :return: 包含每个客户端图像索引的字典
    """
    # 计算每个客户端的数据项数
    num_items = int(len(dataset)/num_users)
    # 初始化客户端字典和所有索引列表
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    # 为每个客户端随机分配数据项
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 更新剩余的索引列表
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

# 定义一个函数，用于从MNIST数据集中随机采样非I.I.D.客户端数据
def mnist_noniid(dataset, num_users):
    """
    从MNIST数据集中采样非I.I.D客户端数据
    :param dataset: 要采样的数据集
    :param num_users: 客户端数量
    :return: 包含每个客户端图像索引的字典
    """
    num_shards, num_imgs = 200, 300  # 设置分片数和每个分片的图像数
    idx_shard = [i for i in range(num_shards)]  # 初始化分片索引列表
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 初始化客户端字典
    idxs = np.arange(num_shards*num_imgs)  # 生成所有索引
    labels = dataset.targets.numpy()  # 获取训练标签

    # 根据标签排序索引
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # 为每个客户端分配数据
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # 随机选择两个分片
        idx_shard = list(set(idx_shard) - rand_set)  # 更新分片列表
        for rand in rand_set:
            # 将选中的分片数据分配给客户端
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users

# 定义一个函数，用于从CIFAR-10数据集中随机采样I.I.D.客户端数据
def cifar_iid(dataset, num_users):
    """
    从CIFAR10数据集中采样I.I.D.客户端数据
    :param dataset: 要采样的数据集
    :param num_users: 客户端数量
    :return: 包含每个客户端图像索引的字典
    """
    num_items = int(len(dataset)/num_users)  # 计算每个客户端的数据项数
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]  # 初始化客户端字典和所有索引列表
    # 为每个客户端随机分配数据项
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        # 更新剩余的索引列表
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

if __name__ == '__main__':
    # 加载MNIST训练数据集，并应用数据转换操作
    dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
    num = 100  # 设置客户端数量
    d = mnist_noniid(dataset_train, num)  # 采样非I.I.D.客户端数据
