#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# import copy
# import torch
# from torch import nn
#
#
# def FedAvg(w):
#     w_avg = copy.deepcopy(w[0])
#     for k in w_avg.keys():
#         for i in range(1, len(w)):
#             w_avg[k] += w[i][k]
#         w_avg[k] = torch.div(w_avg[k], len(w))
#     return w_avg


# 这段代码定义了一个FedAvg函数，它接受一个包含多个模型权重的列表w，并返回这些权重的平均值。


import copy
import torch
from torch import nn


# 定义FedAvg函数，用于计算多个模型权重的平均值
def FedAvg(w):
    # 使用deepcopy创建第一个模型权重的深拷贝，以避免修改原始权重
    w_avg = copy.deepcopy(w[0])

    # 遍历第一个模型权重的所有键（即参数名）
    for k in w_avg.keys():
        # 初始化总和为第一个模型的权重值
        total = w_avg[k]

        # 遍历剩余模型的权重
        for i in range(1, len(w)):
            # 累加第i个模型对应参数的权重值
            total += w[i][k]

        # 计算所有模型对应参数的平均值，并更新到w_avg中
        w_avg[k] = torch.div(total, len(w))

    # 返回包含平均权重的字典
    return w_avg

