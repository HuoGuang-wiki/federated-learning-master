# 指定使用Python 3.6版本，并设置编码为utf-8
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# 这段代码实现了一个联邦学习框架，用于在多个客户端上分布式训练图像分类模型。它首先解析命令行参数以配置训练环境，然后根据指定的数据集（MNIST或CIFAR-10）加载和预处理数据。
# 接着，代码构建了一个全局模型，可以是多层感知机（MLP）或卷积神经网络（CNN），并根据是否为独立同分布（IID）数据采样来划分数据。
# 在训练过程中，代码随机选择部分客户端进行本地模型训练，然后聚合这些本地更新以更新全局模型。
# 最后，代码评估并打印全局模型在训练集和测试集上的性能，并绘制训练损失曲线。整个流程旨在在保护用户数据隐私的同时训练出一个有效的全局模型。


# 导入matplotlib库，并设置为不显示图形界面，只用于生成图片
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 导入其他必要的库
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

# 导入自定义的工具函数和模型定义
from utils.sampling import mnist_iid, mnist_noniid, cifar_iid  # 数据采样函数
from utils.options import args_parser  # 参数解析函数
from models.Update import LocalUpdate  # 客户端本地更新类
from models.Nets import MLP, CNNMnist, CNNCifar  # 模型定义
from models.Fed import FedAvg  # 联邦平均聚合类
from models.test import test_img  # 测试函数

if __name__ == '__main__':
    # 解析命令行参数
    args = args_parser()
    # 设置设备，优先使用GPU，如果没有指定GPU或没有GPU，则使用CPU
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 根据选择的数据集加载训练和测试数据集
    if args.dataset == 'mnist':
        # 定义MNIST数据集的转换操作
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # 加载MNIST训练和测试数据集
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # 根据是否是IID数据采样，选择不同的采样函数
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        # 定义CIFAR-10数据集的转换操作
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # 加载CIFAR-10训练和测试数据集
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        # 对于CIFAR-10，只考虑IID设置
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    # 获取图像尺寸
    img_size = dataset_train[0][0].shape

    # 根据模型和数据集构建全局模型
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        # 计算输入层维度
        len_in = 1
        for x in img_size:
            len_in *= x
        # 构建MLP模型
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    # 打印模型结构
    print(net_glob)
    # 设置模型为训练模式
    net_glob.train()

    # 复制全局模型权重
    w_glob = net_glob.state_dict()

    # 初始化训练过程中的变量
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    test_acc_list = []

    # 如果考虑所有客户端
    if args.all_clients:
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    # 进行多个epoch的训练
    for iter in range(args.epochs):
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        # 随机选择部分客户端进行训练
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            # 创建LocalUpdate对象，进行本地训练
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            # 训练并获取本地模型权重和损失
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(w)
            else:
                w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        # 使用FedAvg聚合本地模型权重
        w_glob = FedAvg(w_locals)

        # 将聚合后的权重加载到全局模型
        net_glob.load_state_dict(w_glob)

        # 打印平均损失
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)








    # # 绘制并保存训练损失曲线
    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

        # 在每个epoch结束后测试模型性能
        acc_test, _ = test_img(net_glob, dataset_test, args)
        test_acc_list.append(acc_test)

    # 绘制并保存训练损失曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), loss_train, marker='o')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 绘制测试准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), test_acc_list, marker='o', color='orange')
    plt.title('Testing Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.tight_layout()  # 调整子图间距
    plt.savefig('./save/fed_training_results.png')

    # 在训练集和测试集上进行测试
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # 打印训练和测试准确率
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
