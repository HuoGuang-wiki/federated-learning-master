# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # Python version: 3.6
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# import torch.optim as optim
# from torchvision import datasets, transforms
# from utils.options import args_parser
# from models.Nets import MLP, CNNMnist, CNNCifar
#
# # 定义了一个测试函数test，它接受一个网络模型和一个数据加载器作为参数。函数用于评估模型在测试集上的性能，计算平均损失和准确率，并打印结果。
# def test(net_g, data_loader):
#     # testing
#     net_g.eval()
#     test_loss = 0
#     correct = 0
#     l = len(data_loader)
#     for idx, (data, target) in enumerate(data_loader):
#         data, target = data.to(args.device), target.to(args.device)
#         log_probs = net_g(data)
#         test_loss += F.cross_entropy(log_probs, target).item()
#         y_pred = log_probs.data.max(1, keepdim=True)[1]
#         correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
#
#     test_loss /= len(data_loader.dataset)
#     print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
#         test_loss, correct, len(data_loader.dataset),
#         100. * correct / len(data_loader.dataset)))
#
#     return correct, test_loss
#
#
# if __name__ == '__main__':
#     # parse args
#     args = args_parser()
#     args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
#
#     torch.manual_seed(args.seed)
#
#     # load dataset and split users
#     if args.dataset == 'mnist':
#         dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))
#         img_size = dataset_train[0][0].shape
#     elif args.dataset == 'cifar':
#         transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)
#         img_size = dataset_train[0][0].shape
#     else:
#         exit('Error: unrecognized dataset')
#
#     # build model
#     if args.model == 'cnn' and args.dataset == 'cifar':
#         net_glob = CNNCifar(args=args).to(args.device)
#     elif args.model == 'cnn' and args.dataset == 'mnist':
#         net_glob = CNNMnist(args=args).to(args.device)
#     elif args.model == 'mlp':
#         len_in = 1
#         for x in img_size:
#             len_in *= x
#         net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)
#     else:
#         exit('Error: unrecognized model')
#     print(net_glob)
#
#     # training
#     optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
#     train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)
#
#     list_loss = []
#     net_glob.train()
#     for epoch in range(args.epochs):
#         batch_loss = []
#         for batch_idx, (data, target) in enumerate(train_loader):
#             data, target = data.to(args.device), target.to(args.device)
#             optimizer.zero_grad()
#             output = net_glob(data)
#             loss = F.cross_entropy(output, target)
#             loss.backward()
#             optimizer.step()
#             if batch_idx % 50 == 0:
#                 print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                     epoch, batch_idx * len(data), len(train_loader.dataset),
#                            100. * batch_idx / len(train_loader), loss.item()))
#             batch_loss.append(loss.item())
#         loss_avg = sum(batch_loss)/len(batch_loss)
#         print('\nTrain loss:', loss_avg)
#         list_loss.append(loss_avg)
#
#     # plot loss
#     plt.figure()
#     plt.plot(range(len(list_loss)), list_loss)
#     plt.xlabel('epochs')
#     plt.ylabel('train loss')
#     plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))
#
#     # testing
#     if args.dataset == 'mnist':
#         dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.1307,), (0.3081,))
#                    ]))
#         test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
#     elif args.dataset == 'cifar':
#         transform = transforms.Compose(
#             [transforms.ToTensor(),
#              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#         dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None, download=True)
#         test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
#     else:
#         exit('Error: unrecognized dataset')
#
#     print('test on', len(dataset_test), 'samples')
#     test_acc, test_loss = test(net_glob, test_loader)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import matplotlib
import os
matplotlib.use('Agg')  # 设置matplotlib的后端为Agg，用于生成图片而不显示
import matplotlib.pyplot as plt  # 导入matplotlib的pyplot模块，用于绘图
import torch  # 导入PyTorch库
import torch.nn.functional as F  # 导入PyTorch的函数库
from torch.utils.data import DataLoader  # 导入PyTorch的数据加载器
import torch.optim as optim  # 导入PyTorch的优化器模块
from torchvision import datasets, transforms  # 从torchvision导入数据集和变换操作
from utils.options import args_parser  # 从utils模块导入参数解析函数
from models.Nets import MLP, CNNMnist, CNNCifar  # 从models模块导入定义的网络模型

# 定义了一个测试函数test，它接受一个网络模型和一个数据加载器作为参数。函数用于评估模型在测试集上的性能，计算平均损失和准确率，并打印结果。
def test(net_g, data_loader):
    # testing
    net_g.eval()  # 将模型设置为评估模式
    test_loss = 0  # 初始化测试损失
    correct = 0  # 初始化正确预测的数量
    l = len(data_loader)  # 获取数据加载器中批次的数量
    for idx, (data, target) in enumerate(data_loader):  # 遍历数据加载器中的每个批次
        data, target = data.to(args.device), target.to(args.device)  # 将数据和目标转移到正确的设备
        log_probs = net_g(data)  # 通过模型获取对数概率
        test_loss += F.cross_entropy(log_probs, target).item()  # 计算交叉熵损失并累加
        y_pred = log_probs.data.max(1, keepdim=True)[1]  # 获取预测结果
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()  # 计算正确预测的数量

    test_loss /= len(data_loader.dataset)  # 计算平均损失
    print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))  # 打印测试结果

    return correct, test_loss  # 返回正确预测的数量和平均损失

if __name__ == '__main__':
    # parse args
    args = args_parser()  # 解析命令行参数
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')  # 设置设备为GPU或CPU

    torch.manual_seed(args.seed)  # 设置随机种子

    # Create log directory if it does not exist
    log_dir = './log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 创建目录

    # load dataset and split users
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))  # 加载MNIST训练集
        img_size = dataset_train[0][0].shape  # 获取图像尺寸
    elif args.dataset == 'cifar':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # 定义CIFAR数据的变换
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, transform=transform, target_transform=None, download=True)  # 加载CIFAR10训练集
        img_size = dataset_train[0][0].shape  # 获取图像尺寸
    else:
        exit('Error: unrecognized dataset')  # 如果数据集不是mnist或cifar，退出程序

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)  # 构建CNN模型用于CIFAR数据集
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)  # 构建CNN模型用于MNIST数据集
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=args.num_classes).to(args.device)  # 构建MLP模型
    else:
        exit('Error: unrecognized model')  # 如果模型不是cnn或mlp，退出程序
    print(net_glob)  # 打印模型结构

    # training
    optimizer = optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)  # 设置优化器
    train_loader = DataLoader(dataset_train, batch_size=64, shuffle=True)  # 创建数据加载器

    list_loss = []  # 初始化损失列表
    list_acc = []  #初始化准确率列表
    net_glob.train()  # 将模型设置为训练模式
    # 遍历指定的训练周期次数
    for epoch in range(args.epochs):
        batch_loss = []  # 初始化一个列表，用于存储每个批次的损失值
        batch_correct = 0  # 初始化一个计数器，用于记录整个批次中预测正确的样本数

        # 遍历数据加载器，获取批次数据和目标标签
        for batch_idx, (data, target) in enumerate(train_loader):
            # 将数据和目标标签移动到指定的设备（如GPU）
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()  # 清空梯度
            output = net_glob(data)  # 前向传播，获取模型输出
            loss = F.cross_entropy(output, target)  # 计算交叉熵损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数
            batch_loss.append(loss.item())  # 将损失值添加到列表中
            pred = output.max(1, keepdim=True)[1]  # 获取预测结果
            batch_correct += pred.eq(target.view_as(pred)).sum().item()  # 更新预测正确的样本数

        # 计算平均损失和准确率
        loss_avg = sum(batch_loss) / len(batch_loss)
        acc = batch_correct / len(train_loader.dataset)
        list_loss.append(loss_avg)  # 将平均损失添加到列表中
        list_acc.append(acc)  # 将准确率添加到列表中
        print('Train Epoch: {} Average loss: {:.4f} Accuracy: {:.2f}%'.format(
            epoch, loss_avg, 100. * acc))  # 打印训练周期、平均损失和准确率

        # 绘制损失和准确率曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)  # 创建一个1行2列的子图，当前是第1个
        plt.plot(range(len(list_loss)), list_loss)  # 绘制损失曲线
        plt.title('Training Loss')  # 设置标题
        plt.xlabel('Epochs')  # 设置x轴标签
        plt.ylabel('Loss')  # 设置y轴标签

        plt.subplot(1, 2, 2)  # 当前是第2个子图
        plt.plot(range(len(list_acc)), list_acc)  # 绘制准确率曲线
        plt.title('Training Accuracy')  # 设置标题
        plt.xlabel('Epochs')  # 设置x轴标签
        plt.ylabel('Accuracy')  # 设置y轴标签

        plt.savefig('./log/nn_{}_{}_{}.png'.format(args.dataset, args.model, args.epochs))  # 保存图像
        plt.close()  # 关闭图像，释放资源

        # 测试模型
        if args.dataset == 'mnist':
            # 如果数据集是MNIST，加载测试数据集
            dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
            # 使用transforms.Compose()将多个图像转换操作组合在一起
            # transforms.ToTensor()将PIL图像或NumPy ndarray转换为FloatTensor，并将图像的像素值从[0, 255]归一化到[0.0, 1.0]
            # transforms.Normalize((0.1307,), (0.3081,))对图像进行标准化，即减去均值0.1307，然后除以标准差0.3081
            # 这些操作有助于模型更好地学习特征
            test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
            # 创建一个DataLoader对象，用于在测试时批量加载数据
            # batch_size=1000指定每个批次包含1000个样本
            # shuffle=False表示在测试时不打乱数据顺序
        elif args.dataset == 'cifar':
            # 如果数据集是CIFAR，加载测试数据集
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_test = datasets.CIFAR10('./data/cifar', train=False, transform=transform, target_transform=None,
                                            download=True)
            test_loader = DataLoader(dataset_test, batch_size=1000, shuffle=False)
        else:
            exit('Error: unrecognized dataset')  # 如果数据集不是MNIST或CIFAR，退出程序

        print('test on', len(dataset_test), 'samples')  # 打印测试样本数量
        test_acc, test_loss = test(net_glob, test_loader)  # 调用测试函数，获取测试准确率和损失