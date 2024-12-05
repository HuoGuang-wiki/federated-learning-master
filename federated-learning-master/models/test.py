


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# 定义一个函数来测试模型在测试集上的性能
def test_img(net_g, datatest, args):
    # 将模型设置为评估模式
    net_g.eval()

    # 初始化测试损失和正确预测的数量
    test_loss = 0
    correct = 0

    # 创建数据加载器，用于批量加载测试数据
    data_loader = DataLoader(datatest, batch_size=args.bs)

    # 获取数据加载器中的批次总数
    l = len(data_loader)

    # 遍历每个批次的数据
    for idx, (data, target) in enumerate(data_loader):
        # 如果使用GPU，则将数据和目标转移到GPU
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()

        # 前向传播，获取模型的预测结果
        log_probs = net_g(data)

        # 计算批次损失，并累加到总损失中
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

        # 获取预测结果中概率最大的类别
        y_pred = log_probs.data.max(1, keepdim=True)[1]

        # 更新正确预测的数量
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    # 计算平均损失
    test_loss /= len(data_loader.dataset)

    # 计算准确率
    accuracy = 100.00 * correct / len(data_loader.dataset)

    # 如果设置了verbose参数，则打印测试结果
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))

    # 返回准确率和平均损失
    return accuracy, test_loss

