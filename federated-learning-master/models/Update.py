


# 这段代码定义了两个类：DatasetSplit和LocalUpdate。
# DatasetSplit类用于从原始数据集中根据索引划分出一部分数据，而LocalUpdate类则用于在客户端上进行模型的本地训练和更新。
# LocalUpdate类中的train方法负责执行训练过程，包括前向传播、损失计算、反向传播和参数更新，并返回更新后的模型参数和平均损失。
import torch
from torch import nn, autograd
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics


# 定义一个用于数据划分的Dataset类
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        """
        初始化DatasetSplit类。

        参数:
        dataset (Dataset): 原始数据集。
        idxs (list): 选中用于划分的数据索引。
        """
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        """
        返回划分后数据集的大小。
        """
        return len(self.idxs)

    def __getitem__(self, item):
        """
        根据item索引获取数据和标签。

        参数:
        item (int): 数据索引。

        返回:
        image (Tensor): 数据图像。
        label (Tensor): 标签。
        """
        image, label = self.dataset[self.idxs[item]]
        return image, label


# 定义一个用于客户端本地更新的类
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        """
        初始化LocalUpdate类。

        参数:
        args (object): 包含训练参数的对象。
        dataset (Dataset): 原始数据集。
        idxs (list): 选中用于训练的数据索引。
        """
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # 创建一个DataLoader，用于本地训练数据的批量加载
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)



    def train(self, net):
        """
        在本地客户端上训练模型。

        参数:
        net (nn.Module): 要训练的模型。

        返回:
        net_params (dict): 更新后的模型参数。
        loss_avg (float): 本地训练的平均损失。
        """
        net.train()
        # 选择优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        # optimizer = torch.optim.Adagrad(net.parameters(), lr=self.args.lr)
        # optimizer = torch.optim.Adadelta(net.parameters(), rho=self.args.rho, eps=self.args.eps)
        # optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr, betas=(self.args.beta1, self.args.beta2), eps=self.args.eps)
        # optimizer = torch.optim.RMSprop(net.parameters(), lr=self.args.lr, alpha=self.args.alpha, eps=self.args.eps)

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

