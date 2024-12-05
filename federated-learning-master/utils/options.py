
# 这段代码定义了一个名为args_parser的函数，它使用argparse库来解析命令行参数。
# 函数首先创建了一个ArgumentParser对象，然后使用add_argument方法添加了一系列参数，每个参数都有其类型、默认值和帮助描述。
# 这些参数涵盖了联邦学习设置、模型配置、数据集信息和其他设置。
# 最后，函数解析命令行参数并返回一个包含所有参数的对象。这个对象可以在程序的其他部分被用来访问这些参数。

import argparse  # 导入argparse模块，用于解析命令行参数
def args_parser():  # 定义函数
    # 创建ArgumentParser对象，用于处理命令行参数
    parser = argparse.ArgumentParser()
    # 添加联邦学习相关的命令行参数
    # 训练的轮数，默认为10
    parser.add_argument('--epochs', type=int, default=100, help="rounds of training")
    # 用户的数量，默认为100
    parser.add_argument('--num_users', type=int, default=100, help="number of users: K")
    # 客户端的比例，默认为0.1
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    # 本地训练的轮数，默认为5
    parser.add_argument('--local_ep', type=int, default=10, help="the number of local epochs: E")
    # 本地批次大小，默认为10
    parser.add_argument('--local_bs', type=int, default=20, help="local batch size: B")
    # 测试批次大小，默认为128
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    # 学习率，默认为0.01
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    # SGD动量，默认为0.5
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    # 训练-测试集的分割类型，可以是'user'或'sample'，默认为'user'
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")



    # 添加优化器相关的命令行参数
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer type: sgd, momentum, adagrad, adadelta, adam, rmsprop')
    # parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--rho', type=float, default=0.9, help='Adadelta and RMSprop hyperparameter (default: 0.9)')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adagrad, Adadelta, and Adam epsilon for numerical stability (default: 1e-8)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 hyperparameter (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 hyperparameter (default: 0.999)')
    parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop hyperparameter (default: 0.99)')

    # 添加模型相关的命令行参数
    # 模型名称，默认为'mlp'
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    # 每种卷积核的数量，默认为9
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    # 卷积核大小，以逗号分隔，默认为'3,4,5'
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
    # 归一化方法，可以是'batch_norm'、'layer_norm'或'None'，默认为'batch_norm'
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    # 卷积网络的滤波器数量，默认为32
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    # 是否使用最大池化而不是步幅卷积，默认为'True'
    parser.add_argument('--max_pool', type=str, default='True', help="Whether use max pooling rather than strided convolutions")

    # 添加其他命令行参数
    # 数据集名称，默认为'mnist'
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    # 是否独立同分布，默认为False
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    # 类别数量，默认为10
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    # 图像的通道数，默认为3
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of images")
    # GPU ID，默认为0，如果设置为-1则使用CPU
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    # 提前停止的轮数，默认为10
    parser.add_argument('--stopping_rounds', type=int, default=10, help='rounds of early stopping')
    # 是否打印详细信息，默认为False
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    # 随机种子，默认为1
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    # 是否对所有客户端进行聚合，默认为False
    parser.add_argument('--all_clients', action='store_true', help='aggregation over all clients')

    # 解析命令行参数
    args = parser.parse_args()
    return args
