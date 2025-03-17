
import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

# 下载并加载CIFAR-10测试数据集
# 参数说明：
# - "../data": 数据集保存的路径
# - train=False: 指定为测试集
# - transform=torchvision.transforms.ToTensor(): 将图像转换为张量格式
# - download=True: 如果数据集不存在，则自动下载
dataset = torchvision.datasets.CIFAR10(
    "../dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 使用DataLoader批量加载数据
# 参数说明：
# - dataset: 已加载的数据集
# - batch_size=64: 每个批次包含64个样本
# - drop_last=True: 如果最后一个批次不满64个样本，则丢弃该批次
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


# 定义一个名为Tudui的神经网络模型类，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的构造函数
        super(Tudui, self).__init__()
        # 定义一个全连接层，输入特征数为196608，输出特征数为10
        # 注意：这里的输入特征数需要根据实际输入数据的形状进行调整
        self.linear1 = Linear(3072, 10)

    # 定义前向传播方法，接收输入张量并返回输出张量
    def forward(self, input):
        output = self.linear1(input)  # 将输入通过全连接层
        return output


# 实例化Tudui模型
tudui = Tudui()

# 遍历DataLoader中的每一个批次的数据
for data in dataloader:
    imgs, targets = data  # 将数据解包为图像和对应的标签
    print("输入图像的形状:", imgs.shape)  # 打印输入图像的形状，通常为 (batch_size, channels, height, width)

    # 将图像展平成一维向量，以适应全连接层的输入要求
    # 参数说明：
    # - imgs: 输入的图像张量
    # - start_dim=1: 从第1维开始展平（保留batch维度）
    output = torch.flatten(imgs, start_dim=1)
    print("展平后的图像形状:", output.shape)  # 打印展平后的图像形状

    # 将展平后的图像通过模型进行前向传播，得到输出
    output = tudui(output)
    print("模型输出的形状:", output.shape)  # 打印模型输出的形状

