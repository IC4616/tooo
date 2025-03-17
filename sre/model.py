
import torch
from torch import nn

# 定义一个名为Tudui的神经网络类，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的构造函数
        super(Tudui, self).__init__()
        # 使用nn.Sequential定义一系列层，构建神经网络的结构
        self.model = nn.Sequential(
            # 第一个卷积层：输入通道数为3（例如RGB图像），输出通道数为32，卷积核大小为5x5，
            # 步幅为1，填充为2，确保输入和输出的空间尺寸相同
            nn.Conv2d(3, 32, 5, 1, 2),
            # 最大池化层，池化核大小为2x2，步幅默认为2，将特征图的高度和宽度减半
            nn.MaxPool2d(2),

            # 第二个卷积层：输入通道数为32，输出通道数为32，卷积核大小为5x5，
            # 步幅为1，填充为2，保持空间尺寸不变
            nn.Conv2d(32, 32, 5, 1, 2),
            # 再次应用最大池化层，进一步减小特征图尺寸
            nn.MaxPool2d(2),

            # 第三个卷积层：输入通道数为32，输出通道数为64，卷积核大小为5x5，
            # 步幅为1，填充为2，保持空间尺寸不变
            nn.Conv2d(32, 64, 5, 1, 2),
            # 第三次应用最大池化层，进一步减小特征图尺寸
            nn.MaxPool2d(2),

            # 将多维的特征图展平为一维向量，以便输入到全连接层
            nn.Flatten(),

            # 第一个全连接层：输入特征数为64 * 4 * 4（经过三次池化后，空间尺寸为4x4，
            # 通道数为64），输出特征数为64
            nn.Linear(64 * 4 * 4, 64),

            # 第二个全连接层：输入特征数为64，输出特征数为10（通常用于分类任务中的10个类别）
            nn.Linear(64, 10)
        )

    def forward(self, x):
        # 将输入数据通过定义好的网络层进行前向传播
        x = self.model(x)
        # 返回最终的输出
        return x


# if __name__ == '__main__':
#     # 创建Tudui类的一个实例
#     tudui = Tudui()
#     # 创建一个输入张量，形状为(64, 3, 32, 32)，表示64个3通道的32x32图像
#     input = torch.ones((64, 3, 32, 32))
#     # 将输入数据传入网络，得到输出
#     output = tudui(input)
#     # 打印输出的形状，验证网络结构是否正确
#     print(output.shape)