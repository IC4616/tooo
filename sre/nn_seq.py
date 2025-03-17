
import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter


# 定义一个名为Tudui的神经网络模型类，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的构造函数
        super(Tudui, self).__init__()
        # 定义一个顺序容器Sequential，包含多个卷积层、池化层、展平层和全连接层
        self.model1 = Sequential(
            # 第一层卷积层：输入通道3（RGB图像），输出通道32，卷积核大小5x5，填充2以保持空间维度不变
            Conv2d(3, 32, kernel_size=5, padding=2),
            # 第一层最大池化层：池化核大小2x2，步幅默认为2
            MaxPool2d(kernel_size=2),

            # 第二层卷积层：输入通道32，输出通道32，卷积核大小5x5，填充2
            Conv2d(32, 32, kernel_size=5, padding=2),
            # 第二层最大池化层：池化核大小2x2
            MaxPool2d(kernel_size=2),

            # 第三层卷积层：输入通道32，输出通道64，卷积核大小5x5，填充2
            Conv2d(32, 64, kernel_size=5, padding=2),
            # 第三层最大池化层：池化核大小2x2
            MaxPool2d(kernel_size=2),

            # 展平层：将多维输入展平为一维，以便输入到全连接层
            Flatten(),

            # 第一层全连接层：输入特征数1024，输出特征数64
            Linear(1024, 64),
            # 第二层全连接层：输入特征数64，输出特征数10（通常用于分类任务的类别数）
            Linear(64, 10)
        )

    # 定义前向传播方法，接收输入张量x并返回输出张量
    def forward(self, x):
        # 将输入x通过定义好的模型层
        x = self.model1(x)
        return x


# 实例化Tudui模型
tudui = Tudui()
# 打印模型的结构
print(tudui)

# 创建一个全为1的张量，模拟输入数据，形状为(batch_size=64, channels=3, height=32, width=32)
input_tensor = torch.ones((64, 3, 32, 32))
# 将输入数据通过模型，得到输出
output = tudui(input_tensor)
# 打印输出的形状，预期为(batch_size=64, num_classes=10)
print(output.shape)

# 创建一个SummaryWriter对象，用于记录日志，日志文件将保存在"../logs_seq"目录下
writer = SummaryWriter("../logs_seq")
# 使用add_graph方法将模型结构和输入示例写入TensorBoard日志
writer.add_graph(tudui, input_tensor)
# 关闭SummaryWriter，确保所有日志数据都被正确写入文件
writer.close()