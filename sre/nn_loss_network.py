
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
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
# - batch_size=1: 每个批次包含1个样本
dataloader = DataLoader(dataset, batch_size=1)


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

# 定义损失函数为交叉熵损失，常用于分类任务
loss = nn.CrossEntropyLoss()

# 遍历DataLoader中的每一个批次的数据
for data in dataloader:
    # 将数据解包为图像和对应的标签
    imgs, targets = data

    # 将输入图像通过模型，得到预测输出
    outputs = tudui(imgs)

    # 计算预测输出与真实标签之间的交叉熵损失
    result_loss = loss(outputs, targets)

    # 打印"ok"表示当前批次处理完成
    print("ok")

    # 可选：打印损失值以监控模型的表现
    # print(f"Loss: {result_loss.item()}")