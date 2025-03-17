from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
# 下载并加载CIFAR10测试数据集，将图像转换为Tensor格式
dataset = torchvision.datasets.CIFAR10(
    "../data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 使用DataLoader创建一个数据加载器，批量大小为64
dataloader = DataLoader(dataset, batch_size=64)


# 定义一个名为Tudui的自定义神经网络模块，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的构造函数
        super(Tudui, self).__init__()
        # 定义第一个卷积层，输入通道为3（RGB图像），输出通道为6，卷积核大小为3x3，
        # 步幅为1，不使用填充（padding=0）
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    # 定义前向传播方法，接收输入张量并返回处理后的输出张量
    def forward(self, x):
        # 将输入x通过第一个卷积层
        x = self.conv1(x)
        return x


# 实例化Tudui类的对象
tudui = Tudui()

# 创建一个SummaryWriter对象，用于将训练过程中的数据写入TensorBoard日志文件
writer = SummaryWriter("../../logs")

# 初始化步数计数器
step = 0
# 遍历DataLoader加载的数据批次
for data in dataloader:
    # 将数据批次解包为图像和标签
    imgs, targets = data
    # 将图像输入到模型tudui中，得到输出
    output = tudui(imgs)

    # 打印输入图像的形状，通常为[批量大小, 通道数, 高度, 宽度]
    print("输入图像的形状:", imgs.shape)  # 例如: torch.Size([64, 3, 32, 32])

    # 打印输出的形状，经过卷积层后，高度和宽度会减少
    print("输出的形状:", output.shape)  # 例如: torch.Size([64, 6, 30, 30])

    # 将输入图像添加到TensorBoard，标签为"input"，当前步数为step
    writer.add_images("input", imgs, step)

    # 由于卷积操作后输出的通道数为6，而原始图像只有3个通道，

    output = torch.reshape(output, (-1, 3, 30, 30))
    # 正确的做法是直接将6通道的输出添加到TensorBoard，
    # 或者选择一个特定的通道进行可视化。以下是直接添加6通道输出的示例：
    writer.add_images("output", output, step)
    # 增加步数计数器
    step += 1

# 关闭SummaryWriter，确保所有数据都被写入日志文件
writer.close()