import  torch
# 导入 PyTorch 库，用于深度学习模型的构建和计算
# 导入 torchvision 库，提供数据集、模型架构和图像转换工具
import torchvision
# 从 torch 模块中导入 nn（神经网络）子模块
from torch import nn
# 从 torch.nn 中导入 MaxPool2d，用于实现最大池化层
from torch.nn import MaxPool2d
# 从 torch.utils.data 导入 DataLoader，用于批量加载数据
from torch.utils.data import DataLoader
# 从 torch.utils.tensorboard 导入 SummaryWriter，用于记录训练过程中的各种信息，便于可视化
from torch.utils.tensorboard import SummaryWriter

# 下载并加载 CIFAR-10 测试数据集，路径为 "data"，不下载训练集，将图像转换为张量格式
dataset = torchvision.datasets.CIFAR10("data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

# 使用 DataLoader 批量加载数据，设置每批大小为 64
dataloader = DataLoader(dataset, batch_size=64)

# 定义一个名为 Tudui 的神经网络模型类，继承自 nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类 nn.Module 的构造函数
        super(Tudui, self).__init__()
        # 定义一个最大池化层，池化核大小为 3x3，不使用 ceil_mode（即不采用向上取整的方式填充）
        self.maxpool = MaxPool2d(kernel_size=3, ceil_mode=False)

    # 定义前向传播方法，接收输入张量 input
    def forward(self, input):
        # 将输入通过最大池化层，得到输出张量 output
        output = self.maxpool(input)
        # 返回输出张量
        return output

# 实例化 Tudui 模型
tudui = Tudui()

# 创建一个 SummaryWriter 对象，用于记录日志，日志文件将保存在 "../logs_maxpool" 目录下
writer = SummaryWriter("data/logs_maxpool")
# 初始化步数计数器
step = 0

# 遍历 DataLoader 中的每一个批次的数据
for data in dataloader:
    # 将数据解包为图像 imgs 和对应的标签 targets
    imgs, targets = data
    # 将输入图像写入 TensorBoard，标签为 "input"，当前步数为 step
    writer.add_images("input", imgs, step)
    # 将输入图像通过模型进行前向传播，得到输出 output
    output = tudui(imgs)
    # 将模型的输出图像写入 TensorBoard，标签为 "output"，当前步数为 step
    writer.add_images("output", output, step)
    # 步数计数器加 1
    step += 1

# 关闭 SummaryWriter，确保所有日志数据都被正确写入文件
writer.close()