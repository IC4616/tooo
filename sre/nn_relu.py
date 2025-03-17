
import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 创建一个2x2的张量，包含正数和负数
input_tensor = torch.tensor([
    [1, -0.5],
    [-1, 3]
], dtype=torch.float32)

# 将输入张量重塑为 (batch_size, channels, height, width) 的形式
# 这里 batch_size=1，channels=1（灰度图像），高度和宽度均为2
input_tensor = torch.reshape(input_tensor, (-1, 1, 2, 2))
# 打印输入张量的形状，输出应为 (1, 1, 2, 2)
print(input_tensor.shape)

# 下载并加载 CIFAR-10 测试数据集，路径为 "../data"，不下载训练集，将图像转换为张量格式
dataset = torchvision.datasets.CIFAR10(
    "../dataset",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

# 使用 DataLoader 批量加载数据，设置每批大小为 64
dataloader = DataLoader(dataset, batch_size=64)

# 定义一个名为 Tudui 的神经网络模型类，继承自 nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类 nn.Module 的构造函数
        super(Tudui, self).__init__()
        # 定义一个 ReLU 激活函数层
        self.relu1 = ReLU()
        # 定义一个 Sigmoid 激活函数层
        self.sigmoid1 = Sigmoid()

    # 定义前向传播方法，接收输入张量 input
    def forward(self, input):
        # 将输入通过 Sigmoid 激活函数层，得到输出张量 output
        output = self.sigmoid1(input)
        # 返回输出张量
        return output

# 实例化 Tudui 模型
tudui = Tudui()

# 创建一个 SummaryWriter 对象，用于记录日志，日志文件将保存在 "../logs_relu" 目录下
writer = SummaryWriter("../logs_relu")
# 初始化步数计数器
step = 0

# 遍历 DataLoader 中的每一个批次的数据
for data in dataloader:
    # 将数据解包为图像 imgs 和对应的标签 targets
    imgs, targets = data
    # 将输入图像写入 TensorBoard，标签为 "input"，当前步数为 step
    writer.add_images("input", imgs, global_step=step)
    # 将输入图像通过模型进行前向传播，得到输出 output
    output = tudui(imgs)
    # 将模型的输出图像写入 TensorBoard，标签为 "output"，当前步数为 step
    writer.add_images("output", output, step)
    # 步数计数器加 1
    step += 1

# 关闭 SummaryWriter，确保所有日志数据都被正确写入文件
writer.close()