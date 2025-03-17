
import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.optim.lr_scheduler import StepLR  # 导入学习率调度器
from torch.optim import SGD  # 明确导入SGD优化器
from torch.utils.data import DataLoader

# 下载并加载CIFAR-10测试数据集
# 参数说明：
# - "../data": 数据集保存的路径
# - train=False: 指定为测试集（此处应为训练集，若要进行训练，请修改为train=True）
# - transform=torchvision.transforms.ToTensor(): 将图像转换为张量格式
# - download=True: 如果数据集不存在，则自动下载
dataset = torchvision.datasets.CIFAR10(
    "../dataset",
    train=False,  # 注意：若要进行训练，请设置为True
    transform=torchvision.transforms.ToTensor(),
    download=True
)

# 使用DataLoader批量加载数据
# 参数说明：
# - dataset: 已加载的数据集
# - batch_size=1: 每个批次包含1个样本（建议增大batch_size以提高训练效率）
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

# 定义优化器为随机梯度下降（SGD），学习率为0.01
optim = SGD(tudui.parameters(), lr=0.01)

# 定义学习率调度器，每30个epoch将学习率乘以0.1（当前未在训练循环中使用）
scheduler = StepLR(optim, step_size=30, gamma=0.1)

# 训练循环，进行20个epoch
for epoch in range(20):
    running_loss = 0.0  # 初始化每个epoch的累计损失为0.0

    # 遍历DataLoader中的每一个批次的数据
    for data in dataloader:
        imgs, targets = data  # 将数据解包为图像和对应的标签

        # 将输入图像通过模型，得到预测输出
        outputs = tudui(imgs)

        # 计算预测输出与真实标签之间的交叉熵损失
        result_loss = loss(outputs, targets)

        # 清零梯度，防止梯度累积导致训练不稳定
        optim.zero_grad()

        # 执行反向传播，计算损失相对于模型参数的梯度
        result_loss.backward()

        # 更新模型参数，基于计算得到的梯度和优化器的策略
        optim.step()

        # 累加当前批次的损失到running_loss
        running_loss += result_loss.item()  # 使用.item()获取标量值

    # 打印每个epoch的总损失
    print(f"Epoch [{epoch + 1}/20], Loss: {running_loss}")

# 训练完成后，可以保存模型参数
# torch.save(tudui.state_dict(), 'tudui_model.pth')

# 如果使用了学习率调度器，可以在每个epoch后更新学习率
# scheduler.step()