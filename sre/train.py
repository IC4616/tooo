
import torchvision
from torch.utils.tensorboard import SummaryWriter

from model import *  # 导入自定义的模型模块（假设模型定义在model.py文件中）
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

# 下载并加载CIFAR-10训练数据集
train_data = torchvision.datasets.CIFAR10(
    root="../dataset",  # 数据存储的根目录
    train=True,  # 指定为训练数据集
    transform=torchvision.transforms.ToTensor(),  # 将图像转换为Tensor
    download=True  # 如果数据不存在，则下载
)

# 下载并加载CIFAR-10测试数据集
test_data = torchvision.datasets.CIFAR10(
    root="../dataset",  # 数据存储的根目录
    train=False,  # 指定为测试数据集
    transform=torchvision.transforms.ToTensor(),  # 将图像转换为Tensor
    download=True  # 如果数据不存在，则下载
)

# 获取训练数据集的长度
train_data_size = len(train_data)
# 获取测试数据集的长度
test_data_size = len(test_data)
# 打印训练数据集的长度
print("训练数据集的长度为：{}".format(train_data_size))
# 打印测试数据集的长度
print("测试数据集的长度为：{}".format(test_data_size))

# 使用DataLoader加载训练数据集，设置批量大小为64
train_dataloader = DataLoader(train_data, batch_size=64)
# 使用DataLoader加载测试数据集，设置批量大小为64
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建Tudui模型的实例
tudui = Tudui()

# 定义损失函数为交叉熵损失，常用于分类任务
loss_fn = nn.CrossEntropyLoss()

# 设置优化器为随机梯度下降（SGD），学习率为0.01
learning_rate = 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练过程中的一些参数
total_train_step = 0  # 记录训练的步数（批次）
total_test_step = 0   # 记录测试的步数（轮次）
epoch = 2            # 训练的总轮数

# 初始化TensorBoard的SummaryWriter，用于记录训练过程中的各种指标
writer = SummaryWriter("../logs_train")

# 开始训练循环，遍历每一个训练轮次
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))

    # 设置模型为训练模式，启用诸如Dropout和BatchNorm等层的训练行为
    tudui.train()
    # 遍历训练数据加载器中的每一个批次
    for data in train_dataloader:
        imgs, targets = data  # 获取图像数据和对应的标签
        outputs = tudui(imgs)  # 将图像输入模型，得到预测输出
        loss = loss_fn(outputs, targets)  # 计算预测输出与真实标签之间的损失

        # 梯度清零，防止梯度累积影响优化过程
        optimizer.zero_grad()
        # 反向传播计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()

        total_train_step += 1  # 训练步数加1
        # 每训练100个批次，打印当前的训练损失，并将损失记录到TensorBoard
        if total_train_step % 100 == 0:
            print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 设置模型为评估模式，禁用诸如Dropout和BatchNorm等层的训练行为
    tudui.eval()
    total_test_loss = 0  # 初始化测试集上的总损失
    total_accuracy = 0   # 初始化测试集上的总正确预测数
    # 在评估模式下进行测试，不计算梯度以提高效率
    with torch.no_grad():
        # 遍历测试数据加载器中的每一个批次
        for data in test_dataloader:
            imgs, targets = data  # 获取图像数据和对应的标签
            outputs = tudui(imgs)  # 将图像输入模型，得到预测输出
            loss = loss_fn(outputs, targets)  # 计算预测输出与真实标签之间的损失
            total_test_loss += loss.item()  # 累加测试损失
            # 计算正确预测的数量，并累加到总正确预测数中
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    # 打印整个测试集上的总损失
    print("整体测试集上的Loss: {}".format(total_test_loss))
    # 打印整个测试集上的正确率（准确率）
    print("整体测试集上的正确率: {}".format(total_accuracy / test_data_size))
    # 将测试集上的总损失记录到TensorBoard
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    # 将测试集上的正确率记录到TensorBoard
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1  # 测试步数加1

    # 保存当前训练轮次下的模型参数，文件名包含当前的轮次数
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

# 关闭TensorBoard的SummaryWriter，确保所有数据都已写入日志文件
writer.close()