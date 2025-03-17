
import torch
import torch.nn.functional as F

# 定义一个5x5的输入矩阵，模拟图像数据
input = torch.tensor([
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])

# 定义一个3x3的卷积核，用于卷积操作
kernel = torch.tensor([
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])

# 将输入张量重塑为形状 (批量大小, 通道数, 高度, 宽度)
# 这里批量大小为1，通道数为1，高度和宽度均为5
input = torch.reshape(input, (1, 1, 5, 5))
# 将卷积核张量重塑为形状 (输出通道数, 输入通道数, 高度, 宽度)
# 这里输出通道和输入通道均为1，高度和宽度均为3
kernel = torch.reshape(kernel, (1, 1, 3, 3))
# 打印输入张量的形状
print("输入张量的形状:", input.shape)
# 打印卷积核的形状
print("卷积核的形状:", kernel.shape)

# 使用F.conv2d进行卷积操作，步幅(stride)为1，不使用填充(padding)
output = F.conv2d(input, kernel, stride=1)
# 打印卷积后的输出结果
print("步幅为1的卷积输出:\n", output)

# 使用F.conv2d进行卷积操作，步幅(stride)为2，不使用填充(padding)
output2 = F.conv2d(input, kernel, stride=2)
# 打印卷积后的输出结果
print("步幅为2的卷积输出:\n", output2)

# 使用F.conv2d进行卷积操作，步幅(stride)为1，并使用填充(padding)为1
# 填充的目的是保持输入和输出的空间维度一致
output3 = F.conv2d(input, kernel, stride=1, padding=1)
# 打印卷积后的输出结果
print("步幅为1且填充为1的卷积输出:\n", output3)