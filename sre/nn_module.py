import torch
from torch import nn
# 定义一个名为Tudui的自定义神经网络模块，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的构造函数
        super().__init__()

    # 定义前向传播方法，接收输入张量并返回处理后的输出张量
    def forward(self, input):
        # 将输入张量的每个元素加1，得到输出张量
        output = input + 1
        return output

# 实例化Tudui类的对象
tudui = Tudui()

# 创建一个标量张量，值为1.0
x = torch.tensor(1.0)

# 将输入张量x传递给模型tudui，得到输出
output = tudui(x)

# 打印输出结果
print(output)