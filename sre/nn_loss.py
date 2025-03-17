
import torch
from torch.nn import L1Loss
from torch import nn

# 创建输入张量，包含三个浮点数元素
inputs = torch.tensor([1, 2, 3], dtype=torch.float32)
# 创建目标张量，包含三个浮点数元素
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

# 将输入张量重塑为 (1, 1, 1, 3) 的形状，模拟单通道的单个样本，具有三个特征
inputs = torch.reshape(inputs, (1, 1, 1, 3))
# 将目标张量重塑为 (1, 1, 1, 3) 的形状，与输入张量的形状保持一致
targets = torch.reshape(targets, (1, 1, 1, 3))

# 实例化L1Loss损失函数，设置reduction='sum'，表示将所有样本的损失求和
loss = L1Loss(reduction='sum')
# 计算输入和目标之间的L1损失
result = loss(inputs, targets)

# 实例化均方误差损失函数（MSELoss）
loss_mse = nn.MSELoss()
# 计算输入和目标之间的均方误差损失
result_mse = loss_mse(inputs, targets)

# 打印L1损失的结果
print(result)
# 打印均方误差损失的结果
print(result_mse)

# 创建一个包含三个浮点数元素的张量x，模拟模型的原始输出（未经过softmax）
x = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)
# 创建一个包含单个整数元素的张量y，表示真实类别的索引（从0开始）
y = torch.tensor([1], dtype=torch.long)  # 注意：CrossEntropyLoss 需要目标为类别索引，而非one-hot编码

# 将张量x重塑为 (1, 3) 的形状，表示一个批次中有一个样本，具有三个类别的预测分数
x = torch.reshape(x, (1, 3))
# 实例化交叉熵损失函数
loss_cross = nn.CrossEntropyLoss()
# 计算输入x和目标y之间的交叉熵损失
# 注意：CrossEntropyLoss 会自动对x进行softmax处理，然后计算交叉熵
result_cross = loss_cross(x, y)
# 打印交叉熵损失的结果
print(result_cross)