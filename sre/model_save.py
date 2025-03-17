
import torch
import torchvision
from torch import nn

# 加载未预训练的VGG16模型
vgg16 = torchvision.models.vgg16(pretrained=False)

# 保存方式1：保存整个模型（包括模型结构和参数）
# 这种方法将模型的结构和参数一起保存到一个文件中
torch.save(vgg16, "vgg16_method1.pth")

# 保存方式2：仅保存模型的参数（官方推荐）
# 这种方法只保存模型的参数（state_dict），不包括模型结构
# 官方推荐这种方式，因为它更灵活，可以在加载参数时使用不同的模型结构
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

# 定义一个自定义的简单神经网络类Tudui
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类nn.Module的构造函数
        super(Tudui, self).__init__()
        # 定义一个卷积层，输入通道数为3，输出通道数为64，卷积核大小为3x3
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)

    # 定义前向传播方法，接收输入张量x并返回输出张量
    def forward(self, x):
        # 将输入通过卷积层，得到输出
        x = self.conv1(x)
        return x

# 实例化自定义模型Tudui
tudui = Tudui()

# 保存自定义模型（方式1）：保存整个模型（包括模型结构和参数）
# 注意：这种方法在某些情况下可能会遇到问题，尤其是在不同环境下加载模型时
torch.save(tudui, "tudui_method1.pth")

# 陷阱说明：
# 使用torch.save保存整个模型（方式1）时，可能会遇到以下问题：
# 1. 模型加载依赖于特定的类定义和模块路径。如果在不同的环境中没有定义相同的类或模块路径，将无法正确加载模型。
# 2. 官方推荐使用方式2（保存state_dict），因为它更灵活，可以在不同的模型实例或不同的环境中加载参数。

# 推荐的模型加载方式示例：

# 加载方式1：加载整个模型
# model = torch.load("vgg16_method1.pth")
# model.eval()

# 加载方式2：加载模型参数到定义好的模型结构中
# vgg16_loaded = torchvision.models.vgg16(pretrained=False)
# vgg16_loaded.load_state_dict(torch.load("vgg16_method2.pth"))
# vgg16_loaded.eval()

# 对于自定义模型Tudui，加载方式如下：

# 加载整个自定义模型（方式1）
# tudui_loaded = torch.load("tudui_method1.pth")
# tudui_loaded.eval()

# 或者，更推荐的方式是先定义模型结构，再加载参数（方式2）：
# tudui_instance = Tudui()
# tudui_instance.load_state_dict(torch.load("tudui_method1.pth"))  # 如果保存的是state_dict
# 但上述保存方式实际上是保存了整个模型，因此需要使用第一种加载方式