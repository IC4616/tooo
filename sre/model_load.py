
import torch
from model_save import *  # 导入自定义的保存模块（如果有定义）

# 方式1：加载整个模型（包括模型结构和参数）
# 使用torch.load加载之前保存的整个VGG16模型
#model = torch.load("vgg16_method1.pth")
#print(model)  # 可选：打印加载的模型结构及参数

# 方式2：仅加载模型的参数（官方推荐）
# 首先实例化一个未预训练的VGG16模型
#vgg16 = torchvision.models.vgg16(pretrained=False)
# 加载之前保存的VGG16模型的参数（state_dict）到实例化的模型中
#vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
model = torch.load("vgg16_method2.pth")  # 这行代码被注释掉，因为只加载参数，不加载整个模型
print(vgg16)  # 可选：打印加载的模型结构及参数

# 陷阱1：加载自定义的整个模型
# 定义一个自定义的简单神经网络类Tudui
# class Tudui(nn.Module):
#     def __init__(self):
#         super(Tudui, self).__init__()
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         return x

# 加载整个自定义模型Tudui（方式1：保存整个模型）
#model_tudui = torch.load('tudui_method1.pth')
#print(model_tudui)  # 打印加载的自定义模型结构及参数

# 陷阱说明：
# 使用torch.save保存整个模型（方式1）时，可能会遇到以下问题：
# 1. 模型加载依赖于特定的类定义和模块路径。如果在不同的环境中没有定义相同的类或模块路径，将无法正确加载模型。
# 2. 官方推荐使用方式2（保存state_dict），因为它更灵活，可以在不同的模型实例或不同的环境中加载参数。

# 推荐的模型加载方式示例：

# 加载方式1：加载整个模型（如果保存的是整个模型）
# model_tudui = torch.load("tudui_method1.pth")
# model_tudui.eval()

# 加载方式2：加载模型参数到定义好的模型结构中（推荐）
# tudui_instance = Tudui()
# tudui_instance.load_state_dict(torch.load("tudui_method2.pth"))  # 需要先保存state_dict
# tudui_instance.eval()

# 注意：
# 上述代码中，tudui_method1.pth 是通过保存整个模型得到的，因此可以直接加载整个模型。
# 如果将来希望更灵活地加载参数，建议保存state_dict而不是整个模型。