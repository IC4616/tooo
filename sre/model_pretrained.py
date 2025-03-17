
import torchvision
from torch import nn

# 注释掉的代码：加载ImageNet数据集（训练集）
# 参数说明：
# - "../data_image_net": 数据集保存的路径
# - split='train': 指定为训练集
# - download=True: 如果数据集不存在，则自动下载
# - transform=torchvision.transforms.ToTensor(): 将图像转换为张量格式
# train_data = torchvision.datasets.ImageNet("../data_image_net", split='train', download=True,
#                                            transform=torchvision.transforms.ToTensor())

# 加载预训练为False的VGG16模型（未预训练）
vgg16_false = torchvision.models.vgg16(pretrained=False)

# 加载预训练为True的VGG16模型（已预训练）
vgg16_true = torchvision.models.vgg16(pretrained=True)

# 打印已预训练的VGG16模型的详细信息
print(vgg16_true)

# 加载CIFAR-10训练数据集
# 参数说明：
# - "../data": 数据集保存的路径
# - train=True: 指定为训练集
# - transform=torchvision.transforms.ToTensor(): 将图像转换为张量格式
# - download=True: 如果数据集不存在，则自动下载
train_data = torchvision.datasets.CIFAR10('../dataset', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 向已预训练的VGG16模型的分类器添加一个新的线性层
# VGG16原本的分类器最后一层是1000类（对应ImageNet的1000个类别），这里添加一个输出10类的线性层以适应CIFAR-10
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
# 打印修改后的VGG16模型的详细信息
print(vgg16_true)

# 打印未预训练的VGG16模型的详细信息
print(vgg16_false)

# 修改未预训练的VGG16模型的分类器的最后一个线性层
# 原分类器的最后一个线性层输入特征数为4096，输出为1000类，这里将其修改为输出10类以适应CIFAR-10
vgg16_false.classifier[6] = nn.Linear(4096, 10)
# 打印修改后的未预训练VGG16模型的详细信息
print(vgg16_false)