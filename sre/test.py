
import torch  # 导入PyTorch库
import torchvision  # 导入torchvision库，用于处理图像数据
from PIL import Image  # 导入PIL库中的Image模块，用于处理图像
from torch import nn  # 导入PyTorch中的神经网络模块

# 定义图像路径
image_path = "../imgs/dog.png"
# 使用PIL库打开图像
image = Image.open(image_path)
# 打印图像对象
print(image)
# 将图像转换为RGB格式
image = image.convert('RGB')

# 定义图像预处理流程，包括调整大小和转换为Tensor
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

# 对图像进行预处理
image = transform(image)
# 打印处理后的图像形状
print(image.shape)

# 定义一个名为Tudui的神经网络类，继承自nn.Module
class Tudui(nn.Module):
    def __init__(self):
        # 调用父类的初始化方法
        super(Tudui, self).__init__()
        # 定义神经网络模型的结构
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),  # 3输入通道，32输出通道，5x5卷积核，步长1，填充2
            nn.MaxPool2d(2),  # 2x2最大池化
            nn.Conv2d(32, 32, 5, 1, 2),  # 32输入通道，32输出通道，5x5卷积核，步长1，填充2
            nn.MaxPool2d(2),  # 2x2最大池化
            nn.Conv2d(32, 64, 5, 1, 2),  # 32输入通道，64输出通道，5x5卷积核，步长1，填充2
            nn.MaxPool2d(2),  # 2x2最大池化
            nn.Flatten(),  # 将多维张量展平为一维
            nn.Linear(64 * 4 * 4, 64),  # 全连接层，输入维度64 * 4 * 4，输出维度64
            nn.Linear(64, 10)  # 全连接层，输入维度64，输出维度10
        )

    # 定义前向传播方法
    def forward(self, x):
        x = self.model(x)
        return x

# 加载预训练模型，并将模型映射到CPU设备
model = torch.load("tudui_0.pth", map_location=torch.device('cpu'))
# 打印加载的模型
print(model)
# 调整图像张量的形状，使其符合模型的输入要求
image = torch.reshape(image, (1, 3, 32, 32))
# 将模型设置为评估模式
model.eval()
# 在不需要计算梯度的上下文中进行推理
with torch.no_grad():
    output = model(image)
# 打印模型的输出
print(output)

# 打印输出中最大值的索引，即预测的类别
print(output.argmax(1))