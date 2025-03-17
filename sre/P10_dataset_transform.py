# 导入torchvision库，用于计算机视觉任务相关的工具和数据集
import torchvision
# 从torch.utils.tensorboard导入SummaryWriter，用于将数据写入TensorBoard日志
from torch.utils.tensorboard import SummaryWriter

# 定义数据集的变换操作，这里仅将图像转换为Tensor
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

# 下载并加载CIFAR10训练集
# 参数说明：
# - root: 数据集存储的根目录
# - train: 是否为训练集
# - transform: 应用于数据的变换
# - download: 如果数据不存在，是否下载
train_set = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=True,
    transform=dataset_transform,
    download=True
)

# 下载并加载CIFAR10测试集
test_set = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=dataset_transform,
    download=True
)

# 以下代码块被注释掉了，可以用于查看数据集的样本信息
# print(test_set[0])  # 打印测试集的第一个样本（图像和标签）
# print(test_set.classes)  # 打印类别名称列表

# 获取测试集的第一个样本
# img, target = test_set[0]
# print(img)  # 打印图像张量
# print(target)  # 打印标签索引
# print(test_set.classes[target])  # 根据标签索引打印类别名称
# img.show()  # 显示图像（需要PIL支持）

# 再次打印测试集的第一个样本（可选）
# print(test_set[0])

# 创建一个SummaryWriter实例，日志文件将保存在"p10"文件夹中
writer = SummaryWriter("p10")

# 循环遍历测试集的前10个样本，并将它们添加到TensorBoard日志中
for i in range(10):
    img, target = test_set[i]  # 获取第i个样本的图像和标签
    writer.add_image("test_set", img, i)  # 将图像添加到TensorBoard，标签为"test_set"，全局步数为i

# 关闭SummaryWriter，确保所有数据都已写入日志文件
writer.close()