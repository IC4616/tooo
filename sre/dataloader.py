# 导入torchvision库，用于计算机视觉任务相关的工具和数据集
import torchvision

# 导入DataLoader，用于批量加载数据
from torch.utils.data import DataLoader

# 导入SummaryWriter，用于将数据写入TensorBoard日志
from torch.utils.tensorboard import SummaryWriter

# 下载并加载CIFAR10测试集
# 参数说明：
# - root: 数据集存储的根目录
# - train: 是否为训练集（False表示测试集）
# - transform: 应用于数据的变换（这里将图像转换为Tensor）
# - download: 如果数据不存在，是否下载
test_data = torchvision.datasets.CIFAR10(
    root="./dataset",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True  # 确保数据集已下载
)

# 创建DataLoader实例，用于批量加载测试数据
# 参数说明：
# - dataset: 要加载的数据集
# - batch_size: 每个批次的样本数量
# - shuffle: 是否在每个epoch开始时打乱数据
# - num_workers: 用于数据加载的子进程数量
# - drop_last: 如果最后一个批次样本不足batch_size，是否丢弃该批次
test_loader = DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,  # 在调试时建议设置为0，避免多线程问题
    drop_last=True
)

# 获取测试数据集中的第一张图片及其对应的标签
img, target = test_data[0]
# 打印图像张量的形状（C, H, W）
print(img.shape)
# 打印图像的标签（类别索引）
print(target)

# 创建一个SummaryWriter实例，日志文件将保存在"dataloader"文件夹中
writer = SummaryWriter("dataloader")

# 进行2个epoch的循环，模拟训练过程
for epoch in range(2):
    step = 0  # 初始化步骤计数器
    # 遍历DataLoader中的每个批次
    for data in test_loader:
        imgs, targets = data  # 获取当前批次的图像和标签
        # 将当前批次的图像添加到TensorBoard日志中
        # 参数说明：
        # - "Epoch: {}".format(epoch): 图像的标签，显示当前属于哪个epoch
        # - imgs: 图像张量
        # - step: 当前的全局步骤数
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1  # 步骤计数器加1

# 关闭SummaryWriter，确保所有数据都已写入日志文件
writer.close()