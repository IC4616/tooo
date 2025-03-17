from torch.utils.data import dataset, Dataset  # 导入PyTorch的Dataset类
from PIL import Image  # 导入PIL库用于图像处理
import os  # 导入os模块用于文件路径操作

class MyData(Dataset):  # 定义自定义数据集类，继承自PyTorch的Dataset

    def __init__(self, root_dir, label_dir):  # 初始化方法，接收根目录和标签目录
        self.root_dir = root_dir  # 保存根目录
        self.label_dir = label_dir  # 保存标签目录（注意：变量名拼写错误，应为label_dir）
        self.path = os.path.join(self.root_dir, label_dir)  # 拼接根目录和标签目录，得到完整路径
        self.img_path = os.listdir(self.path)  # 获取路径下的所有文件名，保存到img_path列表中

    def __getitem__(self, idx):  # 获取单个数据项的方法，接收索引idx
        img_name = self.img_path[idx]  # 根据索引获取文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 拼接完整图像路径（注意：self.label_dir未定义，应为self.lael_dir）
        img = Image.open(img_item_path)  # 打开图像文件（注意：image应为Image，大小写错误）
        label = self.label_dir  # 获取标签（注意：self.label_dir未定义，应为self.lael_dir）
        return img, label  # 返回图像和标签

    def __len__(self):  # 返回数据集长度的方法
        return len(self.img_path)  # 返回图像文件列表的长度（注意：sel应为self，拼写错误）

# 数据集路径和标签定义
root_dir = "dataset/train"  # 根目录
ants_label_dir = "ants"  # 蚂蚁数据标签目录
bees_label_dir = "bees"  # 蜜蜂数据标签目录

# 创建蚂蚁和蜜蜂数据集实例
ants_dataset = MyData(root_dir, "ants")  # 创建蚂蚁数据集
bees_dataset = MyData(root_dir, bees_label_dir)  # 创建蜜蜂数据集

# 合并数据集
train_dataset = ants_dataset + bees_dataset  # 合并蚂蚁和蜜蜂数据集（注意：需要重载+操作符或使用ConcatDataset）
