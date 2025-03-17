from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
# 创建一个SummaryWriter对象，用于写入TensorBoard日志
writer = SummaryWriter("logs")

# 定义图像路径
image_path = "../data/train/ants_image/0013035.jpg"

# 使用PIL库打开图像
img_PIL = Image.open(image_path)

# 将图像转换为NumPy数组
img_array = np.array(img_PIL)

# 打印图像数组的类型
print(type(img_array))

# 打印图像数组的形状
print(img_array.shape)

# 将图像添加到TensorBoard日志中
writer.add_image("test", img_array, 1, dataformats='HWC')

# 循环添加标量数据到TensorBoard日志中
for i in range(100):
    writer.add_scalar("y=3x", 3*i, i)

# 关闭SummaryWriter对象
writer.close()

# tensorboard --logdir=logs(运行)
# tensorboard --logdir=logs --port=6007（只定义显示）