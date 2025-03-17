from PIL import Image  # 导入PIL库中的Image模块
from torch.utils.tensorboard import SummaryWriter  # 导入TensorBoard的SummaryWriter模块
from torchvision import transforms  # 导入torchvision库中的transforms模块

writer = SummaryWriter("logs")  # 创建一个SummaryWriter对象，用于将数据写入TensorBoard日志

img = Image.open("../data/train/ants_image/0013035.jpg")  # 打开一张图片
print(img)  # 打印图片信息

# ToTensor转换器，将PIL图像转换为Tensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("ToTensor", img_tensor)  # 将ToTensor后的图像添加到TensorBoard

# Normalize转换器，对图像进行归一化
print(img_tensor[0][0][0])  # 打印Tensor的一个元素
trans_norm = transforms.Normalize([6, 3, 2], [9, 3, 5])  # 定义Normalize转换器
img_norm = trans_norm(img_tensor)  # 对图像进行归一化
print(img_norm[0][0][0])  # 打印归一化后的Tensor的一个元素
writer.add_image("Normalize", img_norm, 2)  # 将Normalize后的图像添加到TensorBoard

# 可调整大小
print(img.size)
trans_size = transforms.Resize((312, 512))  # 创建一个调整大小的变换，目标尺寸为 (312, 512)
img_resize = trans_size(img)  # 应用调整大小变换到图像
img_resize = trans_tensor(img_resize)  # 将调整大小后的图像转换为张量
writer.add_image("Resize", img_resize, 0)  # 将调整大小后的图像添加到写入器
print(img_resize)

# 组合 - 调整大小 - 2
trans_resize_2 = transforms.Resize(512)  # 创建一个调整大小的变换，目标尺寸为 512
trans_compose = transforms.Compose([trans_resize_2, trans_tensor])  # 创建一个组合变换，包含调整大小和转换为张量
img_resize_2 = trans_compose(img)  # 应用组合变换到图像
writer.add_image("Resize", img_resize_2, 1)  # 将组合变换后的图像添加到写入器

# 随机裁剪
trans_random = transforms.RandomCrop((500, 600))  # 创建随机裁剪变换，裁剪尺寸为(500, 600)
trans_compose_2 = transforms.Compose([trans_random, trans_tensor])  # 将随机裁剪和转换为张量的变换组合在一起
for i in range(10):  # 循环10次
    img_crop = trans_compose_2(img)  # 对图像应用组合变换
    writer.add_image("RandomCropHW", img_crop, i)  # 将裁剪后的图像添加到写入器中，标签为"RandomCropHW"，索引为i

writer.close()  # 关闭SummaryWriter