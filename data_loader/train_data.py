import os
from PIL import Image
from torch.utils import data
from torchvision import transforms
from additional_moduel.common import RGB2YCrCb

# 定义图像变换
to_tensor = transforms.Compose([
    transforms.Resize((640,480)),  # 将图像调整为 128,128
    transforms.ToTensor()            # 转换为张量
])


class MSRS_data(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        dirname = os.listdir(data_dir)  # 获得数据集的子目录
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path  # 获得红外路径
            if sub_dir == 'vi':
                self.vis_path = temp_path  # 获得可见光路径
        self.transform = transform  # 使用传入的transform

        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称

    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称

        # 构建红外图像和可见光图像路径
        inf_image_path = os.path.join(self.inf_path, name)
        vis_image_path = os.path.join(self.vis_path, name)

        # 检查文件扩展名并尝试读取可见光图像（如果是jpg格式就更改为jpg后缀）
        if not os.path.exists(vis_image_path):
            vis_image_path = vis_image_path.replace('.png', '.jpg')  # 将png替换为jpg

        # 如果vis_image_path仍然不存在，抛出错误
        if not os.path.exists(vis_image_path):
            raise FileNotFoundError(f"File not found: {vis_image_path}")

        # 加载图像
        inf_image = Image.open(inf_image_path).convert('L')  # 获取红外图像
        vis_image = Image.open(vis_image_path)  # 获取可见光图像

        # 对图像进行变换（包括调整尺寸为 480x480）
        inf_image = self.transform(inf_image)  # 对红外图像进行变换
        vis_image = self.transform(vis_image)  # 对可见光图像进行变换

        # 将可见光图像转换为YCrCb格式
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)

        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, name

    def __len__(self):
        return len(self.name_list)