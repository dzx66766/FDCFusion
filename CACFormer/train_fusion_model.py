import argparse
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import os
from data_loader.msrs_data import MSRS_data
from models.cls_model import Illumination_classifier
from models.common import gradient, clamp
from new_model.fusion_model import PIAFusion
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau  # 添加学习率调度器

def init_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

# 可视化函数
def save_image(image, path):
    image = image.cpu().detach()  # 将图片从GPU移到CPU，并且断开计算图
    image = T.ToPILImage()(image)  # 转换为PIL图像
    image.save(path)  # 保存为文件

# 可视化权重图像
def plot_weight(weight, epoch, batch_idx, save_path):
    weight = weight.cpu().detach()  # 从GPU移到CPU
    print(weight.shape)

    weight = weight[0]  # 假设batch_size为1，并且权重是单通道图像
    plt.imshow(weight, cmap='hot')
    plt.colorbar()
    plt.savefig(f"{save_path}/weight_epoch_{epoch}_batch_{batch_idx}.png")
    plt.close()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='MSRS',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='pretrained__FMM_size240__resnet')  # 模型存储路径
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=2, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=64, type=int,
                        metavar='N', help='image size of input')
    parser.add_argument('--save_path_image', default='image_save_model')
    parser.add_argument('--loss_weight', default='[3, 7, 50]', type=str,
                        metavar='N', help='loss weight')
    parser.add_argument('--cls_pretrained', default='pretrained/best_cls.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    train_dataset = MSRS_data(args.dataset_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 如果是融合网络
    if args.arch == 'fusion_model':
        model = PIAFusion()
        model = model.cuda()

        # 加载预训练的分类模型
        cls_model = Illumination_classifier(input_channels=3)
        cls_model.load_state_dict(torch.load(args.cls_pretrained))
        cls_model = cls_model.cuda()
        cls_model.eval()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # 添加L2正则化
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)  # 添加学习率调度器

        best_loss = float('inf')
        for epoch in range(args.start_epoch, args.epochs):
            model.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader))
            epoch_loss = 0
            for batch_idx, (vis_image, vis_y_image, _, _, inf_image, _) in enumerate(train_tqdm):
                vis_y_image = vis_y_image.cuda()
                vis_image = vis_image.cuda()
                inf_image = inf_image.cuda()
                optimizer.zero_grad()
                fused_image = model(vis_y_image, inf_image)
                fused_image = clamp(fused_image)

                # 使用预训练的分类模型，得到可见光图片属于白天还是夜晚的概率
                pred = cls_model(vis_image)
                day_p = pred[:, 0]
                night_p = pred[:, 1]
                vis_weight = day_p / (day_p + night_p)
                inf_weight = 1 - vis_weight

                # 计算损失
                loss_illum = F.l1_loss(inf_weight[:, None, None, None] * fused_image,
                                       inf_weight[:, None, None, None] * inf_image) + F.l1_loss(
                    vis_weight[:, None, None, None] * fused_image,
                    vis_weight[:, None, None, None] * vis_y_image)
                loss_aux = F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))
                gradinet_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(inf_image), gradient(vis_y_image)))
                t1, t2, t3 = eval(args.loss_weight)
                loss = t1 * loss_illum + t2 * loss_aux + t3 * gradinet_loss
                # 反向传播和优化
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # 清理GPU缓存
                torch.cuda.empty_cache()

                # 保存中间的一些图片
                if batch_idx == 0:
                    if not os.path.exists(args.save_path_image):
                        os.makedirs(args.save_path_image)
                    fused_image = (fused_image - fused_image.min()) / (fused_image.max() - fused_image.min())
                    fused_image = (fused_image * 255).byte()
                    save_image(fused_image[0], f"{args.save_path_image}/fused_image_epoch_{epoch}_batch_{batch_idx}.png")
                    save_image(vis_image[0], f"{args.save_path_image}/vis_image_epoch_{epoch}_batch_{batch_idx}.png")
                    save_image(inf_image[0], f"{args.save_path_image}/inf_image_epoch_{epoch}_batch_{batch_idx}.png")

            # 更新学习率
            scheduler.step(epoch_loss)

            # 保存最佳模型
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), f'{args.save_path}/best_model.pth')

            print(f"Finished epoch {epoch}, loss: {epoch_loss:.4f}")