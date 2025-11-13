import argparse
import random
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as T
import os
from data_loader.train_data import MSRS_data
from illiminate_models.resnet_cls_models import Illumination_classifier
from additional_moduel.common import gradient, clamp
from model import FDCFusion
import matplotlib.pyplot as plt
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 学习率调度器


def init_seeds(seed=0, cuda=True):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


# 保存图片
def save_image(image, path):
    image = image.cpu().detach()
    image = T.ToPILImage()(image)
    image.save(path)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FDCFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='MSRS',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='pretrained')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N')
    parser.add_argument('--epochs', default=40, type=int, metavar='N')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('-b', '--batch_size', default=8, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', dest='lr')
    parser.add_argument('--image_size', default=64, type=int, metavar='N')
    parser.add_argument('--save_path_image', default='image_save_model')
    parser.add_argument('--loss_weight', default='[3, 7, 50]', type=str, metavar='N')
    parser.add_argument('--cls_pretrained', default='pretrained/best_cls.pth')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--cuda', default=True, type=bool)

    args = parser.parse_args()
    init_seeds(args.seed, args.cuda)

    train_dataset = MSRS_data(args.dataset_path)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.arch == 'fusion_model':
        model = FDCFusion().to(device)

        # 分类器
        cls_model = Illumination_classifier(input_channels=3)
        cls_model.load_state_dict(torch.load(args.cls_pretrained, map_location=device))
        cls_model = cls_model.to(device)
        cls_model.eval()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        # 记录损失
        epoch_losses_total, epoch_losses_illum, epoch_losses_aux, epoch_losses_grad = [], [], [], []

        for epoch in range(args.start_epoch, args.epochs):
            model.train()
            train_tqdm = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch}/{args.epochs}")

            epoch_loss, epoch_loss_illum, epoch_loss_aux, epoch_loss_grad = 0, 0, 0, 0

            for batch_idx, (vis_image, vis_y_image, _, _, inf_image, _) in enumerate(train_tqdm):
                vis_y_image = vis_y_image.to(device)
                vis_image = vis_image.to(device)
                inf_image = inf_image.to(device)

                optimizer.zero_grad()
                fused_image = model(vis_y_image, inf_image)
                fused_image = clamp(fused_image)

                # 分类器预测
                pred = cls_model(vis_image)
                day_p = pred[:, 0]
                night_p = pred[:, 1]
                vis_weight = day_p / (day_p + night_p)
                inf_weight = 1 - vis_weight

                # 三种损失
                loss_illum = F.l1_loss(inf_weight[:, None, None, None] * fused_image,
                                       inf_weight[:, None, None, None] * inf_image) + \
                             F.l1_loss(vis_weight[:, None, None, None] * fused_image,
                                       vis_weight[:, None, None, None] * vis_y_image)
                loss_aux = F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))
                loss_grad = F.l1_loss(gradient(fused_image),
                                      torch.max(gradient(inf_image), gradient(vis_y_image)))

                # 总损失
                t1, t2, t3 = eval(args.loss_weight)
                loss = t1 * loss_illum + t2 * loss_aux + t3 * loss_grad

                # 反向传播
                loss.backward()
                optimizer.step()

                # 累加损失
                epoch_loss += loss.item()
                epoch_loss_illum += loss_illum.item()
                epoch_loss_aux += loss_aux.item()
                epoch_loss_grad += loss_grad.item()

                # 打印每个 batch 的三种损失 + 总损失
                print(f"Epoch [{epoch}/{args.epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Illum Loss: {loss_illum.item():.4f} | "
                      f"Aux Loss: {loss_aux.item():.4f} | "
                      f"Grad Loss: {loss_grad.item():.4f} | "
                      f"Total Loss: {loss.item():.4f}")

                torch.cuda.empty_cache()

                # 保存部分图片
                if batch_idx == 0:
                    if not os.path.exists(args.save_path_image):
                        os.makedirs(args.save_path_image)
                    fused_norm = (fused_image - fused_image.min()) / (fused_image.max() - fused_image.min())
                    fused_norm = (fused_norm * 255).byte()
                    save_image(fused_norm[0], f"{args.save_path_image}/fused_epoch{epoch}_batch{batch_idx}.png")
                    save_image(vis_image[0], f"{args.save_path_image}/vis_epoch{epoch}_batch{batch_idx}.png")
                    save_image(inf_image[0], f"{args.save_path_image}/inf_epoch{epoch}_batch{batch_idx}.png")

            # 更新学习率
            scheduler.step(epoch_loss)
            epoch_losses_total.append(epoch_loss)


            # 保存模型
            torch.save(model.state_dict(), f'{args.save_path}/model_epoch_{epoch}.pth')
            print(f"Finished epoch {epoch}, Epoch Loss: {epoch_loss:.4f}")

        # ===== 训练完成后绘制损失曲线 =====
        plt.figure()
        plt.plot(range(len(epoch_losses_total)), epoch_losses_total, marker='o', label="Total Loss")
        plt.title("Training Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{args.save_path}/loss_curve.png")
        plt.close()
