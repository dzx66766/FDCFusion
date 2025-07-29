"""测试融合网络"""
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from data_loader.msrs_data import MSRS_data
from models.common import YCrCb2RGB, clamp
from evaluate import Evaluator  # 假设Evaluator包含所有评估函数
from utils.img_read_save import image_read_cv2  # 假设你有这个图像读取函数
from new_model.fusion_model import PIAFusion
import torch.nn.functional as F
from PIL import Image


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='TNO',
                        help='Path to dataset (default: TNO)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_model',
                        choices=['fusion_model'])
    parser.add_argument('--save_path', default='results3/fusion',
                        help='Path to save fusion results')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='Number of data loading workers (default: 1)')
    parser.add_argument('--fusion_pretrained', default='/hy-tmp/PIAFusion_pytorch-master/pretrained__FMM_size240__resnet/best_FFM.pth',
                        help='Path to pre-trained fusion model')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed for initializing training.')
    parser.add_argument('--cuda', action='store_true', default=True,
                        help='Use GPU if available. Set to False for CPU.')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {device}")

    init_seeds(args.seed)

    # Load test dataset
    test_dataset = MSRS_data(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Create save directory
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Initialize evaluation metrics
    metric_result = np.zeros(8)

    # Fusion model
    if args.arch == 'fusion_model':
        model = PIAFusion()
        model = model.to(device)
        model.load_state_dict(torch.load(args.fusion_pretrained, map_location=device))
        model.eval()

        test_tqdm = tqdm(test_loader, total=len(test_loader))

        ori_img_folder = args.dataset_path  # Original IR and VI images folder
        eval_folder = args.save_path  # Folder to save fusion results

        with torch.no_grad():
            for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
                vis_y_image = vis_y_image.cuda()
                cb = cb.cuda()
                cr = cr.cuda()
                inf_image = inf_image.cuda()

                # 测试转为Ycbcr的数据再转换回来的输出效果，结果与原图一样，说明这两个函数是没有问题的。
                # t = YCbCr2RGB2(vis_y_image[0], cb[0], cr[0])
                # transforms.ToPILImage()(t).save(name[0])
                fused_image = model(vis_y_image, inf_image)
                fused_image = clamp(fused_image)
                fused_img_path = os.path.join(args.save_path, name[0])
                rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
                rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
                rgb_fused_image = np.clip(rgb_fused_image, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(rgb_fused_image)  # 从 numpy 数组转换回 PIL 图像
                pil_image.save(f'{args.save_path}/{name[0]}')

                # 读取原始IR, VI和融合图像
                ir = image_read_cv2(os.path.join(ori_img_folder, "ir", name[0]), 'GRAY')
                vi = image_read_cv2(os.path.join(ori_img_folder, "vi", name[0]), 'GRAY')
                fi = image_read_cv2(fused_img_path, 'GRAY')

                # 将它们转换为 numpy 数组
                ir = torch.tensor(ir).unsqueeze(0).unsqueeze(0).float()  # 添加批次和通道维度
                vi = torch.tensor(vi).unsqueeze(0).unsqueeze(0).float()  # 添加批次和通道维度
                fi = torch.tensor(fi).unsqueeze(0).unsqueeze(0).float()  # 添加批次和通道维度

                # 使用 F.interpolate 调整尺寸
                ir = F.interpolate(ir, size=(480, 640), mode='bilinear', align_corners=False)
                vi = F.interpolate(vi, size=(480, 640), mode='bilinear', align_corners=False)
                fi = F.interpolate(fi, size=(480, 640), mode='bilinear', align_corners=False)

                # 转换为 numpy 数组
                ir = ir.squeeze().cpu().numpy()
                vi = vi.squeeze().cpu().numpy()
                fi = fi.squeeze().cpu().numpy()

                # 计算融合指标
                metric_result += np.array([
                    Evaluator.EN(fi), Evaluator.SD(fi), Evaluator.SF(fi),
                    Evaluator.MI(fi, ir, vi), Evaluator.SCD(fi, ir, vi),
                    Evaluator.VIFF(fi, ir, vi), Evaluator.Qabf(fi, ir, vi),
                    Evaluator.SSIM(fi, ir, vi)
                ])
        print(f'len of test_loader',len(test_loader))
        # Average metrics
        metric_result /= len(test_loader)

        # Print results
        print("\t\t EN\t SD\t SF\t MI\t SCD\t VIF\t Qabf\t SSIM")
        print('Model Evaluation\t' + '\t'.join(str(np.round(m, 2)) for m in metric_result))
        print("=" * 80)
