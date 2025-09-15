import math
from torch import nn
import torch.nn.functional as F
from additional_moduel.common import reflect_conv


class Illumination_classifier(nn.Module):
    def __init__(self, input_channels, init_weights=True):
        super(Illumination_classifier, self).__init__()
        self.conv1 = reflect_conv(in_channels=input_channels, out_channels=16)
        self.conv2 = reflect_conv(in_channels=16, out_channels=32)
        self.conv3 = reflect_conv(in_channels=32, out_channels=64)
        self.conv4 = reflect_conv(in_channels=64, out_channels=128)

        # 1x1 卷积用于通道调整
        self.channel_adjust = nn.Conv2d(16, 128, kernel_size=1, stride=1, padding=0)

        # 自注意力模块（改进版）
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(128, 128 // 8, kernel_size=1),  # 降维
            nn.ReLU(inplace=True),
            nn.Conv2d(128 // 8, 128, kernel_size=1),  # 升维
            nn.Sigmoid()
        )

        # 批归一化层
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        # 全连接层
        self.linear1 = nn.Linear(in_features=128, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=2)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        activate = nn.LeakyReLU(inplace=False)

        # 第一层卷积
        x1 = activate(self.bn1(self.conv1(x)))

        # 第二层到第四层卷积
        x2 = activate(self.bn2(self.conv2(x1)))
        x3 = activate(self.bn3(self.conv3(x2)))
        x4 = activate(self.bn4(self.conv4(x3)))

        # 调整 x1 的大小并改变通道数以匹配 x4
        x1_resized = F.interpolate(x1, size=x4.shape[2:], mode='bilinear', align_corners=False)
        x1_resized = self.channel_adjust(x1_resized)  # 调整通道数
        x4 = x4 + x1_resized  # 残差连接

        # 自注意力机制增强光照区域
        attention_out = self.attention(x4)
        x = x4 * attention_out + x4  # 在注意力模块后添加残差连接

        # 全局平均池化和全连接
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.linear2(x)
        x = nn.Sigmoid()(x)  # 输出归一化到 [0, 1]
        return x