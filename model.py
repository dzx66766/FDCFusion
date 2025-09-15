import torch
from torch import nn
from additional_moduel.common import reflect_conv
from additional_moduel.dsc import DSC, IDSC
import torch.nn.functional as F
from einops import rearrange
import numbers
import torch.fft as fft


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio):
        super(InvertedResidualBlock, self).__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            # pw
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # dw
            nn.ReflectionPad2d(1),
            nn.Conv2d(hidden_dim, hidden_dim, 3, groups=hidden_dim, bias=False),
            # nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
            # nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self):
        super(DetailNode, self).__init__()
        # Scale is Ax + b, i.e. affine transformation
        self.theta_phi = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_rho = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.theta_eta = InvertedResidualBlock(inp=32, oup=32, expand_ratio=2)
        self.shffleconv = nn.Conv2d(64, 64, kernel_size=1,
                                    stride=1, padding=0, bias=True)

    def separateFeature(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        return z1, z2

    def forward(self, z1, z2):
        z1, z2 = self.separateFeature(
            self.shffleconv(torch.cat((z1, z2), dim=1)))
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class CAFormer(nn.Module):
    def __init__(self, dim, num_heads=8, expansion_ratio=2, qkv_bias=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 多尺度卷积分支（3x3 + 5x5）
        self.conv_3x3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.conv_5x5 = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

        # 十字形窗口注意力（CSWA）
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=qkv_bias)
        self.scale = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 动态路由门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 2, kernel_size=1),  # 输出卷积/注意力的权重
            nn.Softmax(dim=1)
        )

        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        # 多尺度卷积特征
        conv_feat = self.conv_3x3(x) + self.conv_5x5(x)

        # 十字形窗口注意力（水平+垂直）
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv h d) x y -> qkv b h d (x y)', h=self.num_heads, qkv=3)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # 水平注意力
        q_h = rearrange(q, 'b h d (x y) -> b h d x y', x=h)
        k_h = rearrange(k, 'b h d (x y) -> b h d x y', x=h)
        attn_h = (q_h @ k_h.transpose(-2, -1)) * self.scale
        attn_h = attn_h.softmax(dim=-1)
        v_h = rearrange(v, 'b h d (x y) -> b h d x y', x=h)
        out_h = rearrange(attn_h @ v_h, 'b h d x y -> b (h d) x y')

        # 垂直注意力
        q_v = rearrange(q, 'b h d (x y) -> b h d y x', x=h)
        k_v = rearrange(k, 'b h d (x y) -> b h d y x', x=h)
        attn_v = (q_v @ k_v.transpose(-2, -1)) * self.scale
        attn_v = attn_v.softmax(dim=-1)
        v_v = rearrange(v, 'b h d (x y) -> b h d y x', x=h)
        out_v = rearrange(attn_v @ v_v, 'b h d y x -> b (h d) x y')

        # 合并十字形注意力
        attn_feat = out_h + out_v

        # 动态路由门控
        gate_weights = self.gate(x)  # [b, 2, 1, 1]
        conv_weight, attn_weight = gate_weights.chunk(2, dim=1)
        out = conv_weight * conv_feat + attn_weight * attn_feat

        return self.proj(out)


class Mlp(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 ffn_expansion_factor=2,
                 bias=False):
        super().__init__()
        hidden_features = int(in_features * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            in_features, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, in_features, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class BaseFeatureExtraction(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=1., qkv_bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim, 'WithBias')
        self.token_mixer = CAFormer(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = LayerNorm(dim, 'WithBias')
        self.mlp = Mlp(
            in_features=dim,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=qkv_bias
        )

    def forward(self, x):
        x = x + self.token_mixer(self.norm1(x))  # 替换为CAFormer的Token Mixer
        x = x + self.mlp(self.norm2(x))
        return x


class DetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=3):
        super(DetailFeatureExtraction, self).__init__()
        INNmodules = [DetailNode() for _ in range(num_layers)]
        self.net = nn.Sequential(*INNmodules)

    def forward(self, x):
        z1, z2 = x[:, :x.shape[1] // 2], x[:, x.shape[1] // 2:x.shape[1]]
        for layer in self.net:
            z1, z2 = layer(z1, z2)
        return torch.cat((z1, z2), dim=1)


class FIM(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

        self.concat = nn.Conv2d(dim2 * 2, dim2, 1)

        self.fusion = nn.Sequential(
            IDSC(dim2 * 4, dim2),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
            DSC(dim2, dim2),
            nn.BatchNorm2d(dim2),
            nn.GELU(),
            nn.Conv2d(dim2, 128, 1),   # 🔥 输出通道改成 128
            nn.BatchNorm2d(128),
            nn.GELU()
        )

    def forward(self, x, y):
        """
        x: 可见光特征
        y: 红外特征
        """
        b, c, h, w = x.shape
        B, C, H, W = y.shape
        N = H * W

        # 通道对齐
        x = self.trans_c(x)

        # 通道加权
        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        # 交叉注意力 (y→x)
        qy = self.qy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(
            0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        kx = self.kx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(
            0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        vx = self.vx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(
            0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C ** -0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)

        # 交叉注意力 (x→y)
        qx = self.qx(x).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(
            0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        ky = self.ky(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(
            0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)
        vy = self.vy(y).reshape(B, 8, C // 8, H // 4, 4, W // 4, 4).permute(
            0, 3, 5, 1, 4, 6, 2).reshape(B, N // 16, 8, 16, C // 8)

        attny = (qx @ ky.transpose(-2, -1)) * (C ** -0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H // 4, w // 4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)


        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)

        out = torch.cat([x, y, out1, out2], dim=1)
        out = self.fusion(out)
        return out

class FFTSeparation(nn.Module):
    def __init__(self, low_ratio=0.1):
        """
        low_ratio: 保留的低频区域比例 (0.1 表示只保留中心 10% 频率成分作为低频)
        """
        super().__init__()
        self.low_ratio = low_ratio

    def forward(self, x):
        # x: [B,C,H,W]
        B, C, H, W = x.shape
        X = fft.fft2(x, norm="ortho")  # 2D FFT
        X = fft.fftshift(X)  # 把低频移到中心

        # 构造 mask
        cy, cx = H // 2, W // 2
        ly, lx = int(cy * self.low_ratio), int(cx * self.low_ratio)

        mask = torch.zeros((H, W), device=x.device)
        mask[cy - ly:cy + ly, cx - lx:cx + lx] = 1.0  # 低频区域 mask
        mask = mask[None, None, :, :]  # [1,1,H,W]

        # 低频部分
        low_freq = X * mask
        # 高频部分
        high_freq = X * (1 - mask)

        # ifft 回空间域
        low = fft.ifft2(fft.ifftshift(low_freq), norm="ortho").real
        high = fft.ifft2(fft.ifftshift(high_freq), norm="ortho").real

        return low, high


class Shared_FrequencyDecompositionModule(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 dim=64,
                 num_blocks=4,
                 num_heads=8,
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 low_ratio=0.1):
        super(Shared_FrequencyDecompositionModule, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level = nn.Sequential(
            *[TransformerBlock(dim=dim, num_heads=num_heads,
                               ffn_expansion_factor=ffn_expansion_factor,
                               bias=bias, LayerNorm_type=LayerNorm_type)
              for _ in range(num_blocks)]
        )

        self.fft_sep = FFTSeparation(low_ratio=low_ratio)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.encoder_level(x)
        low, high = self.fft_sep(x)
        return low, high


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = reflect_conv(in_channels=128, kernel_size=3, out_channels=128, stride=1, pad=1)
        self.conv2 = reflect_conv(in_channels=128, kernel_size=3, out_channels=64, stride=1, pad=1)
        self.conv3 = reflect_conv(in_channels=64, kernel_size=3, out_channels=32, stride=1, pad=1)
        self.conv4 = reflect_conv(in_channels=32, kernel_size=3, out_channels=16, stride=1, pad=1)
        self.conv5 = nn.Conv2d(in_channels=16, kernel_size=1, out_channels=1, stride=1, padding=0)

    def forward(self, x):
        activate = nn.LeakyReLU()
        x = activate(self.conv1(x))
        x = activate(self.conv2(x))
        x = activate(self.conv3(x))
        x = activate(self.conv4(x))
        x = nn.Tanh()(self.conv5(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
        return x


class Encoder(nn.Module):
    def __init__(self,
                 inp_channels=1,
                 out_channels=1,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 low_ratio=0.1):  # 新增参数
        super(Encoder, self).__init__()

        self.SFDM = Shared_FrequencyDecompositionModule(
            inp_channels=inp_channels,
            dim=dim,
            num_blocks=num_blocks[0],
            num_heads=heads[0],
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
            low_ratio=low_ratio  # 使用传入的 low_ratio
        )

        self.baseFeature = BaseFeatureExtraction(dim=dim, num_heads=heads[2])
        self.detailFeature = DetailFeatureExtraction()

    def forward(self, inp_img):
        low, high = self.SFDM(inp_img)
        base_feature = self.baseFeature(low)
        detail_feature = self.detailFeature(high)
        return base_feature, detail_feature


class FDCFusion(nn.Module):
    def __init__(self, low_ratio_vi=0.15, low_ratio_ir=0.23):
        super(FDCFusion, self).__init__()
        # 可见光 Encoder
        self.encoder_vi = Encoder(low_ratio=low_ratio_vi)
        # 红外 Encoder
        self.encoder_ir = Encoder(low_ratio=low_ratio_ir)

        self.decoder = Decoder()
        self.baseFeature = BaseFeatureExtraction(dim=64, num_heads=8)
        self.detailFeature = DetailFeatureExtraction()
        self.FIM = FIM(dim1=64, dim2=64)

    def forward(self, vi_image, ir_image):
        feature_V_B, feature_V_D = self.encoder_vi(vi_image)
        feature_I_B, feature_I_D = self.encoder_ir(ir_image)

        feature_F_B = self.baseFeature(feature_I_B + feature_V_B)
        feature_F_D = self.detailFeature(feature_I_D + feature_V_D)
        feature = self.FIM(feature_F_B, feature_F_D)
        fused_image = self.decoder(feature)
        return fused_image




