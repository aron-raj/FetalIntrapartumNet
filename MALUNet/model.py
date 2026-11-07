import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math


# ================================
# DepthWise Conv
# ================================
class DepthWiseConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size=3, padding=1, stride=1, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            dim_in,
            dim_in,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            groups=dim_in
        )

        self.norm_layer = nn.GroupNorm(4, dim_in)
        self.conv2 = nn.Conv2d(dim_in, dim_out, kernel_size=1)

    def forward(self, x):
        return self.conv2(self.norm_layer(self.conv1(x)))


# ================================
# Gated Attention Unit
# ================================
class GatedAttentionUnit(nn.Module):
    def __init__(self, in_c, out_c, kernel_size):
        super().__init__()

        self.w1 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size, padding=kernel_size // 2),
            nn.Sigmoid()
        )

        self.w2 = nn.Sequential(
            DepthWiseConv2d(in_c, in_c, kernel_size + 2, padding=(kernel_size + 2) // 2),
            nn.GELU()
        )

        self.wo = nn.Sequential(
            DepthWiseConv2d(in_c, out_c, kernel_size),
            nn.GELU()
        )

        self.cw = nn.Conv2d(in_c, out_c, kernel_size=1)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        return self.wo(x1 * x2) + self.cw(x)


# ================================
# Dilated Gated Attention
# ================================
class DilatedGatedAttention(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 3, padding=1, dilation=1, groups=in_c)
        self.conv2 = nn.Conv2d(in_c, in_c, 3, padding=2, dilation=2, groups=in_c)
        self.conv3 = nn.Conv2d(in_c, in_c, 3, padding=3, dilation=3, groups=in_c)

        self.norm = nn.GroupNorm(4, in_c)
        self.conv_final = nn.Conv2d(in_c, in_c, 1)

        self.gau = GatedAttentionUnit(in_c, out_c, 3)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x = x1 + x2 + x3
        x = F.gelu(self.conv_final(self.norm(x)))

        return self.gau(x)


# ================================
# EA Block
# ================================
class EAblock(nn.Module):
    def __init__(self, in_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, in_c, 1)

        self.k = in_c * 4
        self.linear_0 = nn.Conv1d(in_c, self.k, 1, bias=False)
        self.linear_1 = nn.Conv1d(self.k, in_c, 1, bias=False)

        self.conv2 = nn.Conv2d(in_c, in_c, 1)
        self.norm = nn.GroupNorm(4, in_c)

    def forward(self, x):
        identity = x

        x = self.conv1(x)

        b, c, h, w = x.shape
        x = x.view(b, c, h * w)

        attn = self.linear_0(x)
        attn = F.softmax(attn, dim=-1)

        x = self.linear_1(attn)
        x = x.view(b, c, h, w)

        x = self.norm(self.conv2(x))
        x = x + identity

        return F.gelu(x)


# ================================
# MALUNet
# ================================
class MALUNet(nn.Module):
    def __init__(self, num_classes=3, input_channels=3):
        super().__init__()

        c = [8, 16, 24, 32, 48, 64]

        # Encoder
        self.encoder1 = nn.Conv2d(input_channels, c[0], 3, padding=1)
        self.encoder2 = nn.Conv2d(c[0], c[1], 3, padding=1)
        self.encoder3 = nn.Conv2d(c[1], c[2], 3, padding=1)

        self.encoder4 = nn.Sequential(
            EAblock(c[2]),
            DilatedGatedAttention(c[2], c[3])
        )

        self.encoder5 = nn.Sequential(
            EAblock(c[3]),
            DilatedGatedAttention(c[3], c[4])
        )

        self.encoder6 = nn.Sequential(
            EAblock(c[4]),
            DilatedGatedAttention(c[4], c[5])
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            DilatedGatedAttention(c[5], c[4]),
            EAblock(c[4])
        )

        self.decoder2 = nn.Sequential(
            DilatedGatedAttention(c[4], c[3]),
            EAblock(c[3])
        )

        self.decoder3 = nn.Sequential(
            DilatedGatedAttention(c[3], c[2]),
            EAblock(c[2])
        )

        self.decoder4 = nn.Conv2d(c[2], c[1], 3, padding=1)
        self.decoder5 = nn.Conv2d(c[1], c[0], 3, padding=1)

        self.final = nn.Conv2d(c[0], num_classes, kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):

        # ========================
        # Encoder
        # ========================
        t1 = F.gelu(self.encoder1(x))      # 256
        p1 = F.max_pool2d(t1, 2, 2)        # 128

        t2 = F.gelu(self.encoder2(p1))     # 128
        p2 = F.max_pool2d(t2, 2, 2)        # 64

        t3 = F.gelu(self.encoder3(p2))     # 64
        p3 = F.max_pool2d(t3, 2, 2)        # 32

        t4 = F.gelu(self.encoder4(p3))     # 32
        p4 = F.max_pool2d(t4, 2, 2)        # 16

        t5 = F.gelu(self.encoder5(p4))     # 16
        p5 = F.max_pool2d(t5, 2, 2)        # 8

        bottleneck = F.gelu(self.encoder6(p5))  # 8

        # ========================
        # Decoder
        # ========================
        out = self.decoder1(bottleneck)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = out + t5                     # 16

        out = self.decoder2(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = out + t4                     # 32

        out = self.decoder3(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = out + t3                     # 64

        out = self.decoder4(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = out + t2                     # 128

        out = self.decoder5(out)
        out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        out = out + t1                     # 256

        # Final output
        out = self.final(out)

        # IMPORTANT: No sigmoid for CrossEntropyLoss
        return out