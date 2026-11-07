import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, g):
        g1  = self.W_g(g)
        x1  = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi
class GraphAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.theta = nn.Conv2d(in_ch, in_ch // 2, 1)
        self.phi   = nn.Conv2d(in_ch, in_ch // 2, 1)
        self.g     = nn.Conv2d(in_ch, in_ch, 1)
        self.out   = nn.Conv2d(in_ch, in_ch, 1)
    def forward(self, x):
        B, C, H, W = x.shape
        theta = self.theta(x).view(B, -1, H * W)
        phi   = self.phi(x).view(B, -1, H * W)
        g     = self.g(x).view(B, -1, H * W)
        attn  = torch.bmm(theta.transpose(1, 2), phi)
        attn  = F.softmax(attn, dim=-1)
        out   = torch.bmm(g, attn.transpose(1, 2))
        out   = out.view(B, C, H, W)
        return self.out(out) + x
class FAT_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super().__init__()
        self.e1   = DoubleConv(n_channels, 32)
        self.e2   = DoubleConv(32,  64)
        self.e3   = DoubleConv(64,  128)
        self.e4   = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.graph      = GraphAttention(512)
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.ag1 = AttentionGate(256, 256, 128)
        self.d1  = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.ag2 = AttentionGate(128, 128, 64)
        self.d2  = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.ag3 = AttentionGate(64, 64, 32)
        self.d3  = DoubleConv(128, 64)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.ag4 = AttentionGate(32, 32, 16)
        self.d4  = DoubleConv(64, 32)
        self.out_conv = nn.Conv2d(32, n_classes, 1)
    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        b = self.graph(b)
        d1 = self.up1(b)
        e4 = self.ag1(e4, d1)
        d1 = self.d1(torch.cat([d1, e4], dim=1))
        d2 = self.up2(d1)
        e3 = self.ag2(e3, d2)
        d2 = self.d2(torch.cat([d2, e3], dim=1))
        d3 = self.up3(d2)
        e2 = self.ag3(e2, d3)
        d3 = self.d3(torch.cat([d3, e2], dim=1))
        d4 = self.up4(d3)
        e1 = self.ag4(e1, d4)
        d4 = self.d4(torch.cat([d4, e1], dim=1))
        return self.out_conv(d4)