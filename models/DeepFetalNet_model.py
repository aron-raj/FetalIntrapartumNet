import torch
import torch.nn as nn
import torch.nn.functional as F
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class LightweightCNNBackbone(nn.Module):
    def __init__(self):
        super(LightweightCNNBackbone, self).__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(32, 64, stride=2),
            SEBlock(64, reduction=8)
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=2),
            SEBlock(128, reduction=16)
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(128, 256, stride=2),
            SEBlock(256, reduction=16)
        )
        self.stage4 = nn.Sequential(
            DepthwiseSeparableConv(256, 256, stride=1),
            SEBlock(256, reduction=16)
        )
    def forward(self, x, debug=False):
        x = self.init_conv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x
class GlobalContextBranch(nn.Module):
    def __init__(self, in_channels=256, embed_dim=64, num_heads=4):
        super(GlobalContextBranch, self).__init__()
        self.proj_in = nn.Conv2d(in_channels, embed_dim, 1, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_out = nn.Conv2d(embed_dim, in_channels, 1, bias=False)
    def forward(self, x):
        b, c, h, w = x.shape
        x_proj = self.proj_in(x)
        tokens = x_proj.flatten(2).transpose(1, 2)
        attn_out, _ = self.attn(tokens, tokens, tokens)
        out = self.norm(tokens + attn_out)
        out = out.transpose(1, 2).view(b, -1, h, w)
        out = self.proj_out(out)
        return out
class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels=256):
        super(AdaptiveFusion, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels * 2, in_channels // 2)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // 2, 2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, local_feat, global_feat):
        b = local_feat.size(0)
        stats_local = self.gap(local_feat).view(b, -1)
        stats_global = self.gap(global_feat).view(b, -1)
        stats = torch.cat([stats_local, stats_global], dim=1)
        y = self.fc1(stats)
        y = self.relu(y)
        y = self.fc2(y)
        weights = self.softmax(y) # [B, 2]
        alpha_local = weights[:, 0].view(b, 1, 1, 1)
        alpha_global = weights[:, 1].view(b, 1, 1, 1)
        fused = alpha_local * local_feat + alpha_global * global_feat
        return fused
class AMFAM(nn.Module):
    def __init__(self, in_channels=256):
        super(AMFAM, self).__init__()
        self.dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, groups=in_channels, bias=False)
        self.pw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.dw2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=2, dilation=2, groups=in_channels, bias=False)
        self.pw2 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.dw3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3, dilation=3, groups=in_channels, bias=False)
        self.pw3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels * 3, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 3)
        self.softmax = nn.Softmax(dim=1)
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)    
    def forward(self, x):
        b = x.size(0)
        f1 = self.pw1(self.dw1(x))
        f2 = self.pw2(self.dw2(x))
        f3 = self.pw3(self.dw3(x))
        f_concat = torch.cat([f1, f2, f3], dim=1) 
        y = self.gap(f_concat).view(b, -1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        weights = self.softmax(y)
        alpha1 = weights[:, 0].view(b, 1, 1, 1)
        alpha2 = weights[:, 1].view(b, 1, 1, 1)
        alpha3 = weights[:, 2].view(b, 1, 1, 1)
        f_agg = alpha1 * f1 + alpha2 * f2 + alpha3 * f3
        out = self.fusion(f_agg)
        return out
class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels=256, reduction=16):
        super(AttentionRefinementModule, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        y_c = self.gap(x).view(b, c)
        y_c = self.fc1(y_c)
        y_c = self.relu(y_c)
        y_c = self.fc2(y_c)
        A_c = self.sigmoid_channel(y_c).view(b, c, 1, 1)
        F_c = x * A_c.expand_as(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y_s = torch.cat([avg_out, max_out], dim=1)
        A_s = self.sigmoid_spatial(self.conv_spatial(y_s))
        F_s = x * A_s.expand_as(x)
        F_attention = F_c + F_s
        F_refined = x + F_attention
        return F_refined
class ClassificationHead(nn.Module):
    def __init__(self, in_features=256, num_classes=2):
        super(ClassificationHead, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class M5Model(nn.Module):
    def __init__(self, num_classes=2, use_global=True, use_fusion=True, use_amfam=True, use_arm=True):
        super(M5Model, self).__init__()
        self.use_global = use_global
        self.use_fusion = use_fusion
        self.use_amfam = use_amfam
        self.use_arm = use_arm
        self.backbone = LightweightCNNBackbone()
        if self.use_amfam:
            self.amfam = AMFAM(in_channels=256)
        if self.use_global:
            self.global_branch = GlobalContextBranch(in_channels=256, embed_dim=64, num_heads=4)
        if self.use_fusion and self.use_global:
            self.adaptive_fusion = AdaptiveFusion(in_channels=256)
        if self.use_arm:
            self.arm = AttentionRefinementModule(in_channels=256, reduction=16)
        self.head = ClassificationHead(in_features=256, num_classes=num_classes)
    def forward(self, x):
        features = self.backbone(x)
        local_features = features
        if self.use_amfam:
            local_features = self.amfam(features)
        if self.use_global:
            global_features = self.global_branch(features)
            if self.use_fusion:
                fused = self.adaptive_fusion(local_features, global_features)
            else:
                fused = local_features + global_features 
        else:
            fused = local_features 
        if self.use_arm:
            fused = self.arm(fused)   
        out = self.head(fused)
        return out
if __name__ == '__main__':
    print("="*50)
    print("VERIFICATION: M5 MODEL")
    print("="*50)
    model = M5Model()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"M5 trainable parameters: {total_params:,}")
    print("-" * 50)
    dummy_input = torch.randn(2, 3, 254, 254)
    print("Running Dummy Forward Pass with tensor shape [2, 3, 254, 254]...")
    output = model(dummy_input)
    assert output.shape == (2, 2), f"Output shape is incorrect! Got {output.shape}"
    print("Verified final output is [2, 2].")
    print("="*50)