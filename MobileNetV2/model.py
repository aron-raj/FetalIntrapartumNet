"""
MobileNetV2 Architecture Implementation.
Adapted from: https://github.com/d-li14/mobilenetv2.pytorch
Original Paper: Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.
License: MIT License (d-li14 / tonylins)
"""

import math
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(conv_1x1_bn(inp, hidden_dim))
        layers.extend([
            # Depthwise convolution
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear convolution
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2Classifier(nn.Module):
    """
    MobileNetV2 adapted for JNU-IFM binary classification.
    Produces raw logits of shape [B, num_classes].
    """
    def __init__(self, num_classes=2, pretrained=False, width_mult=1.0):
        super(MobileNetV2Classifier, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        if pretrained:
            # Transfer official ImageNet weights reliably
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        else:
            # Pure modular implementation
            block = InvertedResidual
            input_channel = 32
            last_channel = 1280
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

            input_channel = _make_divisible(input_channel * width_mult, 8)
            self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), 8)

            features = [conv_3x3_bn(3, input_channel, 2)]
            for t, c, n, s in inverted_residual_setting:
                output_channel = _make_divisible(c * width_mult, 8)
                for i in range(n):
                    stride = s if i == 0 else 1
                    features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    input_channel = output_channel
            features.append(conv_1x1_bn(input_channel, self.last_channel))
            self.features = nn.Sequential(*features)

            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        if self.pretrained:
            return self.backbone(x)
        x = self.features(x)
        x = x.mean([2, 3])
        x = self.classifier(x)
        return x

def build_model(num_classes=2, pretrained=False, **kwargs):
    """
    Consistent factory interface for MobileNetV2.
    """
    return MobileNetV2Classifier(num_classes=num_classes, pretrained=pretrained)
