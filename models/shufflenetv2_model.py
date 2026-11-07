import math
import torch
import torch.nn as nn
from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
def channel_shuffle(x, groups=2):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x
class ShuffleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ShuffleBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        branch_features = out_channels // 2
        assert (self.stride != 1) or (in_channels == branch_features << 1)
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, branch_features, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels if (self.stride > 1) else branch_features,
                      branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_features, branch_features, 3, stride=stride, padding=1, groups=branch_features, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        return channel_shuffle(out, groups=2)
class ShuffleNetV2Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ShuffleNetV2Classifier, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        if pretrained:
            self.backbone = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        else:
            stage_repeats = [4, 8, 4]
            stage_out_channels = [24, 116, 232, 464, 1024]
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, stage_out_channels[0], 3, 2, 1, bias=False),
                nn.BatchNorm2d(stage_out_channels[0]),
                nn.ReLU(inplace=True),
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            input_channel = stage_out_channels[0]
            stages = []
            for stage_idx in range(len(stage_repeats)):
                num_repeat = stage_repeats[stage_idx]
                output_channel = stage_out_channels[stage_idx + 1]
                for i in range(num_repeat):
                    stride = 2 if i == 0 else 1
                    stages.append(ShuffleBlock(input_channel, output_channel, stride=stride))
                    input_channel = output_channel
            self.stages = nn.Sequential(*stages)
            self.conv5 = nn.Sequential(
                nn.Conv2d(input_channel, stage_out_channels[-1], 1, 1, 0, bias=False),
                nn.BatchNorm2d(stage_out_channels[-1]),
                nn.ReLU(inplace=True),
            )
            self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(stage_out_channels[-1], num_classes)
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
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stages(x)
        x = self.conv5(x)
        x = self.adaptive_pool(x)
        x = x.squeeze(3).squeeze(2)
        x = self.fc(x)
        return x

def build_model(num_classes=2, pretrained=False, **kwargs):
    return ShuffleNetV2Classifier(num_classes=num_classes, pretrained=pretrained)
