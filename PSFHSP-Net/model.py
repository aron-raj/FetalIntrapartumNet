import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class PSFHSPNet(nn.Module):
    """
    PSFHSP-Net: An Efficient Lightweight Network for Identifying 
    Pubic Symphysis-Fetal Head Standard Plane.
    
    Architecture based on the paper:
    - Initial 7x7 Conv layer (64 channels)
    - 2x BasicBlock0 (64 channels, stride 1)
    - 1x BasicBlock1 (128 channels, stride 2)
    - Omission of standard ResNet layer3 and layer4 (Dimension Reduction modules)
    - Adaptive Avg Pool
    - Fully Connected Layer (2 classes)
    """
    def __init__(self, num_classes=2):
        super(PSFHSPNet, self).__init__()
        self.in_channels = 64

        # Initial single convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Three residual blocks total: 
        # Two BasicBlock0 (no downsampling, 64 channels)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=1)
        
        # One modified BasicBlock1 (with downsampling, 128 channels)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    model = PSFHSPNet(num_classes=2)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PSFHSP-Net Parameters: {params:,}")
    
    x = torch.randn(1, 3, 254, 254)
    out = model(x)
    print(f"Output shape: {out.shape}")
