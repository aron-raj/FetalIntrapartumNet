import torch
import torch.nn as nn
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
class SqueezeNet11Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(SqueezeNet11Classifier, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        weights = SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = squeezenet1_1(weights=weights)
        self.backbone.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
        self.backbone.num_classes = num_classes
    def forward(self, x):
        return self.backbone(x)
def build_model(num_classes=2, pretrained=False, **kwargs):
    return SqueezeNet11Classifier(num_classes=num_classes, pretrained=pretrained)
