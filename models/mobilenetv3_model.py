import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
class MobileNetV3LargeClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(MobileNetV3LargeClassifier, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)
def build_model(num_classes=2, pretrained=False, **kwargs):
    return MobileNetV3LargeClassifier(num_classes=num_classes, pretrained=pretrained)
