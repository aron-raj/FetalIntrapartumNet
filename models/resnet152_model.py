import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights
class ResNet152Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=False):
        super(ResNet152Classifier, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        weights = ResNet152_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet152(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    def forward(self, x):
        return self.backbone(x)
def build_model(num_classes=2, pretrained=False, **kwargs):
    return ResNet152Classifier(num_classes=num_classes, pretrained=pretrained)
