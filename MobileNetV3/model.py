import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class MobileNetV3LargeClassifier(nn.Module):
    """
    MobileNetV3-Large adapted for JNU-IFM binary classification.
    Uses Hardswish activations, Squeeze-and-Excitation attention modules,
    and produces raw logits of shape [B, num_classes].
    """
    def __init__(self, num_classes=2, pretrained=False):
        super(MobileNetV3LargeClassifier, self).__init__()
        self.num_classes = num_classes
        self.pretrained = pretrained

        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)

        # Classifier structure:
        # classifier[0]: Linear(960, 1280)
        # classifier[1]: Hardswish()
        # classifier[2]: Dropout(p=0.2)
        # classifier[3]: Linear(1280, 1000) -> replace with Linear(1280, num_classes)
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def build_model(num_classes=2, pretrained=False, **kwargs):
    """
    Consistent factory interface for MobileNetV3-Large.
    """
    return MobileNetV3LargeClassifier(num_classes=num_classes, pretrained=pretrained)
