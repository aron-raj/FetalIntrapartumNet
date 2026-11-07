from .model import PSFHSPNet
from .init_weights import initialize_with_pretrained

def build_model(num_classes=2, pretrained=False, verbose=False, **kwargs):
    """
    Unified factory interface for PSFHSP-Net.
    When pretrained=True, transfers official ResNet-18 ImageNet weights
    into matching layers (conv1, bn1, layer1, layer2).
    """
    model = PSFHSPNet(num_classes=num_classes)
    if pretrained:
        initialize_with_pretrained(model, verbose=verbose)
    return model
