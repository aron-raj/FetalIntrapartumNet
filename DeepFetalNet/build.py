import warnings
from .model import M5Model

def build_model(num_classes=2, pretrained=False, **kwargs):
    return M5Model(num_classes=num_classes)
