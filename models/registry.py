import torch.nn as nn
from timm.models.registry import register_model
from .fasternet1 import FasterNet


@register_model
def fasternet(**kwargs):
    model = FasterNet(**kwargs)
    return model