import torch.nn as nn
from timm.models.registry import register_model
from .fastermlp import FasterMLP


@register_model
def fastermlp(**kwargs):
    model = FasterMLP(**kwargs)
    return model
