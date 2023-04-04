from typing import Any

from torch import nn

from Modules.raw_net import RawNet
from Modules.res_net import ResNet


def get_model(model_name: str, **model_args: Any) -> nn.Module:
    if model_name == 'ResNet':
        model = ResNet([3, 4], 3, [64, 128])
    elif model_name == 'RawNet':
        model = RawNet([128, 512], 128, 2, 1024, 2)
    else:  # model_name == 'resnet':
        model = ResNet([3, 4], 2, [64, 128])
    return model
