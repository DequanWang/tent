import torch

from torch import nn
from torch.nn import BatchNorm2d

from conf import cfg


def get_norm(out_channels):
    """
    Args:
        norm (str or callable): one of the batch normalization types
        (BatchNorm2d, TrainModeBatchNorm2d, FrozenMeanVarBatchNorm2d).
    Returns:
        nn.Module or None: the normalization module.
    """
    return globals()[cfg.BN.FUNC](out_channels, cfg.BN.EPS, cfg.BN.MOM)


class TrainModeBatchNorm2d(nn.Module):
    __constants__ = ['eps', 'num_features']

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features, self.eps = num_features, eps
        self.register_parameter("weight",
                                nn.Parameter(torch.ones(num_features)))
        self.register_parameter("bias",
                                nn.Parameter(torch.zeros(num_features)))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, affine=True'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        current_mean = x.mean([0, 2, 3])
        current_var = x.var([0, 2, 3], unbiased=False)
        scale = self.weight * (current_var + self.eps).rsqrt()
        bias = self.bias - current_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias


class FrozenMeanVarBatchNorm2d(nn.Module):
    __constants__ = ['eps', 'num_features']

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features, self.eps = num_features, eps
        self.register_parameter("weight",
                                nn.Parameter(torch.ones(num_features)))
        self.register_parameter("bias",
                                nn.Parameter(torch.zeros(num_features)))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def extra_repr(self):
        return '{num_features}, eps={eps}, affine=True'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias
