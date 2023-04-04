from typing import List

import torch
from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                               stride=stride, padding='same', padding_mode='reflect')
        self.norm1 = nn.BatchNorm1d(channels)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                               stride=stride, padding='same', padding_mode='reflect')
        self.norm2 = nn.BatchNorm1d(channels)
        self.activation2 = nn.ReLU()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual
        out = self.activation2(out)
        return out


class Bottleneck(nn.Module):

    def __init__(self, n_blocks: int, channels: int, kernel_size: int, stride: int):
        super().__init__()
        self.bottleneck = nn.Sequential()
        for _ in range(n_blocks):
            self.bottleneck.append(ResidualBlock(channels, kernel_size, stride))
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottleneck(x)


class SincFilters(nn.Module):

    def __init__(self, filters: int, kernel_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=filters, kernel_size=kernel_size)
        self.pool = nn.MaxPool1d(3)
        self.norm = nn.BatchNorm1d(filters)
        self.activation = nn.LeakyReLU()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = torch.abs(out)
        out = self.pool(out)
        out = self.norm(out)
        out = self.activation(out)
        return out


def _make_bottleneck(n_blocks, in_channels: int, out_channels: int, kernel_size: int,
                     stride: int = 1) -> nn.Module:
    bottleneck = Bottleneck(n_blocks, in_channels, kernel_size, stride=stride)
    if in_channels != out_channels:
        bottleneck = nn.Sequential(
            bottleneck,
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_channels)
        )
    return bottleneck


class ResNet(nn.Module):

    def __init__(self, n_res_blocks, kernel_size, channels: List[int]):
        super().__init__()
        self.conv_in = nn.Conv1d(in_channels=1, out_channels=channels[0], kernel_size=kernel_size)
        self.norm = nn.BatchNorm1d(channels[0])
        self.activation1 = nn.ReLU()
        self.pool_in = nn.MaxPool1d(3)
        self.residual_activation = nn.ReLU()
        self.bottlenecks = nn.Sequential()
        channels.append(channels[-1])
        for i in range(len(n_res_blocks)):
            self.bottlenecks.append(_make_bottleneck(n_res_blocks[i], channels[i], channels[i + 1], kernel_size))
        self.pool_out = nn.AvgPool1d(kernel_size=1)
        self.fc = nn.Linear(channels[-1], 2)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        out = self.norm(out)
        out = self.activation1(out)
        out = self.pool_in(out)
        out = self.bottlenecks(out)
        out = self.pool_out(out)
        out = out.mean(-1)
        out = self.fc(out)
        return out
