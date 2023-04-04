import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Union


def to_mel(hz: float) -> np.ndarray:
    return 2595 * np.log10(1 + hz / 700)


def to_hz(mel: np.ndarray) -> np.ndarray:
    return 700 * (10 ** (mel / 2595) - 1)


class SincFilters(nn.Module):

    def __init__(self, kernel_size: int, out_channels: int, stride: int = 1, padding: int = 0, dilation: int = 1,
                 sample_rate: int = 16000, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        self.out_channels = out_channels + 1
        self.kernel_size = kernel_size + (1 - (kernel_size % 2))
        self.sample_rate = sample_rate

        hz = int(sample_rate / 2) * np.linspace(0, 0, 1)
        filters_mel = to_mel(hz)
        filters_mel_max = filters_mel.max()
        filters_mel_min = filters_mel.min()
        fil_band_widths_mel = np.linspace(filters_mel_min, filters_mel_max, self.out_channels + 2)
        fil_band_widths_f = to_hz(fil_band_widths_mel)
        self.freq = fil_band_widths_f[:self.out_channels]

        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels - 1, self.kernel_size)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.freq) - 1):
            fmin = self.freq[i]
            fmax = self.freq[i + 1]
            h_high = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            h_low = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = h_high - h_low

            self.band_pass[i, :] = torch.as_tensor(np.hamming(self.kernel_size)) * torch.as_tensor(hideal)

        band_pass_filter = self.band_pass.to(x.device)

        filters = band_pass_filter.view(self.out_channels - 1, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)


class FMS(nn.Module):

    def __init__(self, in_features: int, out_features: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features, out_features)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.pool(x).view(x.size(0), -1)
        out = self.fc(out)
        out = self.activation(out).view(x.size(0), x.size(1), -1)
        out = x * out + out
        return out


class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.norm1 = nn.BatchNorm1d(in_channels)
        self.activation = nn.LeakyReLU()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,
                               padding_mode='reflect', stride=1)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                               padding_mode='reflect', stride=1)
        if in_channels != out_channels:
            self.conv_down_sample = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                              stride=1, padding=0)
        else:
            self.conv_down_sample = nn.Identity()
        self.pool = nn.MaxPool1d(3)
        self.fms = FMS(out_channels, out_channels)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.norm1(x)
        out = self.activation(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.conv2(out)
        x = self.conv_down_sample(x)
        out += x
        out = self.pool(out)

        out = self.fms(out)
        return out


class RawNet(nn.Module):

    def __init__(self, channels: List[Union[int, List[int]]], kernel_size: int, n_res_blocks: int, gru_node: int,
                 n_gru_layers: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sinc = SincFilters(kernel_size, channels[0])
        self.pool = nn.MaxPool1d(3)
        self.norm1 = nn.BatchNorm1d(channels[0])
        self.activation = nn.LeakyReLU()
        self.res_blocks = nn.Sequential()
        for i in range(n_res_blocks):
            self.res_blocks.append(ResidualBlock(channels[0], channels[0]))
        self.res_blocks.append(ResidualBlock(channels[0], channels[1]))
        for i in range(n_res_blocks):
            self.res_blocks.append(ResidualBlock(channels[1], channels[1]))
        self.norm2 = nn.BatchNorm1d(channels[1])
        self.recurrent = nn.GRU(channels[1], gru_node, n_gru_layers, batch_first=True)
        self.fc1 = nn.Linear(gru_node, gru_node)
        self.fc2 = nn.Linear(gru_node, 2, bias=True)
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.sinc(x)
        out = self.pool(out)
        out = self.activation(out)
        out = F.selu(out)
        out = self.res_blocks(out)
        out = self.norm2(out)
        out = F.selu(out)
        self.recurrent.flatten_parameters()
        out = self.recurrent(out.permute(0, 2, 1))[0]
        out = out[:, -1, :]
        return out
