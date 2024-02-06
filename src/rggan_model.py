import torch.nn as nn
import torch.nn.init as init
import math
from torch.nn.utils.spectral_norm import spectral_norm
from dynamic_layers import MaskedConv2d, MaskedMLP

class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            MaskedConv2d(in_channels, out_channels, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            MaskedConv2d(out_channels, out_channels, (3, 3), stride=1, padding=1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            MaskedConv2d(in_channels, out_channels, (1, 1), stride=1, padding=0)
        )

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class RG_ResGenerator32(nn.Module):
    def __init__(self, z_dim, sparse_train_mode=False):
        super().__init__()
        self.z_dim = z_dim
        self.linear = MaskedMLP(z_dim, 4 * 4 * 256)
        self.sparse_train_mode = sparse_train_mode
        self.blocks = nn.Sequential(
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
            ResGenBlock(256, 256),
        )
        self.output = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            MaskedConv2d(256, 3, (3, 3), stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        # self.set_training_mode()
        z = self.linear(z)
        z = z.view(-1, 256, 4, 4)
        return self.output(self.blocks(z))


class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            MaskedConv2d(in_channels, out_channels, (1, 1), 1, 0))
        self.residual = nn.Sequential(
            MaskedConv2d(in_channels, out_channels, (3, 3), 1, 1),
            nn.ReLU(),
            MaskedConv2d(out_channels, out_channels, (3, 3), 1, 1),
            nn.AvgPool2d(2))
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, MaskedConv2d):
                # init.xavier_uniform_(m.weight, math.sqrt(2))
                # init.zeros_(m.bias)
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, MaskedConv2d):
                # init.xavier_uniform_(m.weight)
                # init.zeros_(m.bias)
                spectral_norm(m)

    def forward(self, x):
        # print(self.residual(x))
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(
                MaskedConv2d(in_channels, out_channels, (1, 1), 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            MaskedConv2d(in_channels, out_channels, (3, 3), 1, 1),
            nn.ReLU(),
            MaskedConv2d(out_channels, out_channels, (3, 3), 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        self.initialize()

    def initialize(self):
        for m in self.residual.modules():
            if isinstance(m, MaskedConv2d):
                spectral_norm(m)
        for m in self.shortcut.modules():
            if isinstance(m, MaskedConv2d):
                spectral_norm(m)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class RG_ResDiscriminator32(nn.Module):
    def __init__(self, sparse_train_mode=False):
        super().__init__()
        self.model = nn.Sequential(
            OptimizedResDisblock(3, 128),
            ResDisBlock(128, 128, down=True),
            ResDisBlock(128, 128),
            ResDisBlock(128, 128),
            nn.ReLU())
        self.linear = MaskedMLP(128, 1, bias=False)
        self.initialize()
        self.sparse_train_mode = sparse_train_mode

    def initialize(self):
        spectral_norm(self.linear)

    def forward(self, x):
        x = self.model(x).sum(dim=[2, 3])
        x = self.linear(x)
        return x

