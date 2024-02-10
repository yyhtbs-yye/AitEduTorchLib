import torch
import torch.nn as nn
import torch.nn.functional as F

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(input_channels, reduced_dim, 1),
            nn.ReLU(),
            nn.Conv2d(reduced_dim, input_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, reduction_ratio=4, survival_prob=0.8):
        super(FusedMBConv, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            SEBlock(mid_channels, max(1, in_channels // reduction_ratio)),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.training and self.use_residual:
            # Stochastic depth for regularization
            if torch.rand(1) < self.survival_prob:
                return x + self.conv(x)
            return x
        else:
            return x + self.conv(x) if self.use_residual else self.conv(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, kernel_size, reduction=4):
        super(MBConvBlock, self).__init__()
        self.use_res_connect = stride == 1 and in_channels == out_channels

        self.expand_conv = nn.Conv2d(in_channels, in_channels * expansion_factor, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(in_channels * expansion_factor)
        self.depthwise_conv = nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, kernel_size, stride=stride, padding=kernel_size//2, groups=in_channels * expansion_factor, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion_factor)
        self.se = SqueezeExcitation(in_channels * expansion_factor, in_channels * expansion_factor // reduction)
        self.project_conv = nn.Conv2d(in_channels * expansion_factor, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x

        x = F.relu6(self.bn0(self.expand_conv(x)))
        x = F.relu6(self.bn1(self.depthwise_conv(x)))
        x = self.se(x)
        x = self.bn2(self.project_conv(x))

        if self.use_res_connect:
            x += identity

        return x

class SEBlock(nn.Module):  # Squeeze-and-Excitation block
    def __init__(self, in_channels, reduced_dim):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_dim, kernel_size=1),
            nn.SiLU(),  # Sigmoid Linear Unit, also known as Swish
            nn.Conv2d(reduced_dim, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)

class FusedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, reduction_ratio=4, survival_prob=0.8):
        super(FusedMBConv, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            SEBlock(mid_channels, max(1, in_channels // reduction_ratio)),
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.training and self.use_residual:
            # Stochastic depth for regularization
            if torch.rand(1) < self.survival_prob:
                return x + self.conv(x)
            return x
        else:
            return x + self.conv(x) if self.use_residual else self.conv(x)

class MBConv(nn.Module):  # Similar to the original MBConv but can include SiLU and SEBlock
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, reduction_ratio=4, survival_prob=0.8):
        super(MBConv, self).__init__()
        self.survival_prob = survival_prob
        self.use_residual = stride == 1 and in_channels == out_channels
        mid_channels = in_channels * expand_ratio

        self.conv = nn.Sequential(
            # Expand
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            # Depthwise
            nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, kernel_size//2, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(),
            SEBlock(mid_channels, max(1, in_channels // reduction_ratio)),
            # Project
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        if self.training and self.use_residual:
            if torch.rand(1) < self.survival_prob:
                return x + self.conv(x)
            return x
        else:
            return x + self.conv(x) if self.use_residual else self.conv(x)
