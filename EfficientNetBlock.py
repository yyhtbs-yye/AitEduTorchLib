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

class FusedMBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion_factor=1, se_ratio=None):
        super(FusedMBConvBlock, self).__init__()
        self.use_se = se_ratio is not None and se_ratio > 0
        mid_channels = in_channels * expansion_factor

        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size, stride=stride, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        if self.use_se:
            num_squeezed_channels = max(1, int(in_channels * se_ratio))
            self.se = SqueezeExcitation(mid_channels, num_squeezed_channels)

        self.project_conv = nn.Conv2d(mid_channels, out_channels, 1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        if self.use_se:
            x = self.se(x)

        x = self.project_conv(x)
        x = self.project_bn(x)
        return x
