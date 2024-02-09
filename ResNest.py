from ResNestBlock import *

class ResNeStBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, groups=32, radix=2, reduction_factor=4, is_first=False):
        super(ResNeStBottleneck, self).__init__()
        self.is_first = is_first
        if is_first:
            self.relu = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.split_attention = SplitAttentionConv2d(
            in_channels if is_first else out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            groups=groups, 
            radix=radix, 
            reduction_factor=reduction_factor
        )
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion)
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size
