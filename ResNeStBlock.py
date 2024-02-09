import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitAttentionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, radix=2, reduction_factor=4):
        super(SplitAttentionConv2d, self).__init__()
        inter_channels = max(in_channels*radix//reduction_factor, 32)
        self.radix = radix
        self.cardinality = groups
        self.channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels*radix, kernel_size, stride, padding, dilation, groups=groups*radix, bias=False)
        self.bn0 = nn.BatchNorm2d(out_channels*radix)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(out_channels, inter_channels, 1, groups=self.cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(inter_channels, out_channels*radix, 1, groups=self.cardinality)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn0(x)
        x = self.relu(x)

        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel//self.radix, dim=1)
            gap = sum(splited)
        else:
            gap = x
        gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)
        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten).view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel//self.radix, dim=1)
            out = sum([att*split for (att, split) in zip(attens, splited)])
        else:
            out = atten * x
        return out.contiguous()

class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super(RadixSoftmax, self).__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x

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
            out_channels, 
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
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        if self.is_first:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
        
        out = self.split_attention(x)
        out = self.conv2(out)
        out = self.bn2(out)

        shortcut = self.shortcut(x)
        out += shortcut
        return F.relu(out)
