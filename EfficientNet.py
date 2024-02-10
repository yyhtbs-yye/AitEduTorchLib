from EfficientNetBlock import *

class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNet, self).__init__()
        # Assuming 'channels' and 'repeats' are lists defining the architecture
        # For EfficientNet-B0, something like:
        # channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        # repeats = [1, 2, 2, 3, 3, 4, 1]
        # strides = [1, 2, 2, 2, 1, 2, 1]
        # kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        # This part of the code would initialize the layers based on the architecture

        self.features = nn.ModuleList([nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(channels[0]),
                                       nn.ReLU(inplace=True)])
        # Add MBConv blocks based on architecture defined in 'channels' and 'repeats'
        for i in range(len(repeats)):
            stride = strides[i]
            in_channels = channels[i]
            out_channels = channels[i + 1]
            for j in range(repeats[i]):
                if j > 0:
                    stride = 1
                    in_channels = out_channels
                self.features.append(MBConvBlock(in_channels, out_channels, expansion_factor=6, stride=stride, kernel_size=kernel_sizes[i]))

        # Final layers
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EfficientNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super(EfficientNetV2, self).__init__()
        # Parameters for EfficientNetV2-M. These need to be adjusted based on the specific variant
        # Note: The following are example values; actual values should match EfficientNetV2-M specifics
        channels = [24, 48, 80, 160, 176, 304, 512, 1280]
        repeats = [2, 4, 4, 6, 9, 15, 5]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 3, 3, 3, 3, 3]
        expansion_factors = [1, 4, 6, 6, 6, 6, 6]  # Example, adjust based on model specifics
        se_ratios = [None, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]  # Squeeze-and-Excitation ratio per block

        self.features = nn.ModuleList([nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
                                       nn.BatchNorm2d(channels[0]),
                                       nn.ReLU(inplace=True)])

        # Build the EfficientNetV2-M blocks
        for i in range(len(repeats)):
            output_channel = channels[i+1]
            for j in range(repeats[i]):
                stride = strides[i] if j == 0 else 1
                in_channel = channels[i] if j == 0 else output_channel
                if i < 2:  # Using FusedMBConvBlock for the first two stages
                    self.features.append(FusedMBConvBlock(in_channel, output_channel, kernel_sizes[i], stride, expansion_factor=expansion_factors[i], se_ratio=se_ratios[i]))
                else:
                    self.features.append(MBConvBlock(in_channel, output_channel, expansion_factor=expansion_factors[i], stride=stride, kernel_size=kernel_sizes[i], reduction=4))  # Assuming MBConvBlock is defined elsewhere

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
