from DenseNetBlock import *
class DenseNet(nn.Module):
    def __init__(self, num_classes=1000, growth_rate=32, block_config=(6, 12, 24, 16), 
                 num_init_features=64, bn_size=4, drop_rate=0):
        super(DenseNet, self).__init__()

        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = nn.Sequential(*[DenseLayer(num_features + i * growth_rate, growth_rate, bn_size, drop_rate) 
                                    for i in range(num_layers)])
            self.features.add_module('denseblock%d' % (i + 1), block)

            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
