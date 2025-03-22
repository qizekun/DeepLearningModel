import torch
import torch.nn as nn


# 定义 ResNet 的基本残差块（Bottleneck Block）
class Bottleneck(nn.Module):
    expansion = 4  # 通道扩展倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)  # 1x1 卷积
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                               bias=False)  # 3x3 卷积
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)  # 1x1 卷积
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 维度匹配的下采样

    def forward(self, x):
        identity = x  # 残差连接

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity  # 残差连接
        out = self.relu(out)

        return out


# 定义 ResNet-50 结构
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        self.in_channels = 64  # 初始输入通道数

        # 初始 7x7 卷积层 + 最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet-50 的四个残差层
        self.layer1 = self._make_layer(64, 3, stride=1)  # 3 个 Bottleneck Block
        self.layer2 = self._make_layer(128, 4, stride=2)  # 4 个 Bottleneck Block
        self.layer3 = self._make_layer(256, 6, stride=2)  # 6 个 Bottleneck Block
        self.layer4 = self._make_layer(512, 3, stride=2)  # 3 个 Bottleneck Block

        # 全局平均池化 + 全连接分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        """构建 ResNet 的层，包括多个 Bottleneck Block"""
        downsample = None

        # 如果 stride > 1 或通道数不匹配，则需要下采样
        if stride != 1 or self.in_channels != out_channels * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * Bottleneck.expansion, kernel_size=1, stride=stride,
                          bias=False),
                nn.BatchNorm2d(out_channels * Bottleneck.expansion)
            )

        layers = []
        layers.append(Bottleneck(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * Bottleneck.expansion  # 更新通道数

        # 添加剩余的 Bottleneck Block
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x