import torch.nn as nn
from module.stoch_norm import StochBatchNorm2d, StochNorm2d
from torchvision import models

__all__ = ['ResNet50_F', 'ResNet50']


class ResNet50_F(nn.Module):
    def __init__(self, pretrained=True, norm_layer=None):
        super(ResNet50_F, self).__init__()
        model_resnet50 = models.resnet50(
            pretrained=pretrained, norm_layer=norm_layer)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

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
        x = x.view(x.size(0), -1)

        return x

    @property
    def output_dim(self):
        return self.__in_features


class ResNet50(nn.Module):
    def __init__(self, pretrained=False, num_classes=1000, norm=nn.BatchNorm2d):
        super(ResNet50, self).__init__()
        self.backbone = ResNet50_F(pretrained=pretrained, norm_layer=norm)
        self.head = nn.Linear(self.backbone.output_dim, num_classes)
        self.head.weight.data.normal_(0, 0.01)
        self.head.bias.data.fill_(0.0)

    def forward_feature(self, x):
        return self.backbone(x)

    def forward(self, x):
        feature = self.forward_feature(x)
        out = self.head(feature)
        return out


if __name__ == "__main__":
    net = ResNet50_F(norm_layer=StochBatchNorm2d)
    # print(net)
    # print(net.bn1.running_mean)
    # print(net.bn1.running_var)
    print(net.bn1.bias)
    print(net.bn1.weight)
    print(net.bn1.k)
