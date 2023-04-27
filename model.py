import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class DeformableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformableConv2d, self).__init__()
        self.padding = padding
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=True)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_channels, 1 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=True)
        nn.init.constant_(self.modulator_conv.weight, 0.) # 0.5
        nn.init.constant_(self.modulator_conv.bias, 0.) # 0.5
        self.regular_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=self.padding, bias=bias)

    def forward(self, x):
        h, w = x.shape[2:]
        max_offset = max(h, w) / 4.
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x, offset=offset, weight=self.regular_conv.weight, bias=self.regular_conv.bias, padding=self.padding, mask=modulator)
        return x

# class Net(nn.Module):
#     def __init__(self, num_classes=576, reid=False):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = DeformableConv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
#         self.bn4 = nn.BatchNorm2d(128)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.reid = reid
#         self.classifier = nn.Linear(128, num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = self.bn1(x)
#         x = F.relu(self.conv2(x))
#         x = self.bn2(x)
#         x = F.relu(self.conv3(x))
#         x = self.bn3(x)
#         x = F.relu(self.conv4(x))
#         x = self.bn4(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         if self.reid:
#             x = x.div(x.norm(p=2, dim=1, keepdim=True))
#             return x
#         x = self.classifier(x)
#         return x

class Net(nn.Module):
    def __init__(self, num_classes=576, reid=False):
        super(Net, self).__init__()
        import torchvision.models as models
        resnet50 = models.resnet50(pretrained=True)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.dcn = DeformableConv2d(2048, 2048, kernel_size=3, stride=1, padding=1, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.reid = reid
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.dcn(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.reid:
            x = x.div(x.norm(p=2, dim=1, keepdim=True))
            return x
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    net = Net()
    x = torch.randn(4, 3, 128, 64) # [batch size, channel, height, width]
    y = net.forward(x)
    print(y)