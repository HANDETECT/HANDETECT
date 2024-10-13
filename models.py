import torch
import torch.nn as nn
from torchvision.models import (
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn,
    resnet18, resnet50, efficientnet_b3, densenet169,
    resnext50_32x4d, mobilenet_v3_small, shufflenet_v2_x2_0,
    squeezenet1_0, squeezenet1_1, inception_v3,
    mnasnet0_5, mnasnet0_75, mnasnet1_0
)
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


import torch
import torch.nn as nn
import torchvision.models as models

class ResNetWithCurvature(nn.Module):
    def __init__(self, num_classes=3, curvature_feature_size=100):  # Set curvature_feature_size to the correct value
        super(ResNetWithCurvature, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        num_resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Replace final fc layer with identity
        
        # Adjusted to 612
        self.fc = nn.Linear(612, num_classes)  # Adjust for combined features

    def forward(self, x, curvature_features):
        x = self.resnet(x)
        combined_features = torch.cat((x, curvature_features), dim=1)
        out = self.fc(combined_features.float())
        return out




class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet18 = resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet18(x)
    

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnet50(x)


class VGG11(nn.Module):
    def __init__(self, num_classes):
        super(VGG11, self).__init__()
        self.vgg11 = vgg11(pretrained=True)
        self.vgg11.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg11(x)


class VGG11BN(nn.Module):
    def __init__(self, num_classes):
        super(VGG11BN, self).__init__()
        self.vgg11_bn = vgg11_bn(pretrained=True)
        self.vgg11_bn.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg11_bn(x)
    

class VGG13(nn.Module):
    def __init__(self, num_classes):
        super(VGG13, self).__init__()
        self.vgg13 = vgg13(pretrained=True)
        self.vgg13.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg13(x)


class VGG13BN(nn.Module):
    def __init__(self, num_classes):
        super(VGG13BN, self).__init__()
        self.vgg13_bn = vgg13_bn(pretrained=True)
        self.vgg13_bn.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg13_bn(x)


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.vgg16 = vgg16(pretrained=True)
        self.vgg16.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16(x)


class VGG16BN(nn.Module):
    def __init__(self, num_classes):
        super(VGG16BN, self).__init__()
        self.vgg16_bn = vgg16_bn(pretrained=True)
        self.vgg16_bn.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg16_bn(x)


class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.vgg19 = vgg19(pretrained=True)
        self.vgg19.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg19(x)


class VGG19BN(nn.Module):
    def __init__(self, num_classes):
        super(VGG19BN, self).__init__()
        self.vgg19_bn = vgg19_bn(pretrained=True)
        self.vgg19_bn.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.vgg19_bn(x)


class EfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3, self).__init__()
        self.efficientnet_b3 = efficientnet_b3(pretrained=True)
        self.efficientnet_b3._fc = nn.Linear(1536, num_classes)

    def forward(self, x):
        return self.efficientnet_b3(x)


class DenseNet169(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()
        self.densenet169 = densenet169(pretrained=True)
        self.densenet169.classifier = nn.Linear(1664, num_classes)

    def forward(self, x):
        return self.densenet169(x)


class ResNext50(nn.Module):
    def __init__(self, num_classes):
        super(ResNext50, self).__init__()
        self.resnext50 = resnext50_32x4d(pretrained=True)
        self.resnext50.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.resnext50(x)


class MobileNetV3Small(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3Small, self).__init__()
        self.mobilenet_v3_small = mobilenet_v3_small(pretrained=True)
        self.mobilenet_v3_small.classifier = nn.Linear(576, num_classes)

    def forward(self, x):
        return self.mobilenet_v3_small(x)


class ShuffleNetV2(nn.Module):
    def __init__(self, num_classes):
        super(ShuffleNetV2, self).__init__()
        self.shufflenet_v2_x2_0 = shufflenet_v2_x2_0(pretrained=True)
        self.shufflenet_v2_x2_0.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        return self.shufflenet_v2_x2_0(x)


class SqueezeNet1_0(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNet1_0, self).__init__()
        self.squeezenet1_0 = squeezenet1_0(pretrained=True)
        self.squeezenet1_0.classifier[1] = nn.Linear(1000, num_classes)

    def forward(self, x):
        return self.squeezenet1_0(x)


class SqueezeNet1_1(nn.Module):
    def __init__(self, num_classes):
        super(SqueezeNet1_1, self).__init__()
        self.squeezenet1_1 = squeezenet1_1(pretrained=True)
        self.squeezenet1_1.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        return self.squeezenet1_1(x)


class InceptionNetV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionNetV3, self).__init__()
        self.inception_v3 = inception_v3(pretrained=True, aux_logits=False)
        self.fc = nn.Linear(2048, num_classes)
        self.required_input_size = 299

    def forward(self, x):
        if x.size(-1) < self.required_input_size or x.size(-2) < self.required_input_size:
            raise ValueError(f"Input size must be at least {self.required_input_size}x{self.required_input_size} pixels")

        x = self.inception_v3(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MNASNet0_5(nn.Module):
    def __init__(self, num_classes):
        super(MNASNet0_5, self).__init__()
        self.mnasnet0_5 = mnasnet0_5(pretrained=True)
        self.mnasnet0_5.classifier[1] = nn.Linear(self.mnasnet0_5.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.mnasnet0_5(x)


class MNASNet0_75(nn.Module):
    def __init__(self, num_classes):
        super(MNASNet0_75, self).__init__()
        self.mnasnet0_75 = mnasnet0_75(pretrained=True)
        self.mnasnet0_75.classifier[1] = nn.Linear(self.mnasnet0_75.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.mnasnet0_75(x)


class MNASNet1_0(nn.Module):
    def __init__(self, num_classes):
        super(MNASNet1_0, self).__init__()
        self.mnasnet1_0 = mnasnet1_0(pretrained=True)

