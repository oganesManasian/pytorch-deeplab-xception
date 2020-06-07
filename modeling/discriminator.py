import torch
import torch.nn as nn
import torch.nn.functional as F

from modeling.backbone import build_backbone
import torch
from torchvision import models


class DiscriminatorTorch(nn.Module):
    def __init__(self, num_classes=2, resnet_version="101", use_pretrained=False):
        super(DiscriminatorTorch, self).__init__()

        if resnet_version == "101":
            self.model = models.resnet101(pretrained=use_pretrained)
        else:
            raise NotImplementedError

        # self.model = torch.hub.load('pytorch/vision:v0.6.0',
        #                             f'resnet{resnet_version}',
        #                             pretrained=use_pretrained)

        # Change first and last layer to match dimensions
        self.model.conv1 = nn.Conv2d(19, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, input):
        return self.model(input)


class Discriminator(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(Discriminator, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        # self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        # self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)

        return x

    # def forward(self, input):
    #     x, low_level_feat = self.backbone(input)
    #     x = self.aspp(x)
    #     x = self.decoder(x, low_level_feat)
    #     x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
    #
    #     return x
    #

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    #
    # def get_1x_lr_params(self):
    #     modules = [self.backbone]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if self.freeze_bn:
    #                 if isinstance(m[1], nn.Conv2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p
    #             else:
    #                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
    #                         or isinstance(m[1], nn.BatchNorm2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p
    #
    # def get_10x_lr_params(self):
    #     modules = [self.aspp, self.decoder]
    #     for i in range(len(modules)):
    #         for m in modules[i].named_modules():
    #             if self.freeze_bn:
    #                 if isinstance(m[1], nn.Conv2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p
    #             else:
    #                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
    #                         or isinstance(m[1], nn.BatchNorm2d):
    #                     for p in m[1].parameters():
    #                         if p.requires_grad:
    #                             yield p


if __name__ == "__main__":
    # model = Discriminator(backbone='resnet', output_stride=16)
    model = DiscriminatorTorch()
    # print(model)
    model.eval()
    input = torch.rand(1, 19, 513, 513)
    output = model(input)
    print(output.size())
