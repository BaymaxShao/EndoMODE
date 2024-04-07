'''
This code for comparison and validation is adapted from the code of the work by A Rau et al. (MICCAI 2022 Challenge/ IEEE TMRB 2023) on SimCol dataset.
The paper: https://ieeexplore.ieee.org/abstract/document/10266754
The code: https://github.com/anitarau/simcol
'''

import torch
import torch.nn as nn
import torchvision.models as tvmodels
from collections import OrderedDict


class PoseNet(nn.Module):
    def __init__(self, out_dim=6):
        super(PoseNet, self).__init__()
        self.out_dim = out_dim
        resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        self.resnet_features = tvmodels.resnet18(weights='IMAGENET1K_V1')
        last_layer = 'layer3'
        resnet_module_list = [getattr(self.resnet_features, l) for l in resnet_feature_layers]
        last_layer_idx = resnet_feature_layers.index(last_layer)
        self.resnet_features = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(512, 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(256, 256, 3, 1, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, 1, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, self.out_dim, 1)
        self.relu = nn.ReLU(inplace=False)
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, img1, img2):
        resnet_f1 = self.resnet_features(img1)
        resnet_f2 = self.resnet_features(img2)
        feat = torch.cat([resnet_f1, resnet_f2], 1)

        out = self.relu(self.convs["squeeze"](feat))

        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)
        pose = 0.01 * out.view(-1, self.out_dim)

        return pose


if __name__ == '__main__':
    input1 = torch.randn((16, 3, 224, 224))
    input2 = torch.randn((16, 3, 224, 224))
    model = PoseNet()
    output = model(input1, input2)
    print(output)




