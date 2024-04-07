'''
This code is written based on the paper of OffstNet (ICRA 2019)
The paper: https://ieeexplore.ieee.org/abstract/document/8793940
'''

import torch
import torch.nn as nn
import torchvision.models as tvmodels


class OffsetNet(nn.Module):
    def __init__(self, out_dim=6):
        super(OffsetNet, self).__init__()
        self.out_dim = out_dim

        resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        self.dep_features = tvmodels.resnet34(weights='IMAGENET1K_V1')
        last_layer = 'layer4'
        resnet_module_list = [getattr(self.dep_features, l) for l in resnet_feature_layers]
        last_layer_idx = resnet_feature_layers.index(last_layer)
        self.sep_features = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])
        self.relu = nn.ReLU(inplace=False)
        self.pose_estimation = nn.Linear(1024, 6)

    def forward(self, img1, img2):
        feat1 = self.sep_features(img1)
        feat2 = self.sep_features(img2)
        feat = torch.cat([feat1, feat2], dim=1)
        feat = feat.mean(3).mean(2)
        out = self.pose_estimation(feat)
        pose = 0.01 * out.view(-1, self.out_dim)
        return pose

if __name__ == '__main__':
    input1 = torch.randn((16, 3, 224, 224))
    input2 = torch.randn((16, 3, 224, 224))
    model = OffsetNet()
    output = model(input1, input2)
    print(output)

