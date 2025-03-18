'''
This is the Pytorch Implementation of our model.
- The code of Separate Feature Extractor (Pre-trained ResNet-18) is written based on the previous works.
- The code of Joint Feature Extractor and Pose Decoder will be released after the paper is accepted.
'''

import torch
import torch.nn as nn
import torchvision.models as tvmodels
from .jfe import jfe34


class EndoMODE(nn.Module):
    def __init__(self, out_dim=6):
        super(NEPose, self).__init__()
        self.out_dim = out_dim

        # Separate Feature Extractor
        resnet_feature_layers = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4']
        self.dep_features = tvmodels.resnet18(weights='IMAGENET1K_V1')
        last_layer = 'layer3'
        resnet_module_list = [getattr(self.dep_features, l) for l in resnet_feature_layers]
        last_layer_idx = resnet_feature_layers.index(last_layer)
        self.sep_features = nn.Sequential(*resnet_module_list[:last_layer_idx + 1])

        # Joint Feature Extractor
        self.joint_features = jfe34()

        # Pose Decoder
        self.squeeze = nn.Conv2d(768, 384, 1)
        self.relu = nn.ReLU(inplace=False)
        self.decoder = PoseDecoder()

    def forward(self, img1, img2):
        # Feature Extraction
        img3 = torch.cat([img1, img2], dim=1)
        f1 = self.sep_features(img1)
        f2 = self.sep_features(img2)
        f3 = self.joint_features(img3)
        feat = torch.cat((f1, f2, f3), dim=1)

        # Pose Decoding
        out = self.relu(self.squeeze(feat))
        out = self.decoder(out)

        pose = 0.01 * out.view(-1, self.out_dim)
        return pose


# class PoseDecoder(nn.Module):
# The code of the pose decoder: Coming Soon...



if __name__ == '__main__':
    input1 = torch.randn((16, 3, 224, 224))
    input2 = torch.randn((16, 3, 224, 224))
    model = NEPose()
    output = model(input1, input2)
    print(output)
