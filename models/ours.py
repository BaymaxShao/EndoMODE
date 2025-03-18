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
        self.feature_channels = 192
        self.out_dim = out_dim
        self.dep_features = dfe(pretrained=True)
        self.joint_features = jfe34()
        self.squeeze = nn.Conv2d(704, 384, 1)
        self.relu = nn.ReLU(inplace=False)
        self.decoder = PoseDecoder()

    def forward(self, img1, img2, flow):
        img3 = torch.cat([img1, img2], dim=1)
        f1 = self.dep_features(img1)
        f2 = self.dep_features(img2)
        fo = self.dep_features(flow)

        # Joint Feature
        f3 = self.joint_features(img3)
        feat = torch.cat((f1, f2, f3, fo), dim=1)

        out = self.relu(self.squeeze(feat))

        out = self.decoder(out)

        pose = 0.01 * out.view(-1, self.out_dim)

        return pose


class PoseDecoder(nn.Module):
    def __init__(self, num_vars=6,
                 depths=[2], dims=[384, 768], drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super(PoseDecoder, self).__init__()
        self.downsample_layers = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.depths = depths
        for i in range(len(dims)-1):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(dims)-1):
            stage = nn.Sequential(
                *[Block(dim=dims[i+1], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], num_vars)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x 

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(3).mean(2)
        x = self.head(x)
        return x


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dim = dim
        self.dwconv = {}
        for i in range(4):
            self.dwconv[i] = nn.Conv2d(dim, dim, kernel_size=2*i+1, padding=i, groups=dim//4)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        xs = [x[:, dim//4*i:dim//4*(i+1), : ,:] for g in range(4)]
        for i in range(4):
            xs[i] = self.dwconv[i](xs[i])
        x = torch.cat(xs, dim=1)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
