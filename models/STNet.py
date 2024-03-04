import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class STNet(nn.Module):
    def __init__(self):
        super(STNet, self).__init__()
        self.spatial = SpaNet()
        self.temporal = TempNet(320, 640, 'gru')
        self.pose_estimation = nn.Linear(320, 6)

    def forward(self, img1, img2, flow, s):
        f_sp = self.spatial(img1, img2, flow)
        f = self.temporal(f_sp, s)
        p = self.pose_estimation(f)
        return p


class SpaNet(nn.Module):
    def __init__(self, num_heads=12):
        super(SpaNet, self).__init__()
        # Layers for dependent feature extraction
        self.sep_encoder = EfficientNet.from_pretrained('efficientnet-b0')
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # Layers for joint feature extraction
        self.attention1 = nn.MultiheadAttention(1280, num_heads)
        self.attention2 = nn.MultiheadAttention(640, num_heads)
        self.conv1 = nn.Conv2d(3840, 1920, kernel_size=1)
        self.conv2 = nn.Conv2d(1920, 960, kernel_size=1)
        self.pose_estimation = nn.Linear(960, 7)

    def forward(self, img_1, img_2, flow):
        # Independent Feature Extraction
        feat_1 = self.sep_encoder.extract_features(img_1)
        feat_1 = self.pooling(feat_1)
        feat_2 = self.sep_encoder.extract_features(img_2)
        feat_2 = self.pooling(feat_2)
        feat_flow = self.sep_encoder.extract_features(flow)
        feat_flow = self.pooling(feat_flow)
        feat_dep = torch.cat([feat_1, feat_2, feat_flow], dim=1)

        # Joint Feature Extraction
        feat_dep_1 = self.conv1(feat_dep)
        feat_joint_1 = self.attention1(feat_dep_1) + feat_dep_1
        feat_dep_2 = self.conv2(feat_joint_1)
        feat_joint_2 = self.attention2(feat_dep_2) + feat_dep_2

        return feat_joint_2


class TempNet(nn.Module):
    def __init__(self, num_hiddens, input_size, rnn_type):
        super(TempNet, self).__init__()
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_size, num_hiddens)
        else:
            self.rnn = nn.LSTM(input_size, num_hiddens)
        self.input_size = self.rnn.input_size
        self.num_hiddens = self.rnn.hidden_size

        if not self.rnn.bidirectional:
            self.num_directions = 1
            self.linear = nn.Linear(self.num_hiddens, self.input_size)
        else:
            self.num_directions = 2
            self.linear = nn.Linear(self.num_hiddens * 2, self.input_size)

    def forward(self, seq, state):

        Y, state = self.rnn(seq, state)

        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, device, batch_size=1):
        if not isinstance(self.rnn, nn.LSTM):
            return torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                               device=device)
        else:
            return (torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device),
                    torch.zeros((self.num_directions * self.rnn.num_layers, batch_size, self.num_hiddens),
                                device=device))
