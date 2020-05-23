import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, fc_hidden1=128, fc_hidden2=128, drop_p=0.3, CNN_embed_dim=100, num_class=15):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)
        self.fc = nn.Linear(CNN_embed_dim, num_class)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            # with torch.no_grad():
            x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
            # TODO:做特征图融合或做全连接融合，两种方式做选择，先做全连接的吧，但是和PANET还是有区别

            x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        # TODO:全连接层最大值融合
        cnn_max,_ = torch.max(cnn_embed_seq, dim=0)
        x = self.fc(cnn_max)
        # return cnn_embed_seq
        return x


class LSTMCNN(nn.Module):
    def __init__(self):
        super(LSTMCNN, self).__init__()
        self.encodercnn = EncoderCNN()
        # self.decoderrnn = DecoderRNN()

    def forward(self, x_3d):
        cnn_embed_seq = self.encodercnn(x_3d)
        return cnn_embed_seq
