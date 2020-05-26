import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, args, fc_hidden1=128, fc_hidden2=128, drop_p=0.3, CNN_embed_dim=100, num_class=15):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        # self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        # self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

        self.fc = nn.Linear(CNN_embed_dim, num_class)

        self.fc2 = nn.Linear(CNN_embed_dim, CNN_embed_dim)
        self.bn2 = nn.BatchNorm1d(CNN_embed_dim, momentum=0.01)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            x = self.resnet(x_3d[:, t, :, :, :])  # ResNet
            # TODO:做特征图融合或做全连接融合，两种方式做选择，先做全连接的吧，但是和PANET还是有区别

            x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            # x = self.bn2(self.fc2(x))
            # x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0)
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        # TODO:全连接层最大值融合
        cnn_max, _ = torch.max(cnn_embed_seq, dim=0)
        x = self.bn2(self.fc2(x))
        x = self.fc(cnn_max)

        # return cnn_embed_seq
        return x


class LSTMCNN(nn.Module):
    def __init__(self, args):
        super(LSTMCNN, self).__init__()
        self.encodercnn = EncoderCNN(args)
        # self.decoderrnn = DecoderRNN()

    def forward(self, x_3d):
        cnn_embed_seq = self.encodercnn(x_3d)
        return cnn_embed_seq



# class MobileNetV2(nn.Module):
#     def __init__(self, args):
#         super(MobileNetV2, self).__init__()
#
#         s1, s2 = 2, 2
#         if args.downsampling == 16:
#             s1, s2 = 2, 1
#         elif args.downsampling == 8:
#             s1, s2 = 1, 1
#
#         # Network is created here, then will be unpacked into nn.sequential
#         self.network_settings = [{'t': -1, 'c': 32, 'n': 1, 's': s1},
#                                  {'t': 1, 'c': 16, 'n': 1, 's': 1},
#                                  {'t': 6, 'c': 24, 'n': 2, 's': s2},
#                                  {'t': 6, 'c': 32, 'n': 3, 's': 2},
#                                  {'t': 6, 'c': 64, 'n': 4, 's': 2},
#                                  {'t': 6, 'c': 96, 'n': 3, 's': 1},
#                                  {'t': 6, 'c': 160, 'n': 3, 's': 2},
#                                  {'t': 6, 'c': 320, 'n': 1, 's': 1},
#                                  {'t': None, 'c': 1280, 'n': 1, 's': 1}]
#         self.num_classes = args.num_classes
#
#         ###############################################################################################################
#
#         # Feature Extraction part
#         # Layer 0
#         self.network = [
#             conv2d_bn_relu6(args.num_channels,
#                             int(self.network_settings[0]['c'] * args.width_multiplier),
#                             args.kernel_size,
#                             self.network_settings[0]['s'], args.dropout_prob)]
#
#         # Layers from 1 to 7
#         for i in range(1, 8):
#             self.network.extend(
#                 inverted_residual_sequence(
#                     int(self.network_settings[i - 1]['c'] * args.width_multiplier),
#                     int(self.network_settings[i]['c'] * args.width_multiplier),
#                     self.network_settings[i]['n'], self.network_settings[i]['t'],
#                     args.kernel_size, self.network_settings[i]['s']))
#
#         # Last layer before flattening
#         self.network.append(
#             conv2d_bn_relu6(int(self.network_settings[7]['c'] * args.width_multiplier),
#                             int(self.network_settings[8]['c'] * args.width_multiplier), 1,
#                             self.network_settings[8]['s'],
#                             args.dropout_prob))
#
#         ###############################################################################################################
#
#         # Classification part
#         self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
#         self.network.append(nn.AvgPool2d(
#             (args.img_height // args.downsampling, args.img_width // args.downsampling)))
#         # self.network.append(nn.AdaptiveAvgPool2d((1, 1)))
#         self.network.append(nn.Dropout2d(args.dropout_prob, inplace=True))
#         self.network.append(
#             nn.Conv2d(int(self.network_settings[8]['c'] * args.width_multiplier), self.num_classes,
#                       1, bias=True))
#
#         self.network = nn.Sequential(*self.network)
#
#         self.initialize()
#
#     def forward(self, x):
#         # Debugging mode
#         # for op in self.network:
#         #     x = op(x)
#         #     print(x.shape)
#         x = self.network(x)
#         x = x.view(-1, self.num_classes)
#         return x
#
#     def initialize(self):
#         """Initializes the model parameters"""
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#                 nn.init.xavier_normal(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant(m.weight, 1)
#                 nn.init.constant(m.bias, 0)
