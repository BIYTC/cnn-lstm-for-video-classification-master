import numpy as np
from dataset import *
import argparse
from utils import *
from torch.utils import data
from torchvision import transforms as T
from model import EncoderCNN
from model import DecoderRNN
from model import LSTMCNN
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
# use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='./data', help='path to dataset')
opt = parser.parse_args()

#
# Parameters
params = {'batch_size': 2,
          'shuffle': True,
          'num_workers': 1}
learning_rate = 1e-4
log_interval = 2  # interval for displaying training info
epochs = 10

# Datasets
partition, labels = load_data(opt.data)

# #pre_processing
# transform = transforms.Compose([
#         transforms.RandomHorizontalFlip(), 
#         transforms.Resize((224,224)),
#         transforms.Normalize((0.485, 0.456, 0.406), 
#                              (0.229, 0.224, 0.225)),
#         transforms.ToTensor()])

# preprocesing
transform = transforms.Compose([transforms.Resize([350, 480]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
training_set = Dataset(partition['train'], labels, transform)
training_generator = data.DataLoader(training_set, **params, collate_fn=training_set.my_collate, drop_last=True)
validation_set = Dataset(partition['val'], labels, transform)
validation_generator = data.DataLoader(validation_set, **params, collate_fn=validation_set.my_collate, drop_last=True)

# defining model
# if use_cuda:
#     encoder_cnn = EncoderCNN().cuda()
#     decoder_rnn = DecoderRNN().cuda()
# else:
#     encoder_cnn = EncoderCNN()
#     decoder_rnn = DecoderRNN()
if use_cuda:
    model = LSTMCNN().cuda()
else:
    model = LSTMCNN()
losses = []
scores = []
checkpoint=torch.load('select_best.pth')

model.eval()
N_count = 0
correct = 0
losses = []
with torch.no_grad():
    for X, y in validation_generator:
        # distribute data to device
        X, y = X.to(device), y.to(device)

        N_count += X.size(0)

        # output = rnn_decoder(cnn_encoder(X))
        output = model(X)
        loss = F.cross_entropy(output, y)
        losses.append(loss)

        y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

