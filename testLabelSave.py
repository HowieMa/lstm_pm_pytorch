# test
from data.handpose_data3 import UCIHandPoseDataset
from model.lstm_pm import LSTM_PM
from src.utils import *

import argparse
import pandas as pd

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

save_dir = 'test_label'

# add parameter
parser = argparse.ArgumentParser(description='Pytorch LSTM_PM with Penn_Action')
parser.add_argument('--learning_rate', type=float, default=8e-6, help='learning rate')
parser.add_argument('--batch_size', default=1, type=int, help='batch size for training')
parser.add_argument('--save_dir', default='ckpt', type=str, help='directory of checkpoint')
parser.add_argument('--cuda', default=1, type=int, help='if you use GPU, set cuda = 1,else set cuda = 0')
parser.add_argument('--temporal', default=4, type=int, help='how many temporals you want ')
args = parser.parse_args()

# hyper parameter
temporal = 5
sample = 2
model_temporal = temporal * (sample + 1) - sample

test_data_dir = '/mnt/data/haoyum/UCIHand/test/test_data'
test_full_data = '/mnt/data/haoyum/UCIHand/test/test_full_data'
test_label_dir = '/mnt/data/haoyum/UCIHand/test/test_label'
model_epo = 35
sigma = 0.04

# load data
test_data = UCIHandPoseDataset(data_dir=test_data_dir, data_dir2=test_full_data,
                               label_dir=test_label_dir, temporal=temporal, train=False)

print 'Test  dataset total number of images sequence is ----' + str(len(test_data))
test_dataset = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


def load_model(model):
    # build model
    net = LSTM_PM(T=model_temporal)
    if torch.cuda.is_available():
        net = net.cuda()
        net = nn.DataParallel(net)  # multi-Gpu

    save_path = os.path.join('ckpt/ucihand_lstm_pm' + str(model)+'.pth')
    state_dict = torch.load(save_path)
    net.load_state_dict(state_dict)
    return net

# **************************************** test all images ****************************************


print '********* test data *********'

net = load_model(model_epo)
net.eval()

label_dict = {}  #
for step, (images, label_map, center_map, empty, imgs) in enumerate(test_dataset):

    images = Variable(images.cuda() if args.cuda else images)           # 4D Tensor
    # Batch_size  *  (temporal * 3)  *  width(368)  *  height(368)
    label_map = Variable(label_map.cuda() if args.cuda else label_map)  # 5D Tensor
    # Batch_size  *  Temporal        * joint *   45 * 45
    center_map = Variable(center_map.cuda() if args.cuda else center_map)  # 4D Tensor
    # Batch_size  *  1          * width(368) * height(368)

    predict_heatmaps = net(images, center_map)  # get a list size: temporal * 4D Tensor
    predict_heatmaps = predict_heatmaps[1:]

    pck_dict = Tests_save_label_imgs(label_map, predict_heatmaps, step,  imgs=imgs,
                    temporal=model_temporal)  # pck dict {0005:[[], [],[]]}

    label_dict = dict(pck_dict.items() + label_dict.items())

print 'sigma ==========> ' + str(sigma)
print '===PCK evaluation in test dataset is ' + str(sum(label_dict.values()) / label_dict.__len__())



















