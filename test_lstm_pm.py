# test
import torch
from torch.utils.data import DataLoader

from data.handpose_data2 import UCIHandPoseDataset
from model.lstm_pm import LSTM_PM
from src.utils import *

import argparse
import pandas as pd

import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

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
test_data_dir = '/mnt/data/haoyum/UCIHand/test/test_data'
test_label_dir = '/mnt/data/haoyum/UCIHand/test/test_label'
device_ids = [0, 1]


# load data
test_data = UCIHandPoseDataset(data_dir=test_data_dir, label_dir=test_label_dir, temporal=temporal, train=False)
print 'Test  dataset total number of images sequence is ----' + str(len(test_data))
test_dataset = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)


def load_model(model):
    # build model
    net = LSTM_PM(T=temporal)
    if torch.cuda.is_available():
        net = net.cuda()
        net = net.cuda(device_ids[0])
        net = nn.DataParallel(net, device_ids=device_ids)  # multi-Gpu

    save_path = os.path.join('ckpt/ucihand_lstm_pm'+str(model)+'.pth')
    state_dict = torch.load(save_path)
    # ******************** transfer from multi-GPU model ********************
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        namekey = k[7:]  # remove `module.`
        new_state_dict[namekey] = v
    # load params
    net.load_state_dict(new_state_dict)
    return net


# **************************************** test all images ****************************************

print '********* test data *********'
model_epo = [10, 15, 20, 25, 30, 35, 40, 45, 50]
for model in model_epo:

    net = load_model(model)
    net.eval()

    sigma = 0.04
    results = []
    for i in range(1):

        result = []
        result.append(sigma)
        pck_all = []
        for step, (images, label_map, center_map, imgs) in enumerate(test_dataset):

            images = Variable(images.cuda() if args.cuda else images)  # 4D Tensor
            # Batch_size  *  (temporal * 3)  *  width(368)  *  height(368)
            label_map = Variable(label_map.cuda() if args.cuda else label_map)  # 5D Tensor
            # Batch_size  *  Temporal        * (joints+1) *   45 * 45
            center_map = Variable(center_map.cuda() if args.cuda else center_map)  # 4D Tensor
            # Batch_size  *  1          * width(368) * height(368)

            predict_heatmaps = net(images, center_map)  # get a list size: temporal * 4D Tensor

            # calculate pck
            pck = lstm_pm_evaluation(label_map, predict_heatmaps, sigma=sigma, temporal=temporal)
            pck_all.append(pck)

            if step % 100 == 0:
                print '--step ...' + str(step)
                print '--pck.....' + str(pck)
                if pck < 0.8:
                    print imgs
            if pck < 0.8:
                save_images(label_map, predict_heatmaps, step, epoch=-1, imgs=imgs, train=False, temporal=5)

        print 'sigma ==========> ' + str(sigma)
        print '===PCK evaluation in test dataset is ' + str(sum(pck_all) / len(pck_all))
        result.append(str(sum(pck_all) / len(pck_all)))
        results.append(result)

        sigma += 0.01

    results = pd.DataFrame(results)
    results.to_csv('ckpt/' + str(model) + 'test_pck.csv')












