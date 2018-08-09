import argparse
from torchvision import transforms
from model.lstm_pm import LSTM_PM
from data.handpose_data2 import UCIHandPoseDataset

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np

from torch.autograd import Variable
from torch.utils.data import DataLoader
import os
import scipy.misc

# hyper parameter
temporal = 5
data_dir = '..dataset/frames/train_data'
label_dir = '..dataset/label/train_label'
predict_dir = ''
loss_dir = ''


# add parameter
parser = argparse.ArgumentParser(description='Pytorch LSTM_PM with Penn_Action')
parser.add_argument('--learning_rate', type=float, default=8e-5, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='batch size for training')
parser.add_argument('--epochs', default=30, type=int, help='number of epochs for training')
parser.add_argument('--begin_epoch', default=0, type=int, help='how many epochs the model has been trained')
parser.add_argument('--save_dir', default='checkpoint', type=str, help='directory of checkpoint')
parser.add_argument('--cuda', default=1, type=int, help='if you use GPU, set cuda = 1,else set cuda = 0')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

transform = transforms.Compose([transforms.ToTensor()])

# Build dataset
dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir, temporal=temporal)
print 'Dataset total number of images sequence is ----' + str(len(dataset))

# Data Loader
train_dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
# Build model
net = LSTM_PM(T=temporal)
if args.cuda:
    net = net.cuda()

save_losses = []

def train():
    optimizer = optim.Adam(params=net.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    net.train()
    for epoch in range(args.begin_epoch, args.epochs + 1):
        print 'epoch............' + str(epoch)

        for step, (images, label_map, center_map) in enumerate(train_dataset):
            print '--step .....' + str(step)
            images = Variable(images.cuda() if args.cuda else images)               # 4D Tensor
            # Batch_size  *  (temporal * 3)  *  width(368)  *  height(368)
            label_map = Variable(label_map.cuda() if args.cuda else label_map)      # 5D Tensor
            # Batch_size  *  Temporal        * (joints+1) *   45 * 45
            center_map = Variable(center_map.cuda() if args.cuda else center_map)   # 4D Tensor
            # Batch_size  *  1          * width(368) * height(368)

            optimizer.zero_grad()

            predict_heatmaps = net(images, center_map)  # get a list size: temporal

            loss_list = []  # loss of 21 joints and total loss
            total_loss = 0
            for i in range(len(predict_heatmaps)):       # for each temporal
                predict = predict_heatmaps[i]            # 4D Tensor  Batch_size * (joints+1) * 45 * 45
                target = label_map[:, i, :, :, :]        # 4D Tensor  Batch_size * (joints+1) * 45 * 45
                each_loss = criterion(predict, target)   # calculate loss

                loss_list.append(float(each_loss))  # save loss of each joint
                total_loss += each_loss

            if step % 100 == 0:                          # save images
                save_prediction(label_map, predict_heatmaps, step, temporal, epoch)

            total_loss.backward()
            optimizer.step()

            print '--loss ' + str(float(total_loss.data[0]))
            loss_list.append(float(total_loss.data[0]))
            save_losses.append(loss_list)

    torch.save(net.state_dict(), os.path.join(args.save_dir, 'penn_lstm_pm{:d}.pth'.format(epoch)))


def save_prediction(label_map, predict_heatmaps, step, temporals, epoch):
    """
    :param label_map:           5D Tensor    Batch_size  *  Temporal * (joints+1) *   45 * 45
    :param predict_heatmaps:
    :param step:
    :param t:
    :param epoch:
    :return:
    """
    for b in range(args.batch_size):  # for each batch
        output = np.ones((50 * 2, 50 * temporals))  # each temporal save an image
        for t in range(temporals):  # for each temporal
            pre = np.zeros((45, 45))  #
            gth = np.zeros((45, 45))

            for i in range(21):
                pre += predict_heatmaps[t][b, i, :, :].data.cpu().numpy()
                gth += label_map[b, t, i, :, :].data.cpu().numpy()
            output[0:45,  50 * t: 50 * t + 45] = gth
            output[50:95, 50 * t: 50 * t + 45] = pre
            scipy.misc.imsave('epoch'+str(epoch) + '_step'+str(step) + '_batch' + str(b) + '.jpg', output)
    return


if __name__ =='__main__':
    train()








