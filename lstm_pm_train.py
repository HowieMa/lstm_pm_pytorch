# https://github.com/HowieMa/lstm_pm_pytorch.git

import argparse
import json
import numpy as np
import os
import scipy.misc

from model.lstm_pm import LSTM_PM
from data.handpose_data2 import UCIHandPoseDataset

import torch
import torch.optim as optim
import torch.nn as nn

from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms

# hyper parameter
temporal = 5
train_data_dir = '/home/haoyum/UCIHand/train/train_data'
train_label_dir = '/home/haoyum/UCIHand/train/train_label'
test_data_dir = '/home/haoyum/UCIHand/test/test_data'
test_label_dir = '/home/haoyum/UCIHand/test/test_label'


# add parameter
parser = argparse.ArgumentParser(description='Pytorch LSTM_PM with Penn_Action')
parser.add_argument('--learning_rate', type=float, default=8e-6, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='batch size for training')
parser.add_argument('--epochs', default=30, type=int, help='number of epochs for training')
parser.add_argument('--begin_epoch', default=0, type=int, help='how many epochs the model has been trained')
parser.add_argument('--save_dir', default='ckpt', type=str, help='directory of checkpoint')
parser.add_argument('--cuda', default=1, type=int, help='if you use GPU, set cuda = 1,else set cuda = 0')
parser.add_argument('--temporal', default=4, type=int, help='how many temporals you want ')
args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

transform = transforms.Compose([transforms.ToTensor()])

# Build dataset
train_data = UCIHandPoseDataset(data_dir=train_data_dir, label_dir=train_label_dir, temporal=temporal, train=True)
test_data = UCIHandPoseDataset(data_dir=test_data_dir, label_dir=test_label_dir, temporal=temporal, train=False)


print 'Train dataset total number of images sequence is ----' + str(len(train_data))
print 'Test  dataset total number of images sequence is ----' + str(len(test_data))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_dataset = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

# Build model
net = LSTM_PM(T=temporal)
if args.cuda:
    net = net.cuda()


def loss_history_init():
    loss_history = {}
    for b in range(args.batch_size):  # each person
        loss_history['batch'+str(b)] = {}
        for t in range(temporal):
            loss_history['batch'+str(b)]['temporal'+str(t)] = []
    loss_history['total'] = 0.0
    return loss_history


def train():
    # initialize optimizer
    optimizer = optim.SGD(params=net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=40000, gamma=0.333)

    optimizer = optim.Adam(params=net.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))

    criterion = nn.MSELoss(size_average=True)  # loss average

    for epoch in range(args.begin_epoch, args.epochs + 1):
        net.train()
        print 'epoch....................' + str(epoch)

        for step, (images, label_map, center_map, imgs) in enumerate(train_dataset):

            if step % 10 ==0:
                print '--step .....' + str(step)
                print '--loss ' + str(float(total_loss.data[0]))

            images = Variable(images.cuda() if args.cuda else images)               # 4D Tensor
            # Batch_size  *  (temporal * 3)  *  width(368)  *  height(368)
            label_map = Variable(label_map.cuda() if args.cuda else label_map)      # 5D Tensor
            # Batch_size  *  Temporal        * (joints+1) *   45 * 45
            center_map = Variable(center_map.cuda() if args.cuda else center_map)   # 4D Tensor
            # Batch_size  *  1          * width(368) * height(368)

            optimizer.zero_grad()

            predict_heatmaps = net(images, center_map)  # get a list size: temporal * 4D Tensor

            # ******************** calculate and save loss of each joints ********************
            total_loss = save_loss(predict_heatmaps, label_map, criterion,train=True)

            # ******************** save training heat maps per 100 steps ********************
            if step % 100 == 0:
                save_images(label_map, predict_heatmaps, step, temporal, epoch, imgs)

            # backward
            total_loss.backward()
            optimizer.step()
            scheduler.step()

        # ******************** test per 10 epochs ********************
        if epoch % 10 == 0:
            pck_all = []
            net.eval()
            print '********* test data *********'
            for step, (images, label_map, center_map, imgs) in enumerate(test_dataset):
                images = Variable(images.cuda() if args.cuda else images)  # 4D Tensor
                # Batch_size  *  (temporal * 3)  *  width(368)  *  height(368)
                label_map = Variable(label_map.cuda() if args.cuda else label_map)  # 5D Tensor
                # Batch_size  *  Temporal        * (joints+1) *   45 * 45
                center_map = Variable(center_map.cuda() if args.cuda else center_map)  # 4D Tensor
                # Batch_size  *  1          * width(368) * height(368)

                predict_heatmaps = net(images, center_map)  # get a list size: temporal * 4D Tensor
                total_loss = save_loss(predict_heatmaps, label_map, criterion, train=False)
                if step % 10 ==0:
                    print '--step .....' + str(step)
                    print '--loss ' + str(float(total_loss.data[0]))

                # calculate pck
                pck_all.append(evaluation(label_map, predict_heatmaps))

            print 'PCK evaluation in test dataset is ' + str(sum(pck_all) / len(pck_all))

    torch.save(net.state_dict(), os.path.join(args.save_dir, 'penn_lstm_pm{:d}.pth'.format(epoch)))
    print 'train done!'


def save_loss(predict_heatmaps, label_map, criterion, train):
    loss_save = loss_history_init()
    total_loss = 0
    for b in range(label_map.shape[0]):  # for each batch (person)
        for t in range(temporal):  # for each temporal
            for i in range(21):  # for each joint
                predict = predict_heatmaps[t][b, i, :, :]  # 2D Tensor
                target = label_map[b, t, i, :, :]
                tmp_loss = criterion(predict, target)  # average MSE loss
                loss_save['batch' + str(b)]['temporal' + str(t)].append(float(tmp_loss))
                total_loss += tmp_loss

    total_loss = total_loss / (label_map.shape[0] * temporal * 21.0)
    loss_save['total'] = float(total_loss.data[0])

    if train is True:
        json.dump(loss_save, open('ckpt/'+'train_loss.json', 'wb'))
    else:
        json.dump(loss_save, open('ckpt/' + 'test_loss.json', 'wb'))

    return total_loss


def save_images(label_map, predict_heatmaps, step, temporals, epoch, imgs):
    """
    :param label_map:
    :param predict_heatmaps:    5D Tensor    Batch_size  *  Temporal * (joints+1) *   45 * 45
    :param step:
    :param temporals:
    :param epoch:
    :param imgs: list [(), (), ()] temporal * batch_size
    :return:
    """

    for b in range(label_map.shape[0]):                    # for each batch (person)
        output = np.ones((50 * 2, 50 * temporals))      # each temporal save a single image
        seq = imgs[0][b].split('/')[-2]                    # sequence name 001L0
        img = ""
        for t in range(temporals):  # for each temporal
            im = imgs[t][b].split('/')[-1][1:5]            # image name 0005
            img += '_' + im
            pre = np.zeros((45, 45))  #
            gth = np.zeros((45, 45))

            for i in range(21):
                pre += predict_heatmaps[t][b, i, :, :].data.cpu().numpy()    # 2D
                gth += label_map[b, t, i, :, :].data.cpu().numpy()           # 2D

            output[0:45,  50 * t: 50 * t + 45] = gth
            output[50:95, 50 * t: 50 * t + 45] = pre

        if not os.path.exists('ckpt/epoch'+str(epoch)):
            os.mkdir('ckpt/epoch'+str(epoch))

        scipy.misc.imsave('ckpt/epoch'+str(epoch) + '/s'+str(step) + '_b' + str(b) + seq + img + '.jpg', output)


def evaluation(label_map, predict_heatmaps):
    pck_eval = []
    for b in range(label_map.shape[0]):        # for each batch (person)
        for t in range(temporal):           # for each temporal
            target = np.asarray(label_map[b, t, :, :, :].data)
            predict = np.asarray(predict_heatmaps[t][b, :, :, :].data)
            pck_eval.append(PCK(predict, target))

    return sum(pck_eval) / float(len(pck_eval))  #


def PCK(predict, target, label_size=45, sigma=0.04):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than
    :param predict:         3D numpy       22 * 45 * 45
    :param target:          3D numpy       22 * 45 * 45
    :param label_size:
    :param sigma:
    :return:
    """
    pck = 0
    for i in range(predict.shape[0]):
        pre_x, pre_y = np.where(predict[i, :, :] == np.max(predict[i, :, :]))
        tar_x, tar_y = np.where(target[i, :, :] == np.max(target[i, :, :]))

        dis = np.sqrt((pre_x[0] - tar_x[0])**2 + (pre_y[0] - tar_y[0])**2)
        if dis < sigma * label_size:
            pck += 1
    return pck / float(predict.shape[0])


if __name__ =='__main__':
    train()








