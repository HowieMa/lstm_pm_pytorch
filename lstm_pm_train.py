import argparse
from torchvision import transforms
from model.lstm_pm import LSTM_PM
from data.handpose_data import UCIHandPoseDataset

import torch
import torch.optim as optim
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
import os

temporal = 4

parser = argparse.ArgumentParser(description='Pytorch LSTM_PM with Penn_Action')

parser.add_argument('--learning_rate', type=float, default=8e-5, help='learning rate')
parser.add_argument('--batch_size', default=4, type=int, help='batch size for training')
parser.add_argument('--epochs', default=50, type=int, help='number of epochs for training')
parser.add_argument('--begin_epoch', default=0, type=int, help='how many epochs the model has been trained')
parser.add_argument('--save_dir',default='checkpoint', type = str,help='directory of checkpoint')
parser.add_argument('--cuda', default=0, type=int, help='if you use GPU, set cuda = 1,else set cuda = 0')


args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

transform = transforms.Compose([transforms.ToTensor()])


data_dir = 'dataset/frames/001'
label_dir = 'dataset/label/001'

dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir, temporal=temporal)

train_dataset = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

net = LSTM_PM(T=4)

if args.cuda:
    net = net.cuda()


def train():
    optimizer = optim.Adam(params=net.parameters(), lr=args.learning_rate, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    net.train()
    for epoch in range(args.begin_epoch, args.epochs + 1):
        print 'epoch............' + str(epoch)

        for idx, (images, label_map, center_map) in enumerate(train_dataset):

            images = Variable(images.cuda() if args.cuda else images)
            label_map = Variable(label_map.cuda() if args.cuda else label_map)
            center_map = Variable(center_map.cuda() if args.cuda else center_map)


            optimizer.zero_grad()

            predict_heatmaps = net(images, center_map)  # list
            
            ## *******************************************

            loss = 0
            for i in range(len(predict_heatmaps)):
                predict = predict_heatmaps[i]
                target = label_map[:,i, :, :, :]
                loss += criterion(predict, target)

            loss.backward()
            ## *******************************************

            optimizer.step()

            print loss.data[0]


    torch.save(net.state_dict(), os.path.join(args.save_dir, 'penn_lstm_pm{:d}.pth'.format(epoch)))



if __name__ =='__main__':
    train()






