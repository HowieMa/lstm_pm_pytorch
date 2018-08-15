import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

#%matplotlib inline
import matplotlib.pyplot as plt

import argparse
from torchvision import transforms
from model.lstm_pm import LSTM_PM
from data.handpose_data2 import UCIHandPoseDataset

import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

from torch.autograd import Variable
from torch.utils.data import DataLoader
import os

import scipy.misc
from IPython.core.debugger import set_trace
import json
import numpy as np

nb_temporal = 4
nb_epochs=20
batch_size=4


transform = transforms.Compose([transforms.ToTensor()])

data_dir = '/mnt/UCIHand/train/train_data'
label_dir = '/mnt/UCIHand/train/train_label'
dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir, temporal=nb_temporal,train=True)
train_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)
net = LSTM_PM(T=nb_temporal).cuda()
#net.load_state_dict(torch.load('/home/danningx/code/8-15/checkpoint/penn_lstm_pm_9'))
net = torch.nn.DataParallel(net)



def history_dict_init():
    history_dict={}
    for i in range(batch_size):
        history_dict['train'+str(i+1)]=[]
        history_dict['test'+str(i+1)] = []
    history_dict['train'] = []
    history_dict['test'] = []
    history_dict['lr'] = []
    return history_dict


def save_image(save_path, nb_temporal, pred_heatmap, label_map, size=45, nb_heatmap=21):
    # predict_heatmaps shape: [nb_temporal, batch_size,22,45,45]
    for i in range(pred_heatmap[0].shape[0]):# each batch (person)
        if label_map is None:
            s = np.zeros((size, size*nb_temporal)) 
        else:
            s = np.zeros((size*2, size*nb_temporal)) 
            
        for j in range(len(pred_heatmap)):  # each temporal
            for k in range(nb_heatmap):
                if label_map is None:
                    s[:, j*45:(j+1)*45]+=predict_heatmaps[j].cpu().data.numpy()[i,k,:,:]
                else:
                    s[:45, j*45:(j+1)*45]+=label_map[i , j, k, :, :]
                    s[45:, j*45:(j+1)*45]+=predict_heatmaps[j].cpu().data.numpy()[i,k,:,:]
        scipy.misc.imsave(save_path+str(i+1)+'.jpg', s)
        
        
history_dict = history_dict_init()
save_json_path = "/home/danningx/code/8-15/train_history.json"
save_weight_path = "/home/danningx/code/8-15/checkpoint/penn_lstm_pm_"
save_runtime_heatmap_path = "/home/danningx/code/8-15/runtime_heatmaps/"



optimizer = optim.Adam(params=net.parameters(), lr=8e-6, betas=(0.5, 0.999))
criterion = nn.MSELoss()


for epoch in range(nb_epochs):
    print 'epoch......................' + str(epoch+1)
    
    net.train(True)
    for idx, (images, label_map, center_map, _) in enumerate(train_dataset):  #2579 itr per epoch(tmp=4, batchsize=4)
        
        images = Variable(images.cuda())        
        label_map = Variable(label_map.cuda())
        center_map = Variable(center_map.cuda())
        
        optimizer.zero_grad()
        predict_heatmaps = net(images, center_map)  # list
        train_loss = 0
        for i in range(len(predict_heatmaps)):
            predict = predict_heatmaps[i]
            target = label_map[:,i, :, :, :]
            tmp_loss = criterion(predict, target) 
            history_dict['train'+str(i+1)].append(float(tmp_loss))
            train_loss += tmp_loss * batch_size

        history_dict['train'].append(float(train_loss))
        train_loss.backward()
        ## *******************************************
        optimizer.step()
        #scheduler.step()
        if idx%10==0:
            print "Epoch: "+str(epoch+1)+" itr: "+str(idx+1)+" loss: "+str(float(train_loss))
            torch.save(net.state_dict(),save_weight_path+str(epoch))

    for param_group in optimizer.param_groups:
        #print param_group['lr']
        history_dict['lr'].append(param_group['lr'])
        
    #save loss history and parameters after each epoch
    json.dump(history_dict, open(save_json_path, 'wb')) 
    
    #***********************save train heatmap after each 20 epoch**********************************
    #if epoch%2 == 0:
    tmp_path = save_runtime_heatmap_path+'/'+'train'+'/e'+str(epoch)+'_bat'
    save_image(tmp_path, nb_temporal, predict_heatmaps, label_map)

