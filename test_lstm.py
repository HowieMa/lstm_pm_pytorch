
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

from torchvision import transforms
from model.lstm_pm import LSTM_PM
from data.handpose_data2 import UCIHandPoseDataset

import torch
import torch.nn as nn

from torch.autograd import Variable
from torch.utils.data import DataLoader
import os

import scipy.misc
import json
import numpy as np
import os
import utils


# In[2]:


nb_temporal = 4
batch_size=4
ckpt_path = '/home/danningx/code/8-15/checkpoint'
run = [20,25,30,35,40,45,50]
pck_sigma = 0.04
avg_pck_savepath = '/home/danningx/code/8-15/pck.json'
img_save_path = '/home/danningx/code/8-15/runtime_heatmaps/test/'


# In[3]:


transform = transforms.Compose([transforms.ToTensor()])
data_dir = '/mnt/UCIHand/test/test_data'
label_dir = '/mnt/UCIHand/test/test_label'
dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir, temporal=nb_temporal,train=False)
test_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)

net = LSTM_PM(T=nb_temporal).cuda()
net = torch.nn.DataParallel(net)

ckpt_list = os.listdir(ckpt_path)


# In[7]:


pck_history = {}
for ckpt_name in ckpt_list:
    if int(ckpt_name.split('_')[-1]) in run:
        print 'Testing ckeckpoint '+ ckpt_name.split('_')[-1] + '*****************************************'
        pck_all = []
        img_name = []
        net.load_state_dict(torch.load(os.path.join(ckpt_path, ckpt_name)))
        net.eval()
        for step, (images, label_map, center_map, imgs) in enumerate(test_dataset):
            images = Variable(images.cuda())
            label_map = Variable(label_map.cuda())
            center_map = Variable(center_map.cuda()) 
            
            predict_heatmaps = net(images, center_map) 

            pck = utils.lstm_pm_evaluation(label_map, predict_heatmaps, sigma=pck_sigma, temporal=4)
            pck_all.append(pck)
            img_name.append(imgs)
            
            if step%50==0:
                utils.save_image(img_save_path+'stp'+str(step)+'_b', nb_temporal, predict_heatmaps, label_map)
                print "pck: " + str(pck)
                
        avg_pck = sum(pck_all)/float(len(pck_all))
        print "checkpoint "+ckpt_name.split('_')[-1]+" : "+avg_pck
        pck_history[int(ckpt_name.split('_')[-1])] = {'avg':avg_pck,'pck_all':pck_all, 'img_name':img_name}
        json.dump(pck_history, open(avg_pck_savepath, 'wb')) 
        
        

