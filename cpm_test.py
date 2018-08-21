
# coding: utf-8

# In[7]:


import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cpm_model 
import numpy as np
from torch.autograd import Variable
from handpose_data_cpm import UCIHandPoseDataset
from torchvision import transforms
from torch.utils.data import DataLoader
import json
#%matplotlib inline
import matplotlib.pyplot as plt
import utils
from collections import OrderedDict


# In[5]:


#**************hyper parameters needed to fill manully******************************
#nb_epochs=100
batch_size=1
nb_joint = 21
nb_stage = 6
background = False
out_c = nb_joint+1 if background else nb_joint
heat_weight = 45 * 45 * out_c / 1.0
path_root = '/home/danningx/danningx/cpm_xdn/8-18/test_all/'
ckpt_path = '/home/danningx/danningx/cpm_xdn/8-18/checkpoint/cpm_30'
#************************************************************************************


# In[19]:


save_json_path = os.path.join(path_root, "test_history.json")
save_test_heatmap_path = utils.mkdir(os.path.join(path_root, "test_heatmaps"))

transform = transforms.Compose([transforms.ToTensor()])
#test data loader
data_dir = '/mnt/UCIHand/test/test_data'
label_dir = '/mnt/UCIHand/test/test_label'
dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir)
test_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# In[20]:



net = cpm_model.CPM(out_c=nb_joint, background=background).cuda()
state_dict = torch.load(ckpt_path)

# create new OrderedDict that does not contain `module.`
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    namekey = k[7:]  # remove `module.`
    new_state_dict[namekey] = v
# load params
net.load_state_dict(new_state_dict)
net = torch.nn.DataParallel(net)


# In[24]:


test_history = {}
net.eval()
runtime_pck= []
img_name = []
for idx, (images, label_map, center_map, imgs) in enumerate(test_dataset):
    images = Variable(images.cuda())
    label_map = Variable(label_map.cuda())
    center_map = Variable(center_map.cuda()) 
    center_map = center_map[:,0,:,:]

    predict_heatmaps = net(images, center_map) 
    runtime_pck.append(utils.cpm_evaluation(label_map, predict_heatmaps, sigma=0.04))
    img_name.append(imgs)

    utils.save_image_cpm(save_test_heatmap_path+'idx_'+str(idx)+'_b', predict_heatmaps, label_map)
    if idx%100 == 0:
        print str(idx)+' '+str(runtime_pck[-1])
avg_pck = sum(runtime_pck)/float(len(runtime_pck))


print " avg pck : "+str(avg_pck)
test_history = {'avg':avg_pck,'pck_all':runtime_pck, 'img_name':img_name}
json.dump(test_history, open(save_json_path, 'wb')) 

