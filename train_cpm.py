
# coding: utf-8

# In[1]:


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


# In[2]:


#**************hyper parameters needed to fill manully******************************
nb_epochs=100
batch_size=4
nb_joint = 21
nb_stage = 6
background = False
out_c = nb_joint+1 if background else nb_joint
heat_weight = 45 * 45 * out_c / 1.0
path_root = '/home/danningx/danningx/cpm_xdn/8-18/'
#************************************************************************************


# In[3]:


save_json_path = os.path.join(path_root, "train_history.json")
save_weight_path = os.path.join(utils.mkdir(os.path.join(path_root, "checkpoint")), "cpm_")
save_runtime_heatmap_path = utils.mkdir(os.path.join(path_root, "runtime_heatmaps/train"))
save_test_heatmap_path = utils.mkdir(os.path.join(path_root, "runtime_heatmaps/test"))

transform = transforms.Compose([transforms.ToTensor()])

#train data loader
data_dir = '/mnt/UCIHand/train/train_data'
label_dir = '/mnt/UCIHand/train/train_label'
dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir)
train_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#test data loader
data_dir = '/mnt/UCIHand/test/test_data'
label_dir = '/mnt/UCIHand/test/test_label'
dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir)
test_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# In[4]:


net = cpm_model.CPM(out_c=nb_joint, background=background).cuda()
net = torch.nn.DataParallel(net)

optimizer = torch.optim.Adam(params=net.parameters(), lr=8e-6, betas=(0.5, 0.999))
criterion = nn.MSELoss()


# In[9]:

loss_history = utils.init_loss_history_cpm()
for epoch in range(nb_epochs):
    print 'epoch......................' + str(epoch+1)
    net.train(True)
    runtime_pck = []
    for idx, (images, label_map, center_map, _) in enumerate(train_dataset): 
        images = Variable(images.cuda())        
        label_map = Variable(label_map.cuda())
        center_map = Variable(center_map.cuda())
        center_map = center_map[:,0,:,:]
        
        optimizer.zero_grad()
        
        runtime_loss = utils.init_runtime_loss_cpm()
        predict_heatmaps = net(images, center_map)
        for b, batch in enumerate(predict_heatmaps):
            for s, stg_heat in enumerate(batch): # each stage
                runtime_loss[s] += criterion(stg_heat, label_map[b]) * heat_weight
        
        loss = sum(runtime_loss)#/batch_size
        loss.backward()
        optimizer.step()
        runtime_pck.append(utils.cpm_evaluation(label_map, predict_heatmaps, sigma=0.04))
        
        
        # write loss hitory
        for s in range(nb_stage):
            loss_history['stage'+str(s+1)].append(float(runtime_loss[s])/float(batch_size))
        loss_history['all'].append(float(sum(runtime_loss)/float(batch_size)))
        
        if idx%20 ==0 :
            print "Epoch: "+str(epoch+1)+" itr: "+str(idx+1)+" loss: "+str(float(loss)/batch_size)+ "  pck: "+str(sum(runtime_pck)/float(len(runtime_pck)))
            #break
            
    loss_history['train_pckall'].append(sum(runtime_pck)/float(len(runtime_pck)))
    for param_group in optimizer.param_groups:
        #print param_group['lr']
        loss_history['lr'].append(param_group['lr'])
    #save weights/loss history after each epoch       
    torch.save(net.state_dict(),save_weight_path+str(epoch))
    json.dump(loss_history, open(save_json_path, 'wb')) 
    if epoch%5 == 0:
        tmp_path = save_runtime_heatmap_path+'/e'+str(epoch+1)+'_'
        utils.save_image_cpm(tmp_path, predict_heatmaps, label_map)
        
    # ****************************test *********************************
    if epoch%5 ==4:
        net.eval()
        print 'Testing epoch '+ str(epoch+1) + '*****************************************'
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
            
            if idx%100==0:
                utils.save_image_cpm(save_test_heatmap_path+'/e'+str(epoch+1)+'_stp'+str(idx)+'_b', predict_heatmaps, label_map)
                #break
        avg_pck = sum(runtime_pck)/float(len(runtime_pck))
                
        
        print "Test Epoch "+str(epoch+1)+"  pck : "+str(avg_pck)
        loss_history['test_pck'][epoch+1] = {'avg':avg_pck,'pck_all':runtime_pck, 'img_name':img_name}
        
        

        
        
