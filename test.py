# test
import torch
from data.handpose_data2 import UCIHandPoseDataset
from model.lstm_pm import LSTM_PM

temporal = 4 
data_dir = 'dataset/frames/001'
label_dir = 'dataset/label/001'

dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir, temporal=temporal)


net = LSTM_PM(T=temporal)
net.load_state_dict(torch.load('checkpoint/penn_lstm_pm20.pth'))

net.eval()

images, label_maps, center_map = dataset[2]
images = images.unsqueeze(0)

center_map = center_map.unsqueeze(0)

heatmaps = net(images, center_map)

import scipy.misc
import numpy as np

k = 0
for maps in heatmaps:  # 4
    maps = maps.squeeze(0)  # 22 45 45
    print maps.shape
    a = maps.data.numpy()
    s = np.zeros((45,45))
    for i in range(22):
        s += a[i,:,:]
        scipy.misc.imsave('testimg/img'+str(k)+'_'+str(i) +'.jpg', a[i,:,:] )
    k+=1


# criterion
def PCK(predict, target):
    return 0
