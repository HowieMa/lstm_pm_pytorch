'''
rebuild dataset
split train and test dataset

'''

import os
import numpy as np
import scipy.io

frame_dir = 'Penn_Action/frames'
label_dir = 'Penn_Action/labels'
train_dir = 'Penn_Action/train'
test_dir = 'Penn_Action/test'

nums = os.listdir(label_dir)

for idx, num in enumerate(nums, 1):
    print idx
    data = scipy.io.loadmat(os.path.join(label_dir, num))
    num = num.split('.')[0]
    
    npy_Data = dict()
    npy_Data['framepath'] = os.path.join(frame_dir, num)
    npy_Data['dimensions'] = list(data['dimensions'][0][0:2])
    npy_Data['pose'] = str(data['pose'][0])
    npy_Data['nframes'] = data['nframes'][0][0]
    npy_Data['action'] = str(data['action'][0])
    npy_Data['x'] = data['x']
    npy_Data['y'] = data['y']
    npy_Data['bbox'] = data['bbox']
    npy_Data['visibility'] = data['visibility']
    npy_Data['seq'] = int(num)

    if data['train'][0][0] == -1:
        save_dir = os.path.join(test_dir, num + '.npy')
    else:
        save_dir = os.path.join(train_dir, num + '.npy')
    np.save(save_dir, npy_Data)















