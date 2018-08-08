import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image


class UCIHandPoseDataset(Dataset):
    def __init__(self, data_dir, label_dir, temporal = 7, transform = None, sigma=1):
        self.height = 368
        self.width = 368

        self.seqs = os.listdir(data_dir)    # L00, L01, L02,... , R01, R02, R03,...
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.temporal = temporal
        self.transform = transform
        self.joints = 21                    # 21 heat maps
        self.sigma=sigma

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        '''
        :param idx:
        :return:
        images          Tensor      temporal * 3 * width(368) * height(368)
        label_map       Tensor      temporal * (joints + 1) * width/8(46) * height/8(46)
        center_map      Tensor      1 * width(368) * height(368)
        '''

        label_size = self.width / 8 - 1
        seq = self.seqs[idx]         # e.g L00
        image_path = os.path.join(self.data_dir, seq)  #
        label_path = os.path.join(self.label_dir, seq)

        imgs = os.listdir(image_path)
        labels = json.load(open(label_path + '.json'))
        img_num = len(imgs)

        # initialize
        images = torch.zeros(self.temporal * 3, self.width, self.height)
        label_maps = torch.zeros(self.temporal, self.joints + 1, label_size, label_size)

        start_index = np.random.randint(0, img_num - self.temporal + 1)  #

        for i in range(self.temporal):                          # get temporal images
            img = imgs[i + start_index]                         # L0005.jpg
            im = Image.open(image_path + '/' + img)             # read image
            #set_trace()
            w, h, c = np.asarray(im).shape                                  # 256 * 256 * 3
            ratio_x = self.width / float(w)
            ratio_y = self.height / float(h)                    # 368 / 256 = 1.4375

            im = im.resize((self.width, self.height))                # unit8      368 * 368 * 3
            images[(i*3):(i*3+3), :, :] = transforms.ToTensor()(im)  # Tensor     3 * 368 * 368
            label = labels[img.split('.')[0][1:]]                  # list       21 * 2
            label_maps[i, :, :, :] = self.genLabelMap(label, label_size=label_size,
                                                      joints=self.joints, ratio_x=ratio_x, ratio_y=ratio_y)
        # generate the Gaussian heat map
        center_map = self.genCenterMap(x=self.width / 2.0, y=self.height / 2.0, sigma=21,
                                       size_w=self.width, size_h=self.height)
        center_map = torch.from_numpy(center_map)
        center_map = center_map.unsqueeze_(0)
  
        return images.float(), label_maps.float(), center_map.float()


    def genCenterMap(self, x, y, sigma, size_w, size_h):
        '''
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        '''
        
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d

    def genLabelMap(self, label, label_size, joints, ratio_x, ratio_y):
        '''
        generate 22 heatmaps
        :param label:           list            21 * 2
        :param boxsize:         int             368
        :param stride:          int             8
        :param joints:          int             21
        :return: heatmap        Tensor          (joints+1) * boxsize/stride * boxsize/stride
        '''
        #print label_size
        label_maps = torch.zeros(joints + 1, label_size, label_size)  # Tensor

        for i in range(len(label)):
            lbl = label[i]                  # [x, y]
            x = lbl[0] * ratio_x/8.0         # modify the label
            y = lbl[1] * ratio_y/8.0
            heatmap = self.genCenterMap(x, y, sigma=self.sigma, size_w=label_size, size_h=label_size)

            label_maps[i, :, :] = torch.from_numpy(np.transpose(heatmap))

        label_maps[joints, :, :] = torch.from_numpy(np.transpose(heatmap))  # !!!
        return label_maps                       # Tensor          (joints + 1) * boxsize/stride * boxsize/stride




