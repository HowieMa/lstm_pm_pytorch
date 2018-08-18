import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image


class UCIHandPoseDataset(Dataset):

    def __init__(self, data_dir, label_dir, joints=21, transform=None, sigma=1):
        self.height = 368
        self.width = 368

        self.seqs = os.listdir(data_dir)  # 001L00, 001L01, L02,... 151L08, R01, R02, R03,...
        self.data_dir = data_dir
        self.label_dir = label_dir

        self.transform = transform
        self.joints = joints  # 21 heat maps
        self.sigma = sigma  # gaussian center heat map sigma

        self.images_dir = []
        self.gen_imgs_dir()

    def gen_imgs_dir(self):
        """
        get directory of all images
        :return:
        """

        for seq in self.seqs:
            if seq == '.DS_Store':
                continue
            image_path = os.path.join(self.data_dir, seq)  #
            imgs = os.listdir(image_path)  # [0005.jpg, 0011.jpg......]
            for i in range(len(imgs)):
                self.images_dir.append(image_path + '/' + imgs[i])  #

        print 'total numbers of image is ' + str(len(self.images_dir))

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        images          3D Tensor      3                *   height(368)      *   weight(368)
        label_map       3D Tensor      (joints + 1)     *   label_size(45)   *   label_size(45)
        center_map      3D Tensor      1                *   height(368)      *   weight(368)
        """

        label_size = self.width / 8 - 1         # 45
        img = self.images_dir[idx]           # # '.../001L0/L0005.jpg'

        seq = img.split('/')[-2]            # 001L0
        label_path = os.path.join(self.label_dir, seq)
        labels = json.load(open(label_path + '.json'))

        # get image
        im = Image.open(img)                # read image
        w, h, c = np.asarray(im).shape      # weight 256 * height 256 * 3
        ratio_x = self.width / float(w)
        ratio_y = self.height / float(h)    # 368 / 256 = 1.4375
        im = im.resize((self.width, self.height))                       # unit8      weight 368 * height 368 * 3
        image = transforms.ToTensor()(im)   # 3D Tensor  3 * height 368 * weight 368

        # get label map
        label = labels[img.split('/')[-1][1:5]]         # 0005  list       21 * 2
        lbl = self.genLabelMap(label, label_size=label_size, joints=self.joints, ratio_x=ratio_x, ratio_y=ratio_y)
        label_maps = torch.from_numpy(lbl)

        # generate the Gaussian heat map
        center_map = self.genCenterMap(x=self.width / 2.0, y=self.height / 2.0, sigma=21,
                                       size_w=self.width, size_h=self.height)
        center_map = torch.from_numpy(center_map)
        center_map = center_map.unsqueeze_(0)

        return image.float(), label_maps.float(), center_map.float(), img

    def genCenterMap(self, x, y, sigma, size_w, size_h):
        """
        generate Gaussian heat map
        :param x: center point
        :param y: center point
        :param sigma:
        :param size_w: image width
        :param size_h: image height
        :return:            numpy           w * h
        """
        gridy, gridx = np.mgrid[0:size_h, 0:size_w]
        D2 = (gridx - x) ** 2 + (gridy - y) ** 2
        return np.exp(-D2 / 2.0 / sigma / sigma)  # numpy 2d

    def genLabelMap(self, label, label_size, joints, ratio_x, ratio_y):
        """
        generate label heat map
        :param label:               list            21 * 2
        :param label_size:          int             45
        :param joints:              int             21
        :param ratio_x:             float           1.4375
        :param ratio_y:             float           1.4375
        :return:  heatmap           numpy           joints * boxsize/stride * boxsize/stride
        """
        # initialize
        label_maps = np.zeros((joints, label_size, label_size))
        background = np.zeros((label_size, label_size))

        # each joint
        for i in range(len(label)):
            lbl = label[i]                      # [x, y]
            x = lbl[0] * ratio_x / 8.0          # modify the label
            y = lbl[1] * ratio_y / 8.0
            heatmap = self.genCenterMap(y, x, sigma=self.sigma, size_w=label_size, size_h=label_size)  # numpy
            background += heatmap               # numpy
            label_maps[i, :, :] = np.transpose(heatmap)  # !!!

        return label_maps  # numpy           label_size * label_size * (joints + 1)


## test case
# data = UCIHandPoseDataset(data_dir='/Users/mahaoyu/UCI/HandsPoseDataset/train/train_data',
#                           label_dir='/Users/mahaoyu/UCI/HandsPoseDataset/train/train_label')
#
# img, label, center, name = data[1]
#
# print img.shape
# print label.shape
# print center.shape
# print name
#
#
# lab = np.asarray(label)
# out = np.zeros(((45, 45)))
# for i in range(21):
#     out += lab[i,:, :]