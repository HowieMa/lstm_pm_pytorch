import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from PIL import Image


class UCIHandPoseDataset(Dataset):

    def __init__(self, data_dir, label_dir, train, temporal=5, joints=21, transform=None, sigma=1):
        self.height = 368
        self.width = 368

        self.seqs = os.listdir(data_dir)  # 001L00, 001L01, L02,... 151L08, R01, R02, R03,...
        self.data_dir = data_dir
        self.label_dir = label_dir

        self.temporal = temporal * 6
        self.transform = transform
        self.joints = joints  # 21 heat maps
        self.sigma = sigma  # gaussian center heat map sigma

        self.temporal_dir = []

        self.train = train
        if self.train is True:
            self.gen_temporal_dir(temporal)

    def gen_temporal_dir(self, step):
        """
        build temporal directory in order to guarantee get all images has equal chance to be trained
        for train dataset, make each image has the same possibility to be trained

        :param step: for training set, step = 1, for test set, step = temporal
        :return:
        """

        for seq in self.seqs:
            if seq == '.DS_Store':
                continue
            image_path = os.path.join(self.data_dir, seq)  #
            imgs = os.listdir(image_path)  # [0005.jpg, 0011.jpg......]
            imgs.sort()

            img_num = len(imgs)
            if img_num < self.temporal:
                continue  # ignore sequences whose length is less than temporal

            for i in range(0, img_num - self.temporal + 1, step):
                tmp = []
                for k in range(i, i + self.temporal):
                    tmp.append(os.path.join(image_path, imgs[k]))
                self.temporal_dir.append(tmp)  #

        self.temporal_dir.sort()
        print 'total numbers of image sequence is ' + str(len(self.temporal_dir))

    def __len__(self):
        if self.train is True:
            length = len(self.temporal_dir)/self.temporal
        else:
            length = len(self.temporal_dir)
        return length

    def __getitem__(self, idx):
        """
        :param idx:
        :return:
        images          3D Tensor      (temporal * 3)   *   height(368)   *   weight(368)
        label_map       4D Tensor      temporal         *   joints        *   label_size(45)   *   label_size(45)
        center_map      3D Tensor      1                *   height(368)   *   weight(368)
        imgs            list of image directory
        """
        label_size = self.width / 8 - 1         # 45

        imgs = self.temporal_dir[idx]           # ['.../001L0/L0005.jpg', '.../001L0/L0011.jpg', ... ]
        imgs.sort()
        seq = imgs[0].split('/')[-2]            # 001L0
        label_path = os.path.join(self.label_dir, seq)
        labels = json.load(open(label_path + '.json'))

        # initialize
        images = torch.zeros(self.temporal * 3, self.width, self.height)
        label_maps = torch.zeros(self.temporal, self.joints, label_size, label_size)

        for i in range(self.temporal):          # get temporal images
            img = imgs[i]                       # '.../001L0/L0005.jpg'

            # get image
            im = Image.open(img)                # read image
            w, h, c = np.asarray(im).shape      # weight 256 * height 256 * 3
            ratio_x = self.width / float(w)
            ratio_y = self.height / float(h)    # 368 / 256 = 1.4375

            im = im.resize((self.width, self.height))                       # unit8      weight 368 * height 368 * 3
            images[(i * 3):(i * 3 + 3), :, :] = transforms.ToTensor()(im)   # 3D Tensor  3 * height 368 * weight 368
            # ToTensor function will normalize data

            # get label map
            img_num = img.split('/')[-1][1:5]

            if img_num in labels: # for images without label, set label to zero
                label = labels[img_num]         # 0005  list       21 * 2
                lbl = self.genLabelMap(label, label_size=label_size, joints=self.joints, ratio_x=ratio_x, ratio_y=ratio_y)
                label_maps[i, :, :, :] = torch.from_numpy(lbl)

        # generate the Gaussian heat map
        center_map = self.genCenterMap(x=self.width / 2.0, y=self.height / 2.0, sigma=21,
                                       size_w=self.width, size_h=self.height)
        center_map = torch.from_numpy(center_map)
        center_map = center_map.unsqueeze_(0)

        return images.float(), label_maps.float(), center_map.float(), imgs

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
            label_maps[i, :, :] = np.transpose(heatmap)

        return label_maps  # numpy           label_size * label_size * joints


# test case

if __name__ == '__main__':
    temporal = 5
    data_dir = '../dataset/frames/001'
    label_dir = '../dataset/label/001'

    dataset = UCIHandPoseDataset(data_dir=data_dir, label_dir=label_dir, temporal=temporal)

    a = dataset.temporal_dir
    images, label_maps,center_map =  dataset[2]
    print images.shape  # (5*3) * 368 * 368
    print label_maps.shape  # 5 21 45 45

