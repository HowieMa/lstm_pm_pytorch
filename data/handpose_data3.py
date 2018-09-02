"""
train model without label
"""
import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset
import numpy as np
import json
import math
from PIL import Image


class UCIHandPoseDataset(Dataset):

    def __init__(self, data_dir, data_dir2, label_dir, train=True, sample=2, temporal=5, joints=21, sigma=1):
        """
        :param data_dir: 6 frames data
        :param data_dir2: all frames data
        :param label_dir:
        :param train:
        :param sample:
        :param temporal:
        :param joints:
        :param transform:
        :param sigma:
        """
        self.height = 368
        self.width = 368

        self.seqs = os.listdir(data_dir)  # 001L00, 001L01, L02,... 151L08, R01, R02, R03,...
        self.data_dir = data_dir
        self.data_dir2 = data_dir2
        self.label_dir = label_dir

        self.temporal = temporal  # insert 5 images in each two labeled picture
        self.joints = joints  # 21 heat maps
        self.sigma = sigma  # gaussian center heat map sigma

        self.sample = sample
        self.temporal_dir = []

        self.train = train
        if self.train is True:
            self.gen_temporal_dir(1)

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
        length = len(self.temporal_dir)/self.temporal
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
        real_temporal = self.temporal * (self.sample + 1) - self.sample  # truly number of images
        imgs = self.temporal_dir[idx]           # ['.../001L0/L0005.jpg', '.../001L0/L0011.jpg', ... ]
        imgs.sort()

        # get label in json form
        seq = imgs[0].split('/')[-2]            # 001L0
        label_path = os.path.join(self.label_dir, seq)
        labels = json.load(open(label_path + '.json'))

        all_frame_path = os.path.join(self.data_dir2, seq)  #
        all_frames = os.listdir(all_frame_path)  # [0001.jpg, 0002.jpg......]
        all_frames.sort()

        # Initialize Tensor
        images = torch.zeros(real_temporal * 3, self.width, self.height)
        label_maps = torch.zeros(real_temporal, self.joints, label_size, label_size)


        # *******************   get image with label   *******************
        img_count = 0
        pre_img = imgs[0][-8:-4]  # initial

        img = imgs[0]
        im = Image.open(img)  # read image
        w, h, c = np.asarray(im).shape  # weight 256 * height 256 * 3
        ratio_x = self.width / float(w)
        ratio_y = self.height / float(h)  # 368 / 256 = 1.4375
        im = im.resize((self.width, self.height))  # unit8      weight 368 * height 368 * 3

        images[(img_count * 3):(img_count * 3 + 3), :, :] = transforms.ToTensor()(im)
        # 3D Tensor  3 * height 368 * weight 368 ToTensor function will normalize data

        # get label map
        img_num = img.split('/')[-1][1:5]
        if img_num in labels:  # for images without label, set label to zero
            label = labels[img_num]  # 0005  list       21 * 2
            lbl = self.genLabelMap(label, label_size=label_size, joints=self.joints, ratio_x=ratio_x, ratio_y=ratio_y)
            label_maps[img_count, :, :, :] = torch.from_numpy(lbl)

        img_count += 1
        # get all img
        for i in range(1, self.temporal):          # get temporal images
            img = imgs[i]                       # '.../001L0/L0005.jpg'

            # *****************   get image without label   *****************
            img_name = img[-8:-4]               # 0005 name of image with label
            sample_wait_list = []
            samples = []
            for frame in all_frames:
                if pre_img < frame[-8:-4] < img_name:
                    sample_wait_list.append(frame)
            sample_wait_list.sort()
            step = int(math.ceil((len(sample_wait_list)+2) / float(self.sample+ 2)))  #

            for s in range(step - 1, len(sample_wait_list), step):
                samples.append(sample_wait_list[s])

            for sap in samples:
                img_path = os.path.join(all_frame_path, sap)
                im = Image.open(img_path)
                im = im.resize((self.width, self.height))  # unit8      weight 368 * height 368 * 3
                images[(img_count * 3):(img_count * 3 + 3), :, :] = transforms.ToTensor()(im)
                img_count += 1
            pre_img = img_name

            #  *******************   get image with label   *******************
            im = Image.open(img)  # read image
            w, h, c = np.asarray(im).shape  # weight 256 * height 256 * 3
            ratio_x = self.width / float(w)
            ratio_y = self.height / float(h)  # 368 / 256 = 1.4375
            im = im.resize((self.width, self.height))  # unit8      weight 368 * height 368 * 3

            images[(img_count * 3):(img_count * 3 + 3), :, :] = transforms.ToTensor()(im)
            # 3D Tensor  3 * height 368 * weight 368 ToTensor function will normalize data

            # get label map
            img_num = img.split('/')[-1][1:5]

            if img_num in labels:  # for images without label, set label to zero
                label = labels[img_num]  # 0005  list       21 * 2
                lbl = self.genLabelMap(label, label_size=label_size, joints=self.joints, ratio_x=ratio_x,
                                       ratio_y=ratio_y)
                label_maps[img_count, :, :, :] = torch.from_numpy(lbl)
            img_count += 1

        # ***************** generate the Gaussian heat map  *****************
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
    import scipy.misc
    temporal = 4
    data_dir = '../dataset/train_data'
    data_dir2 = '../dataset/train_full_data'
    label_dir = '../dataset/train_label'

    dataset = UCIHandPoseDataset(data_dir=data_dir, data_dir2=data_dir2, sample=2, label_dir=label_dir, temporal=temporal)

    a = dataset.temporal_dir
    images, label_maps, center_map, imgs = dataset[3]
    print images.shape  # (5*3) * 368 * 368
    print label_maps.shape  # 5 21 45 45
    print imgs
    np.asarray(images)
    label_maps = np.asarray(label_maps)

    # ***************** draw label map *****************
    out_labels = np.ones((45, 50 * label_maps.shape[0]))
    for i in range(label_maps.shape[0]):
        out = np.zeros((45,45))
        for o in range(21):
            out += label_maps[i, o, :, :]

        out_labels[:, i * 50:i * 50 + 45] = out
        scipy.misc.imsave('label.jpg', out_labels)

    # ***************** draw image *****************
    im_size = 368
    target = Image.new('RGB', (label_maps.shape[0]*im_size, im_size))
    left = 0
    right = im_size
    for i in range(label_maps.shape[0]):
        im = images[i*3:i*3 + 3, :, :]
        im = transforms.ToPILImage()(im)
        target.paste(im, (left, 0, right, im_size))
        left += im_size
        right += im_size
    target.save('img.jpg')


