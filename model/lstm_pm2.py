"""
Just try somehing new.....
not finish yet ...
"""
import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvNet1(nn.Module):
    def __init__(self, outc):
        super(ConvNet1, self).__init__()
        self.conv1_convnet1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)  # 3 * 368 * 368
        self.pool1_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_convnet1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3_convnet1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4_convnet1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_convnet1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_convnet1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_convnet1 = nn.Conv2d(512, self.outclass, kernel_size=1)  # 512 * 45 * 45
        self.outclass = outc

    def forward(self, image):
        """
        :param image: 3 * 368 * 368
        :return: initial_heatmap out_class * 45 * 45
        """
        x = self.pool1_convnet1(F.relu(self.conv1_convnet1(image)))  # output 128 * 184 * 184
        x = self.pool2_convnet1(F.relu(self.conv2_convnet1(x)))  # output 128 * 92 * 92
        x = self.pool3_convnet1(F.relu(self.conv3_convnet1(x)))  # output 128 * 45 * 45
        x = F.relu(self.conv4_convnet1(x))  # output 32 * 45 * 45
        x = F.relu(self.conv5_convnet1(x))  # output 512 * 45 * 45
        x = F.relu(self.conv6_convnet1(x))  # output 512 * 45 * 45
        initial_heatmap = self.conv7_convnet1(x)  # output (class + 1) * 45 * 45
        return initial_heatmap


class ConvNet2(nn.Module):
    def __init__(self):
        super(ConvNet2, self).__init__()
        self.conv1_convnet2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)  # 3 * 368 * 368
        self.pool1_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_convnet2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)  # 128 * 184 * 184
        self.pool2_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3_convnet2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)  # 128 * 92 * 92
        self.pool3_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4_convnet2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)  # 32 * 45 * 45

    def forward(self, image):
        """
        :param image: 3 * 368 * 368
        :return: Fs(.) features 32 * 45 * 45
        """
        x = self.pool1_convnet2(F.relu(self.conv1_convnet2(image)))  # output 128 * 184 * 184
        x = self.pool2_convnet2(F.relu(self.conv2_convnet2(x)))  # output 128 * 92 * 92
        x = self.pool3_convnet2(F.relu(self.conv3_convnet2(x)))  # output 128 * 45 * 45
        x = F.relu(self.conv4_convnet2(x))  # output 32 * 45 * 45


class ConvNet3(nn.Module):
    def __init__(self, outc):
        super(ConvNet3, self).__init__()
        self.Mconv1_convnet3 = nn.Conv2d(48, 128, kernel_size=11, padding=5)
        self.Mconv2_convnet3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_convnet3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_convnet3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_convnet3 = nn.Conv2d(128, self.outclass, kernel_size=1, padding=0)
        self.outclass = outc

    def forward(self, input):
        """
        :param input: 48 * 45 * 45
        :return:
        """
        x = F.relu(self.Mconv1_convnet3(input))  # output 128 * 45 * 45
        x = F.relu(self.Mconv2_convnet3(x))  # output 128 * 45 * 45
        x = F.relu(self.Mconv3_convnet3(x))  # output 128 * 45 * 45
        x = F.relu(self.Mconv4_convnet3(x))  # output 128 * 45 * 45
        x = self.Mconv5_convnet3(x)  # output (class+1) * 45 * 45
        return x  # heatmap (class+1) * 45 * 45


class LSTM0(nn.Module):
    def __init__(self, outc):
        super(LSTM0, self).__init__()
        self.conv_gx_lstm0 = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1)
        self.outclass = outc

    def forward(self, x):
        gx = self.conv_gx_lstm0(x)
        ix = self.conv_ix_lstm0(x)
        ox = self.conv_ox_lstm0(x)

        gx = F.tanh(gx)
        ix = F.sigmoid(ix)
        ox = F.sigmoid(ox)

        cell1 = F.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1


class LSTMPoseMachine(nn.Module):
    def __init__(self, outclass=21, T=5):
        super(LSTMPoseMachine, self).__init__()
        self.outclass = outclass
        self.T = 5
        self.stage1 = ConvNet1(outclass)


class LSTM_PM(nn.Module):
    def __init__(self, outclass=21, T=7):
        super(LSTM_PM, self).__init__()
        self.outclass = outclass
        self.T = T
        self.pool_center_lower = nn.AvgPool2d(kernel_size=9, stride=8)

        # conv_net1
        self.conv1_convnet1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)  # 3 * 368 * 368
        self.pool1_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_convnet1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3_convnet1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4_convnet1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_convnet1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_convnet1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_convnet1 = nn.Conv2d(512, self.outclass, kernel_size=1)  # 512 * 45 * 45

        # conv_net2
        self.conv1_convnet2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)  # 3 * 368 * 368
        self.pool1_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_convnet2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)  # 128 * 184 * 184
        self.pool2_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3_convnet2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)  # 128 * 92 * 92
        self.pool3_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4_convnet2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)  # 32 * 45 * 45

        # conv_net3
        self.Mconv1_convnet3 = nn.Conv2d(48, 128, kernel_size=11, padding=5)
        self.Mconv2_convnet3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_convnet3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_convnet3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_convnet3 = nn.Conv2d(128, self.outclass, kernel_size=1, padding=0)

        # lstm
        self.conv_ix_lstm = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        # initial lstm
        self.conv_gx_lstm0 = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(32 + 1 + self.outclass, 48, kernel_size=3, padding=1)

    def convnet1(self, image):
        '''
        :param image: 3 * 368 * 368
        :return: initial_heatmap out_class * 45 * 45
        '''
        x = self.pool1_convnet1(F.relu(self.conv1_convnet1(image)))  # output 128 * 184 * 184
        x = self.pool2_convnet1(F.relu(self.conv2_convnet1(x)))  # output 128 * 92 * 92
        x = self.pool3_convnet1(F.relu(self.conv3_convnet1(x)))  # output 128 * 45 * 45
        x = F.relu(self.conv4_convnet1(x))  # output 32 * 45 * 45
        x = F.relu(self.conv5_convnet1(x))  # output 512 * 45 * 45
        x = F.relu(self.conv6_convnet1(x))  # output 512 * 45 * 45
        initial_heatmap = self.conv7_convnet1(x)  # output (class + 1) * 45 * 45
        return initial_heatmap

    def convnet2(self, image):
        '''
        :param image: 3 * 368 * 368
        :return: Fs(.) features 32 * 45 * 45
        '''
        x = self.pool1_convnet2(F.relu(self.conv1_convnet2(image)))  # output 128 * 184 * 184
        x = self.pool2_convnet2(F.relu(self.conv2_convnet2(x)))  # output 128 * 92 * 92
        x = self.pool3_convnet2(F.relu(self.conv3_convnet2(x)))  # output 128 * 45 * 45
        x = F.relu(self.conv4_convnet2(x))  # output 32 * 45 * 45
        return x  # output 32 * 45 * 45

    def convnet3(self, hide_t):
        """
        :param h_t: 48 * 45 * 45
        :return: heatmap   out_class * 45 * 45
        """
        x = F.relu(self.Mconv1_convnet3(hide_t))  # output 128 * 45 * 45
        x = F.relu(self.Mconv2_convnet3(x))  # output 128 * 45 * 45
        x = F.relu(self.Mconv3_convnet3(x))  # output 128 * 45 * 45
        x = F.relu(self.Mconv4_convnet3(x))  # output 128 * 45 * 45
        x = self.Mconv5_convnet3(x)  # output (class+1) * 45 * 45
        return x  # heatmap (class+1) * 45 * 45

    def lstm(self, heatmap, features, centermap, hide_t_1, cell_t_1):
        '''
        :param heatmap:     class * 45 * 45
        :param features:    32 * 45 * 45
        :param centermap:   1 * 45 * 45
        :param hide_t_1:    48 * 45 * 45
        :param cell_t_1:    48 * 45 * 45
        :return:
        hide_t:    48 * 45 * 45
        cell_t:    48 * 45 * 45
        '''
        xt = torch.cat([heatmap, features, centermap], dim=1)  # (32+ class+1 +1 ) * 45 * 45

        gx = self.conv_gx_lstm(xt)  # output: 48 * 45 * 45
        gh = self.conv_gh_lstm(hide_t_1)  # output: 48 * 45 * 45
        g_sum = gx + gh
        gt = F.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)  # output: 48 * 45 * 45
        oh = self.conv_oh_lstm(hide_t_1)  # output: 48 * 45 * 45
        o_sum = ox + oh
        ot = F.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)  # output: 48 * 45 * 45
        ih = self.conv_ih_lstm(hide_t_1)  # output: 48 * 45 * 45
        i_sum = ix + ih
        it = F.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)  # output: 48 * 45 * 45
        fh = self.conv_fh_lstm(hide_t_1)  # output: 48 * 45 * 45
        f_sum = fx + fh
        ft = F.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * F.tanh(cell_t)

        return cell_t, hide_t

    def lstm0(self, x):
        gx = self.conv_gx_lstm0(x)
        ix = self.conv_ix_lstm0(x)
        ox = self.conv_ox_lstm0(x)

        gx = F.tanh(gx)
        ix = F.sigmoid(ix)
        ox = F.sigmoid(ox)

        cell1 = F.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1

    def stage2(self, image, cmap, heatmap, cell_t_1, hide_t_1):
        '''
        :param image:               3 * 368 * 368
        :param cmap: gaussian       1 * 368 * 368
        :param heatmap:             out_class * 45 * 45
        :param cell_t_1:            48 * 45 * 45
        :param hide_t_1:            48 * 45 * 45
        :return:
        new_heatmap:                out_class * 45 * 45
        cell_t:                     48 * 45 * 45
        hide_t:                     48 * 45 * 45
        '''
        features = self.convnet2(image)
        centermap = self.pool_center_lower(cmap)
        cell_t, hide_t = self.lstm(heatmap, features, centermap, hide_t_1, cell_t_1)
        new_heat_map = self.convnet3(hide_t)
        return new_heat_map, cell_t, hide_t

    def stage1(self, image, cmap):
        '''
        :param image:                3 * 368 * 368
        :param cmap:                 1 * 368 * 368
        :return:
        heatmap:                     out_class * 45 * 45
        cell_t:                      48 * 45 * 45
        hide_t:                      48 * 45 * 45
        '''
        initial_heatmap = self.convnet1(image)
        features = self.convnet2(image)
        centermap = self.pool_center_lower(cmap)

        x = torch.cat([initial_heatmap, features, centermap], dim=1)
        cell1, hide1 = self.lstm0(x)
        heatmap = self.convnet3(hide1)
        return initial_heatmap, heatmap, cell1, hide1

    def forward(self, images, center_map):
        '''

        :param images:      Tensor      T * 3 * w(368) * h(368)
        :param center_map:  Tensor      1 * 368 * 368
        :return:
        heatmaps            list        (T + 1 ) * out_class * 45 * 45
        '''
        image = images[:, 0:3, :, :]

        heat_maps = []
        initial_heatmap, heatmap, cell, hide = self.stage1(image, center_map)  # initial heat map
        heat_maps.append(initial_heatmap)
        heat_maps.append(heatmap)

        #
        for i in range(1, self.T):
            image = images[:, (3 * i):(3 * i + 3), :, :]
            heatmap, cell, hide = self.stage2(image, center_map, heatmap, cell, hide)
            heat_maps.append(heatmap)
        return heat_maps


# test case
if __name__ == '__main__':
    Conv_Net3()
    # net = LSTM_PM(T=4)
    # a = torch.randn(4, 3, 368, 368)
    # c = torch.randn(1, 368, 368)
    # maps = net(a, c)
    # for m in maps:
    #     print m.shape

