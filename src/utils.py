import json
import numpy as np
import os
import scipy.misc


def loss_history_init(temporal=5):
    loss_history = {}
    for t in range(temporal):
        loss_history['temporal'+str(t)] = []
    loss_history['total'] = 0.0
    return loss_history


def save_loss(predict_heatmaps, label_map, epoch, step, criterion, train, temporal=5, save_dir='ckpt/'):
    loss_save = loss_history_init(temporal=temporal)

    predict = predict_heatmaps[0]
    target = label_map[:, 0, :, :, :]
    initial_loss = criterion(predict, target)  # loss initial
    total_loss = initial_loss

    for t in range(temporal):
        predict = predict_heatmaps[t + 1]
        target = label_map[:, t, :, :, :]
        tmp_loss = criterion(predict, target)  # loss in each stage
        total_loss += tmp_loss
        loss_save['temporal' + str(t)] = float('%.8f' % tmp_loss)

    total_loss = total_loss
    loss_save['total'] = float(total_loss)

    # save loss to file
    if train is True:
        if not os.path.exists(save_dir + 'loss_epoch' + str(epoch)):
            os.mkdir(save_dir + 'loss_epoch' + str(epoch))
        json.dump(loss_save, open(save_dir + 'loss_epoch' + str(epoch) + '/s' + str(step).zfill(4) + '.json', 'wb'))

    else:
        if not os.path.exists(save_dir + 'loss_test/'):
            os.mkdir(save_dir + 'loss_test/')
        json.dump(loss_save, open(save_dir + 'loss_test/' + str(step).zfill(4) + '.json', 'wb'))

    return total_loss


def save_images(label_map, predict_heatmaps, step, epoch, imgs, train, pck=1, temporal=5, save_dir='ckpt/'):
    """
    :param label_map:
    :param predict_heatmaps:    5D Tensor    Batch_size  *  Temporal * joints *   45 * 45
    :param step:
    :param temporal:
    :param epoch:
    :param train:
    :param imgs: list [(), (), ()] temporal * batch_size
    :return:
    """

    for b in range(label_map.shape[0]):                     # for each batch (person)
        output = np.ones((50 * 2, 50 * temporal))           # cd .. temporal save a single image
        seq = imgs[0][b].split('/')[-2]                     # sequence name 001L0
        img = ""
        for t in range(temporal):                           # for each temporal
            im = imgs[t][b].split('/')[-1][1:5]             # image name 0005
            img += '_' + im
            pre = np.zeros((45, 45))  #
            gth = np.zeros((45, 45))
            for i in range(21):                             # for each joint
                pre += np.asarray(predict_heatmaps[t][b, i, :, :].data)  # 2D
                gth += np.asarray(label_map[b, t, i, :, :].data)         # 2D

            output[0:45,  50 * t: 50 * t + 45] = gth
            output[50:95, 50 * t: 50 * t + 45] = pre

        if train is True:
            if not os.path.exists(save_dir + 'epoch'+str(epoch)):
                os.mkdir(save_dir + 'epoch'+str(epoch))
            scipy.misc.imsave(save_dir + 'epoch'+str(epoch) + '/s'+str(step) + '_b' + str(b) + '_' + seq + img + '.jpg', output)
        else:

            if not os.path.exists(save_dir + 'test'):
                os.mkdir(save_dir + 'test')
            scipy.misc.imsave(save_dir + 'test' + '/s' + str(step) + '_b' + str(b) + '_'
                              + seq + img + '_' + str(round(pck, 4)) + '.jpg', output)


def lstm_pm_evaluation(label_map, predict_heatmaps, sigma=0.04, temporal=5):
    pck_eval = []
    empty = np.zeros((21, 45, 45))                                      # 3D numpy 21 * 45 * 45
    for b in range(label_map.shape[0]):        # for each batch (person)
        for t in range(temporal):           # for each temporal
            target = np.asarray(label_map[b, t, :, :, :].data)          # 3D numpy 21 * 45 * 45
            predict = np.asarray(predict_heatmaps[t][b, :, :, :].data)  # 3D numpy 21 * 45 * 45
            if not np.equal(empty, target):
                pck_eval.append(PCK(predict, target, sigma=sigma))

    return sum(pck_eval) / float(len(pck_eval))  #


def PCK(predict, target, label_size=45, sigma=0.04):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than  the value is 1, otherwise it is 0
    :param predict:         3D numpy       21 * 45 * 45
    :param target:          3D numpy       21 * 45 * 45
    :param label_size:
    :param sigma:
    :return: 0/21, 1/21, ...
    """
    pck = 0
    for i in range(predict.shape[0]):
        pre_x, pre_y = np.where(predict[i, :, :] == np.max(predict[i, :, :]))
        tar_x, tar_y = np.where(target[i, :, :] == np.max(target[i, :, :]))

        dis = np.sqrt((pre_x[0] - tar_x[0])**2 + (pre_y[0] - tar_y[0])**2)
        if dis < sigma * label_size:
            pck += 1
    return pck / float(predict.shape[0])


def draw_loss(epoch):
    all_losses = os.listdir('ckpt/loss_epoch'+str(epoch))
    losses = []

    for loss_j in all_losses:
        loss = json.load('ckpt/loss_epoch'+str(epoch) + '/' +loss_j)
        a = loss['total']
        losses.append(a)









