import json
import numpy as np
import os
import scipy.misc


def loss_history_init_lstm(temporal=5):
    loss_history = {}
    for t in range(temporal):
        loss_history['temporal'+str(t)] = []
    loss_history['total'] = 0.0
    return loss_history

def init_loss_history_cpm(nb_stage=6):
    loss_history = {}
    for i in range(nb_stage):
        loss_history['stage'+str(i+1)] = []
    loss_history['all'] = []
    loss_history['train_pckall'] = []
    loss_history['test_pck'] = {}
    loss_history['lr'] = []
    return loss_history

def init_runtime_loss_cpm(nb_stage = 6):
    runtime_loss = []
    for i in range(nb_stage):
        runtime_loss.append(0)   #loss for each stage 
    return runtime_loss

def save_loss(predict_heatmaps, label_map, epoch, step, criterion, train, temporal=5, save_dir='ckpt/'):
    loss_save = loss_history_init(temporal=temporal)
    total_loss = 0

    for t in range(temporal):
        predict = predict_heatmaps[t]
        target = label_map[:, t, :, :, :]
        tmp_loss = criterion(predict, target)
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

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def save_image(save_path, nb_temporal, pred_heatmap, label_map, size=45, nb_heatmap=21):
    # predict_heatmaps shape: [nb_temporal, batch_size,22,45,45]
    for i in range(pred_heatmap[0].shape[0]):# each batch (person)
        if label_map is None:
            s = np.zeros((size, size*nb_temporal)) 
        else:
            s = np.zeros((size*2, size*nb_temporal)) 
            
        for j in range(len(pred_heatmap)):  # each temporal
            for k in range(nb_heatmap):
                if label_map is None:
                    s[:, j*45:(j+1)*45]+=pred_heatmap[j].cpu().data.numpy()[i,k,:,:]
                else:
                    s[:45, j*45:(j+1)*45]+=label_map[i , j, k, :, :]
                    s[45:, j*45:(j+1)*45]+=pred_heatmap[j].cpu().data.numpy()[i,k,:,:]
        scipy.misc.imsave(save_path+str(i+1)+'.jpg', s)
        
def save_image_cpm(save_path, pred_heatmap, label_map, size=45, nb_heatmap=21, nb_stage =6):
    # predict_heatmaps shape: [nb_temporal, batch_size,22,45,45]
    for i in range(pred_heatmap.shape[0]):# each batch (person)
        if label_map is None:
            s = np.zeros((size, size*nb_stage)) 
        else:
            s = np.zeros((size, size*(1+nb_stage))) 
            
        for j in range(nb_stage):  # each stage
            for k in range(nb_heatmap):
                s[:, j*45:(j+1)*45]+=pred_heatmap[i,j,k,:,:].cpu().data.numpy()
                if label_map is not None:
                    s[:, -45:]+=label_map[i , k, :, :]
        scipy.misc.imsave(save_path+str(i+1)+'.jpg', s)
        
def lstm_pm_evaluation(label_map, predict_heatmaps, sigma=0.04, temporal=5):
    pck_eval = []
    for b in range(label_map.shape[0]):        # for each batch (person)
        for t in range(temporal):           # for each temporal
            target = np.asarray(label_map[b, t, :, :, :].data)
            predict = np.asarray(predict_heatmaps[t][b, :, :, :].data)
            pck_eval.append(PCK(predict, target, sigma=sigma))

    return sum(pck_eval) / float(len(pck_eval))  #


def cpm_evaluation(label_map, predict_heatmaps,nb_stage=6, sigma=0.04):
    pck_eval = []
    for b in range(label_map.shape[0]):        # for each batch (person)
        for s in range(nb_stage):
            target = np.asarray(label_map[b, :, :, :].data)
            predict = np.asarray(predict_heatmaps[b,s, :, :, :].data)
            pck_eval.append(PCK(predict, target, sigma=sigma))

    return sum(pck_eval) / float(len(pck_eval))  #

def PCK(predict, target, label_size=45, sigma=0.04):
    """
    calculate possibility of correct key point of one single image
    if distance of ground truth and predict point is less than sigma, than
    :param predict:         3D numpy       22 * 45 * 45
    :param target:          3D numpy       22 * 45 * 45
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








