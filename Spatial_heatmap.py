import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
import mne
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json
import random
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchsummary import summary
import tqdm
import sys, time, copy
import torch.utils.data
from experiments.config import config
from sklearn.model_selection import LeaveOneOut
from experiments import utils as exp_utils


base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)
from stl2g.preprocessing.config import CONSTANT
from stl2g.preprocessing.OpenBMI import raw
from stl2g.preprocessing.BCIIV2A import raw as raw_bci2a
from stl2g.utils import get_loaders
from stl2g.model.L2GNet import L2GNet
from experiments.make_montage import plot_montage, set_montage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes



def L2GNet_prepare_training(spatial_div_dict, temporal_div_dict ,d_model_dic,  head_dic, d_ff, n_layers, dropout, lr,
                  clf_class, domain_class, ch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = L2GNet(spatial_div_dict, temporal_div_dict ,d_model_dic,  head_dic, d_ff, n_layers, dropout,
                  clf_class, domain_class, ch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    model_type = 'L2GNet'
    dataSet = 'OpenBMI'
    model_path = '/home/alk/L2G-MI/checkpoints/OpenBMI/L2GNet/test/test.pth'
    path = CONSTANT[dataSet]['raw_path']
    num_ch = len(CONSTANT[dataSet]['sel_chs'])
    batch_size = config[model_type][dataSet]['batch_size']
    epochs = config[model_type][dataSet]['epochs']
    lr = config[model_type][dataSet]['lr']
    dropout = config[model_type][dataSet]['dropout']
    d_model_dict = config[model_type][dataSet]['d_model_dict']
    head_dict = config[model_type][dataSet]['head_dict']
    d_ff = config[model_type][dataSet]['d_ff']
    n_layers = config[model_type][dataSet]['n_layers']
    clf_class = config[model_type][dataSet]['num_class']
    domain_class = CONSTANT[dataSet]['n_subjs']
    alpha_scala = config[model_type][dataSet]['alpha_scala']
    spatial_local_dict = CONSTANT[dataSet]['spatial_ch_group']
    temporal_local_dict = CONSTANT[dataSet]['temporal_ch_region']
    # 加载
    sel_chs = CONSTANT[dataSet]['sel_chs']
    id_ch_selected = raw.chanel_selection(sel_chs)
    div_id = raw.channel_division(spatial_local_dict)
    spatial_region_split = raw.region_id_seg(div_id, id_ch_selected)
    ret_dict = {}
    # 取subject 数据
    subjects = [[4], [5], [6], [8], [15]]
    for subject in subjects:
        train_X, train_y, train_domain_y = raw.load_data_batchs(path, 1, subject, clf_class,
                                                                id_ch_selected, 0.1)
        # 找到y中不同类别的索引
        sub_lab_dic = {}
        sub_lab_dic_t = {}
        for label in [0, 1]:
            class_indices = np.argwhere(train_y == label)
            # 选择每个类别的第一个索引
            sample_indices = class_indices[:, 0]
            # 在X中取出对应的样本
            selected_X = train_X[sample_indices]
            selected_y = train_y[sample_indices]
            # inp = inp.unsqueeze(1)
            model, optimizer, lr_scheduler, criterion, device, criterion_domain = \
                L2GNet_prepare_training(spatial_region_split, temporal_local_dict, d_model_dict, head_dict, d_ff,
                                        n_layers, dropout, lr,
                                        clf_class, domain_class, num_ch)
            inp = torch.from_numpy(selected_X).type(torch.cuda.FloatTensor).to(device)
            model.load_state_dict(torch.load(model_path))
            # 获取整个网络的权重
            model.eval()
            params = model.state_dict()
            # 假设原始数据为input_data，维度为（1，20，400）
            inp.requires_grad = True
            output1, output2 = model(inp, 0.01)
            output = output1.sum()
            model.zero_grad()
            output.backward()
            grads = inp.grad
            grads = torch.abs(grads)
            # 可视化梯度*输入
            grads = grads.cpu().numpy()
            # print(params)
            # 定义反卷积层
            grads = grads - grads.min()
            grads = grads / grads.max()
            # 对 axis=2 求均值，压缩维度 0
            # 取某一个时间点的数据
            grads = grads[:,:, 100] # Right hand 可以
            # grads = grads[:, :, 150]  # Right hand 可以
            result = np.mean(grads, axis=0)
            # avg_grads = np.mean(grads, axis=0)
            # result = np.mean(avg_grads, axis=1)
            # 然后使用 dict() 函数将元组列表转换为字典
            eeg_dict = dict(zip(sel_chs, list(result)))
            sub_lab_dic[label] = eeg_dict
        ret_dict[subject[0]] = sub_lab_dic

    reMyWeight1, myinfo, my_chLa_index = set_montage(ret_dict[4][0])
    reMyWeight2, myinfo, my_chLa_index = set_montage(ret_dict[4][1])
    reMyWeight3, myinfo, my_chLa_index = set_montage(ret_dict[5][0])
    reMyWeight4, myinfo, my_chLa_index = set_montage(ret_dict[5][1])
    reMyWeight5, myinfo, my_chLa_index = set_montage(ret_dict[6][0])
    reMyWeight6, myinfo, my_chLa_index = set_montage(ret_dict[6][1])
    reMyWeight7, myinfo, my_chLa_index = set_montage(ret_dict[8][0])
    reMyWeight8, myinfo, my_chLa_index = set_montage(ret_dict[8][1])
    reMyWeight9, myinfo, my_chLa_index = set_montage(ret_dict[15][0])
    reMyWeight10, myinfo, my_chLa_index = set_montage(ret_dict[15][1])

    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(15, 6), gridspec_kw=dict(top=0.9),
                           sharex=True, sharey=True)
    weight_dic = {
        0: reMyWeight1,
        1: reMyWeight2,
        2: reMyWeight3,
        3: reMyWeight4,
        4: reMyWeight5,
        5: reMyWeight6,
        6: reMyWeight7,
        7: reMyWeight8,
        8: reMyWeight9,
        9: reMyWeight10,
    }
    plt_num = 0
    for i in range(2):
        for j in range(5):
            im, cn = mne.viz.plot_topomap(weight_dic[plt_num], myinfo, cmap='coolwarm', axes=ax[i, j], show=False)
            plt_num += 1
            # if j == 3:
            #     plt.colorbar(im, ax=ax[i, j+1])
    # 调整行与行之间的间距
    plt.subplots_adjust(hspace=0.1, wspace=0.1)  # 可根据需要调整具体的值
    ax[0, 0].set_title('Subject 4', fontweight='bold', fontsize=16)
    ax[0, 1].set_title('Subject 5', fontweight='bold', fontsize=16)
    ax[0, 2].set_title('Subject 6', fontweight='bold', fontsize=16)
    ax[0, 3].set_title('Subject 8', fontweight='bold', fontsize=16)
    ax[0, 4].set_title('Subject 15', fontweight='bold', fontsize=16)
    ax[0, 0].set_ylabel('Left hand', fontweight='bold', fontsize=16)
    ax[1, 0].set_ylabel('Right hand', fontweight='bold', fontsize=16)
    # 调整行与行之间的间距
    plt.subplots_adjust(hspace=0.01, wspace=0.01)  # 可根据需要调整具体的值
    plt.tight_layout()  # 调整布局
    # 绘制所有图像后，添加色彩指示条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # 调整参数以适应您的布局
    cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label('Colorbar Label', rotation=270, labelpad=15)  # 设置色彩指示条的标签
    plt.savefig('./figs/spatial_region_active.png',bbox_inches='tight', dpi=300)
    # plt.show()

print(ret_dict)
print('finish')
