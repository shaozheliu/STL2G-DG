import numpy as np
import matplotlib.pyplot as plt

import os
from PIL import Image
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
from experiments.make_montage import plot_montage
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())


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
    # 取subject 数据
    subjects = [[2]]
    figid = 0
    for subject in subjects:
        train_X, train_y, train_domain_y = raw.load_data_batchs(path, 1, subject, clf_class,
                                                                id_ch_selected, 0.1)
        # 找到y中不同类别的索引
        sub_lab_dic = {}
        sub_lab_dic_t = {}
        for label in [0]:
            class_indices = np.argwhere(train_y == label)
            # 选择每个类别的第一个索引
            sample_indices = class_indices[:, 0][0]
            # 在X中取出对应的样本
            selected_X = train_X[sample_indices]
            selected_y = train_y[sample_indices]
            inp = torch.FloatTensor(np.expand_dims(selected_X, axis=0))
            inp = inp.unsqueeze(1)
            model, optimizer, lr_scheduler, criterion, device, criterion_domain = \
                L2GNet_prepare_training(spatial_region_split, temporal_local_dict, d_model_dict, head_dict, d_ff,
                                        n_layers, dropout, lr,
                                        clf_class, domain_class, num_ch)
            model.load_state_dict(torch.load(model_path))
            # 获取整个网络的权重
            params = model.state_dict()
            # print(params)
            # 定义反卷积层
            deconv = nn.ConvTranspose2d(in_channels=20, out_channels=1, kernel_size=(1, 40), stride=(1, 40), padding=0)
            # a = params['L2G.L2G.Local_spatial_conved.S_region_convFrontal Central.conv1.weight']
            deconv.weight.data = params['L2G.L2G.Local_spatial_conved.S_region_convFrontal Central.conv1.weight']
            # 生成heatmap
            output = deconv(inp)  # 反卷积操作
            print(output.shape)
            # finallayer_name = 'L2G'
            # model.eval()  # 使用eval()属性
            # features_blobs = []  # 后面用于存放特征图
            # # # 获取 features 模块的输出
            # model._modules.get(finallayer_name).register_forward_hook(hook_feature)
            # # # 获取权重
            # net_name = []
            # params = []
            # for name, param in model.named_parameters():
            #     net_name.append(name)
            #     params.append(param)
            # print(net_name[-9], net_name[-10])  # classifier.1.bias classifier.1.weight
            # print(len(params))  #
            # weight_softmax = np.squeeze(params[-12].data.numpy())  # shape:(8, 528)
            # weight_layer2 = np.squeeze(params[-8].data.numpy())  # shape:(4, 8)
            # logit_class, logit_domain = net(inp, alpha_scala)  # 计算输入图片通过网络后的输出值
            # print(logit_class.shape)
            # print(np.squeeze(params[-12].data.numpy()).shape)
            # print(features_blobs[0].shape)  # 特征图大小为　(1, 22, 24)  这个就是我们最后一层feature map的维度，(batch，通道数，filter_w, filter_h)
            #
            # np.save(f'./{subject[0]}_{label}_data.npy', features_blobs[0])
            # # plot_montage(chan_names_dict, figid)
            # figid += 1



print('finish')
