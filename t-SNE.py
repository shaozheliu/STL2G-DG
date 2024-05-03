import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import manifold, datasets
from sklearn.preprocessing import MinMaxScaler
import random
from sklearn.manifold import TSNE
import os
import sys
from functools import partial
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
import tqdm
import sys, time, copy
from sklearn.metrics import accuracy_score, cohen_kappa_score, precision_score, recall_score, roc_auc_score,classification_report
import torch.utils.data
from experiments.config import config
from sklearn.model_selection import LeaveOneOut

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)
from stl2g.preprocessing.config import CONSTANT
from stl2g.preprocessing.OpenBMI import raw
from stl2g.utils import get_loaders
from stl2g.model.L2GNet import L2GNet
from stl2g.model.EEGNet import EEGNet
from torchvision import datasets
from stl2g.preprocessing.OpenBMI import raw


def plot_embedding_2d(X, y, title=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Define colors for y values (red for class 0, grey for class 1)
    colors = ['red' if label == 0 else 'grey' for label in y]

    # Plot scatter plot with colors based on y
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1], color=colors[i])

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
    plt.show()



def L2GNet_prepare_training(spatial_div_dict, temporal_div_dict ,d_model_dic,  head_dic, d_ff, n_layers, dropout, lr,
                  clf_class, domain_class):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = L2GNet(spatial_div_dict, temporal_div_dict ,d_model_dic,  head_dic, d_ff, n_layers, dropout,
                  clf_class, domain_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain

def EEGNet_prepare_training(org_ch, lr, dropout):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EEGNet(org_ch, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain


# 训练函数，待调参数为神经网络隐藏层的神经元数hiddenLayer
def train_iris(dataSet, model_type):

    # 参数
    spatial_local_dict = CONSTANT[dataSet]['spatial_ch_group']
    temporal_div_dict = CONSTANT[dataSet]['temporal_ch_region']
    d_model_dict = config[model_type][dataSet]['d_model_dict']
    head_dict = config[model_type][dataSet]['head_dict']
    path = CONSTANT[dataSet]['raw_path']
    clf_class = config[model_type][dataSet]['num_class']
    domain_class = CONSTANT[dataSet]['n_subjs']
    d_ff = config[model_type][dataSet]['d_ff']
    n_layers = config[model_type][dataSet]['n_layers']
    batch_size = config[model_type][dataSet]['batch_size']
    lr = config[model_type][dataSet]['lr']
    dropout = config[model_type][dataSet]['dropout']
    num_ch = len(CONSTANT[dataSet]['sel_chs'])
    # 加载数据
    train_subs = [i for i in range(6,7)]
    test_sub = [i for i in range(2,3)]
    sel_chs = CONSTANT[dataSet]['sel_chs']
    id_ch_selected = raw.chanel_selection(sel_chs)
    div_id = raw.channel_division(spatial_local_dict)
    spatial_region_split = raw.region_id_seg(div_id, id_ch_selected)
    train_X, train_y, train_domain_y = raw.load_data_batchs(path, 1, train_subs, clf_class,
                                                            id_ch_selected, 0.1)
    test_X, test_y, test_domain_y = raw.load_data_batchs(path, 1, test_sub, clf_class, id_ch_selected, 0.1)
    # 数据标准化
    X_train_mean = train_X.mean(0)
    X_train_var = np.sqrt(train_X.var(0))
    train_X -= X_train_mean
    train_X /= X_train_var
    test_X -= X_train_mean
    test_X /= X_train_var

    # load model
    if model_type == 'L2GNet':
        model, optimizer, lr_scheduler, criterion, device, criterion_domain = \
            L2GNet_prepare_training(spatial_region_split, temporal_div_dict, d_model_dict, head_dict, d_ff, n_layers,
                                    dropout, lr,
                                    clf_class, domain_class)
        input_data = torch.from_numpy(train_X).to(device, dtype=torch.float32)
        model_path = '/home/alk/L2G-MI/checkpoints/OpenBMI/L2GNet/test/test.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # 提取某一层的输出
        output_L2G = model.L2G(input_data)
        outfeature = output_L2G.detach().cpu().numpy()
    elif model_type == 'EEGNet':
        model, optimizer, lr_scheduler, criterion, device, criterion_domain = EEGNet_prepare_training(num_ch, lr,
                                                                                                      dropout)
        input_data = torch.from_numpy(train_X).to(device, dtype=torch.float32)

    tsne2d = TSNE(n_components=2, init='pca', random_state=0)
    X_tsne_2d = tsne2d.fit_transform(outfeature)
    plot_embedding_2d(X_tsne_2d[:, 0:2], train_y, "t-SNE 2D")
    print("Computing t-SNE embedding")

    # raw data
    # input_X = train_X.reshape(200, -1)




if __name__ == '__main__':
    # sys.path.append(r"\home\alk\L2G-MI\stl2g")
    model_type = 'L2GNet'
    dataSet = 'OpenBMI'
    train_iris(dataSet, model_type)