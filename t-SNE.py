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
from stl2g.model.DeepConvNet import DeepConvNettest
from stl2g.model.ShallowNet import Shallow_Net
from torchvision import datasets
from stl2g.preprocessing.OpenBMI import raw
from stl2g.preprocessing.BCIIV2A import raw as raw_bci2a

def plot_embedding_2d(X, y, ax=None):
    """Plot an embedding X with the class label y colored by the domain d."""
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    # Define colors for y values (red for class 0, grey for class 1, green for class 2, blue for class 3)
    colors = ['red' if label == 0 else 'grey' if label == 1 else 'green' if label == 2 else 'blue' for label in y]

    # Plot scatter plot with colors based on y
    for i in range(X.shape[0]):
        ax.scatter(X[i, 0], X[i, 1], color=colors[i])

    # Remove x-axis and y-axis ticks
    ax.set_xticks([])
    ax.set_yticks([])



def L2GNet_prepare_training(spatial_div_dict, temporal_div_dict ,d_model_dic,  head_dic, d_ff, n_layers, dropout, lr,
                  clf_class, domain_class, nb_class):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = L2GNet(spatial_div_dict, temporal_div_dict ,d_model_dic,  head_dic, d_ff, n_layers, dropout,
                  clf_class, domain_class, nb_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain

def EEGNet_prepare_training(org_ch, lr, dropout, nb_class):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EEGNet(org_ch, dropout, nb_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain

def DeepConvNet_prepare_training(org_ch, lr, dropout, nb_class):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DeepConvNettest(org_ch, dropout, nb_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain


def ShallowNet_prepare_training(org_ch, lr, dropout, nb_class):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Shallow_Net(org_ch, dropout, nb_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain


# 训练函数，待调参数为神经网络隐藏层的神经元数hiddenLayer
def train_iris(dataSetlist, model_type):
    data_dic = {}
    y_dic = {}
    for dataSet in dataSetlist:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # 参数
        spatial_local_dict = CONSTANT[dataSet]['spatial_ch_group']
        temporal_div_dict = CONSTANT[dataSet]['temporal_ch_region']
        try:
            d_model_dict = config[model_type][dataSet]['d_model_dict']
        except KeyError:
            d_model_dict = None
        try:
            head_dict = config[model_type][dataSet]['head_dict']
        except KeyError:
            head_dict = None
        path = CONSTANT[dataSet]['raw_path']
        clf_class = config[model_type][dataSet]['num_class']
        domain_class = CONSTANT[dataSet]['n_subjs']
        try:
            d_ff = config[model_type][dataSet]['d_ff']
        except KeyError:
            d_ff = None
        try:
            n_layers = config[model_type][dataSet]['n_layers']
        except KeyError:
            n_layers = None
        batch_size = config[model_type][dataSet]['batch_size']
        lr = config[model_type][dataSet]['lr']
        dropout = config[model_type][dataSet]['dropout']
        num_ch = len(CONSTANT[dataSet]['sel_chs'])
        # 加载数据
        sel_chs = CONSTANT[dataSet]['sel_chs']
        if dataSet == 'OpenBMI':
            train_subs = [i for i in range(6, 7)]
            train_subs_conv = [i for i in range(5, 6)]
            id_ch_selected = raw.chanel_selection(sel_chs)
            div_id = raw.channel_division(spatial_local_dict)
            spatial_region_split = raw.region_id_seg(div_id, id_ch_selected)
            train_X, train_y, train_domain_y = raw.load_data_batchs(path, 1, train_subs, clf_class,
                                                                    id_ch_selected, 0.1)
            train_X_conv, train_y, train_domain_y = raw.load_data_batchs(path, 1, train_subs_conv, clf_class,
                                                                    id_ch_selected, 0.1)
            model_path1 = '/home/alk/L2G-MI/checkpoints/OpenBMI/L2GNet/test/test.pth'
            model_path2 = '/home/alk/L2G-MI/checkpoints/OpenBMI/EEGNet/val/EEGNet.pth'
            # model_path3 = '/home/alk/L2G-MI/checkpoints/OpenBMI/DeepConvNet/val/convNet.pth'
            model_path3 = '/home/alk/L2G-MI/checkpoints/OpenBMI/EEGNet/val/EEGNet.pth'
            model_path4 = '/home/alk/L2G-MI/checkpoints/OpenBMI/ShallowNet/val/ShallowNet.pth'
        elif dataSet == 'BCIIV2A':
            train_subs = [i for i in range(9, 10)]
            id_ch_selected = raw_bci2a.chanel_selection(sel_chs)
            div_id = raw_bci2a.channel_division(spatial_local_dict)
            spatial_region_split = raw_bci2a.region_id_seg(div_id, id_ch_selected)
            train_X, train_y, train_domain_y = raw_bci2a.load_data_batchs(path, 1, train_subs, clf_class,
                                                                id_ch_selected, 100)
            model_path1 = '/home/alk/L2G-MI/checkpoints/BCIIV2A/L2GNet/test/save.pth'
            model_path2 = '/home/alk/L2G-MI/checkpoints/BCIIV2A/EEGNet/test/save.pth'
            model_path3 = '/home/alk/L2G-MI/checkpoints/BCIIV2A/DeepConvNet/save/save.pth'
            model_path4 = '/home/alk/L2G-MI/checkpoints/BCIIV2A/ShallowNet/save/save.pth'
        # 数据标准化
        X_train_mean = train_X.mean(0)
        X_train_var = np.sqrt(train_X.var(0))
        train_X -= X_train_mean
        train_X /= X_train_var
        X_train_conv_mean = train_X_conv.mean(0)
        X_train_conv_var = np.sqrt(train_X_conv.var(0))
        train_X_conv -= X_train_conv_mean
        train_X_conv /= X_train_conv_var

        input_data = torch.from_numpy(train_X).to(device, dtype=torch.float32)
        input_data_conv =  torch.from_numpy(train_X_conv).to(device, dtype=torch.float32)
        # load model
        if dataSet == 'OpenBMI':
            M_l2g, optimizer, lr_scheduler, criterion, device, criterion_domain = \
                    L2GNet_prepare_training(spatial_region_split, temporal_div_dict, d_model_dict, head_dict, d_ff, n_layers,
                                            dropout, lr,
                                            clf_class, domain_class, num_ch)
            M_l2g.load_state_dict(torch.load(model_path1))
            M_l2g.eval()
            # L2G
            output_L2G = M_l2g.L2G(input_data)
            output_L2G = output_L2G.detach().cpu().numpy()

            # EEGNet
            M_EEG, optimizer, lr_scheduler, criterion, device, criterion_domain = EEGNet_prepare_training(num_ch, lr, dropout, clf_class)
            M_EEG.load_state_dict(torch.load(model_path2))
            M_EEG.eval()
            eeg_1 = input_data.unsqueeze(1).permute(0,1,3,2)
            eeg_2 = M_EEG.conv1(eeg_1)
            eeg_3 =M_EEG.batchnorm1(eeg_2)
            eeg_4 = eeg_3.permute(0,3,1,2)
            eeg_5 = M_EEG.padding1(eeg_4)
            eeg_6 = M_EEG.conv2(eeg_5)
            eeg_7 = M_EEG.pooling2(eeg_6)
            eeg_8 = M_EEG.padding2(eeg_7)
            eeg_9 = M_EEG.conv3(eeg_8)
            eeg_10 = M_EEG.batchnorm3(eeg_9)
            output_EEG = M_EEG.pooling3(eeg_10)
            output_EEG = output_EEG.reshape(200,-1)
            output_EEG = output_EEG.detach().cpu().numpy()

            # ConvNet
            # M_conv, optimizer, lr_scheduler, criterion, device, criterion_domain = DeepConvNet_prepare_training(num_ch, lr, dropout, clf_class)
            # M_conv.load_state_dict(torch.load(model_path3))
            # M_conv.eval()
            # input_data_re = input_data.unsqueeze(1)
            # layer_index = 5  # 选择要提取特征图的层的索引，这里假设是第二层（索引从0开始）
            # intermediate_model = nn.Sequential(*list(M_conv.children())[:layer_index + 1])  # 选择要提取的层及之前的层
            # output_Conv = intermediate_model(input_data_re)
            # output_Conv = torch.mean(output_Conv, dim=1)
            # output_Conv = output_Conv.reshape(output_Conv.shape[0],-1)
            # output_Conv = output_Conv.detach().cpu().numpy()
            M_conv, optimizer, lr_scheduler, criterion, device, criterion_domain = EEGNet_prepare_training(num_ch, lr,
                                                                                                          dropout,
                                                                                                          clf_class)
            M_conv.load_state_dict(torch.load(model_path2))
            M_conv.eval()
            eeg_1 = input_data_conv.unsqueeze(1).permute(0, 1, 3, 2)
            eeg_2 = M_conv.conv1(eeg_1)
            eeg_3 = M_conv.batchnorm1(eeg_2)
            eeg_4 = eeg_3.permute(0, 3, 1, 2)
            eeg_5 = M_conv.padding1(eeg_4)
            eeg_6 = M_conv.conv2(eeg_5)
            eeg_7 = M_conv.pooling2(eeg_6)
            eeg_8 = M_conv.padding2(eeg_7)
            eeg_9 = M_conv.conv3(eeg_8)
            eeg_10 = M_conv.batchnorm3(eeg_9)
            output_Conv = M_conv.pooling3(eeg_10)
            output_Conv = output_Conv.reshape(200, -1)
            output_Conv = output_Conv.detach().cpu().numpy()


            # ShallowNet
            M_shallow, optimizer, lr_scheduler, criterion, device, criterion_domain = ShallowNet_prepare_training(num_ch, lr, dropout, clf_class)
            M_shallow.load_state_dict(torch.load(model_path4))
            M_shallow.eval()
            input_data_re = input_data.unsqueeze(1)
            layer_index = 4  # 选择要提取特征图的层的索引，这里假设是第二层（索引从0开始）
            intermediate_model = nn.Sequential(*list(M_shallow.children())[:layer_index + 1])  # 选择要提取的层及之前的层
            output_shallow = intermediate_model(input_data_re)

            output_shallow = output_shallow.detach().cpu().numpy()
            output_shallow = output_shallow.reshape(output_shallow.shape[0],-1)

            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            X_raw = tsne2d.fit_transform(train_X.reshape(train_X.shape[0], -1))
            X_l2g = tsne2d.fit_transform(output_L2G)
            X_eeg = tsne2d.fit_transform(output_EEG)
            X_conv = tsne2d.fit_transform(output_Conv)
            X_shallow = tsne2d.fit_transform(output_shallow)

        elif dataSet == 'BCIIV2A':
            M_l2g, optimizer, lr_scheduler, criterion, device, criterion_domain = \
                L2GNet_prepare_training(spatial_region_split, temporal_div_dict, d_model_dict, head_dict, d_ff,
                                        n_layers,
                                        dropout, lr,
                                        clf_class, domain_class, num_ch)
            M_l2g.load_state_dict(torch.load(model_path1))
            M_l2g.eval()
            # L2G
            output_L2G = M_l2g.L2G(input_data)
            output_L2G = output_L2G.detach().cpu().numpy()

            # EEGNet
            M_EEG, optimizer, lr_scheduler, criterion, device, criterion_domain = EEGNet_prepare_training(num_ch, lr,
                                                                                                          dropout,
                                                                                                          clf_class)
            M_EEG.load_state_dict(torch.load(model_path2))
            M_EEG.eval()
            eeg_1 = input_data.unsqueeze(1).permute(0, 1, 3, 2)
            eeg_2 = M_EEG.conv1(eeg_1)
            eeg_3 = M_EEG.batchnorm1(eeg_2)
            eeg_4 = eeg_3.permute(0, 3, 1, 2)
            eeg_5 = M_EEG.padding1(eeg_4)
            eeg_6 = M_EEG.conv2(eeg_5)
            eeg_7 = M_EEG.pooling2(eeg_6)
            eeg_8 = M_EEG.padding2(eeg_7)
            eeg_9 = M_EEG.conv3(eeg_8)
            eeg_10 = M_EEG.batchnorm3(eeg_9)
            output_EEG = M_EEG.pooling3(eeg_10)
            output_EEG = output_EEG.reshape(output_EEG.shape[0], -1)
            output_EEG = output_EEG.detach().cpu().numpy()

            # ConvNet
            M_conv, optimizer, lr_scheduler, criterion, device, criterion_domain = DeepConvNet_prepare_training(num_ch,
                                                                                                                lr,
                                                                                                                dropout,
                                                                                                                clf_class)
            M_conv.load_state_dict(torch.load(model_path3))
            M_conv.eval()
            input_data_re = input_data.unsqueeze(1)
            layer_index = 5  # 选择要提取特征图的层的索引，这里假设是第二层（索引从0开始）
            intermediate_model = nn.Sequential(*list(M_conv.children())[:layer_index + 1])  # 选择要提取的层及之前的层
            output_Conv = intermediate_model(input_data_re)
            output_Conv = torch.mean(output_Conv, dim=1)
            output_Conv = output_Conv.reshape(output_Conv.shape[0], -1)
            output_Conv = output_Conv.detach().cpu().numpy()

            # ShallowNet
            M_shallow, optimizer, lr_scheduler, criterion, device, criterion_domain = ShallowNet_prepare_training(
                num_ch, lr, dropout, clf_class)
            M_shallow.load_state_dict(torch.load(model_path4))
            M_shallow.eval()
            input_data_re = input_data.unsqueeze(1)
            layer_index = 4  # 选择要提取特征图的层的索引，这里假设是第二层（索引从0开始）
            intermediate_model = nn.Sequential(*list(M_shallow.children())[:layer_index + 1])  # 选择要提取的层及之前的层
            output_shallow = intermediate_model(input_data_re)

            output_shallow = output_shallow.detach().cpu().numpy()
            output_shallow = output_shallow.reshape(output_shallow.shape[0], -1)

            tsne2d = TSNE(n_components=2, init='pca', random_state=0)
            X_raw = tsne2d.fit_transform(train_X.reshape(train_X.shape[0],-1))
            X_l2g = tsne2d.fit_transform(output_L2G)
            X_eeg = tsne2d.fit_transform(output_EEG)
            X_conv = tsne2d.fit_transform(output_Conv)
            X_shallow = tsne2d.fit_transform(output_shallow)
        data_dic[dataSet] = [X_raw, X_eeg, X_conv, X_shallow, X_l2g]
        y_dic[dataSet] = train_y

    # 创建画布和子图
    fig, axs = plt.subplots(5, 2, figsize=(8, 15))
    # 在每个子图中绘制不同的数据
    for i in range(5):
        for j in range(2):
            ax = axs[i, j]
            if i == 0:
                if j == 0:
                    ax.set_title('BCIIV 2A', fontweight='bold')
                    ax.set_ylabel('Raw Data', fontweight='bold')
                    plot_embedding_2d(data_dic[dataSetlist[0]][0][:, 0:2], y_dic[dataSetlist[0]], ax)
                else:
                    ax.set_title('OpenBMI', fontweight='bold')
                    plot_embedding_2d(data_dic[dataSetlist[1]][0][:, 0:2], y_dic[dataSetlist[1]], ax)

                # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
            elif i == 1:
                # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
                if j == 0:
                    plot_embedding_2d(data_dic[dataSetlist[0]][1][:, 0:2], y_dic[dataSetlist[0]], ax)
                    ax.set_ylabel('EEGNet', fontweight='bold')
                else:
                    plot_embedding_2d(data_dic[dataSetlist[1]][1][:, 0:2],  y_dic[dataSetlist[1]], ax)
            elif i == 2:
                # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
                if j == 0:
                    ax.set_ylabel('DeepConvNet', fontweight='bold')
                    plot_embedding_2d(data_dic[dataSetlist[0]][2][:, 0:2], y_dic[dataSetlist[0]], ax)
                else:
                    plot_embedding_2d(data_dic[dataSetlist[1]][2][:, 0:2],  y_dic[dataSetlist[1]], ax)
            elif i == 3:
                # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
                if j == 0:
                    plot_embedding_2d(data_dic[dataSetlist[0]][3][:, 0:2], y_dic[dataSetlist[0]], ax)
                    ax.set_ylabel('MMCNN', fontweight='bold')
                else:
                    plot_embedding_2d(data_dic[dataSetlist[1]][3][:, 0:2],  y_dic[dataSetlist[1]], ax)
            elif i == 4:
                # ax.text(0.05, 0.95, 'acc: 0.85\nauc: 0.92', transform=ax.transAxes, fontsize=15, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, pad=3))
                if j == 0:
                    ax.set_ylabel('STL2G-DG', fontweight='bold')
                    plot_embedding_2d(data_dic[dataSetlist[0]][4][:, 0:2], y_dic[dataSetlist[0]], ax)
                else:
                    plot_embedding_2d(data_dic[dataSetlist[1]][4][:, 0:2],  y_dic[dataSetlist[1]], ax)
    # 创建共用的图例
    # handles, labels = axs[0,0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2)
    print("Computing t-SNE embedding")
    # 自动调整子图布局
    plt.tight_layout()
    # 保存每个子图为单独的图像文件
    plt.savefig(f'./experiments/tSNE.png', dpi=300)
    plt.show()





if __name__ == '__main__':
    # sys.path.append(r"\home\alk\L2G-MI\stl2g")
    model_type = 'L2GNet'
    datalist = ['OpenBMI', 'OpenBMI']
    train_iris(datalist, model_type)
