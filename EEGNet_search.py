import random
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
from stl2g.model.EEGNet import EEGNet

base_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(base_dir)
from stl2g.preprocessing.config import CONSTANT
from stl2g.preprocessing.OpenBMI import raw
from stl2g.utils import get_loaders
from stl2g.model.BackBone import BlackNet
from ray import tune, train
from ray.air.config import RunConfig
from ray import tune
import ray
import matplotlib.pyplot as plt
from ray.tune.schedulers import AsyncHyperBandScheduler


os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def EEGNet_prepare_training(spatial_div_dict, temporal_div_dict, dropout, lr,
                  clf_class):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EEGNet(spatial_div_dict, temporal_div_dict, dropout, clf_class).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, criterion_domain, device



# 一次epoch的训练函数
def train_func(model, optimizer, criterion, criterion_domain, train_loader, device):
    model.train()
    for i, (inputs, labels, domain_labels) in enumerate(train_loader):
        inputs = inputs.type(torch.cuda.FloatTensor).to(device)
        labels = labels.type(torch.cuda.LongTensor).to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(torch.softmax(outputs, dim=1), 1)
            # label predictor loss
            loss = criterion(outputs, labels)
            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

def test_func(model, X, y, device):
    model.eval()
    inputs = torch.from_numpy(X).type(torch.cuda.FloatTensor).to(device)
    labels = torch.from_numpy(y).type(torch.cuda.FloatTensor).to(device)
    outputs = model(inputs)
    proba, preds = torch.max(torch.sigmoid(outputs), 1)
    te = torch.softmax(outputs, 1)[:, 1]
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    acc = accuracy_score(labels, preds)
    return acc


# 训练函数，待调参数为神经网络隐藏层的神经元数hiddenLayer
def train_iris(hyper_parms):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 参数
    spatial_local_dict = CONSTANT[dataSet]['spatial_ch_group']
    temporal_div_dict = CONSTANT[dataSet]['temporal_ch_region']
    path = CONSTANT[dataSet]['raw_path']
    clf_class = config[model_type][dataSet]['num_class']
    d_model_dic = config[model_type][dataSet]['d_model_dict']
    head_dic = config[model_type][dataSet]['head_dict']
    domain_class = 3
    batch_size = config[model_type][dataSet]['batch_size']
    # batch_size = 10
    # 加载数据
    train_subs = [i for i in range(1,10)]
    test_sub = [i for i in range(1,10)]
    sel_chs = CONSTANT[dataSet]['sel_chs']
    id_ch_selected = raw.chanel_selection(sel_chs)
    div_id = raw.channel_division(spatial_local_dict)
    spatial_region_split = raw.region_id_seg(div_id, id_ch_selected)
    model, optimizer, lr_scheduler, criterion, criterion_domain, device = \
        L2G_prepare_training(spatial_region_split, temporal_div_dict, d_model_dic, head_dic, hyper_parms['d_ff'], hyper_parms['n_layers'], hyper_parms['dropout'],
                             hyper_parms['lr'], clf_class, domain_class)
    # Black_prepare_training(spatial_region_split, temporal_div_dict, hyper_parms['dropout'], hyper_parms['lr'], clf_class)
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

    # DataLoader
    dataloaders = get_loaders(train_X, train_y, train_domain_y, test_X, test_y, test_domain_y, batch_size=batch_size)
    train_sample = dataloaders['train'].dataset.X.shape[0]
    test_sample = dataloaders['test'].dataset.X.shape[0]

    while True:
        train_func(model, optimizer, criterion, criterion_domain, dataloaders['train'], device)
        acc = test_func(model, test_X, test_y, device)
        metrics= {'acc':acc}
        ray.train.report(metrics)



if __name__ == '__main__':
    # sys.path.append(r"\home\alk\L2G-MI\stl2g")

    model_type = 'L2GNet'
    dataSet = 'OpenBMI'
    log_path = f'rayResults/{dataSet}/{model_type}'
    for directory in [log_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    outputfile = open(
        f'{log_path}/exp_{model_type}_rayres.txt',
        'w')
    sys.stdout = outputfile

    # 参数搜索空间,在16，32，64中选择hiddenLayer
    hyper_parms = {
            'd_ff': tune.grid_search([1,2,3,4]),
            'n_layers': tune.grid_search([1,2,3,4]),
            # 'lr': tune.choice([0.001, 0.003]),
            # 'dropout' : tune.choice([0.001, 0.003]),
            'dropout' : tune.loguniform(0.001, 0.003),
            'lr': tune.loguniform(1e-4, 1e-1),
            # 'dropout' : tune.choice([0.001, 0.003]),
            # 'lr': tune.choice([0.001, 0.003]),
            # 'lr': tune.loguniform(1e-4, 1e-1),

    }

    sched = AsyncHyperBandScheduler()  # 采用的优化方法
    resources_per_trial = {"gpu": 1}  # 分配调参时的计算资源
    # 创建参数优化器
    tuner = tune.Tuner(
            tune.with_resources(train_iris, resources=resources_per_trial),
            tune_config=tune.TuneConfig(
                metric="acc",
                mode="max",
                scheduler=sched,
            ),
            run_config=RunConfig(
                name="TuneTest",
                local_dir="./rayResults",
                stop={
                    "acc": 0.80,
                    "training_iteration": 150,
                },
            ),
            param_space=hyper_parms,
        )
    # 进行参数优化
    results = tuner.fit()
    bestResult = results.get_best_result(metric="acc", mode="max")
    print(bestResult.config)
    print(bestResult)
    # bestResult.metrics_dataframe.plot("training_iteration", "acc")
    # plt.show()
    outputfile.close()  # 关闭文件
    # storagePath = "./rayResults/TuneTest"
    # tuner = tune.Tuner.restore(path=storagePath, trainable=train_iris)
    # res = tuner.get_results()
    # bestResult = res.get_best_result(metric="acc", mode="max")
    # print(bestResult.config)
    # bestResult.metrics_dataframe.plot("training_iteration", "acc")
    # plt.show()

