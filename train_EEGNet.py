import random
import os
import sys
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
from stl2g.model.EEGNet import EEGNet



def EEGNet_prepare_training(org_ch, lr, dropout):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = EEGNet(org_ch, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criterion = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    return model, optimizer, lr_scheduler, criterion, device, criterion_domain


def train_model_with_domain(model, criterion, criterion_domain, optimizer,lr_scheduler, device, dataloaders, n_epoch,
                            args={'dataset_sizes': {'train': 1800, 'val': 200, 'test':304}},
                            ):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in tqdm.tqdm(range(n_epoch)):
        sys.stdout.flush()
        print('Epoch {}/{}'.format(epoch + 1, 300))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            print(phase)
            # Iterate over data.
            # for i, (inputs, labels, domain_labels) in enumerate(dataloaders[phase]):
            for i, (inputs, labels, domain_labels) in enumerate(dataloaders[phase]):
                inputs = inputs.type(torch.cuda.FloatTensor).to(device)
                labels = labels.type(torch.cuda.LongTensor).to(device)
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(torch.softmax(outputs, dim=1), 1)
                    # label predictor loss
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == label_idx)   # 这里源码有问题
                running_corrects += torch.sum(preds.data == labels.data)  # 这里源码有问题

            # if phase == 'train':
            #     scheduler.step(epoch=epoch)

            epoch_loss = running_loss / args['dataset_sizes'][phase]
            epoch_acc = running_corrects / args['dataset_sizes'][phase]

            print('Predictor {} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test_evaluate(model, device, X, y, model_name):
    inputs = torch.from_numpy(X).type(torch.cuda.FloatTensor).to(device)
    labels = torch.from_numpy(y).type(torch.cuda.FloatTensor).to(device)
    if model_name == "Mymodel_2b":
        outputs, domain_outputs = model(inputs, 0.1)
    else:
        outputs = model(inputs)
    # te = torch.softmax(outputs, dim=1)
    proba, preds = torch.max(torch.sigmoid(outputs), 1)
    te = torch.softmax(outputs, 1)[:, 1]
    labels = labels.cpu().numpy()
    preds = preds.cpu().numpy()
    te = te.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    ka = cohen_kappa_score(labels, preds)
    prec = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    roc_auc = roc_auc_score(labels, te)
    return acc, ka, prec, recall, roc_auc

def subject_independent_validation(dataSet, subjects, org_ch, batch_size, epochs, lr, model_name, dropout):
    acc_ls = []
    ka_ls = []
    prec_ls = []
    recall_ls = []
    roc_auc_ls = []
    loo = LeaveOneOut()
    subidx = [i for i in range(1,subjects+1)]
    for i, (train_idx, test_idx) in enumerate(loo.split(subidx)):
        print(f"---------Start Fold {i} process---------")
        train_subs = [subidx[sub_idx] for sub_idx in train_idx]
        test_sub = [subidx[sub_idx] for sub_idx in test_idx]
        print(f'train subjects are: {train_subs}, test subject is: {test_sub}')
        sel_chs = CONSTANT[dataSet]['sel_chs']
        id_ch_selected = raw.chanel_selection(sel_chs)
        # 加载数据
        train_X, train_y, train_domain_y = raw.load_data_batchs(path, session, train_subs, num_class,
                                                                       id_ch_selected, 0.1)
        test_X, test_y, test_domain_y = raw.load_data_batchs(path, session, test_sub, num_class, id_ch_selected, 0.1)
        # 数据标准化
        X_train_mean = train_X.mean(0)
        X_train_var = np.sqrt(train_X.var(0))
        train_X -= X_train_mean
        train_X /= X_train_var
        test_X -= X_train_mean
        test_X /= X_train_var

        # DataLoader
        dataloaders = get_loaders(train_X, train_y, train_domain_y, test_X, test_y, test_domain_y, batch_size = batch_size)
        train_sample = dataloaders['train'].dataset.X.shape[0]
        test_sample = dataloaders['test'].dataset.X.shape[0]
        dataset = {'dataset_sizes': {'train': train_sample, 'test': test_sample}}
        model, optimizer, lr_scheduler, criterion, device, criterion_domain = EEGNet_prepare_training(org_ch, lr, dropout)
        print(summary(model, input_size=[(train_X.shape[1], train_X.shape[2])]))
        best_model = train_model_with_domain(model, criterion, criterion_domain, optimizer, lr_scheduler, device, dataloaders, epochs, dataset)
        acc, ka, prec, recall, roc_auc = test_evaluate(best_model, device, test_X, test_y, model_name)
        acc_ls.append(acc)
        ka_ls.append(ka)
        prec_ls.append(prec)
        recall_ls.append(recall)
        roc_auc_ls.append(roc_auc)

        print(f'The accuracy is: {acc_ls}, cross-subject acc is: {np.mean(acc_ls)} \n')
        print(f'The acohen_kappa_score is: {ka_ls}, cross-subject acohen_kappa_score is: {np.mean(ka_ls)} \n')
        print(f'The precision is: {prec_ls}, cross-subject precision is: {np.mean(prec_ls)} \n')
        print(f'The recall is: {recall_ls}, cross-subject recall is: {np.mean(recall_ls)} \n')
        print(f'The roc_auc is: {roc_auc_ls}, cross-subject roc is: {np.mean(roc_auc_ls)} \n')

if __name__ == '__main__':
    # sys.path.append(r"\home\alk\L2G-MI\stl2g")
    model_type = 'EEGNet'
    dataSet = 'OpenBMI'
    path = CONSTANT[dataSet]['raw_path']
    subject = CONSTANT[dataSet]['n_subjs']
    num_ch = len(CONSTANT[dataSet]['sel_chs'])
    batch_size = config[model_type][dataSet]['batch_size']
    epochs = config[model_type][dataSet]['epochs']
    lr = config[model_type][dataSet]['lr']
    dropout = config[model_type][dataSet]['dropout']
    # subject = 3
    num_class = 2
    session = 1
    log_path =  f'logs/{dataSet}/{model_type}'
    for directory in [log_path]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    outputfile = open(
        f'{log_path}/exp_{model_type}_numsubjects_{subject}_dropout{dropout}_numch{num_ch}.txt', 'w')
    sys.stdout = outputfile
    loo = LeaveOneOut()
    subject_independent_validation(dataSet, subject, num_ch, batch_size, epochs, lr, model_type, dropout)
    outputfile.close()  # 关闭文件

