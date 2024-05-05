import numpy as np
import scipy.io as sio
from collections import Counter
from stl2g.utils import resampling, butter_bandpass_filter
from stl2g.preprocessing.config import CONSTANT
from sklearn.model_selection import train_test_split


CONSTANT = CONSTANT['BCIIV2A']
n_chs = CONSTANT['n_chs']
n_trials = 2*CONSTANT['n_trials']
window_len = CONSTANT['trial_len']*CONSTANT['orig_smp_freq']
orig_chs = CONSTANT['orig_chs']
trial_len = CONSTANT['trial_len']
orig_smp_freq = CONSTANT['orig_smp_freq']
path = CONSTANT['raw_path']
start = CONSTANT['MI']['start']
stop = CONSTANT['MI']['stop']

def read_raw(PATH, subject, training, num_class, id_chosen_chs):
    data = np.zeros((n_trials, n_chs, window_len))
    label = np.zeros(n_trials)

    NO_valid_trial = 0
    if training:
        mat = sio.loadmat(PATH + '/A0' + str(subject) + 'T.mat')['data']
    else:
        mat = sio.loadmat(PATH + '/A0' + str(subject) + 'E.mat')['data']
    for ii in range(0, mat.size):
        mat_1 = mat[0, ii]
        mat_2 = [mat_1[0, 0]]
        mat_info = mat_2[0]
        _X = mat_info[0]
        _trial = mat_info[1]
        _y = mat_info[2]
        _fs = mat_info[3]
        _classes = mat_info[4]
        _artifacts = mat_info[5]
        _gender = mat_info[6]
        _age = mat_info[7]
        for trial in range(0, _trial.size):
            # num_class = 2: picked only class 1 (left hand) and class 2 (right hand) for our propose
            if (_y[trial][0] <= num_class):
                data[NO_valid_trial, :, :] = np.transpose(_X[int(_trial[trial]):(int(_trial[trial]) + window_len),
                                                          1:21])  # selected merely motor cortices region
                label[NO_valid_trial] = int(_y[trial])
                NO_valid_trial += 1
    data = data[0:NO_valid_trial, id_chosen_chs, :]
    label = label[0:NO_valid_trial] - 1  # -1 to adjust the values of class to be in range 0 and 1
    return data, label


def load_crop_data(PATH, subject, start, stop, new_smp_freq, num_class, id_chosen_chs):
    start_time = int(start*new_smp_freq) # 2*
    stop_time = int(stop*new_smp_freq) # 6*
    X_train, y_tr = read_raw(PATH=PATH, subject=subject,
                             training=True, num_class=num_class, id_chosen_chs=id_chosen_chs)
    X_test, y_te = read_raw(PATH=PATH, subject=subject,
                            training=False, num_class=num_class, id_chosen_chs=id_chosen_chs)
    if new_smp_freq < orig_smp_freq:
        X_train = resampling(X_train, new_smp_freq, trial_len)
        X_test = resampling(X_test, new_smp_freq, trial_len)
    X_train = X_train[:,:,start_time:stop_time]
    X_test = X_test[:,:,start_time:stop_time]
    print("Verify dimension training {} and testing {}".format(X_train.shape, X_test.shape))
    return X_train, y_tr, X_test, y_te


def chanel_selection(sel_chs):
    chs_id = []
    for name_ch in sel_chs:
        ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
        chs_id.append(ch_id)
        print('chosen_channel:', name_ch, '---', 'Index_is:', ch_id)
    return chs_id

def channel_division(chs_group):
    orig_chs = CONSTANT['orig_chs']
    div_id = {}
    for region in chs_group:
        chs_list = []
        for name_ch in chs_group[region]:
            ch_id = np.where(np.array(orig_chs) == name_ch)[0][0]
            chs_list.append(ch_id)
        div_id[region] = chs_list
        print(f'region {region} division ---, Index is {chs_list}')
    return div_id

def region_id_seg(div_id, id_ch_selected):
    region_split = {}
    for region in div_id:
        chs_list = []
        for id_ch in div_id[region]:
            ch_id = np.where(np.array(id_ch_selected) == id_ch)[0][0]
            chs_list.append(ch_id)
        region_split[region] = chs_list
    return region_split




def load_data_batchs(path, session, subject_list, num_class, id_ch_selected, new_smp_freq):
    train_subjects_data = []
    test_subjects_data = []
    train_subjects_label = []
    test_subjects_label = []
    domain_label_train = []
    domain_label_test = []
    for subject in subject_list:
        X_train, y_tr, X_test, y_te = load_crop_data(
            PATH=path, subject=subject, start=start, stop=stop, new_smp_freq=new_smp_freq, num_class=num_class,
            id_chosen_chs=id_ch_selected)
        domain_label_tr = np.ones(y_tr.shape[0], dtype=int) * (subject-min(subject_list))
        domain_label_te = np.ones(y_te.shape[0], dtype=int) * (subject-min(subject_list))
        # 逐个将subject数据写入最终输出的list中
        if len(train_subjects_data) == 0:
            train_subjects_data = X_train
            test_subjects_data = X_test
            train_subjects_label = y_tr
            test_subjects_label = y_te
            domain_label_train = domain_label_tr
            domain_label_test = domain_label_te
        else:
            train_subjects_data = np.concatenate((test_subjects_data, X_train), axis=0)
            test_subjects_data = np.concatenate((test_subjects_data, X_test), axis=0)
            train_subjects_label = np.concatenate((train_subjects_label, y_tr), axis=0)
            test_subjects_label = np.concatenate((test_subjects_label, y_te), axis=0)
            domain_label_train = np.concatenate((domain_label_train, domain_label_tr), axis=0)
            domain_label_test = np.concatenate((domain_label_test, domain_label_te), axis=0)
    # 注意：实验设置非跨被试情况下，可以注释掉不用合并
    subject_data = np.concatenate((train_subjects_data, test_subjects_data), axis=0)
    subject_label = np.concatenate((train_subjects_label, test_subjects_label), axis=0).astype(int)
    subject_domain_label = np.concatenate((domain_label_train, domain_label_test), axis=0)
    print('total_data_shape:', subject_data.shape)
    return subject_data, subject_label, subject_domain_label


if __name__ == '__main__':
    subject = 1
    start = CONSTANT['MI']['start'] # 2
    stop = CONSTANT['MI']['stop']  # 6
    sel_chs = CONSTANT['sel_chs']
    chs_group = CONSTANT['spatial_ch_group']
    new_smp_freq = 100
    num_class = 4
    id_ch_selected = chanel_selection(sel_chs)
    div_id = channel_division(chs_group)
    region_split = region_id_seg(div_id, id_ch_selected)
    # X_train, y_tr, X_test, y_te = load_crop_data(
    #     PATH=path, subject=subject, start=start, stop=stop, new_smp_freq=new_smp_freq, num_class=num_class,
    #     id_chosen_chs=id_ch_selected)
    subject_data, subject_label, subject_domain_label = load_data_batchs(path, 1, [1], num_class, id_ch_selected, new_smp_freq)
