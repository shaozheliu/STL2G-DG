import numpy as np
import scipy.io as sio
from collections import Counter
from stl2g.utils import resampling, butter_bandpass_filter
from stl2g.preprocessing.config import CONSTANT
from sklearn.model_selection import train_test_split

CONSTANT = CONSTANT['OpenBMI']


def read_raw(PATH, session, subject, num_class, id_ch_selected):
    mat_file_name = PATH + '/sess' + str(session).zfill(2) + '_subj' + str(subject).zfill(2) + '_EEG_MI.mat'
    mat = sio.loadmat(mat_file_name)
    print('This is data from: ', mat_file_name)
    if num_class == 2:
        raw_train_data = mat['EEG_MI_train'][0]['smt'][0]
        raw_train_data = (np.swapaxes(raw_train_data, 0, 2))[id_ch_selected]
        raw_train_data = np.swapaxes(raw_train_data, 0, 1)
        print('raw_train_data_shape:', raw_train_data.shape)
        raw_test_data = mat['EEG_MI_test'][0]['smt'][0]
        raw_test_data = np.swapaxes(raw_test_data, 0, 2)[id_ch_selected]
        raw_test_data = np.swapaxes(raw_test_data, 0, 1)
        print('raw_test_data_shape:', raw_test_data.shape)
        label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0] - 1
        label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0] - 1
        return raw_train_data, label_train_data, raw_test_data, label_test_data

    elif num_class == 3:
        raw_train_data = __segment_data(mat, type_data='train')
        raw_train_data = np.take(raw_train_data, id_ch_selected, axis=2)
        raw_train_data = np.swapaxes(raw_train_data, 1, 2)
        print('raw_train_data_shape:', raw_train_data.shape)
        raw_test_data = __segment_data(mat, type_data='test')
        raw_test_data = np.take(raw_test_data, id_ch_selected, axis=2)
        raw_test_data = np.swapaxes(raw_test_data, 1, 2)
        print('raw_test_data_shape:', raw_test_data.shape)
        label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0] - 1
        label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0] - 1
        return raw_train_data, label_train_data, raw_test_data, label_test_data

    elif num_class == "transitory_mi":
        raw_train_data = __segment_data_whole_period(mat, type_data='train')
        raw_train_data = np.take(raw_train_data, id_ch_selected, axis=2)
        raw_train_data = np.swapaxes(raw_train_data, 1, 2)
        print('raw_train_data_shape:', raw_train_data.shape)
        raw_test_data = __segment_data_whole_period(mat, type_data='test')
        raw_test_data = np.take(raw_test_data, id_ch_selected, axis=2)
        raw_test_data = np.swapaxes(raw_test_data, 1, 2)
        print('raw_test_data_shape:', raw_test_data.shape)
        label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0] - 1
        label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0] - 1
        return raw_train_data, label_train_data, raw_test_data, label_test_data


def load_crop_data(PATH, n_subjs, new_smp_freq, num_class, MI_len, id_chosen_chs, start_mi=None, stop_mi=None):
    if num_class == 2:
        print("Two-class MI data is downloading")
        orig_smp_freq = CONSTANT['orig_smp_freq']  # 1000
        n_trials = CONSTANT['n_trials_2_class']  # 100
        sessions = [1, 2]
        n_chs = len(id_chosen_chs)
        X_train, y_train = np.zeros((n_subjs, len(sessions), n_trials, n_chs, int(new_smp_freq * MI_len))), np.zeros(
            (n_subjs, len(sessions), n_trials))
        X_test, y_test = np.zeros((n_subjs, len(sessions), n_trials, n_chs, int(new_smp_freq * MI_len))), np.zeros(
            (n_subjs, len(sessions), n_trials))
        for id_sub, subject in enumerate(range(1, n_subjs + 1)):
            for id_se, sess in enumerate(sessions):
                X_tr, y_tr, X_te, y_te = read_raw(PATH, sess, subject, num_class, id_chosen_chs)
                X_tr_resam = resampling(X_tr, new_smp_freq, MI_len)
                X_te_resam = resampling(X_te, new_smp_freq, MI_len)
                X_train[id_sub, id_se, :, :, :] = X_tr_resam
                X_test[id_sub, id_se, :, :, :] = X_te_resam
                y_train[id_sub, id_se, :] = y_tr
                y_test[id_sub, id_se, :] = y_te
        return X_train.reshape(n_subjs, -1, n_chs, int(new_smp_freq * MI_len)), y_train.reshape(n_subjs,
                                                                                                -1), X_test.reshape(
            n_subjs, -1, n_chs, int(new_smp_freq * MI_len)), y_test.reshape(n_subjs, -1)
    elif num_class == 3:
        print("Three-class MI data is downloading")
        orig_smp_freq = CONSTANT['orig_smp_freq']  # 1000
        n_trials = CONSTANT['n_trials_3_class']  # 150
        sessions = [1, 2]
        n_chs = len(id_chosen_chs)
        MI_len = 4
        X_train, y_train = np.zeros((n_subjs, len(sessions), n_trials, n_chs, int(new_smp_freq * MI_len))), np.zeros(
            (n_subjs, len(sessions), n_trials))
        X_test, y_test = np.zeros((n_subjs, len(sessions), n_trials, n_chs, int(new_smp_freq * MI_len))), np.zeros(
            (n_subjs, len(sessions), n_trials))
        for id_sub, subject in enumerate(range(1, n_subjs + 1)):
            for id_se, sess in enumerate(sessions):
                X_tr, y_tr, X_te, y_te = read_raw(PATH, sess, subject, num_class, id_chosen_chs)
                X_tr_addon, y_tr_addon = __add_on_resting(X_tr, y_tr, orig_smp_freq)
                X_te_addon, y_te_addon = __add_on_resting(X_te, y_te, orig_smp_freq)
                X_tr_resam = resampling(X_tr_addon, new_smp_freq, MI_len)
                X_te_resam = resampling(X_te_addon, new_smp_freq, MI_len)
                X_train[id_sub, id_se, :, :, :] = X_tr_resam
                X_test[id_sub, id_se, :, :, :] = X_te_resam
                y_train[id_sub, id_se, :] = y_tr_addon
                y_test[id_sub, id_se, :] = y_te_addon
        return X_train.reshape(n_subjs, -1, n_chs, int(new_smp_freq * MI_len)), y_train.reshape(n_subjs,
                                                                                                -1), X_test.reshape(
            n_subjs, -1, n_chs, int(new_smp_freq * MI_len)), y_test.reshape(n_subjs, -1)

    elif num_class == "transitory_mi" and start_mi != None and stop_mi != None:
        print("Two-class transitory MI data is downloading with the time interval of {} s and {} s".format(start_mi,
                                                                                                           stop_mi))
        orig_smp_freq = CONSTANT['orig_smp_freq']  # 1000
        n_trials = CONSTANT['n_trials_2_class']  # 100
        sessions = [1, 2]
        n_chs = len(id_chosen_chs)
        MI_len = 4
        X_test, y_test = np.zeros((n_subjs, len(sessions), n_trials, n_chs, int(new_smp_freq * MI_len))), np.zeros(
            (n_subjs, len(sessions), n_trials))
        for id_sub, subject in enumerate(range(1, n_subjs + 1)):
            for id_se, sess in enumerate(sessions):
                X_tr, y_tr, X_te, y_te = read_raw(PATH, sess, subject, num_class, id_chosen_chs)
                X_te_transient, y_te_transient = __transitory_mi(X_te, y_te, orig_smp_freq, start_mi, stop_mi)
                X_te_resam = resampling(X_te_transient, new_smp_freq, MI_len)
                X_test[id_sub, id_se, :, :, :] = X_te_resam
                y_test[id_sub, id_se, :] = y_te_transient
        return X_test.reshape(n_subjs, -1, n_chs, int(new_smp_freq * MI_len)), y_test.reshape(n_subjs, -1)


def chanel_selection(sel_chs):
    orig_chs = CONSTANT['orig_chs']
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


def __segment_data(mat_arr, type_data):
    data = mat_arr['EEG_MI_' + type_data][0]['x'][0]
    t = mat_arr['EEG_MI_' + type_data][0]['t'][0][0]
    low_cut = 0
    high_cut = CONSTANT['trial_len']  # 8s
    orig_smp_freq = CONSTANT['orig_smp_freq']  # 1000
    orig_n_chs = CONSTANT['n_chs']  # 62
    n_trials = CONSTANT['n_trials_2_class']  # 100
    data_seg = np.zeros((n_trials, high_cut * orig_smp_freq, orig_n_chs))
    # print('This pre-processing is for task {} low cut {} high cut {}'.format(task,low_cut,high_cut))
    for i in range(n_trials):
        start_pos = t[i] + (low_cut * orig_smp_freq)
        stop_pos = t[i] + (high_cut * orig_smp_freq)
        data_seg[i, :, :] = data[start_pos:stop_pos, :]
    return data_seg


def __segment_data_whole_period(mat_arr, type_data):
    data = mat_arr['EEG_MI_' + type_data][0]['x'][0]
    t = mat_arr['EEG_MI_' + type_data][0]['t'][0][0]
    low_cut = -3
    high_cut = CONSTANT['trial_len']  # 8s
    data_len = 11
    orig_smp_freq = CONSTANT['orig_smp_freq']  # 1000
    orig_n_chs = CONSTANT['n_chs']  # 62
    n_trials = CONSTANT['n_trials_2_class']  # 100
    data_seg = np.zeros((n_trials, data_len * orig_smp_freq, orig_n_chs))
    # print('This pre-processing is for task {} low cut {} high cut {}'.format(task,low_cut,high_cut))
    for i in range(n_trials):
        start_pos = t[i] + (low_cut * orig_smp_freq)
        stop_pos = t[i] + (high_cut * orig_smp_freq)
        # print("Debugg the selected period is:", (stop_pos-start_pos)/orig_smp_freq)
        data_seg[i, :, :] = data[start_pos:stop_pos, :]
    return data_seg


def __add_on_resting(X, y, smp_freq):
    print("MI Right, MI Left and Resting EEG Segmentation Process is being processed...")
    print("This data contains {} time ponts with sampling frequency of {} Hz.".format(X.shape[2], smp_freq))
    start_pos_mi = int(CONSTANT['MI']['start'] * smp_freq)  # 0s
    stop_pos_mi = int(CONSTANT['MI']['stop'] * smp_freq)  # 4s
    start_pos_rest = int(CONSTANT['MI']['stop'] * smp_freq)  # 4s
    stop_pos_rest = int(CONSTANT['trial_len'] * smp_freq)  # 8s
    index_class1 = np.where(y == 0)[0]
    index_class2 = np.where(y == 1)[0]
    X_class1, y_class1 = X[index_class1], y[index_class1]
    X_class2, y_class2 = X[index_class2], y[index_class2]
    # Split data into resting and MI signals
    X_mi_class1 = X_class1[:, :, start_pos_mi:stop_pos_mi]
    X_mi_class2 = X_class2[:, :, start_pos_mi:stop_pos_mi]
    X_rest_class1 = X_class1[:, :, start_pos_rest:stop_pos_rest]
    X_rest_class2 = X_class2[:, :, start_pos_rest:stop_pos_rest]
    # Choose a half of resting data to keep balancing the number of classes in our data
    X_rest_class1_50per, _, _, _ = train_test_split(X_rest_class1, y_class1, random_state=42, test_size=0.5)
    X_rest_class2_50per, _, _, _ = train_test_split(X_rest_class2, y_class2, random_state=42, test_size=0.5)
    X_rest_all = np.concatenate((X_rest_class1_50per, X_rest_class2_50per), axis=0)
    # Build class for resting data
    y_rest_all = np.full(X_rest_all.shape[0], 2)
    # Combine all classes again
    X_new_all = np.concatenate((X_mi_class1, X_mi_class2, X_rest_all), axis=0)
    y_new_all = np.concatenate((y_class1, y_class2, y_rest_all), axis=0)
    return X_new_all, y_new_all


def __transitory_mi(X, y, smp_freq, start, stop):
    print("MI Right ang MI Left EEG including transitory period is being processed...")
    print("This data contains {} time ponts with sampling frequency of {} Hz.".format(X.shape[2], smp_freq))
    start_pos_mi = int(start * smp_freq)
    stop_pos_mi = int(stop * smp_freq)
    # Segment needed MI period
    X_mi = X[:, :, start_pos_mi:stop_pos_mi]
    return X_mi, y


def load_data_batchs2(PATH, session, subject_list, num_class, id_ch_selected, new_smp_freq, bands=[8,30], order=5):
    train_subjects_data = []
    test_subjects_data = []
    train_subjects_label = []
    test_subjects_label = []
    domain_label_train = []
    domain_label_test = []
    for subject in subject_list:
        mat_file_name = PATH + '/sess' + str(session).zfill(2) + '_subj' + str(subject).zfill(2) + '_EEG_MI.mat'
        mat = sio.loadmat(mat_file_name)
        print('This is data from: ', mat_file_name)
        if num_class == 2:
            raw_train_data = mat['EEG_MI_train'][0]['smt'][0]
            raw_train_data = (np.swapaxes(raw_train_data, 0, 2))[id_ch_selected]
            raw_train_data = np.swapaxes(raw_train_data, 0, 1)
            raw_train_data = butter_bandpass_filter(raw_train_data, bands[0], bands[1], 100, order)
            raw_train_data = resampling(raw_train_data, new_smp_freq, raw_train_data.shape[2])
            print('raw_train_data_shape:', raw_train_data.shape)
            raw_test_data = mat['EEG_MI_test'][0]['smt'][0]
            raw_test_data = np.swapaxes(raw_test_data, 0, 2)[id_ch_selected]
            raw_test_data = np.swapaxes(raw_test_data, 0, 1)
            raw_test_data = butter_bandpass_filter(raw_test_data, bands[0], bands[1], 100, order)
            raw_test_data = resampling(raw_test_data, new_smp_freq, raw_test_data.shape[2])
            print('raw_test_data_shape:', raw_test_data.shape)
            label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0] - 1
            label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0] - 1
            domain_label_tr = np.ones(label_train_data.shape[0], dtype=int) * (subject - min(subject_list))
            domain_label_te = np.ones(label_test_data.shape[0], dtype=int) * (subject - min(subject_list))
        # 逐个将subject数据写入最终输出的list中
        if len(train_subjects_data) == 0:
            train_subjects_data = raw_train_data
            test_subjects_data = raw_test_data
            train_subjects_label = label_train_data
            test_subjects_label = label_test_data
            domain_label_train = domain_label_tr
            domain_label_test = domain_label_te
        else:
            train_subjects_data = np.concatenate((test_subjects_data, raw_train_data), axis=0)
            test_subjects_data = np.concatenate((test_subjects_data, raw_test_data), axis=0)
            train_subjects_label = np.concatenate((train_subjects_label, label_train_data), axis=0)
            test_subjects_label = np.concatenate((test_subjects_label, label_test_data), axis=0)
            domain_label_train = np.concatenate((domain_label_train, domain_label_tr), axis=0)
            domain_label_test = np.concatenate((domain_label_test, domain_label_te), axis=0)
    # 注意：实验设置非跨被试情况下，可以注释掉不用合并
    subject_data = np.concatenate((train_subjects_data, test_subjects_data), axis=0)
    subject_label = np.concatenate((train_subjects_label, test_subjects_label), axis=0)
    subject_domain_label = np.concatenate((domain_label_train, domain_label_test), axis=0)
    return subject_data, subject_label, subject_domain_label



def load_data_batchs(PATH, session, subject_list, num_class, id_ch_selected, new_smp_freq):
    train_subjects_data = []
    test_subjects_data = []
    train_subjects_label = []
    test_subjects_label = []
    domain_label_train = []
    domain_label_test = []
    for subject in subject_list:
        mat_file_name = PATH + '/sess' + str(session).zfill(2) + '_subj' + str(subject).zfill(2) + '_EEG_MI.mat'
        mat = sio.loadmat(mat_file_name)
        # print('This is data from: ', mat_file_name)
        if num_class == 2:
            raw_train_data = mat['EEG_MI_train'][0]['smt'][0]
            raw_train_data = (np.swapaxes(raw_train_data, 0, 2))[id_ch_selected]
            raw_train_data = np.swapaxes(raw_train_data, 0, 1)
            raw_train_data = resampling(raw_train_data, new_smp_freq, raw_train_data.shape[2])
            # print('raw_train_data_shape:', raw_train_data.shape)
            raw_test_data = mat['EEG_MI_test'][0]['smt'][0]
            raw_test_data = np.swapaxes(raw_test_data, 0, 2)[id_ch_selected]
            raw_test_data = np.swapaxes(raw_test_data, 0, 1)
            raw_test_data = resampling(raw_test_data, new_smp_freq, raw_test_data.shape[2])
            # print('raw_test_data_shape:', raw_test_data.shape)
            label_train_data = mat['EEG_MI_train'][0]['y_dec'][0][0] - 1
            label_test_data = mat['EEG_MI_test'][0]['y_dec'][0][0] - 1
            domain_label_tr = np.ones(label_train_data.shape[0], dtype=int) * (subject-min(subject_list))
            domain_label_te = np.ones(label_test_data.shape[0], dtype=int) * (subject-min(subject_list))
        # 逐个将subject数据写入最终输出的list中
        if len(train_subjects_data) == 0:
            train_subjects_data = raw_train_data
            test_subjects_data = raw_test_data
            train_subjects_label = label_train_data
            test_subjects_label = label_test_data
            domain_label_train = domain_label_tr
            domain_label_test = domain_label_te
        else:
            train_subjects_data = np.concatenate((test_subjects_data, raw_train_data), axis=0)
            test_subjects_data = np.concatenate((test_subjects_data, raw_test_data), axis=0)
            train_subjects_label = np.concatenate((train_subjects_label, label_train_data), axis=0)
            test_subjects_label = np.concatenate((test_subjects_label, label_test_data), axis=0)
            domain_label_train = np.concatenate((domain_label_train, domain_label_tr), axis=0)
            domain_label_test = np.concatenate((domain_label_test, domain_label_te), axis=0)
    # 注意：实验设置非跨被试情况下，可以注释掉不用合并
    subject_data = np.concatenate((train_subjects_data, test_subjects_data), axis=0)
    subject_label = np.concatenate((train_subjects_label, test_subjects_label), axis=0)
    subject_domain_label = np.concatenate((domain_label_train, domain_label_test), axis=0)
    print('total_data_shape:', subject_data.shape)
    return subject_data, subject_label, subject_domain_label


if __name__ == '__main__':
    path = CONSTANT['raw_path']
    chs_group = CONSTANT['spatial_ch_group']
    session = 1
    subject_list = [5]
    num_class = 2
    sel_chs = CONSTANT['sel_chs']
    # 筛选我们需要的channel

    id_ch_selected = chanel_selection(sel_chs)
    div_id = channel_division(chs_group)
    region_split = region_id_seg(div_id, id_ch_selected)
    # raw_train_data, label_train_data, \
    # raw_test_data, label_test_data = read_raw(path, session, subject, num_class, id_ch_selected)
    subject_data, subject_label, subject_domain_label = load_data_batchs(path, session, subject_list, num_class, id_ch_selected, 0.1)
    label_test = Counter(subject_label)
    print('Counter \n', Counter(subject_label))
    print(f'训练数据维度:{subject_data.shape}')
