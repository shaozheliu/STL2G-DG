import numpy as np
from scipy import signal
import numpy as np
import torch
import mne
from scipy.signal import butter, filtfilt

from torch.utils.data import Dataset, DataLoader


def resampling(data, new_smp_freq, data_len):
    if len(data.shape) != 3:
        raise Exception('Dimesion error', "--> please use three-dimensional input")
    new_smp_point = int(data_len*new_smp_freq)
    data_resampled = np.zeros((data.shape[0], data.shape[1], new_smp_point))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data_resampled[i,j,:] = signal.resample(data[i,j,:], new_smp_point)
    return data_resampled


def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


class load_data(Dataset):
    def __init__(self, X, y, y_domain, train=True):
        """

        :param X: 输入数据X
        :param y: 真实标签
        :param y_domain: 域标签，如果不需要域标签输入：None
        :param train: 标识是否是training流程
        """
        self.X = X
        self.y = y
        self.y_domain = y_domain
        self.train = train

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.train:
            x = x[:, 0:4000]
        else:
            x = x[:, 0:4000]
        # 是否有domain判断

        return x, self.y[idx], self.y_domain[idx]


def get_loaders(train_X, train_y, train_domain_y, test_X, test_y, test_domain_y, batch_size = 250):
    train_set, test_set = load_data(train_X, train_y, train_domain_y, True), \
                                   load_data(test_X, test_y, test_domain_y, False)
    data_loader_train = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        num_workers=0,
        # pin_memory=True,
        drop_last=False,
        shuffle= True
    )
    data_loader_test = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            num_workers=0,
            # pin_memory=True,
            drop_last=False,
            shuffle=False
    )
    dataloaders = {
        'train': data_loader_train,
        'test': data_loader_test
    }
    return dataloaders



