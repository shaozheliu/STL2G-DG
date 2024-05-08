import torch.nn.functional as F
import torch.nn as nn
import torch

class BackBoneNet(nn.Module):
    def __init__(self, ch, dropout):
        super(BackBoneNet, self).__init__()
        ##----------------ShallowNet 基准网络------------------#
        self.dropout = dropout
        # Layer 1
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 40), padding=(0,6))   # 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 40), padding=(0, 6))
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(ch, 1))
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 4))

        # Layer 2
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 20), padding=(0, 10), bias=False)  # 10
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 20), padding=(0, 4), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(2)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 8))


    def forward(self, x):
        x = x.unsqueeze(1)  # sample, 1, eeg_channel, timepoints
        # Layer 1
        x = self.conv1(x)  # sample, out_channels, eeg_channel, time_dim
        x = self.conv1_2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = x.reshape(-1, 2 * 8)
        return x

class BackBoneNet_s_shallow(nn.Module):
    def __init__(self, ch, dropout):
        super(BackBoneNet_s_shallow, self).__init__()
        ##----------------ShallowNet 基准网络------------------#
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 20), padding=0)
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(ch, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 8))

        # Layer 2
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 20), padding=(0, 10), bias=False)  # 10
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 4), stride=(1,8))
        self.batchnorm3 = nn.BatchNorm2d(10)
        self.pooling3 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 4))


    def forward(self, x):
        x = x.unsqueeze(1)  # sample, 1, eeg_channel, timepoints
        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = self.conv3(x)  # 测试用
        x = x.reshape(x.size(0), -1)
        return x

class BackBoneNet_t_shallow(nn.Module):
    def __init__(self, ch, dropout):
        super(BackBoneNet_t_shallow, self).__init__()
        ##----------------ShallowNet 基准网络------------------#
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 15), padding=0)
        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(ch, 1), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 2))

        # Layer 2
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 20), padding=(0, 10), bias=False)  # 10
        self.conv3 = nn.Conv2d(in_channels=20, out_channels=10, kernel_size=(1, 12), stride=(1,4))


    def forward(self, x):
        x = x.unsqueeze(1)  # sample, 1, eeg_channel, timepoints
        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = self.conv3(x)  # 测试用
        x = x.reshape(x.size(0), -1)
        return x



class BackBoneNet_t(nn.Module):
    def __init__(self, ch, dropout):
        super(BackBoneNet_t, self).__init__()
        ##----------------ShallowNet 基准网络------------------#
        self.dropout = dropout
        # Layer 1
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 10), padding=(0,2))  # 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 10), padding=(0, 0))
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(ch, 1))
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2))

        # Layer 2
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 5), padding=(0), bias=False)   # 10
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=(1, 10), padding=(0), bias=False)  # 10
        self.batchnorm2 = nn.BatchNorm2d(2)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 4))


    def forward(self, x):
        x = x.unsqueeze(1)  # sample, 1, eeg_channel, timepoints
        # Layer 1
        x = self.conv1(x)  # sample, out_channels, eeg_channel, time_dim
        x = self.conv1_2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = x.reshape(-1, 2 * 8)
        return x


class Spatial_Local_Conv(nn.Module):
    def __init__(self, division, dropout):
        super(Spatial_Local_Conv, self).__init__()
        self.division = division
        self.dropout = dropout

        for i in self.division.keys():
            rigion_conv = BackBoneNet_s_shallow(len(self.division[i]), dropout)
            setattr(self, f'S_region_conv{i}', rigion_conv)

    def forward(self, x):
        encoded_regions = []

        # 根据给定的region进行channel维度的切分
        for i in self.division.keys():
            region = x[:, self.division[i], :]
            encoded_region = getattr(self, f'S_region_conv{i}')(region)
            encoded_region = encoded_region.unsqueeze(1)
            encoded_regions.append(encoded_region)
        return encoded_regions

class Temporal_Local_Conv(nn.Module):
    def __init__(self, division, dropout):
        super(Temporal_Local_Conv, self).__init__()
        self.division = division
        self.dropout = dropout

        for i in self.division.keys():
            rigion_conv = BackBoneNet_t_shallow(20, dropout)
            setattr(self, f'T_region_conv{i}', rigion_conv)

    def forward(self, x):
        encoded_regions = []

        # 根据给定的region进行channel维度的切分
        for i in self.division.keys():
            region = x[:, :, self.division[i]]
            encoded_region = getattr(self, f'T_region_conv{i}')(region)
            encoded_region = encoded_region.unsqueeze(1)
            encoded_regions.append(encoded_region)
        return encoded_regions





class BlackNet(nn.Module):
    def __init__(self, spatial_div_dict, temporal_div_dict, dropout, nb_class):
        super(BlackNet, self).__init__()
        self.st_len = len(spatial_div_dict) + len(temporal_div_dict)
        # 空间维度backbone分块卷积
        self.Local_spatial_conved = Spatial_Local_Conv(spatial_div_dict, dropout)
        self.Local_temporal_conved = Temporal_Local_Conv(temporal_div_dict, dropout)
        self.fc1 = nn.Linear(self.st_len * 60, nb_class)

    def forward(self, x):
        S_Region_tensor = torch.cat(self.Local_spatial_conved(x), dim=1)
        T_Region_tensor = torch.cat(self.Local_temporal_conved(x), dim=1)
        fusion = torch.cat([S_Region_tensor, T_Region_tensor], dim=1)
        ret = fusion.reshape(-1, self.st_len * 60)
        ret = self.fc1(ret)
        return  ret



if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 20, 400))
    s_division = {
        '1': [i for i in range(5)],
        '2': [i for i in range(5, 15)],
        '3': [i for i in range(15, 20)],
    }
    t_division = {
        '1': [i for i in range(100)],
        '2': [i for i in range(100, 200)],
        '3': [i for i in range(200, 300)],
        '4': [i for i in range(300, 400)],
    }
    # model = BackBoneNet(30, dropout=0.2)
    # model = Spatial_Local_Conv(s_division, dropout=0.3)
    model = BlackNet(s_division, t_division, dropout=0.3, nb_class=4)

    output = model(inp)