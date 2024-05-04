import torch.nn.functional as F
import torch.nn as nn
import torch

class DeepConvNettest(nn.Module):
    def __init__(self, ch, dropout, nb_class):
        super(DeepConvNettest, self).__init__()
        ##----------------ShallowNet 基准网络------------------#
        self.dropout = dropout
        self.nb_class = nb_class
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(1, 40), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(10)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 4))

        # Layer 2
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(1, 20), padding=(0, 2), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(20)
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 4))

        self.fc1 = nn.Linear(30 * 51, self.nb_class)

    def forward(self, x):
        x = x.unsqueeze(1)  # sample, 1, eeg_channel, timepoints
        # Layer 1
        x = self.conv1(x)  # sample, out_channels, eeg_channel, time_dim
        x = self.batchnorm1(x)
        x = F.relu(x)
        x = self.pooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = torch.mean(x, dim=1)
        x = x.reshape(-1, 30 * 51)
        # FC Layer
        x = self.fc1(x)
        return x

class DeepConvNet(nn.Module):
    def __init__(self, channels = 22, dropout = 0.2, nb_class = 4):
        super(DeepConvNet, self).__init__()
        self.dropout = dropout
        self.nb_class = nb_class
        # Layer 1
        self.conv1 = nn.Conv2d(1, 25, (1, 5), padding=0)
        self.conv1_2 =nn.Conv2d(25,25,(channels,1))
        self.batchnorm1 = nn.BatchNorm2d(25, False)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=(1,2) ,stride=(1,2))

        # Layer 2
        self.conv2 = nn.Conv2d(25, 50, (1, 5))
        self.batchnorm2 = nn.BatchNorm2d(50, False)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=(1,2) ,stride=(1,2))

        # Layer 3
        self.conv3 = nn.Conv2d(50, 100, (1, 5))
        self.batchnorm3 = nn.BatchNorm2d(100, False)
        self.maxpooling3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Layer 4
        self.conv4 = nn.Conv2d(100, 200, (1, 5))
        self.batchnorm4 = nn.BatchNorm2d(200, False)
        self.maxpooling4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(100*46, self.nb_class)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv1_2(x)
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.maxpooling1(x)
        x = F.dropout(x, self.dropout)
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.maxpooling2(x)
        x = F.dropout(x, self.dropout)

        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.maxpooling3(x)
        x = F.dropout(x, self.dropout)

        # Flatten
        x = x.reshape(-1,100*46)
        # FC Layer
        x = self.fc1(x)
        #
        # # FC Layer2
        # x = F.elu(self.fc2(x))

        return x

if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 30, 1000))
    model = DeepConvNettest(30, dropout=0.2, nb_class=2)
    # model = DeepConvNet(20, dropout=0.2, nb_class=2)

    output = model(inp)