import torch
import torch.nn as nn

import torch.nn.functional as F

class Shallow_Net(nn.Module):
    def __init__(self, channels = 30, dropout = 0.2, nb_class = 2):
        super(Shallow_Net, self).__init__()
        self.dropout = dropout
        self.nb_class = nb_class
        # Layer 1
        self.conv1 = nn.Conv2d(1, 10, (1, 15), padding=0)

        # Layer 2
        self.conv2 = nn.Conv2d(10, 40, (channels, 1),bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        # self.pooling2 = nn.AvgPool2d(kernel_size=(1,35), stride=(1,7))
        self.pooling2 = nn.AvgPool2d(kernel_size=(1, 20), stride=(1, 2))

        self.conv3 = nn.Conv2d(40, 20, kernel_size=(1,4), stride=(1,8))
        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        # self.fc1 = nn.Linear(137*40, self.nb_class)
        self.fc1 = nn.Linear(460, 2)

    def forward(self, x):
        x = x.unsqueeze(1)
        # Layer 1
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        x = self.pooling2(x)
        x = F.dropout(x, self.dropout)
        x = self.conv3(x)  # 测试用
        x = x.reshape(-1,x.shape[1]*x.shape[3])
        # FC Layer
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 30, 400))
    model = Shallow_Net(30, dropout=0.2, nb_class=2)
    # model = DeepConvNet(20, dropout=0.2, nb_class=2)

    output = model(inp)