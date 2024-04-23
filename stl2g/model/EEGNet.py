
import torch.nn.functional as F
import torch
import torch.nn as nn



class EEGNet(nn.Module):
    def __init__(self, kernel_length, dropout):
        super(EEGNet, self).__init__()
        self.T = 120
        self.kernel_length = kernel_length
        self.dropout = dropout
        # Layer 1
        # self.conv1 = nn.Conv2d(1, 16, (1, self.kernel_length), padding=(0,  self.kernel_length//2))
        self.conv1 = nn.Conv2d(1, 16, (1, self.kernel_length), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, False)

        # Layer 2
        self.padding1 = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2 = nn.Conv2d(1, 4, (2, 64))
        self.batchnorm2 = nn.BatchNorm2d(4, False)
        self.pooling2 = nn.MaxPool2d(2, 4)

        # Layer 3
        self.padding2 = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3 = nn.Conv2d(4, 4, (8, 32))
        self.batchnorm3 = nn.BatchNorm2d(4, False)
        self.pooling3 = nn.MaxPool2d((2, 4))

        # FC Layer
        # NOTE: This dimension will depend on the number of timestamps per sample in your data.
        # I have 120 timepoints.
        self.fc1 = nn.Linear(4 * 2 * 16, 2)

    def forward(self, x):
        x = x.unsqueeze(1)    # (2,1,20,400)
        x = x.permute(0, 1, 3, 2) # (sample,1,400,20)
        # Layer 1
        x = F.elu(self.conv1(x))  # (sample,16,400,1)
        x = self.batchnorm1(x)
        x = F.dropout(x, self.dropout)
        # Transpose
        x = x.permute(0, 3, 1, 2)   # (sample,1,16,400)

        # Layer 2
        x = self.padding1(x)        # (sample,1,17,433)
        x = F.elu(self.conv2(x))    # (sample,4,16,370)
        x = self.batchnorm2(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling2(x)

        # Layer 3
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, self.dropout)
        x = self.pooling3(x)

        # FC Layer
        x = x.reshape(-1, 4 * 2 * 16)
        x = F.sigmoid(self.fc1(x))
        return x


if __name__ == "__main__":
    inp = torch.autograd.Variable(torch.randn(2, 20, 400))
    model = EEGNet(kernel_length = 20, dropout=0.2)
    output = model(inp)


