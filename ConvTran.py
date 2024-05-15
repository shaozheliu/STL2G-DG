import torch
import torch.nn as nn

class S_Backbone_test(nn.Module):
    def __init__(self, ch, dropout):
        super(S_Backbone_test, self).__init__()
        self.dropout = dropout
        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(1, 40), padding=(0))
        self.deconv1 = nn.ConvTranspose2d(in_channels=20, out_channels=1, kernel_size=(1, 40),
                                          padding=0)

    def forward(self, x):
        x = x.unsqueeze(1)  # sample, 1, eeg_channel, timepoints
        # Layer 1
        x = self.conv1(x)  # sample, out_channels, eeg_channel, timepoints
        heatmap = self.deconv1(x)  # sample, 1, eeg_channel, timepoints

        return heatmap

if __name__ == '__main__':
    inp = torch.autograd.Variable(torch.randn(1, 3, 400))
    model = S_Backbone_test(3, 0.3)
    out = model(inp)