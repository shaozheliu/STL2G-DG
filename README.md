# L2GNet


## 可调整的超参数
### CONSTANT
- n_subjs 用的被试数（一个被试200个training），我最终会用54 
- temporal_ch_region  整体在1，400，每个分段要连续；

### CONFIG
- 'lr': 0.002,
- 'batch_size': 250,
- 'epochs': 300,
- 'dropout':0.3,
- 'd_model_dict' :{
                        'spatial':40,
                        'temporal':40,
                        'st_fusion':40
                        },
                    'head_dict' :{
                        'spatial': 2,
                         'temporal': 1,
                        'st_fusion':1
                    }, head_dict可以调整，且必须可以整除model_dict
- 'd_ff': 10
- 'n_layers': 2,

### 模型内部控制整体参数量
- n_layers、d_ff 会影响参数量
- L2GNet中可以调整的量为
  - S_Backbone_test和TBackbone_test的卷积kernel值，调整的话可以跟我沟通我来调整
    


### L2GNet  32*32 的时空Transformer  0.62 


### L2GNet  32*32 的时空Transformer  0.62 
```python
class S_Backbone_test(nn.Module):
    def __init__(self, ch, dropout):
        super(S_Backbone_test, self).__init__()
        self.dropout = dropout
        # Layer 1
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 40), padding=(0,6))   # 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 40), padding=(0, 6))
        self.conv1_2 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(ch, 1))
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 10), stride=(1, 4))

        # Layer 2
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 20), padding=(0, 10), bias=False)  # 10
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 20), padding=(0, 4), bias=False)
        self.batchnorm2 = nn.BatchNorm2d(4)
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
        x = x.reshape(-1, 4 * 8)
        return x

class T_Backbone_test(nn.Module):
    def __init__(self, ch, dropout):
        super(T_Backbone_test, self).__init__()
        self.dropout = dropout
        # Layer 1
        # self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(1, 10), padding=(0,2))  # 10
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(1, 10), padding=(0, 0))
        self.conv1_2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(ch, 1))
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.pooling1 = nn.AvgPool2d(kernel_size=(1, 5), stride=(1, 2))

        # Layer 2
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(1, 5), padding=(0), bias=False)   # 10
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(1, 10), padding=(0), bias=False)  # 10
        self.batchnorm2 = nn.BatchNorm2d(4)
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
        x = x.reshape(-1, 4 * 8)
        return x
```