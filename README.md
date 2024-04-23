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