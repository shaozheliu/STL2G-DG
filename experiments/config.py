config = {
        'EEGNet': {
            'OpenBMI': {
                    'input_shape': (1,20,400),
                    'data_format': 'NDCT',
                    'num_class': 2,
                    'lr': 0.001,
                    'batch_size': 300,
                    'epochs': 300,
                    'dropout':0.3,
            },
        },
        'DeepConvNet': {
            'OpenBMI': {
                    'input_shape': (1,20,400),
                    'data_format': 'NDCT',
                    'num_class': 2,
                    'lr': 0.001,
                    'batch_size': 300,
                    'epochs': 300,
                    'dropout':0.3,
            },
        },
        'ShallowNet': {
            'OpenBMI': {
                    'input_shape': (1,20,400),
                    'data_format': 'NDCT',
                    'num_class': 2,
                    'lr': 0.001,
                    'batch_size': 300,
                    'epochs': 300,
                    'dropout':0.3,
            },
        },
        'STL2G': {
            'OpenBMI': {
                    'input_shape': (1,20,400),
                    'data_format': 'NDCT',
                    'num_class': 2,
                    'domain_class': 18,
                    'lr': 0.001,
                    'batch_size': 250,
                    'epochs': 300,
                    'dropout':0.3,
                    'd_model_dict':{'spatial':30, 'temporal':30},
                    'head_dict':{'spatial':2, 'temporal':2},
                    'd_ff': 2,
                    'n_layers': 3,
                    'alpha_scala': 1
            },
        },
        'L2GNet': {
            'OpenBMI': {
                    'input_shape': (1,20,400),
                    'data_format': 'NDCT',
                    'num_class': 2,
                    'domain_class': 18,
                    'lr': 0.0015,
                    'batch_size': 250,
                    'epochs': 200,
                    'dropout':0.3,
                    'd_model_dict' :{
                        'spatial':64,
                        'temporal':64,
                        'st_fusion':64
                        },
                    'head_dict' :{
                        'spatial': 2,
                         'temporal': 2,
                        'st_fusion':2
                    },
                    'd_ff': 1,
                    'n_layers': 2,
                    'alpha_scala': 1
            },
        },
        'BlackNet': {
            'OpenBMI': {
                    'input_shape': (1,20,400),
                    'data_format': 'NDCT',
                    'num_class': 2,
                    'domain_class': 18,
                    'lr': 0.002,
                    'batch_size': 250,
                    'epochs': 200,
                    'dropout':0.3,
                    'd_model_dict' :{
                        'spatial':40,
                        'temporal':40,
                        'st_fusion':40
                        },
                    'head_dict' :{
                        'spatial': 2,
                         'temporal': 1,
                        'st_fusion':1
                    },
                    'd_ff': 10,
                    'n_layers': 2,
                    'alpha_scala': 1
            },
        }
}
