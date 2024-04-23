CONSTANT = {
    'OpenBMI': {
        'raw_path': '/home/alk/L2G-MI/datasets/OpenBMI/raw', # raw data path
        # 'n_subjs': 54,
        'n_subjs':20,  # 我想用的
        'n_trials_2_class': 100,
        'n_trials_3_class': 150,
        'n_chs': 62,
        'orig_smp_freq': 1000,                  # Original sampling frequency  (Hz)
        'trial_len': 8,                         # 8s (cut-off)
        'MI': {
            'start': 0,                         # start at time = 0 s
            'stop': 4,                          # stop at time = 4 s
            'len': 4,                           # 4s
        },
        'orig_chs': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7',
                    'C3','Cz','C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3',
                    'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4', 'C5', 'C1',
                    'C2', 'C6', 'CP3','CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h', 'TTP7h',
                    'TP7', 'TPP9h', 'FT10','FTT10h','TPP8h', 'TP8', 'TPP10h', 'F9', 'F10',
                    'AF7', 'AF3', 'AF4', 'AF8', 'PO3','PO4'],
        # 'sel_chs': ['FC5', 'FC3', 'FC1', 'FC2', 'FC4','FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5',
        #             'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'],  ## save
        'sel_chs': ['Fz', 'FC1', 'FC2','F3', 'F7', 'FC3', 'FC5','F4', 'F8', 'FC4', 'FC6'
                    ,'C1', 'Cz', 'C2','C3', 'C5','C4','C6','CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2',
                    'CP3', 'CP5', 'P3','CP4', 'CP6', 'P4'],
        'spatial_ch_group':{
            'Frontal Central': ['Fz', 'FC1', 'FC2'],  # 额叶
            'Left Frontal': ['F3', 'F7', 'FC3', 'FC5'],  # 左额叶
            'Right Frontal': ['F4', 'F8', 'FC4', 'FC6'], # 右额叶
            'Central':['C1', 'Cz', 'C2'],                # 中央区
            'Left Central':['C3', 'C5'],                 # 左中央区
            'Right Central':['C4','C6'],                 # 右中央区
            'Central Parietal': ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2'],  # 顶叶
            'Left Parietal': ['CP3', 'CP5', 'P3'],     # 左顶叶
            'Right Parietal':['CP4', 'CP6', 'P4'],     # 右顶叶
        },
        'temporal_ch_group':{
            '1':[i for i in range(8)],
            '2':[i for i in range(8,16)],
        },
        'temporal_ch_region':{
            '1':[i for i in range(100)],
            '2':[i for i in range(100,200)],
            '3':[i for i in range(200,300)],
            '4':[i for i in range(300,400)],
        }   # L2G独享
    }
}