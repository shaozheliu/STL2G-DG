import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne




# 加载自定义的excel文件，制作自己的montage
# myStardart = pd.read_excel('sensor_dataframe.xlsx', index_col=0) # 读取自己的电极定位文件
# ch_names = np.array(myStardart.index)                              # 电极名称
# position = np.array(myStardart)                                    # 电极坐标位置
# sensorPosition = dict(zip(ch_names, position))                     # 制定为字典的形式
# myMontage = mne.channels.make_dig_montage(ch_pos=sensorPosition)
# myMontage.plot()
# plt.show()
# #
# myWeight = {'Fz': 0.297512650489807,
#           'FC3': 0.131990149617195, 'FC1': 0.36110034584999, 'FCz': 1, 'FC2':0, 'FC4': 0.288021147251129,
#           'C5': 0.231157153844833, 'C3': 0.323453158140182, 'C1': 0.726558804512023, 'Cz': 0.41, 'C2': 0.46, 'C4': 0.99, 'C6':0.15,
#           'CP3':0.23, 'CP1': -0.16, 'CPz': -0.16, 'CP2': -0.16, 'CP4': -0.16,
#           'P1': -0.16, 'Pz': -0.16, 'P2': -0.16,
#           'POz': -0.16,}

def plot_montage(myWeight, id):
    myStardart = pd.read_excel('home/alk/L2G-MI/experiments/sensor_dataframe.xlsx', index_col=0)  # 读取自己的电极定位文件
    ch_names = np.array(myStardart.index)  # 电极名称
    position = np.array(myStardart)  # 电极坐标位置
    sensorPosition = dict(zip(ch_names, position))  # 制定为字典的形式
    myMontage = mne.channels.make_dig_montage(ch_pos=sensorPosition)
    # myMontage.plot()
    plt.show()
    norMyWeiht = myWeight.copy()
    # Step 1: 获取需要归一化的值
    values = [v for v in norMyWeiht.values()]

    # Step 2: 找到最小值和最大值
    min_value = min(values)
    max_value = max(values)

    # Step 3: 对值进行归一化计算
    for key, value in norMyWeiht.items():
        normalized_value = (value - min_value) / (max_value - min_value)
        norMyWeiht[key] = normalized_value

    # 查看脑地形图矩阵导联位置
    my_chLa_index = ch_names.tolist()
    print('脑地形图矩阵导联顺序:',my_chLa_index)

    # 重构自定义矩阵顺序
    reMyWeight = []
    for key in my_chLa_index:
        val = myWeight[key]
        reMyWeight.append(val)
    myinfo = mne.create_info(
            ch_names = my_chLa_index,
            ch_types = ['eeg']*30,   # 通道个数
            sfreq = 400)            # 采样频率
    myinfo.set_montage(myMontage)

    im, cn = mne.viz.plot_topomap(reMyWeight,
                                  myinfo,
                                  names = my_chLa_index,
                                  # vlim=(-2, 2)
                                  show=False
                                 )

    plt.colorbar(im)
    plt.title('My Montage')
    plt.savefig(f'./fig{id}')
    plt.show()


def set_montage(myWeight):
    myStardart = pd.read_excel('/home/alk/L2G-MI/experiments/sensor_dataframe.xlsx', index_col=0)  # 读取自己的电极定位文件
    ch_names = np.array(myStardart.index)  # 电极名称
    position = np.array(myStardart)  # 电极坐标位置
    sensorPosition = dict(zip(ch_names, position))  # 制定为字典的形式
    myMontage = mne.channels.make_dig_montage(ch_pos=sensorPosition)
    # myMontage.plot()
    plt.show()
    norMyWeiht = myWeight.copy()
    # Step 1: 获取需要归一化的值
    values = [v for v in norMyWeiht.values()]

    # Step 2: 找到最小值和最大值
    min_value = min(values)
    max_value = max(values)

    # Step 3: 对值进行归一化计算
    for key, value in norMyWeiht.items():
        normalized_value = (value - min_value) / (max_value - min_value)
        norMyWeiht[key] = normalized_value

    # 查看脑地形图矩阵导联位置
    my_chLa_index = ch_names.tolist()
    print('脑地形图矩阵导联顺序:', my_chLa_index)

    # 重构自定义矩阵顺序
    reMyWeight = []
    for key in my_chLa_index:
        val = myWeight[key]
        reMyWeight.append(val)
    myinfo = mne.create_info(
        ch_names=my_chLa_index,
        ch_types=['eeg'] * 30,  # 通道个数
        sfreq=1000)  # 采样频率
    myinfo.set_montage(myMontage)
    return reMyWeight, myinfo, my_chLa_index


if __name__ == '__main__':

    plt_dict = {0: {0: {'Fz': 0.19789375, 'FC3': 0.94231135, 'FC1': 0.646623, 'FCz': 0.457127, 'FC2': 0.50451696, 'FC4': 0.62706476, 'C5': 0.66050076, 'C3': 0.10842439, 'C1': 0.5659989, 'Cz': 0.795953, 'C2': 0.6696056, 'C4': 0.42250824, 'C6': 0.91856885, 'CP3': 0.21180056, 'CP1': 0.67597103, 'CPz': 1.0, 'CP2': 0.0, 'CP4': 0.32046497, 'P1': 0.122908846, 'Pz': 0.793518, 'P2': 0.78748363, 'POz': 0.7968124},
                    1: {'Fz': 0.3857744, 'FC3': 0.7373408, 'FC1': 0.5312523, 'FCz': 0.7761056, 'FC2': 0.90752214, 'FC4': 0.5777697, 'C5': 0.60525775, 'C3': 0.0, 'C1': 0.79691494, 'Cz': 0.7581965, 'C2': 0.662511, 'C4': 0.58496284, 'C6': 1.0, 'CP3': 0.5106992, 'CP1': 0.75347143, 'CPz': 0.7827314, 'CP2': 0.77230805, 'CP4': 0.22756031, 'P1': 0.6719665, 'Pz': 0.59333104, 'P2': 0.6402745, 'POz': 0.76156384},
                    2: {'Fz': 0.23026013, 'FC3': 0.9146818, 'FC1': 0.4339726, 'FCz': 0.5630181, 'FC2': 0.56479216, 'FC4': 0.4605075, 'C5': 0.7266833, 'C3': 0.0, 'C1': 0.75963885, 'Cz': 0.5892282, 'C2': 0.8273483, 'C4': 0.53168815, 'C6': 0.64506084, 'CP3': 0.4705196, 'CP1': 0.83426726, 'CPz': 1.0, 'CP2': 0.054937318, 'CP4': 0.2085642, 'P1': 0.21804678, 'Pz': 0.66141754, 'P2': 0.56074154, 'POz': 0.9842595},
                    3: {'Fz': 0.31174082, 'FC3': 0.3392657, 'FC1': 0.108118184, 'FCz': 0.87305254, 'FC2': 0.75729036, 'FC4': 0.0, 'C5': 0.5876589, 'C3': 0.5823405, 'C1': 0.7767113, 'Cz': 0.79769415, 'C2': 0.74725574, 'C4': 0.33089355, 'C6': 0.5700621, 'CP3': 0.4089708, 'CP1': 0.5714552, 'CPz': 1.0, 'CP2': 0.5791549, 'CP4': 0.25886607, 'P1': 0.61650735, 'Pz': 0.009705976, 'P2': 0.6322547, 'POz': 0.49183422}},
                4: {0: {'Fz': 0.36378133, 'FC3': 1.0, 'FC1': 0.62846076, 'FCz': 0.3807393, 'FC2': 0.57641745, 'FC4': 0.5093618, 'C5': 0.6682698, 'C3': 0.688599, 'C1': 0.8425348, 'Cz': 0.8092322, 'C2': 0.68884724, 'C4': 0.8114946, 'C6': 0.6256634, 'CP3': 0.0, 'CP1': 0.8589068, 'CPz': 0.9806896, 'CP2': 0.18103176, 'CP4': 0.40576714, 'P1': 0.17196466, 'Pz': 0.11039718, 'P2': 0.38807803, 'POz': 0.33409366},
                    1: {'Fz': 0.4092098, 'FC3': 0.72591054, 'FC1': 0.28705496, 'FCz': 0.75610536, 'FC2': 1.0, 'FC4': 0.4676234, 'C5': 0.8045061, 'C3': 0.6294294, 'C1': 0.61740094, 'Cz': 0.6031444, 'C2': 0.6492385, 'C4': 0.62138504, 'C6': 0.7005668, 'CP3': 0.99320334, 'CP1': 0.51293504, 'CPz': 0.88604397, 'CP2': 0.5331482, 'CP4': 0.8046738, 'P1': 0.377901, 'Pz': 0.0, 'P2': 0.4598482, 'POz': 0.08492447},
                    2: {'Fz': 0.5089359, 'FC3': 0.853678, 'FC1': 0.79962635, 'FCz': 1.0, 'FC2': 0.9289074, 'FC4': 0.504564, 'C5': 0.70947033, 'C3': 0.40018255, 'C1': 0.82232285, 'Cz': 0.80062944, 'C2': 0.7618578, 'C4': 0.80343133, 'C6': 0.37566274, 'CP3': 0.47237173, 'CP1': 0.7855873, 'CPz': 0.81647086, 'CP2': 0.0, 'CP4': 0.8554816, 'P1': 0.4507452, 'Pz': 0.45054445, 'P2': 0.66440564, 'POz': 0.7524026},
                    3: {'Fz': 0.3939626, 'FC3': 0.14808495, 'FC1': 0.43990394, 'FCz': 1.0, 'FC2': 0.99140006, 'FC4': 0.19974658, 'C5': 0.49907988, 'C3': 0.5946731, 'C1': 0.220637, 'Cz': 0.39103845, 'C2': 0.19134437, 'C4': 0.39236653, 'C6': 0.35315225, 'CP3': 0.0, 'CP1': 0.16613029, 'CPz': 0.32368436, 'CP2': 0.21901703, 'CP4': 0.27669466, 'P1': 0.4393514, 'Pz': 0.37156636, 'P2': 0.6972033, 'POz': 0.12280501}},
                7: {0: {'Fz': 0.3435401, 'FC3': 0.5132321, 'FC1': 0.04993872, 'FCz': 0.48858738, 'FC2': 0.41255715, 'FC4': 0.32270476, 'C5': 0.73291284, 'C3': 0.6757962, 'C1': 1.0, 'Cz': 0.95644546, 'C2': 0.80412316, 'C4': 0.25063044, 'C6': 0.27899235, 'CP3': 0.40283024, 'CP1': 0.7617216, 'CPz': 0.9812712, 'CP2': 0.56236655, 'CP4': 0.48710808, 'P1': 0.05924014, 'Pz': 0.27819318, 'P2': 0.0, 'POz': 0.30932564},
                    1: {'Fz': 0.5076661, 'FC3': 1.0, 'FC1': 0.51105064, 'FCz': 0.35560042, 'FC2': 0.7181053, 'FC4': 0.39260358, 'C5': 0.57471824, 'C3': 0.4165601, 'C1': 0.41373196, 'Cz': 0.30704266, 'C2': 0.3517236, 'C4': 0.34105045, 'C6': 0.6496037, 'CP3': 0.51230526, 'CP1': 0.21393868, 'CPz': 0.7001374, 'CP2': 0.3167248, 'CP4': 0.4895645, 'P1': 0.3391499, 'Pz': 0.0, 'P2': 0.97585976, 'POz': 0.43889013},
                    2: {'Fz': 0.29843247, 'FC3': 0.0, 'FC1': 0.5170863, 'FCz': 0.96607965, 'FC2': 1.0, 'FC4': 0.67976326, 'C5': 0.48681664, 'C3': 0.7035528, 'C1': 0.32104686, 'Cz': 0.32092375, 'C2': 0.15333264, 'C4': 0.19273163, 'C6': 0.48081985, 'CP3': 0.25643924, 'CP1': 0.21828575, 'CPz': 0.49250844, 'CP2': 0.78472185, 'CP4': 0.6863696, 'P1': 0.32575476, 'Pz': 0.5639532, 'P2': 0.15925047, 'POz': 0.4572752},
                    3: {'Fz': 0.6642909, 'FC3': 0.6330117, 'FC1': 0.32400894, 'FCz': 0.7688947, 'FC2': 1.0, 'FC4': 0.2070083, 'C5': 0.87665176, 'C3': 0.60574734, 'C1': 0.26105183, 'Cz': 0.0950408, 'C2': 0.02921845, 'C4': 0.6662094, 'C6': 0.54566175, 'CP3': 0.101483166, 'CP1': 0.0, 'CPz': 0.3852995, 'CP2': 0.5478407, 'CP4': 0.62388146, 'P1': 0.400715, 'Pz': 0.07657118, 'P2': 0.37778813, 'POz': 0.14814812}}
                }


    reMyWeight1, myinfo, my_chLa_index = set_montage(plt_dict[0][1])
    reMyWeight2, myinfo, my_chLa_index = set_montage(plt_dict[0][1])
    mne.viz.plot_topomap(reMyWeight1,
                         myinfo,
                         names=my_chLa_index,
                         # vlim=(-2, 2)
                         show=False
                         )
    plt.show()
