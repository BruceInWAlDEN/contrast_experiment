'''
make_data: 制作数据集，数据集保存为txt. 每次在每类中读取适量的数据
see_loss: 查看损失
'''
import pandas as pd
import os
import glob
import torch
import matplotlib.pyplot as plt
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def see_loss(path):
    '''
    :param metric: 损失字典路径
    :return: 将结果写如 result.xlsx
    '''
    data = torch.load(path, map_location='cpu')
    print(data.keys())
    for i in data.keys():
        print(i, ',', str(data[i]).strip("[]").replace('tensor', '').replace("(", '').replace(")", '').replace(', requires_grad=True',''))


def make_data(config):
    '''
    先将所有路径读取保存
    --data
        path1.txt
        path2.txt
        ...
    txt：
    中第一行：每次读取的数量，总数据量
    后面每一行：路径，标签
    '''

    key = ['0-20/n_0', '20-50/n_1', '50-100/n_2', '100/n_5', 'n_8', 'n_9']
    df = pd.read_excel(config)
    if not os.path.exists('data'):
        os.mkdir('data')
    for i in range(len(df)):
        if df['flag'][i] == 1 and pd.isnull(df['3th'][i]) and df['1th'][i][0] == 'H':

            # -----------K:盘数据没有

            num = int(df[key[0]][i] * df['inter_ratio'][i])  # 读取数量0-20/n_o * inter_ratio
            class_path = os.path.join(df['1th'][i], df['2th'][i]).replace('H:', '/mnt/160_h')
            paths = glob.glob(os.path.join(class_path, '*.*'))
            paths = [img_d for img_d in paths if '.tif' in img_d or '.png' in img_d or '.jpg' in img_d]
            label = df['label'][i]  # 标签

            print('============', class_path, '==', len(paths), '================')
            with open('data/' + str(i) + '.txt', 'a') as file:
                file.write('{:.0f},{:.0f},{:.0f}\n'.format(num, len(paths), label))
                for img in paths:
                    file.write(img + '\n')


see_loss('check/unsupervised_fine_tuned_metric')