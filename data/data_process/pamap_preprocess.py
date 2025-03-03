# encoding=utf-8
"""
    Created on 10:38 2018/12/17
    @author: Hangwei Qian
    Adapted from: https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
"""
import os
import numpy as np
import torch
import pickle as cp
from torch.utils.data import Dataset, DataLoader
# from utils import get_sample_weights, opp_sliding_window
# from sliding_window import sliding_window
from collections import Counter
# os.chdir(sys.path[0])
# sys.path.append('../')
from utils import *

NUM_FEATURES = 52

def build_npydataset_readme(path):
    '''
        构建数据集readme
    '''
    datasets = sorted(os.listdir(path)) 
    curdir = os.curdir # 记录当前地址
    os.chdir(path) # 进入所有npy数据集根目录
    with open('readme.md', 'w') as w:
        for dataset in datasets:
            if not os.path.isdir(dataset):
                continue
            x_train = np.load('%s/x_train.npy' % (dataset))
            x_test = np.load('%s/x_test.npy' % (dataset))
            y_train = np.load('%s/y_train.npy' % (dataset))
            y_test = np.load('%s/y_test.npy' % (dataset))
            category = len(set(y_test.tolist()))
            d = Counter(y_test)
            new_d = {} # 顺序字典
            for i in range(category):
                new_d[i] = d[i]
            log = '\n===============================================================\n%s\n   x_train shape: %s\n   x_test shape: %s\n   y_train shape: %s\n   y_test shape: %s\n\n共【%d】个类别\ny_test中每个类别的样本数为 %s\n' % (dataset, x_train.shape, x_test.shape, y_train.shape, y_test.shape, category, new_d)
            w.write(log)
    os.chdir(curdir) # 返回原始地址


def save_npy_data(dataset_name, root_dir, xtrain, xtest, ytrain, ytest):
    '''
        dataset_name: 数据集
        root_dir: 数据集保存根目录
        xtrain: 训练数据 : array  [n1, window_size, modal_leng]
        xtest: 测试数据 : array   [n2, window_size, modal_leng]
        ytrain: 训练标签 : array  [n1,]
        ytest: 测试标签 : array   [n2,]
    '''
    path = os.path.join(root_dir, dataset_name)
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(path + '/x_train.npy', xtrain)
    np.save(path + '/x_test.npy', xtest)
    np.save(path + '/y_train.npy', ytrain)
    np.save(path + '/y_test.npy', ytest)
    print('\n.npy数据【xtrain，xtest，ytrain，ytest】已经保存在【%s】目录下\n' % (root_dir))
    build_npydataset_readme(root_dir)
    
def opp_sliding_window(data_x, data_y, ws, ss): # window size, step size
    data_x = sliding_window(data_x,(ws,data_x.shape[1]),(ss,1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y,ws,ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


NUM_FEATURES = 52
SLIDING_WINDOW_LEN = 128
SLIDING_WINDOW_STEP = 64

class data_loader_pamap2(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        sample, target = self.samples[index], self.labels[index]
        return sample, target

    def __len__(self):
        return len(self.samples)


def normalize(x):
    """Normalizes all sensor channels by mean substraction,
    dividing by the standard deviation and by 2.

    :param x: numpy integer matrix
        Sensor data
    :return:
        Normalized sensor data
    """
    x = np.array(x, dtype=np.float32)
    m = np.mean(x, axis=0)
    x -= m
    std = np.std(x, axis=0)
    std += 0.000001

    x /= std
    return x


def complete_HR(data):
    """Sampling rate for the heart rate is different from the other sensors. Missing
    measurements are filled

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        HR channel data
    """

    pos_NaN = np.isnan(data)
    idx_NaN = np.where(pos_NaN == False)[0]
    data_no_NaN = data * 0
    for idx in range(idx_NaN.shape[0] - 1):
        data_no_NaN[idx_NaN[idx]: idx_NaN[idx + 1]] = data[idx_NaN[idx]]

    data_no_NaN[idx_NaN[-1]:] = data[idx_NaN[-1]]

    return data_no_NaN


def divide_x_y(data):
    """Segments each sample into time, labels and sensor channels

    :param data: numpy integer matrix
        Sensor data
    :return: numpy integer matrix, numpy integer array
        Time and labels as arrays, sensor channels as matrix
    """
    data_t = data[:, 0]
    data_y = data[:, 1]
    data_x = data[:, 2:]

    return data_t, data_x, data_y

def adjust_idx_labels(data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function adjust the labels picking the labels
    for the protocol settings

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    data_y[data_y == 24] = 0
    data_y[data_y == 12] = 8
    data_y[data_y == 13] = 9
    data_y[data_y == 16] = 10
    data_y[data_y == 17] = 11

    return data_y


def del_labels(data_t, data_x, data_y):
    """The pamap2 dataset contains in total 24 action classes. However, for the protocol,
    one uses only 16 action classes. This function deletes the nonrelevant labels

    18 ->

    :param data_y: numpy integer array
        Sensor labels
    :return: numpy integer array
        Modified sensor labels
    """

    idy = np.where(data_y == 0)[0]
    labels_delete = idy

    idy = np.where(data_y == 8)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 9)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 10)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 11)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 18)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 19)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    idy = np.where(data_y == 20)[0]
    labels_delete = np.concatenate([labels_delete, idy])

    return np.delete(data_t, labels_delete, 0), np.delete(data_x, labels_delete, 0), np.delete(data_y, labels_delete, 0)


def downsampling(data_t, data_x, data_y):
    """Recordings are downsamplied to 30Hz, as in the Opportunity dataset

    :param data_t: numpy integer array
        time array
    :param data_x: numpy integer array
        sensor recordings
    :param data_y: numpy integer array
        labels
    :return: numpy integer array
        Downsampled input
    """

    idx = np.arange(0, data_t.shape[0], 3)

    return data_t[idx], data_x[idx], data_y[idx]


def process_dataset_file(data):
    """Function defined as a pipeline to process individual Pamap2 files

    :param data: numpy integer matrix
        channel data: samples in rows and sensor channels in columns
    :return: numpy integer matrix, numy integer array
        Processed sensor data, segmented into samples-channel measurements (x) and labels (y)
    """

    # Data is divided in time, sensor data and labels
    data_t, data_x, data_y = divide_x_y(data)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # nonrelevant labels are deleted
    data_t, data_x, data_y = del_labels(data_t, data_x, data_y)

    print("data_x shape {}".format(data_x.shape))
    print("data_y shape {}".format(data_y.shape))
    print("data_t shape {}".format(data_t.shape))

    # Labels are adjusted
    data_y = adjust_idx_labels(data_y)
    data_y = data_y.astype(int)

    if data_x.shape[0] != 0:
        HR_no_NaN = complete_HR(data_x[:, 0])
        data_x[:, 0] = HR_no_NaN

        data_x[np.isnan(data_x)] = 0

        data_x = normalize(data_x)

    else:
        data_x = data_x
        data_y = data_y
        data_t = data_t

        print("SIZE OF THE SEQUENCE IS CERO")

    data_t, data_x, data_y = downsampling(data_t, data_x, data_y)

    return data_x, data_y


def load_data_pamap2():
    import os
    data_dir = './'
    saved_filename = 'pamap2_processed.data'
    if os.path.isfile( data_dir + saved_filename ) == True:
        data = np.load(data_dir + saved_filename, allow_pickle=True)
        X_train = data[0][0]
        y_train = data[0][1]

        # X_validation = data[1][0]
        # y_validation = data[1][1]

        X_test = data[1][0]
        y_test = data[1][1]

        return X_train, y_train, X_test, y_test
    else:
        dataset = './'
        # File names of the files defining the PAMAP2 data.
        PAMAP2_DATA_FILES = ['PAMAP2_Dataset/Protocol/subject101.dat',  # 0
                             'PAMAP2_Dataset/Optional/subject101.dat',  # 1
                             'PAMAP2_Dataset/Protocol/subject102.dat',  # 2
                             'PAMAP2_Dataset/Protocol/subject103.dat',  # 3
                             'PAMAP2_Dataset/Protocol/subject104.dat',  # 4
                             'PAMAP2_Dataset/Protocol/subject107.dat',  # 5
                             'PAMAP2_Dataset/Protocol/subject108.dat',  # 6
                             'PAMAP2_Dataset/Optional/subject108.dat',  # 7
                             'PAMAP2_Dataset/Protocol/subject109.dat',  # 8
                             'PAMAP2_Dataset/Optional/subject109.dat',  # 9
                             'PAMAP2_Dataset/Protocol/subject105.dat',  # 10
                             'PAMAP2_Dataset/Optional/subject105.dat',  # 11
                             'PAMAP2_Dataset/Protocol/subject106.dat',  # 12
                             'PAMAP2_Dataset/Optional/subject106.dat',  # 13
                             ]

        X_train = np.empty((0, NUM_FEATURES))
        y_train = np.empty((0))

        X_test = np.empty((0, NUM_FEATURES))
        y_test = np.empty((0))

        counter_files = 0

        print('Processing dataset files ...')
        for filename in PAMAP2_DATA_FILES:
            if counter_files < 12:
                # Train partition
                try:
                    print('Train... file {0}'.format(filename))
                    data = np.loadtxt(dataset + filename)
                    print('Train... data size {}'.format(data.shape))
                    x, y = process_dataset_file(data)
                    print(x.shape)
                    print(y.shape)
                    X_train = np.vstack((X_train, x))
                    y_train = np.concatenate([y_train, y])
                except KeyError:
                    print('ERROR: Did not find {0} in zip file'.format(filename))

            else:
                # Testing partition
                try:
                    print('Test... file {0}'.format(filename))
                    data = np.loadtxt(dataset + filename)
                    print('Test... data size {}'.format(data.shape))
                    x, y = process_dataset_file(data)
                    print(x.shape)
                    print(y.shape)
                    X_test = np.vstack((X_test, x))
                    y_test = np.concatenate([y_test, y])
                except KeyError:
                    print('ERROR: Did not find {0} in zip file'.format(filename))

            counter_files += 1

        print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape))
        X, y = sliding_window(X_train,y_train,SLIDING_WINDOW_LEN,0.5)
        X_, y_ = sliding_window(X_test, y_test, SLIDING_WINDOW_LEN, 0.5)
        SAVE_PATH=True
        if SAVE_PATH: # 数组数据保存目录
            save_npy_data(
                dataset_name='PAMAP2',
                root_dir=os.path.abspath('./data'),
                xtrain=X,
                xtest=X_,
                ytrain=y,
                ytest=y_
            )
        #return X_train, y_train, X_val, y_val, X_test, y_test


    #print("Final datasets with size: | train {0} | test {1} | ".format(X_train.shape, X_test.shape))
 
    # return X_train, y_train,X_test, y_test

    # SLIDING_WINDOW_LEN = 128
    # SLIDING_WINDOW_STEP = 64
    # X_train, y_train = opp_sliding_window(X_train, y_train, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    # X_test, y_test = opp_sliding_window(X_test, y_test, SLIDING_WINDOW_LEN, SLIDING_WINDOW_STEP)
    # np.save('../data/pamapl/' + 'x_train.npy', X_train)
    # np.save('../data/pamapl/' + 'y_train.npy', y_train)
    # np.save('../data/pamapl/' + 'x_test.npy', X_test)
    # np.save('../data/pamapl/' + 'y_test.npy', y_test)
    # obj = [(X_train, y_train), (X_test, y_test)]
    # f = file(os.path.join(target_filename), 'wb')
    # return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    load_data_pamap2()