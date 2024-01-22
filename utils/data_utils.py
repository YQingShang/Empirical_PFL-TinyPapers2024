import numpy as np
import os
import torch
import pandas as pd


def read_data(datapath,dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(datapath,dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join(datapath,dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data
def std_data(array):
    array_r = array.astype(float)
    for col_idx in range(array.shape[1]):
        col = array[:, col_idx]  
        if len(set(col))!=2:
            col = (col-np.mean(col))/np.std(col) # standardization
        array_r[:, col_idx] = col
    return(array_r)


def read_data_tabular(datapath,dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join(datapath,dataset, 'train/')
        train_file = train_data_dir + str(idx) + '_train.csv'

        train_tmp=pd.read_csv(train_file,index_col=0)
        X = train_tmp.drop(columns='label').to_numpy()
        X = std_data(X)
        y = train_tmp['label'].to_numpy()
        train_data={'x':X,'y':y}

        return train_data

    else:
        test_data_dir = os.path.join(datapath,dataset, 'test/')
        test_file = test_data_dir + str(idx) + '_test.csv'

        test_tmp=pd.read_csv(test_file,index_col=0)
        X = test_tmp.drop(columns='label').to_numpy()
        X = std_data(X)
        y = test_tmp['label'].to_numpy()
        test_data={'x':X,'y':y}

        return test_data
    
def read_client_data_tabular( datapath,dataset, idx, is_train=True):
    if is_train:
        train_data = read_data_tabular(datapath,dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data_tabular(datapath,dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data
    

def read_client_data(datapath,dataset, idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(datapath,dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(datapath,dataset, idx)
    elif 'Categorical' in dataset:
        return read_client_data_tabular(datapath,dataset, idx)
    
    if is_train:
        train_data = read_data(datapath,dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(datapath,dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data



