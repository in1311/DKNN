import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random
import numpy as np


def get_para_info(optim_type, loss_type, hidden_neurons, lr, pe_weight, top_n, batch_size):
    para_info = ''
    if optim_type!='adam':
        para_info += ',' + optim_type
    if loss_type!='rmse':
        para_info += ',' + loss_type
    if hidden_neurons!= [4, 256, 256, 16]:
        para_info += ',' + str(hidden_neurons)
    if lr!= 0.0001:
        para_info += ',lr=' + str(lr)
    if pe_weight!= 0.8:
        para_info += ',pe_w=' + str(pe_weight)
    if top_n!= 16:
        para_info += ',top_n=' + str(top_n)
    if batch_size!= 128:
        para_info += ',bs=' + str(batch_size)
    return para_info

class DataSet():
    def __init__(self, data):
        super(DataSet, self).__init__()
        self.data = data
        self.train_data, self.val_data, self.test_data = self.load_dataset(self.data)
        self.train_num = len(self.train_data)

    def scaler_data(self):

        self.scaler_dist = MinMaxScaler(feature_range=(0, 1))
        self.scaler_features = StandardScaler()
        self.scaler_label = StandardScaler()
        data_train_scaler = self.train_data.copy()
        data_val_scaler = self.val_data.copy()
        data_test_scaler = self.test_data.copy()

        data_train_scaler.index = data_train_scaler.values[:,0]
        data_val_scaler.index = data_val_scaler.values[:,0]
        data_train_scaler.iloc[:,0] = range(0, len(self.train_data))
        data_val_scaler.iloc[:,0] = range(0, len(self.val_data))

        data_train_scaler.iloc[:,1:3] = self.scaler_dist.fit_transform(self.train_data.values[:,1:3])
        data_val_scaler.iloc[:, 1:3] = self.scaler_dist.transform(self.val_data.values[:, 1:3])
        if data_test_scaler.empty is not True:
            data_test_scaler.iloc[:, 1:3] = self.scaler_dist.transform(self.test_data.values[:, 1:3])
            data_test_scaler.index = data_test_scaler.values[:,0]
            data_test_scaler.iloc[:,0] = range(0, len(self.test_data))

        data_train_scaler.iloc[:,3:-1] = self.scaler_features.fit_transform(self.train_data.values[:,3:-1])
        data_val_scaler.iloc[:,3:-1] = self.scaler_features.transform(self.val_data.values[:,3:-1])
        if data_test_scaler.empty is not True:
            data_test_scaler.iloc[:,3:-1] = self.scaler_features.transform(self.test_data.values[:,3:-1])

        data_train_scaler.iloc[:,-1] = self.scaler_label.fit_transform(self.train_data.values[:,-1].reshape(-1,1))
        data_val_scaler.iloc[:,-1] = self.scaler_label.transform(self.val_data.values[:,-1].reshape(-1,1))
        if data_test_scaler.empty is not True:
            data_test_scaler.iloc[:,-1] = self.scaler_label.transform(self.test_data.values[:,-1].reshape(-1,1))

        return {'train':data_train_scaler, 'val':data_val_scaler, 'test':data_test_scaler}

    def load_dataset(self, all_data):
        if 'z_trend' in all_data.columns:
            all_data = all_data.drop('z_trend',axis=1, inplace=False)
        train_data = all_data[all_data['dataset'] == 'train']
        val_data = all_data[all_data['dataset'] == 'val']
        test_data = all_data[all_data['dataset'] == 'test']
        train_data = train_data.drop('dataset',axis=1, inplace=False)
        val_data = val_data.drop('dataset',axis=1, inplace=False)
        test_data = test_data.drop('dataset',axis=1, inplace=False)

        return train_data, val_data, test_data

    def get_data(self):
        return self.train_data, self.val_data, self.test_data


class DIAGNOSIS():
    def __init__(self, out, labels):
        super(DIAGNOSIS, self).__init__()
        self.v_rmse = self.rmse(out, labels)
        self.v_mse = self.mse(out, labels)
        self.v_mae = self.mae(out, labels)
        self.v_mape = self.mape(out, labels)
        self.v_r2 = self.R2(out, labels)

    def rmse(self, out, labels):
        y = 0
        if type(out)==np.ndarray:
            y = np.sqrt(np.mean((out - labels)**2))
        elif type(out)==torch.Tensor:
            y = torch.sqrt(torch.mean(torch.pow(torch.abs(out - labels), 2)))
        return y

    def mse(self, out, labels):
        y = 0
        if type(out)==np.ndarray:
            y = np.mean((out - labels) ** 2)
        elif type(out) == torch.Tensor:
            y = torch.mean(torch.pow(torch.abs(out - labels), 2))
        return y

    def mape(self, outs, labels):
        y = 0
        if type(outs)==np.ndarray:
            y = np.mean(np.abs((outs - labels) / labels))
        elif type(outs) == torch.Tensor:
            y = torch.mean(torch.abs((outs - labels) / labels))
        return y

    def mae(self, outs, labels):
        y = 0
        if type(outs)==np.ndarray:
            y = np.mean(np.abs(outs - labels))
        elif type(outs) == torch.Tensor:
            y = torch.mean(torch.abs(outs - labels))
        return y

    def R2(self, outs, labels):
        y = 0
        if type(outs)==np.ndarray:
            SStot=np.sum((labels-np.mean(labels))**2)
            SSres=np.sum((outs-labels)**2)
            y = 1-SSres/SStot
        elif type(outs) == torch.Tensor:
            SStot=torch.sum((labels-torch.mean(labels))**2)
            SSres=torch.sum((outs-labels)**2)
            y = 1-SSres/SStot
        return y

    def get(self):
        return self.v_rmse, self.v_mse, self.v_mae, self.v_mape


class EarlyStopping:
    """ Reference: https://github.com/Cai-Yichao/torch_backbones/blob/master/utils/earlystopping.py """
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, save_model_dir):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_model_dir)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, save_model_dir)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, save_model_dir):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if save_model_dir is not None:
            torch.save(model.state_dict(), save_model_dir + 'checkpoint.pt')
        self.val_loss_min = val_loss

def pe_dis_cal(pe_x, pe_know):
    num_x = pe_x.shape[0]
    num_know = pe_know.shape[0]
    pe_dis = torch.matmul(pe_x, pe_know.T)
    return pe_dis

def rmse(out, labels):
    y = torch.sqrt(torch.mean(torch.pow(torch.abs(out - labels), 2)))
    return y

def mse(out, labels):
    y = torch.mean(torch.pow(torch.abs(out - labels), 2))
    return y

def mae(out, labels):
    y = torch.mean(torch.abs(out - labels))
    return y

def mape(out, labels):
    y = torch.mean(torch.abs((out - labels) / labels))
    return y