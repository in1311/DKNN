import numpy as np
import torch
import numpy as np
import torch.nn as nn


class DIAGNOSIS():
    ##### Diagnostics based on outputs and labels #####
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
        mask = abs(labels-0) > 1e-8  #  avoid the denominator to be zero
        if type(outs)==np.ndarray:
            y = np.mean(np.abs((outs - labels) / labels)[mask])
        elif type(outs) == torch.Tensor:
            y = torch.mean(torch.abs((outs - labels) / labels)[mask])
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
    ##### EarlyStopping. Reference: https://github.com/Cai-Yichao/torch_backbones/blob/master/utils/earlystopping.py #####
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
        ##### Saves model when validation loss decrease. #####
        if self.verbose:
            print(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if save_model_dir is not None:
            torch.save(model.state_dict(), save_model_dir + 'checkpoint.pt')
        self.val_loss_min = val_loss


def pe_dis_cal(pe_x, pe_know):
    ##### Calculate positional vector dot product similarity #####
    pe_dis = torch.matmul(pe_x, pe_know.T)
    return pe_dis

class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()
        return

    def forward(self, outs, labels):
        loss = torch.mean(torch.abs((outs - labels) / labels))
        return loss

class MAE(nn.Module):
    def __init__(self):
        super(MAE, self).__init__()
        return

    def forward(self, outs, labels):
        loss = torch.mean(torch.abs(outs - labels))
        return loss

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
        return

    def forward(self, outs, labels):
        loss = torch.mean(torch.square((outs - labels)))
        return loss

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()
        return

    def forward(self, outs, labels):
        loss = torch.sqrt(torch.mean(torch.pow(torch.abs(outs - labels), 2)))
        return loss
