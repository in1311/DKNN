import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from model.utils import DIAGNOSIS
from model.loaddataset import DataSet
from model.net import DKNN


def visualize_field_pre(filepath):
    ##### Visualize the predicted random field #####
    data = pd.read_csv(filepath)
    columns=['z', 'predict']
    titles=['true', 'prediction']
    fig = plt.figure(figsize=(15,5))
    for i in range(2):
        ax = plt.subplot(1, 2, i+1)
        plt.imshow(data[columns[i]].values.reshape(100, 100))
        plt.colorbar()
        plt.gca().invert_yaxis()
        plt.title(titles[i], fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath[0: filepath.rfind('/')] + '/RFresult_visualize.png', dpi=300, bbox_inches='tight')

def predict(modelname, datapath, train_info, hidden_neurons, pe_weight, top_n, is_save_result=True):
    d_input, d_model, d_trend = hidden_neurons
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ##### read the sampled dataset #####
    datafile = datapath[datapath.rfind('/')+1:]
    datafilename = datafile[0:datafile.rfind('.')]
    sampledata = pd.read_csv(datapath)
    dataset = DataSet(sampledata)
    data_scaler = dataset.scaler_data()
    data_train_scaler = data_scaler['train']

    known_coods_scaler = torch.from_numpy((data_train_scaler.values[:, 1:3].astype(float))).to(torch.float32).to(device)
    known_feature_scaler = torch.from_numpy(data_train_scaler.values[:, 3:3+d_input].astype(float)).to(torch.float32).to(device)
    known_z = torch.from_numpy(data_train_scaler.values[:, -1].astype(float)).to(torch.float32).to(device)

    ##### read the random field #####
    RFname = datafilename[0: datafilename[0: datafilename.rfind('_')].rfind('_')]
    RFdata = pd.read_csv(datapath[0:datapath[0: datapath.rfind('/')].rfind('/')+1] + RFname + '.csv')
    RFdata_scaler = RFdata.copy()
    if 'z_trend' in RFdata_scaler.columns:
        RFdata_scaler = RFdata_scaler.drop('z_trend',axis=1, inplace=False)

    RFdata_scaler.iloc[:, 1:3] = dataset.scaler_coods.transform(RFdata_scaler.values[:, 1:3])
    RFdata_scaler.iloc[:, 3:-1] = dataset.scaler_features.transform(RFdata_scaler.values[:, 3:-1])
    RFdata_scaler.iloc[:,-1] = dataset.scaler_label.transform(RFdata_scaler['z'].values.reshape(-1,1))
    
    RF_dataloader = DataLoader(RFdata_scaler.values, shuffle=False, batch_size=500, drop_last=False)

    ##### Model initialization #####
    net = DKNN(d_input=d_input, d_model=d_model, d_q=d_model, d_k=d_model, d_v=d_model, known_num=len(known_z),d_trend=d_trend, top_k=top_n, pe_weight=pe_weight)
    net.cal_pe_know(known_feature_scaler, known_coods_scaler)
    net.to(device)
    result_dir = './results/' + modelname + '/' + datafilename  + '/' + train_info + '/'
    net.load_state_dict(torch.load(result_dir + 'checkpoint.pth',map_location=torch.device(device)))

    ##### predict random field #####
    RFresult = []
    with torch.no_grad():
        net.eval()
        for i in tqdm(RF_dataloader):
            i = i.to(torch.float32)
            input_feature = i[:, 3:3+d_input].to(device)
            input_feature[:,-1] = 0
            input_coods = i[:, 1:3].to(device)
            if net.pe_val is not None:
                input_pe = net.pe_val[i[:,0].type(torch.long)]
            else:
                input_pe = net.position_rep(input_feature, input_coods[:, 0], input_coods[:, 1])
            output, _, _ = net(input_coods, input_feature, input_pe, known_coods_scaler, known_feature_scaler, known_z)
            RFresult.extend(output.cpu().detach().numpy())

    ##### diagnose the output #####
    RFresult_inverse = dataset.scaler_label.inverse_transform(np.array(RFresult).reshape(-1,1))
    RFtarget_inverse = dataset.scaler_label.inverse_transform(np.array(RFdata_scaler['z']).reshape(-1,1))
    RF_diag_inverse = DIAGNOSIS(RFresult_inverse, RFtarget_inverse)
    RF_rmse_inverse, RF_mse_inverse, RF_mae_inverse, RF_mape_inverse = RF_diag_inverse.get()
    RF_diag = DIAGNOSIS(np.array(RFresult), np.array(RFdata_scaler['z']))

    print('RF_rmse:{:.4f}, RF_mse:{:.4f}, RF_mae:{:.4f}, RF_mape:{:.4f}, RF_R2:{:.4f}'.format(RF_diag.v_rmse, RF_diag.v_mse, RF_diag.v_mae, RF_diag.v_mape, RF_diag.v_r2))
    print('Inverse')
    print('MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%, RF_mse:{:.4f}, RF_R2:{:.4f}'.format(RF_mae_inverse, RF_rmse_inverse, RF_mape_inverse*100, RF_mse_inverse, RF_diag_inverse.v_r2))
    
    if is_save_result is True:
        ##### save the predict result and diagnoses #####
        RFdata['predict'] = np.array(RFresult_inverse)
        RFdata.to_csv(result_dir + 'RFresult.csv', index=False)

        with open(result_dir + 'RFresult_diag.txt', 'w') as f:
            f.write('-----------Diagnosis-------------')
            f.write('\rRF_rmse:{:.4f}, RF_mse:{:.4f}, RF_mae:{:.4f}, RF_mape:{:.4f}'.format(RF_diag.v_rmse, RF_diag.v_mse, RF_diag.v_mae, RF_diag.v_mape))
            f.write('\rinverse: MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%, RF_mse:{:.4f}, RF_R2:{:.4f}'.format(RF_mae_inverse, RF_rmse_inverse, RF_mape_inverse*100, RF_mse_inverse, RF_diag_inverse.v_r2))
    
    # os.remove(result_dir + 'checkpoint.pth')
    return [RF_mae_inverse, RF_rmse_inverse, RF_mape_inverse], result_dir + 'RFresult.csv'