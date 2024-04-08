import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.utils import DIAGNOSIS
from utils.loaddataset import DataSet
from model.net import DKNN


def visualize_field_pre(filepath):
    ##### Visualize the predicted random field #####
    data = pd.read_csv(filepath)
    columns=['target', 'predict']
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

def predict(modelname, datapath, train_info, hidden_neurons, pe_weight, top_k, is_save_result=True):
    d_input, d_model, d_trend = hidden_neurons
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ##### read the sampled dataset #####
    # read the sampled dataset
    sampledata = pd.read_csv(datapath)

    # get the name of sampled dataset
    datafile = datapath[datapath.rfind('/')+1:]
    datafilename = datafile[0:datafile.rfind('.')]

    ###### preprocess the sampled dataset #####
    # Scaling the data
    dataset = DataSet(sampledata)
    data_scaler = dataset.scaler_data()

    # Take the train set as known points (observed locations) and transform to tensor
    data_train_scaler = data_scaler['train']
    known_coods_scaler = torch.from_numpy((data_train_scaler.values[:, 1:3].astype(float))).to(torch.float32).to(device)
    known_feature_scaler = torch.from_numpy(data_train_scaler.values[:, 3:3+d_input].astype(float)).to(torch.float32).to(device)

    ##### read the random field #####
    # get the name of radom field file
    RFname = datafilename[0: datafilename[0: datafilename.rfind('_')].rfind('_')]

    # read the random field
    RFdata = pd.read_csv('./Data/random_field/' + RFname + '.csv')
    print(RFdata.head())

    ##### Process the random field data
    RFdata_scaler = RFdata.copy()

    # Remove the redundant column
    if 'trend' in RFdata_scaler.columns:
        RFdata_scaler = RFdata_scaler.drop('trend',axis=1, inplace=False)

    # scale the data
    RFdata_scaler.iloc[:, 1:3] = dataset.scaler_coods.transform(RFdata_scaler.values[:, 1:3])
    RFdata_scaler.iloc[:, 3:-1] = dataset.scaler_features.transform(RFdata_scaler.values[:, 3:-1])
    RFdata_scaler.iloc[:,-1] = dataset.scaler_label.transform(RFdata_scaler.values[:, -1].reshape(-1,1))

    # get the dataloader
    RF_dataloader = DataLoader(RFdata_scaler.values, shuffle=False, batch_size=500, drop_last=False)
    print(RFdata_scaler.describe())

    ##### Load the best model parameters #####
    save_dir = './results/' + modelname + '/' + datafilename  + '/' + train_info
    
    ##### Model initialization #####
    # Define the DKNN model 
    net = DKNN(d_input=d_input, d_model=d_model, known_num=dataset.train_num, d_trend=d_trend, top_k=top_k, pe_weight=pe_weight)

    # load the best model parameters
    net.load_state_dict(torch.load(save_dir + '/checkpoint.pth',map_location=torch.device(device)))

    # Calculate positional embedding before training to increase the speed of training
    net.cal_pe_know(known_feature_scaler, known_coods_scaler)
    net.cal_pe_unknow(torch.from_numpy(RFdata_scaler.values[:, 3:3+d_input].astype(float)).to(torch.float32).to(device),
                        torch.from_numpy(RFdata_scaler.values[:, 1:3].astype(float)).to(torch.float32).to(device)) 
    net.to(device)

    ##### predict random field #####
    RFoutput = []
    with torch.no_grad():
        net.eval()  # set the model to evaluation mode
        for i in tqdm(RF_dataloader):
            # model input
            i = i.to(torch.float32)
            input_feature = i[:, 3:3+d_input].to(device)
            input_feature[:,-1] = 0
            input_coods = i[:, 1:3].to(device)
            if net.pe_unknow is not None:
                input_pe = net.pe_unknow[i[:,0].type(torch.long)]
            else:
                input_pe = net.position_rep(input_feature, input_coods[:, 0], input_coods[:, 1])
            # model execution
            output, _ = net(input_coods, input_feature, input_pe, known_coods_scaler, known_feature_scaler)
            RFoutput.extend(output.cpu().detach().numpy())

    # reverse the output
    RFoutput_inverse = dataset.scaler_label.inverse_transform(np.array(RFoutput).reshape(-1,1))

    # diagnose the reversed output
    RF_diag_inverse = DIAGNOSIS(RFoutput_inverse, RFdata['target'].values.reshape(-1,1))
    RF_rmse_inverse, RF_mse_inverse, RF_mae_inverse, RF_mape_inverse = RF_diag_inverse.get()

    # print diagnostic results
    print('MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%, RF_mse:{:.4f}, RF_R2:{:.4f}'.format(RF_mae_inverse, RF_rmse_inverse, RF_mape_inverse*100, RF_mse_inverse, RF_diag_inverse.v_r2))
    
    if is_save_result is True:
        ##### save the predict result and diagnostic results #####
        # save the predict result
        RFdata['predict'] = np.array(RFoutput_inverse)
        RFdata.to_csv(save_dir + '/RFresult.csv', index=False)

        # save diagnostic results
        with open(save_dir + '/RFresult_diag.txt', 'w') as f:
            f.write('-----------Diagnosis-------------')
            f.write('\r MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%, RF_mse:{:.4f}, RF_R2:{:.4f}'.format(RF_mae_inverse, RF_rmse_inverse, RF_mape_inverse*100, RF_mse_inverse, RF_diag_inverse.v_r2))

        print(RFdata.head())
    
    # os.remove(result_dir + 'checkpoint.pth')
    return [RF_mae_inverse, RF_rmse_inverse, RF_mape_inverse], save_dir + '/RFresult.csv'