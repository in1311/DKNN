import datetime, os 
import pandas as pd
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.utils import DIAGNOSIS, EarlyStopping
from model.loaddataset import DataSet
from model.net import DKNN, RMSE, MSE, MAE, MAPE


def train(modelname, datapath, 
          batch_size=128, lr=0.0001, hidden_neurons=[4, 256, 16], pe_weight=0.8, top_k=16, 
          loss_type='rmse', optim_type='adam', if_summary=False, if_save_model=False):
    
    d_input, d_model, d_trend = hidden_neurons

    start_time = datetime.datetime.now()
    train_info = start_time.strftime("%m%d_%H%M%S")

    datafile = datapath[datapath.rfind('/')+1:]
    datafilename = datafile[0:datafile.rfind('.')]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    save_dir = './results/' +  modelname + '/' + datafilename + '/' + train_info
    
    ##### read and scaler data #####
    data = pd.read_csv(datapath)
    dataset = DataSet(data)
    data_scaler = dataset.scaler_data()
    data_train_scaler = data_scaler['train']
    data_val_scaler = data_scaler['val']
    print('data_train:')
    print(dataset.get_data()[0].describe())
    print('data_train_scaler:')
    print(data_train_scaler.describe())

    train_dataloader = DataLoader(data_train_scaler.values.astype(float), shuffle=True, batch_size=batch_size, drop_last=False)
    val_dataloader = DataLoader(data_val_scaler.values.astype(float), shuffle=False, batch_size=batch_size, drop_last=False) 

    known_coods_scaler = data_train_scaler.values[:, 1:3]
    known_feature_scaler = data_train_scaler.values[:, 3:3+d_input]
    known_z = data_train_scaler.values[:, -1]
    known_coods_scaler = torch.from_numpy((known_coods_scaler.astype(float))).to(torch.float32).to(device)
    known_feature_scaler = torch.from_numpy(known_feature_scaler.astype(float)).to(torch.float32).to(device)
    known_z = torch.from_numpy(known_z.astype(float)).to(torch.float32).to(device)

    ##### define DKNN model #####
    model = DKNN(d_input=d_input, d_model=d_model, d_q=d_model, d_k=d_model, d_v=d_model, known_num=dataset.train_num, d_trend=d_trend, top_k=top_k, pe_weight=pe_weight)
    model.cal_pe_know(known_feature_scaler, known_coods_scaler)
    model.cal_pe_val(torch.from_numpy(data_val_scaler.values[:, 3:3+d_input].astype(float)).to(torch.float32).to(device),
                     torch.from_numpy(data_val_scaler.values[:, 1:3].astype(float)).to(torch.float32).to(device)) 
    model.to(device)

    ##### loss function ##### 
    if loss_type=='mae':
        criterion = MAE()
    elif loss_type=='mse':
        criterion = MSE()
    elif loss_type=='rmse':
        criterion = RMSE()
    elif loss_type=='mape':
        criterion = MAPE()
    
    ##### optimizer #####
    if optim_type=='adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    elif optim_type=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.8,  verbose=True, patience=5, min_lr=1e-6)

    ##### tensorboard summary #####
    if if_summary:
        writer = SummaryWriter(save_dir + '/summary/')
    
    ##### early stopping #####
    patience = 30
    early_stopping = EarlyStopping(patience, verbose=True)

    ##### start train #####
    print(train_info + '  start')
    epoch = 300
    total_train_step = 0
    total_val_step = 0
    save_model_dir = None
    min_loss = float('inf')
    best_epoch = float('inf')
    best_rmse_inverse,best_mae_inverse,best_mape_inverse = float('inf'),float('inf'),float('inf')
    
    for e in tqdm(range(epoch)):
        ##### training #####
        total_train_loss = 0
        train_num = 0
        model.train()  
        for step, i in enumerate(train_dataloader):
            i = i.to(torch.float32)
            input_pe = model.pe_know[i[:,0].type(torch.long)]
            input_coods = i[:, 1:3].to(device)
            input_feature = i[:, 3:3+d_input].to(device)
            target = i[:, -1].to(device)
            optimizer.zero_grad()
            output, _, _ = model(input_coods, input_feature, input_pe, known_coods_scaler, known_feature_scaler, known_z)
            ##### calculate loss and optimize #####
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            ##### diagnose the output #####
            diag = DIAGNOSIS(output, target)
            l_rmse, l_mse, l_mae, l_mape = diag.get()
            if if_summary:
                total_train_step = total_train_step + 1
                writer.add_scalar("train_loss", loss, total_train_step)
                writer.add_scalar("train_rmse", l_rmse, total_train_step)
                writer.add_scalar("train_mae", l_mae, total_train_step)
                writer.add_scalar("train_mape", l_mape, total_train_step)
            total_train_loss = total_train_loss + loss.item()
            train_num +=1
            if step % 3 == 0:
                print('\n--------train epoch {},step {},loss: {:.5f}, RMSE:{:.5f},MSE:{:.5f}, MAE:{:.5f}, MAPE:{:.5f}--'
                        .format(e, step, loss.item(), l_rmse, l_mse, l_mae, l_mape))
        total_train_loss = total_train_loss/train_num
        if if_summary:
            writer.add_scalar("total_train_loss", total_train_loss, e)

        ##### validating #####
        total_vali_loss = 0
        l2p = 0
        l3p = 0
        l4p = 0
        lrmse_inverse,lmae_inverse,lmape_inverse = 0,0,0
        val_num = 0
        with torch.no_grad():
            model.eval()
            for step, i in enumerate(val_dataloader):
                i = i.to(torch.float32)
                input_pe = model.pe_val[i[:,0].type(torch.long)]
                input_coods = i[:, 1:3].to(device)
                input_feature = i[:, 3:3+d_input].to(device)
                if d_input == 4:
                    input_feature[:,-1] = 0
                target = i[:, -1].to(device)
                output, _, _ = model(input_coods, input_feature, input_pe, known_coods_scaler, known_feature_scaler, known_z)
                ###### diagnose the output ######
                val_loss = criterion(output, target)
                val_diag = DIAGNOSIS(output, target)
                output_inverse = dataset.scaler_label.inverse_transform(output.cpu().reshape(-1, 1))
                target_inverse = dataset.scaler_label.inverse_transform(target.cpu().reshape(-1, 1))
                val_diag_inverse = DIAGNOSIS(output_inverse, target_inverse)
                t_rmse, t_mse, t_mae, t_mape = val_diag.get()
                total_vali_loss += val_loss.item()
                l2p += t_rmse
                l3p += t_mse
                l4p += t_mape
                lrmse_inverse += val_diag_inverse.v_rmse
                lmae_inverse += val_diag_inverse.v_mae
                lmape_inverse += val_diag_inverse.v_mape
                total_val_step = total_val_step + 1
                val_num = val_num + 1
            final_loss = total_vali_loss / val_num
            final_rmse = l2p / val_num
            final_mse = l3p / val_num
            final_mape = l4p / val_num
            final_rmse_inverse = lrmse_inverse / val_num
            final_mae_inverse = lmae_inverse / val_num
            final_mape_inverse = lmape_inverse / val_num
            ##### adjust learning rate #####
            scheduler.step(final_loss)
            if if_summary:
                writer.add_scalar("total_vali_loss", final_loss, e)
                writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], e)
            end_time = datetime.datetime.now()
            print('--------vali loss: {:.5f}, RMSE:{:.5f},MSE:{:.5f}, MAPE:{:.5f}, RMSE_inv:{:.5f}, epoch time:{:.2f}m--'
                .format(final_loss, final_rmse, final_mse, final_mape, final_rmse_inverse, (end_time-start_time).seconds/60))
            if final_loss < min_loss:
                min_loss = final_loss
                best_epoch = e
                best_rmse_inverse = final_rmse_inverse
                best_mae_inverse = final_mae_inverse
                best_mape_inverse = final_mape_inverse
                ##### save the best model parameters and info #####
                if if_save_model:
                    save_model_dir = save_dir + '/'
                    if not os.path.exists(save_model_dir):
                        os.makedirs(save_model_dir)
                    torch.save(model.state_dict(), save_model_dir + 'checkpoint.pth')
                    end_time = datetime.datetime.now()
                    with open(save_model_dir + 'checkpoint_info.txt', 'w') as f:
                        f.write('-----------model_info-------------')
                        f.write('\rmodel name:  dknn\r')
                        f.write('train info:  {}\r'.format(train_info))
                        f.write('min loss:  {}\r'.format(min_loss))
                        f.write('best_epoch:  {}\r'.format(best_epoch))
                        f.write('best_rmse_inverse:  {}\r'.format(best_rmse_inverse))
                        f.write('best inverse MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%\r'
                            .format(best_mae_inverse,best_rmse_inverse,best_mape_inverse*100))
                        f.write('spend time: {} minutes\r'.format((end_time-start_time).seconds/60))
                        f.write('epoch: {}\r'.format(e))
            ##### early stopping #####
            early_stopping(final_loss, model, save_model_dir=None)
            if early_stopping.early_stop:
                print("Early stopping")
                break
    if if_summary:
        writer.close()
    ##### end train and print result #####
    print(train_info + '  end')
    print('******** {} *********'.format(datafilename))
    print('final min loss:', min_loss)
    print('best_epoch:', best_epoch)
    print('best inverse MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}%'
        .format(best_mae_inverse,best_rmse_inverse,best_mape_inverse*100))
    end_time = datetime.datetime.now()
    print('spend time: {:3f} minutes'.format((end_time-start_time).seconds/60))
    best_inverse = [best_mae_inverse,best_rmse_inverse,best_mape_inverse]
    return train_info, min_loss, best_epoch, best_inverse