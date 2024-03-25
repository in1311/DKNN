from model.train import train
from model.predict import predict, visualize_field_pre


if __name__ == '__main__':

    datafile ='SRF5s1_112_rand0.05s10_s5.csv'  #  sampled dataset
    
    ##### hyperparameter #####
    batch_size = 128  # batch size
    lr = 0.0001  # learning rate
    hidden_neuron = [4, 256, 16]  # [input dimension, model dimension, trend dimension]
    pe_weight = 0.8  # weight of positional vector 
    top_k = 400  # top k nearest neighbors
    loss_type = 'rmse'  # loss function type
    optim_type='adam'  # optimizer type
    if_summary = True  # if save the training summary or not
    if_save_model = True  # if save the best model or not

    ##### train #####
    modelname = 'DKNN'
    datapath = './Data/random_field/dataset/' + datafile
    train_info, min_loss, best_epoch, best_inverse = train(modelname, datapath, batch_size, lr, hidden_neuron, 
                                                            pe_weight, top_k, loss_type=loss_type, optim_type=optim_type, 
                                                            if_summary=if_summary, if_save_model=if_save_model) 
    with open('./results/train_log.txt', 'a', encoding='utf-8') as f:
        f.write('\rtrain_info: {} /**/ datafile: {}\rmin_loss({}): {:.5f}; best_epoch: {:.5f}; best_rmse_inverse: {:.5f}; best_inverse MAE/RMSE/MAPE: {:.2f}/{:.2f}/{:.2f}% '\
        '/**/ hidden_neurons: {}; pe_weight: {}; model: {}; batch_size: {}; lr: {}; optim_type: {}; top_n:{}\r'
                .format(train_info, datafile, loss_type, min_loss, best_epoch, best_inverse[1], best_inverse[0], best_inverse[1], best_inverse[2]*100,
                        hidden_neuron, pe_weight, modelname, batch_size, lr, optim_type, top_k))
    
    ##### predict #####
    if if_save_model is True:
        #  load the best model and predict the whole random field
        RF_diag, result_path = predict(modelname, datapath, train_info, hidden_neuron, pe_weight=pe_weight, top_n=top_k, is_save_result=True)
        #  visualize the predicted random field
        visualize_field_pre(result_path)