import torch
import torch.nn as nn
import math
from utils import pe_dis_cal


class DKNN(nn.Module):
    def __init__(self, d_input, d_model, d_q, d_k, d_v, knownn_num, d_trend, top_n):
        super(DKNN, self).__init__()
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.knownn_num = knownn_num
        self.d_trend = d_trend
        self.d_pe = d_q
        self.AttRe = Attribute_Rep(d_input, d_model)
        self.Pgrn = PGRN(d_model, d_q)
        self.Ssan = SSAN(d_model, d_q, d_k, d_v, top_n)
        self.MetaPN = MetaPN(d_pe=self.d_pe, d_model=d_model, d_trend=d_trend)
        self.pe_know = None
        self.pe_val = None
    
    def kriging_decoder(self, z_know, var_know, var_pre, pe_pre_trend, pe_know_drift):
        device = str(var_know.device) 
        k = pe_know_drift.shape[-1]
        batch_size = var_pre.shape[0]
        sys_mat_know = torch.zeros(self.knownn_num + k, self.knownn_num + k).to(device)
        sys_mat_pre = torch.zeros(batch_size, self.knownn_num + k).to(device)

        sys_mat_know[0:self.knownn_num, 0:self.knownn_num] = var_know
        sys_mat_know[0:self.knownn_num, self.knownn_num:self.knownn_num + k] = pe_know_drift
        sys_mat_know[self.knownn_num:self.knownn_num + k, 0:self.knownn_num] = pe_know_drift.T

        sys_mat_pre[0:batch_size, 0:self.knownn_num] = var_pre
        sys_mat_pre[0:batch_size, self.knownn_num:self.knownn_num + k] = pe_pre_trend
        try:
            sys_mat_know_inv = torch.linalg.inv(sys_mat_know)
        except:
            sys_mat_know_inv = torch.linalg.pinv(sys_mat_know)
        lamda = torch.matmul(sys_mat_pre, sys_mat_know_inv.T)
        lamda = lamda[:, :-k]
        z_pre = torch.matmul(lamda, z_know)
        self.lamda = lamda

        trend_pre = torch.sum(pe_pre_trend,-1) / k

        return z_pre, trend_pre

    def cal_pe_know(self, feature_know, know_coods):
        self.pe_know = self.position_rep(feature_know, know_coods[:,0], know_coods[:,1])
        return 
    
    def cal_pe_val(self, feature, coods):
        self.pe_val = self.position_rep(feature, coods[:,0], coods[:,1])
        return 

    def get_pe_weight(self, weight):
        self.Ssan.pe_weight = nn.Parameter(torch.Tensor([weight]), requires_grad=False)
        print('******** init pe_weight:{} **********'.format(self.Ssan.pe_weight.data))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def position_rep(self, input_vec, coodx, coody):
        d = self.d_pe
        if d % 4 != 0:
            print('d % 4 != 0')
            return 0
        device = str(input_vec.device)
        batch_size = input_vec.shape[0]
        Rmat = torch.zeros((batch_size, d, d)).to(device)
        for each in range(batch_size):
            for i in range(int(d / 4)):
                theta = 10 ** (-4 * i / d) * math.pi / 2
                coodx_ = coodx * theta
                coody_ = coody * theta
                Rmat[each,i * 4,i * 4] = torch.cos(coodx_[each])
                Rmat[each,i * 4,i * 4 + 1] = -torch.sin(coodx_[each])
                Rmat[each,i * 4 + 1,i * 4] = torch.sin(coodx_[each])
                Rmat[each,i * 4 + 1,i * 4 + 1] = torch.cos(coodx_[each])
                Rmat[each,i * 4 + 2,i * 4 + 2] = torch.cos(coody_[each])
                Rmat[each,i * 4 + 2,i * 4 + 3] = -torch.sin(coody_[each])
                Rmat[each,i * 4 + 3,i * 4 + 2] = torch.sin(coody_[each])
                Rmat[each,i * 4 + 3,i * 4 + 3] = torch.cos(coody_[each])
        ones_vec = torch.ones([batch_size, d, 1]).to(device)
        pos_embedding = torch.bmm(Rmat, ones_vec)
        # del Rmat, ones_vec
        return pos_embedding.squeeze() 
    
    def forward(self, input_coods, input_features, input_pe, know_coods, know_features, known_y):
        vec_unknow = self.AttRe(input_features)
        vec_know = self.AttRe(know_features)

        # Ssan
        pe_unknow = input_pe
        var_unknow = self.Ssan(vec_unknow, vec_know, pe_unknow, self.pe_know)
        var_know = self.Ssan(vec_know, vec_know, self.pe_know, self.pe_know)
        
        # Diagonal Zeroing
        a = torch.diag_embed(torch.diag(var_know))
        var_know = var_know - a
        
        # PGRN
        known_z = torch.mean(self.Pgrn(vec_know, self.pe_know),-1)
            
        # Meta-PN
        know_drift = self.MetaPN(know_coods, self.pe_know)
        unknow_drift = self.MetaPN(input_coods, pe_unknow)
        
        out, out_trend = self.kriging_decoder(known_z, var_know, var_unknow, unknow_drift, know_drift)
        
        return out, var_know, out_trend


class MetaPN(nn.Module):
    def __init__(self, d_pe, d_model, d_trend):
        super(MetaPN, self).__init__()
        self.d_model = d_model
        self.d_trend = d_trend
        self.d_pe = d_pe
        self.ml1_w = Fcn(self.d_pe, d_model * 2)
        self.ml1_b = Fcn(self.d_pe, d_model)
        self.ml2_w = Fcn(self.d_pe, d_model * d_model)
        self.ml2_b = Fcn(self.d_pe, d_model)
        self.ml3_w = Fcn(self.d_pe, d_model * d_trend)
        self.ml3_b = Fcn(self.d_pe, d_trend)
        self.relu = nn.PReLU()
        
    def forward(self, coods, pes):
        batch_size = pes.shape[0]
        # layer1
        w1 = self.ml1_w(pes).reshape(batch_size, 2, self.d_model)
        b1 = self.ml1_b(pes).reshape(batch_size, 1, self.d_model)
        x = torch.bmm(coods.unsqueeze(1), w1) + b1  
        x = self.relu(x)
        # layer2
        w2 = self.ml2_w(pes).reshape(batch_size, self.d_model, -1) 
        b2 = self.ml2_b(pes).reshape(batch_size, 1, -1) 
        x = torch.bmm(x, w2) + b2  
        x = self.relu(x)
        # layer_reg
        w3 = self.ml3_w(pes).reshape(batch_size, self.d_model, -1) 
        b3 = self.ml3_b(pes).reshape(batch_size, 1, -1) 
        x = torch.bmm(x, w3) + b3  
        return x.squeeze()


class SSAN(nn.Module):
    def __init__(self, d_model, q, k, v, top_n):
        super(SSAN, self).__init__()
        self.d_model = d_model
        self.d_q = q
        self.d_k = k
        self.d_v = v

        self.map_query = nn.Linear(self.d_model, self.d_q, bias=False)
        self.map_key = nn.Linear(self.d_model, self.d_k, bias=False)
        self.map_value = nn.Linear(self.d_model, self.d_v)

        self.Att = ATTENTION(q=self.d_q, k=self.d_k, v=self.d_v)
        self.top_n = top_n
        self.pe_weight = nn.Parameter(torch.FloatTensor([0.8]), requires_grad=False)
        
    def sparse(self, pe_q, pe_k, att_scores):
        pe_sims = pe_dis_cal(pe_q, pe_k)
        pe_sims = pe_sims / math.sqrt(self.d_q)
        if type(self.top_n) == float:
            self.top_n = int(len(pe_k) * self.top_n)
        if self.top_n!=0:
            indices_to_remove = pe_sims < torch.topk(pe_sims, self.top_n)[0][..., -1, None]  
            att_scores[indices_to_remove] = 0  
            # pe_sims[indices_to_remove] = 0  
        return att_scores, pe_sims

    def forward(self, x_q, x_kv, pe_q, pe_kv, if_outscore=True):

        pe_q1 = pe_q * self.pe_weight
        pe_kv1 = pe_kv * self.pe_weight
        x_q = x_q * torch.abs(1-self.pe_weight)
        x_kv = x_kv * torch.abs(1-self.pe_weight)

        residual = x_q + pe_q1
        query = self.map_query(x_q + pe_q1) + residual

        residual = x_kv + pe_kv1
        key = self.map_key(x_kv + pe_kv1) + residual

        value = self.map_value(x_kv + pe_kv1)

        # cal attention score
        att_scores = self.Att(query, key, value, if_outscore)
        att_scores, _ = self.sparse(pe_q, pe_kv, att_scores)

        return att_scores.squeeze()


class ATTENTION(nn.Module):
    def __init__(self, q, k, v):
        super(ATTENTION, self).__init__()
        self.dim_q = q
        self.dim_k = k
        self.dim_v = v
        self.softmax = nn.Softmax(dim=-1)
        self.score = None

    def forward(self, query, key, value, if_outscore=True):
        score = torch.matmul(query, key.permute(1, 0)) / math.sqrt(self.dim_q)
        if if_outscore is True:
            return score
        else:
            score = self.softmax(score)
            return torch.matmul(score, value.permute(1, 0))


class PGRN(nn.Module):
    def __init__(self, d_input, d_out):
        super(PGRN, self).__init__()
        self.fcn1 = nn.Linear(d_input, d_out)
        self.fcn12 = nn.Linear(d_input, d_out)
        self.fcn2 = nn.Linear(d_out, d_out)
        self.fcn3 = nn.Linear(d_out, d_out)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(d_out)

    def forward(self, x, pe):
        residual = x
        x = self.fcn1(x) + self.fcn12(pe)
        x = self.fcn2(x) * self.sigmoid(self.fcn3(x)) + residual
        x = self.bn(x)
        return x
    
        
class Attribute_Rep(nn.Module):
    def __init__(self, d_feature, d_model):
        super(Attribute_Rep, self).__init__()
        self.linear1 = Fcn(d_feature, d_model)
        self.linear2 = Fcn(d_model, d_model)
        self.linear3 = Fcn(d_model, d_model)
        self.relu = nn.PReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


class Fcn(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Fcn, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.fc = nn.Linear(self.in_feature, self.out_feature)

    def forward(self, x):
        x = self.fc(x)
        return x


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
