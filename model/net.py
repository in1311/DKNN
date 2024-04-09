import torch
import torch.nn as nn
import math
from utils.utils import pe_dis_cal


class DKNN(nn.Module):
    def __init__(self, d_input, d_model, known_num, d_trend, top_k, pe_weight):
        super(DKNN, self).__init__()
        """
        The initializer.

        ----------
        Parameters
        d_input: int
            The dimension of input features.
        d_model: int
            The dimension of embedding space (hidden units of each dense layer).
        known_num: int
            The number of known points (observed locations).
        d_trend: int
            The dimension of trend vector.
        top_k: int
            top k nearest neighbors.
        pe_weight: float
            The weight of positional vector.
        """
        self.d_model = d_model
        self.d_trend = d_trend
        self.known_num = known_num
        self.AttRe = Attribute_Rep(d_input, d_model)
        self.Pgrn = PGRN(d_model)
        self.Ssan = SSAN(d_model, top_k, pe_weight)
        self.MetaPN = MetaPN(d_model, d_trend)
        self.pe_know = None
        self.pe_unknow = None
    
    def kriging_decoder(self, z_know, cov_know, cov_unknow, trend_unknow, trend_know):
        
        ##### kriging decoder and interpolation  #####
        device = str(cov_know.device) 
        k = trend_know.shape[-1]
        batch_size = cov_unknow.shape[0]

        # Kriging system matrix
        sys_mat_know = torch.zeros(self.known_num + k, self.known_num + k).to(device)
        sys_mat_unknow = torch.zeros(batch_size, self.known_num + k).to(device)

        sys_mat_know[0:self.known_num, 0:self.known_num] = cov_know
        sys_mat_know[0:self.known_num, self.known_num:self.known_num + k] = trend_know
        sys_mat_know[self.known_num:self.known_num + k, 0:self.known_num] = trend_know.T

        sys_mat_unknow[0:batch_size, 0:self.known_num] = cov_unknow
        sys_mat_unknow[0:batch_size, self.known_num:self.known_num + k] = trend_unknow
        
        ##### Solving the K-equation #####
        try:
            sys_mat_know_inv = torch.linalg.inv(sys_mat_know)
        except:
            sys_mat_know_inv = torch.linalg.pinv(sys_mat_know)
        lamda = torch.matmul(sys_mat_unknow, sys_mat_know_inv.T)
        lamda = lamda[:, :-k]
        self.lamda = lamda
        
        ##### Estimated based on interpolation formula #####
        # Residual output
        residual_pre = torch.matmul(lamda, z_know)

        # Trend output
        trend_pre = torch.sum(trend_unknow,-1) / k

        # interpolation output
        prediction = residual_pre + trend_pre

        return prediction, trend_pre

    def cal_pe_know(self, feature_know, know_coods):
        self.pe_know = self.position_rep(feature_know, know_coods[:,0], know_coods[:,1])
        return 
    
    def cal_pe_unknow(self, feature, coods):
        self.pe_unknow = self.position_rep(feature, coods[:,0], coods[:,1])
        return 

    def reset_parameters(self):
        # Initialize the weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.normal_(m.weight, mean=0, std=0.1)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def position_rep(self, input_vec, coodx, coody):
        ##### Positional representation #####
        d = self.d_model
        if d % 4 != 0:
            print('d % 4 != 0')
            return 0
        device = str(input_vec.device)
        batch_size = input_vec.shape[0]
        # Positional encoding
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
    
    def forward(self, input_coods, input_features, input_pe, know_coods, know_features):
        """ 
        Forward process of DKNN

        ----------
        Parameters
        input_coods: Tensor with shape [batch_size, 2]
        input_features: Tensor with shape [batch_size, d_input]
        input_pe: Tensor with shape [batch_size, d_model]
        know_coods: Tensor with shape [know_num, 2]
        know_features: Tensor with shape [know_num, d_input]

        -------
        Returns
        output: Tensor with shape [batch_size, 1]
        out_trend: Tensor with shape [batch_size, 1]
        """

        ##### attribute representation #####
        vec_unknow = self.AttRe(input_features)  # attribute representation of unknown points
        vec_know = self.AttRe(know_features)  # attribute representation of known points

        ##### SSAN #####
        cov_unknow = self.Ssan(vec_unknow, vec_know, input_pe, self.pe_know)  # covariance matrix of unknown points
        cov_know = self.Ssan(vec_know, vec_know, self.pe_know, self.pe_know)  # covariance matrix of known points
        # Zero the diagonal of the matrix cov_know
        cov_know = cov_know - torch.diag_embed(torch.diag(cov_know))
        
        ##### PGRN #####
        known_z = torch.mean(self.Pgrn(vec_know, self.pe_know),-1)  # PGRN output of known points
            
        ##### Meta-PN #####
        trend_know = self.MetaPN(know_coods, self.pe_know)  # trend matrix of known points
        trend_unknow = self.MetaPN(input_coods, input_pe)  # trend matrix of unknown points
        
        ##### kriging decoder and interpolation #####
        output, out_trend = self.kriging_decoder(known_z, cov_know, cov_unknow, trend_unknow, trend_know)
        
        return output, out_trend


class MetaPN(nn.Module):
    ##### meta polynomial network #####
    def __init__(self, d_model, d_trend):
        """
        The initializer.

        ----------
        Parameters
        d_model: int
            The dimension of hidden units of each dense layer.
        d_trend: int
            The dimension of trend vector.
        """
        super(MetaPN, self).__init__()
        self.d_model = d_model
        self.d_trend = d_trend
        self.d_pe = d_model
        self.ml1_w = Dense(self.d_pe, d_model * 2)
        self.ml1_b = Dense(self.d_pe, d_model)
        self.ml2_w = Dense(self.d_pe, d_model * d_model)
        self.ml2_b = Dense(self.d_pe, d_model)
        self.ml3_w = Dense(self.d_pe, d_model * d_trend)
        self.ml3_b = Dense(self.d_pe, d_trend)
        self.relu = nn.PReLU()
        
    def forward(self, coods, pe):
        """ 
        Forward process of MetaPN

        ----------
        Parameters
        coods: Tensor with shape [batch_size, 2] or [know_num, 2]
        pe: Tensor with shape [batch_size, d_pe] or [know_num, d_pe]

        -------
        Returns
        output: Tensor with shape [batch_size, d_trend] or [know_num, d_trend]
        """
        batch_size = pe.shape[0]
        # layer1
        w1 = self.ml1_w(pe).reshape(batch_size, 2, self.d_model)
        b1 = self.ml1_b(pe).reshape(batch_size, 1, self.d_model)
        x = torch.bmm(coods.unsqueeze(1), w1) + b1  
        x = self.relu(x)
        # layer2
        w2 = self.ml2_w(pe).reshape(batch_size, self.d_model, -1) 
        b2 = self.ml2_b(pe).reshape(batch_size, 1, -1) 
        x = torch.bmm(x, w2) + b2  
        x = self.relu(x)
        # layer_reg
        w3 = self.ml3_w(pe).reshape(batch_size, self.d_model, -1) 
        b3 = self.ml3_b(pe).reshape(batch_size, 1, -1) 
        x = torch.bmm(x, w3) + b3  
        output = x.squeeze()
        return output


class SSAN(nn.Module):
    ##### spatial sparse attention network #####
    def __init__(self, d_model, top_k, pe_weight):
        """
        The initializer.

        ----------
        Parameters
        d_model: int
            The dimension of hidden units of each dense layer.
        top_k: int
            Top k nearest neighbors.
        pe_weight: float
            The weight of positional vector.
        """
        super(SSAN, self).__init__()
        self.d_model = d_model
        self.d_q = d_model
        self.d_k = d_model
        self.d_v = d_model
        self.map_query = nn.Linear(self.d_model, self.d_q, bias=False)
        self.map_key = nn.Linear(self.d_model, self.d_k, bias=False)

        self.top_k = top_k
        self.pe_weight= nn.Parameter(torch.Tensor([pe_weight]), requires_grad=False)

    def sparse(self, pe_q, pe_k, att_scores):
        ##### sparsification #####
        pe_sims = pe_dis_cal(pe_q, pe_k)
        pe_sims = pe_sims / math.sqrt(self.d_q)
        if type(self.top_k) == float:
            self.top_k = int(len(pe_k) * self.top_k)
        if self.top_k!=0:
            indices_to_remove = pe_sims < torch.topk(pe_sims, self.top_k)[0][..., -1, None]  
            att_scores[indices_to_remove] = 0  
            # pe_sims[indices_to_remove] = 0  
        return att_scores, pe_sims

    def forward(self, ae_q, ae_kv, pe_q, pe_kv):
        """ 
        Forward process of SSAN

        ----------
        Parameters
        ae_q: Tensor with shape [batch_size, d_model] or [know_num, d_model]
        ae_kv: Tensor with shape [know_num, d_model]
        pe_q: Tensor with shape [batch_size, d_model] or [know_num, d_model]
        pe_kv: Tensor with shape [know_num, d_model]

        -------
        Returns
        att_scores: Tensor with shape [batch_size, know_num] or [know_num, know_num]
        """
        pe_q1 = pe_q * self.pe_weight
        pe_kv1 = pe_kv * self.pe_weight
        ae_q = ae_q * torch.abs(1-self.pe_weight)
        ae_kv = ae_kv * torch.abs(1-self.pe_weight)

        residual = ae_q + pe_q1
        query = self.map_query(ae_q + pe_q1) + residual

        residual = ae_kv + pe_kv1
        key = self.map_key(ae_kv + pe_kv1) + residual

        ##### cal attention score #####
        att_scores = torch.matmul(query, key.permute(1, 0)) / math.sqrt(self.d_model)
        att_scores, _ = self.sparse(pe_q, pe_kv, att_scores)
        att_scores = att_scores.squeeze()

        return att_scores


class PGRN(nn.Module):
    ##### position-aware gated residual network #####
    def __init__(self, d_model):
        """
        The initializer.

        ----------
        Parameters
        d_model: int
            The dimension of embedding space.
        """
        super(PGRN, self).__init__()
        self.dense1 = Dense(d_model, d_model)
        self.dense12 = Dense(d_model, d_model)
        self.dense2 = Dense(d_model, d_model)
        self.dense3 = Dense(d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        self.bn = nn.BatchNorm1d(d_model)

    def forward(self, ae, pe):
        """ 
        Forward process of PGRN

        ----------
        Parameters
        ae: Tensor with shape [know_num, d_model]
        pe: Tensor with shape [know_num, d_model]

        -------
        Returns
        output: Tensor with shape [know_num, d_model]
        """
        residual = ae
        ae = self.dense1(ae) + self.dense12(pe)
        ae = self.dense2(ae) * self.sigmoid(self.dense3(ae)) + residual
        output = self.bn(ae)
        return output
    
        
class Attribute_Rep(nn.Module):
    ##### Attribute representation (numerical) #####
    def __init__(self, d_input, d_model):
        """
        The initializer.

        ----------
        Parameters
        d_input: int
            The dimension of input features.
        d_model: int
            The dimension of embedding space.
        """
        super(Attribute_Rep, self).__init__()
        self.dense1 = Dense(d_input, d_model)
        self.dense2 = Dense(d_model, d_model)
        self.dense3 = Dense(d_model, d_model)
        self.relu = nn.PReLU()

    def forward(self, input_features):
        """ 
        Forward process of Attribute_Rep

        ----------
        Parameters
        input_features: Tensor with shape [batch_size, d_input] or [know_num, d_input]

        -------
        Returns
        output: Tensor with shape [batch_size, d_model] or [know_num, d_model]
        """
        x = self.dense1(input_features)
        x = self.relu(x)
        x = self.dense2(x)
        x = self.relu(x)
        output = self.dense3(x)
        return output


class Dense(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Dense, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.fc = nn.Linear(self.in_feature, self.out_feature)

    def forward(self, x):
        x = self.fc(x)
        return x