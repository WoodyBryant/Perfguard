import numpy as np
import torch
import torch.nn as nn
import os
from ImportConfig import Config
# from get_data import *
# import joblib
config = Config()

def preprocess_adj(A):
    '''
    Pre-process adjacency matrix
    :param A: adjacency matrix
    :return:
    '''
    batch_size  = A.shape[0]
    final_list = []
    for i in range(batch_size):
        tmp_A = A[i,:,:].squeeze()
        I = np.eye(tmp_A.shape[0])
        A_hat = tmp_A + I # add self-loops
        D_hat_diag = np.sum(A_hat, axis=1)
        D_hat_diag_inv_sqrt = np.power(D_hat_diag, -0.5)
        D_hat_diag_inv_sqrt[np.isinf(D_hat_diag_inv_sqrt)] = 0.
        D_hat_inv_sqrt = np.diag(D_hat_diag_inv_sqrt)
        final = np.dot(np.dot(D_hat_inv_sqrt, A_hat), D_hat_inv_sqrt)
        final_list.append(final)
    final_result = np.stack(final_list,axis = 0)
    return final_result

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim) 
        if acti:
            self.acti = nn.ReLU(inplace=True)
        else:
            self.acti = None

    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)
        
class Attention_Plus(nn.Module):
    def __init__(self, embed_dim, k_dim):
        super(Attention_Plus, self).__init__()
        self.linear1 = nn.Linear(embed_dim, k_dim)
        self.linear2 = nn.Linear(embed_dim, k_dim)

    def forward(self,plan1,plan2):
        # B*M*F，B*M‘*F
        W_s1 = self.linear1(plan1)
        W_s2 = self.linear2(plan2)
        # B*M*M'
        # print(W_s1.shape,W_s2.shape)
        E = torch.sigmoid(torch.bmm(W_s1.float(), W_s2.permute(0,2,1).float()))
        Y = torch.bmm(E,plan2)
        X = (torch.mul(plan1,Y)+Y-plan1)/2.0
        return X,Y
    
class NeuralTensorNetwork(nn.Module):
    def __init__(self, embedding_size, tensor_dim, dropout = 0.5):
        super(NeuralTensorNetwork, self).__init__()
        self.tensor_dim = tensor_dim
        self.T1 = nn.Parameter(torch.Tensor(embedding_size * embedding_size * tensor_dim))
        self.T1.data.normal_(mean=0.0, std=0.02)
        self.W1 = nn.Linear(embedding_size * 2, tensor_dim)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=dropout)
        self.bn = nn.BatchNorm1d(tensor_dim)

    def forward(self, emb1,emb2 ):
        R = self.tensor_Linear(emb1, emb2, self.T1, self.W1)
        R = self.tanh(R)
        R = self.dropout(R)
        return R

    def tensor_Linear(self, emb1, emb2, tensor_layer, linear_layer):
        # b*1*d
        batch_size,_,emb_size = emb1.size()
        emb1_2 = torch.cat((emb1, emb2), dim=2)
        # b*1*tensor_dim
        linear_product = linear_layer(emb1_2).view(batch_size,-1)

        # b*1*d  d*(d*k) -> b*1*(d*k)
        tensor_product = emb1.view(batch_size,emb_size).mm(tensor_layer.view(emb_size, -1)).view(batch_size,1,emb_size,-1)
        # |tensor_product| = (batch_size, unknown_dim * tensor_dim)
        # 1*k*d
        tensor_product = tensor_product.view(batch_size,-1,emb_size).bmm(emb2.view(batch_size,emb_size,_)).view(batch_size,-1)
        tensor_product = tensor_product.contiguous()

        result = tensor_product + linear_product
        # |result| = (batch_size, tensor_dim)
        return self.bn(result)

class PerfGuard(nn.Module):
    def __init__(self,input_dim,embed_dim,tensor_dim,p):
        super(PerfGuard, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, embed_dim)
        self.dropout = nn.Dropout(p)
        self.attention = Attention_Plus(embed_dim = embed_dim,k_dim = embed_dim)
        self.ntn = NeuralTensorNetwork(embedding_size = embed_dim,tensor_dim = tensor_dim)
        self.linear = nn.Linear(tensor_dim, 1)
        self._feature_generator = None

    def forward(self, A1,A2,X1,X2):
        A1 = torch.from_numpy(preprocess_adj(A1)).float().cuda(config.device)
        X1 = self.dropout(torch.from_numpy(X1).cuda(config.device))
        F1 = torch.bmm(A1.float(), X1.float())
        output_gcn1= self.gcn_layer1(F1)

        A2 = torch.from_numpy(preprocess_adj(A2)).float().cuda(config.device)
        X2 = self.dropout(torch.from_numpy(X2).cuda(config.device))
        F2 = torch.bmm(A2.float(), X2.float())
        output_gcn2 = self.gcn_layer1(F2)

        # B*M*N 
        X , Y= self.attention(output_gcn1,output_gcn2)

        # transform the dim of X and Y into B*1*N'
        X = torch.mean(X,axis = 1).view(X.shape[0],1,-1)
        Y = torch.mean(Y,axis = 1).view(Y.shape[0],1,-1)
        ntn_output = self.ntn(X,Y)
        final_output = torch.sigmoid(self.linear(ntn_output).view(-1))
        return final_output
    
            

