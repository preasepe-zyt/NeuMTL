import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# torch.Size([128, 67, 376]) torch.Size([128, 96, 107])
#  k = 32, protein_hidden_dim = 107, drug_hidden_dim  = 376

class Task_Specific_Attention(torch.nn.Module):# durg : B x 128 x 45, target : B x L x 128
      def __init__(self, k = 256, protein_hidden_dim = 86, drug_hidden_dim  = 658): #, num_tasks
        super(Task_Specific_Attention, self).__init__()
        # paramters for Mutual-Attention 
        self.W_b = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(protein_hidden_dim, drug_hidden_dim)))
        self.W_x = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, drug_hidden_dim)))
        self.W_p = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k, protein_hidden_dim)))
        self.w_hx = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k,1)))
        self.w_hp = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(k,1)))
        
      def forward(self, x, target):

        x = x.permute(0, 2, 1)

        C = F.tanh(torch.matmul(target, torch.matmul(self.W_b, x))) # B x L x 45

        H_c = F.tanh(torch.matmul(self.W_x, x) + torch.matmul(torch.matmul(self.W_p, target.permute(0, 2, 1)), C))          # B x k x 45
        H_p = F.tanh(torch.matmul(self.W_p, target.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_x, x), C.permute(0, 2, 1)))# B x k x L

        a_c_weight = torch.matmul(torch.t(self.w_hx), H_c)
        a_p_weight = torch.matmul(torch.t(self.w_hp), H_p)


        a_c = F.softmax(a_c_weight, dim=2) # B x 1 x 45
        a_p = F.softmax(a_p_weight, dim=2) # B x 1 x L

        c = torch.squeeze(torch.matmul(a_c, x.permute(0, 2, 1)))     # B x 128
        p = torch.squeeze(torch.matmul(a_p, target))                # B x 128

        return c, p, a_c_weight, a_p_weight
