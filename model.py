import torch

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from typing import Optional, Dict
import math

from utils import Tokenizer
from einops.layers.torch import Rearrange

from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F


from Task_Specific_Attention  import Task_Specific_Attention
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from shared_attention import *
import torchvision.models as models
from gated_attention import Attn_Net_Gated

class Encoder(torch.nn.Module):
    def __init__(self, Drug_Features, dropout, Final_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = 512
        self.GraphConv1 = GCNConv(Drug_Features, Drug_Features * 2)
        self.GraphConv2 = GCNConv(Drug_Features * 2, Drug_Features * 3)
        self.GraphConv3 = GCNConv(Drug_Features * 3, Drug_Features * 4)



    def forward(self, data):
       x, edge_index, batch, num_nodes, affinity = data.x, data.edge_index, data.batch, data.c_size, data.y
       a = affinity.view(-1, 1)
       GCNConv = self.GraphConv1(x, edge_index)
       GCNConv = self.GraphConv2(GCNConv, edge_index)
       GCNConv  = self.GraphConv3(GCNConv, edge_index)

       return GCNConv 


class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # 双向 GRU
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers,
                          batch_first=True, bidirectional=True)
                          
        self.relu = nn.ReLU()

    def forward(self, x):
        # Embedding
        embed = self.embedding(x) 
        # GRU
        gru_out, _ = self.gru(embed) 
        gru_out = self.relu(gru_out)
        return gru_out

        
class Classification_Module(torch.nn.Module):  
    def __init__(self, num_features_xd=0, output_dim=1, dropout = 0):
         super(Classification_Module, self).__init__()
         self.fc_g1   = torch.nn.Linear(num_features_xd, 512)
         self.fc_g2   = torch.nn.Linear(512, 256)
         self.relu    = nn.ReLU()
         self.dropout = nn.Dropout(dropout)
         self.out     = nn.Linear(256, output_dim)
         self.sigmoid = nn.Sigmoid()

#     #---------------------------------------
    def forward(self, x):
         x   = self.fc_g1(x)
         x   = self.relu(x)
         x   = self.dropout(x)
         x   = self.fc_g2(x)
         x   = self.relu(x)
         x   = self.dropout(x)
         out = self.out(x)
         out = self.sigmoid(out)
         return out




class GatedCNN(nn.Module):
    def __init__(self, Protein_Features, Num_Filters, Embed_dim, Final_dim, K_size):
        super(GatedCNN, self).__init__()
        self.Protein_Embed = nn.Embedding(Protein_Features + 1, Embed_dim)
        self.Protein_Conv1 = nn.Conv1d(in_channels=1000, out_channels=Num_Filters, kernel_size=K_size)
        self.Protein_Gate1 = nn.Conv1d(in_channels=1000, out_channels=Num_Filters, kernel_size=K_size)
        self.Protein_Conv2 = nn.Conv1d(in_channels=Num_Filters, out_channels=Num_Filters * 2, kernel_size=K_size)
        self.Protein_Gate2 = nn.Conv1d(in_channels=Num_Filters, out_channels=Num_Filters * 2, kernel_size=K_size)
        self.Protein_Conv3 = nn.Conv1d(in_channels=Num_Filters * 2, out_channels=Num_Filters * 3, kernel_size=K_size)
        self.Protein_Gate3 = nn.Conv1d(in_channels=Num_Filters * 2, out_channels=Num_Filters * 3, kernel_size=K_size)

        self.relu = nn.ReLU()

    def forward(self, data):
        target = data.target
        Embed = self.Protein_Embed(target)
        conv1 = self.Protein_Conv1(Embed)  
        gate1 = torch.sigmoid(self.Protein_Gate1(Embed))
        GCNN1_Output = conv1 * gate1
        GCNN1_Output = self.relu(GCNN1_Output)
        #GATED CNN 2ND LAYER
        conv2 = self.Protein_Conv2(GCNN1_Output)
        gate2 = torch.sigmoid(self.Protein_Gate2(GCNN1_Output)) 
        GCNN2_Output = conv2 * gate2                            
        GCNN2_Output = self.relu(GCNN2_Output)
        #GATED CNN 3RD LAYER
        conv3 = self.Protein_Conv3(GCNN2_Output)
        gate3 = torch.sigmoid(self.Protein_Gate3(GCNN2_Output))
        GCNN3_Output = conv3 * gate3                           
        GCNN3_Output = self.relu(GCNN3_Output)

        return GCNN3_Output 

        
class FC(torch.nn.Module):
    def __init__(self, output_dim, n_output, dropout=0):
        super(FC, self).__init__()
        self.FC_layers = nn.Sequential(
            nn.Linear(output_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, n_output)
        )

    def forward(self, Combined):

        Pridection = self.FC_layers(Combined)
        return Pridection


# MAin CLass
class NeuMTL(torch.nn.Module):
    def __init__(self, tokenizer, device): #, num_tasks
        super(NeuMTL, self).__init__()
        self.node_feature = 94
        self.protein_f = 25
        self.filters = 32
        self.kernel = 8
        self.output_dim = 128
        self.encoder_dropout = 0.2

        
        #Encoder
        self.encoder = Encoder(Drug_Features=self.node_feature, dropout=self.encoder_dropout, Final_dim=self.output_dim)
        self.rnn =  RNNEncoder(1000, embed_dim = 128, hidden_dim = 256)
        self.resnet18 = models.resnet18(pretrained=True).to(device)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        for param in self.resnet18.layer4.parameters():
            param.requires_grad = True
        num_features = self.resnet18.fc.in_features
        self.resnet18.fc =torch.nn.Linear(num_features, 512)
        #classification weight
        encoder_transformer_layer_graph = nn.TransformerEncoderLayer(d_model=376, nhead=8, dim_feedforward=512,
                                                             dropout=0.35, batch_first=True )
        self.transformer_graph = nn.TransformerEncoder(encoder_layer=encoder_transformer_layer_graph, num_layers=2)

        encoder_transformer_layer_seq = nn.TransformerEncoderLayer(d_model=512, nhead=8, dim_feedforward=512,
                                                               dropout=0.35, batch_first=True )
        self.transformer_seq = nn.TransformerEncoder(encoder_layer=encoder_transformer_layer_seq, num_layers=2)
        self.attn_graph = AttnPooling(input_dim=376)
        self.attn_seq = AttnPooling(input_dim=512)
        #FC
        self.class_fc_bbb = Classification_Module(num_features_xd=512, output_dim=1, dropout=0)
        self.class_fc_neu1 = Classification_Module(num_features_xd=512, output_dim=1, dropout=0)
        self.class_fc_neu2 = Classification_Module(num_features_xd=512, output_dim=1, dropout=0)
        self.class_fc_neu3 = Classification_Module(num_features_xd=512, output_dim=1, dropout=0)
        self.layernorm_cla = nn.LayerNorm(512)
        self.graph = torch.nn.Linear(376, 512)

                                      
        #regression encoder
        self.cnn = GatedCNN(Protein_Features=self.protein_f, Num_Filters=self.filters, 
                            Embed_dim=self.output_dim, Final_dim=self.output_dim, K_size=self.kernel)

        #Task_Specific_Attention 
        self.Task_Specific_Attention_graph = Task_Specific_Attention(drug_hidden_dim  = 376, k = 256, protein_hidden_dim = 107)
        self.Task_Specific_Attention_seq = Task_Specific_Attention(drug_hidden_dim  = 512, k = 256, protein_hidden_dim = 107)

        self.gated_attention_graph = Attn_Net_Gated(L=483, D=512, dropout=0.1, out=483)
        self.gated_attention_seq = Attn_Net_Gated(L=619, D=512, dropout=0.1, out=619)
        # Fully connected layer
        
        self.final_FC_graph = FC(output_dim=483, n_output=1, dropout=0)
        self.final_FC_seq = FC(output_dim=619, n_output=1, dropout=0)
        
        self.layernorm_protein  = nn.LayerNorm(107)
        
        self.layernorm_graph = nn.LayerNorm(483)
        self.layernorm_seq = nn.LayerNorm(619)
        

    def regression(self, data):
        batch_Protein = self.cnn(data)
        # Encode the input graph
        graph = self.encoder(data)
        graph_max = gmp(graph, data.batch) 
        graph, mask = to_dense_batch(graph, data.batch) 
        smiles_seq = self.rnn(data.smiles_seq)
        seq_max = torch.max(smiles_seq, dim=1)[0]
        protein = torch.max(batch_Protein, dim=1)[0]
        #early
        #graph
        drug_graph, protein_graph, _ , _ = self.Task_Specific_Attention_graph(graph, batch_Protein)
        drug_graph = drug_graph+graph_max
        protein_graph = protein_graph+protein
        Combined_graph = torch.cat((drug_graph, protein_graph), 1) 
        Combined_graph = self.layernorm_graph(Combined_graph)
        Combined_graph = self.gated_attention_graph(Combined_graph)
        Prediction_graph = self.final_FC_graph(Combined_graph)
        
        #seq
        drug_seq, protein_seq, _ , _ = self.Task_Specific_Attention_seq(smiles_seq, batch_Protein)
        drug_seq = drug_seq+seq_max
        protein_seq = protein_seq+protein
        Combined_seq = torch.cat((drug_seq, protein_seq), 1) 
        Combined_seq = self.layernorm_seq(Combined_seq)
        Combined_seq = self.gated_attention_seq(Combined_seq)
        Prediction_seq = self.final_FC_seq(Combined_seq)
        #later
        Prediction = (Prediction_graph+Prediction_seq)/2
        
        return Prediction
        
    def BBB(self, data2):
        graph = self.encoder(data2)
        graph, mask = to_dense_batch(graph, data2.batch)
        graph = self.transformer_graph(graph)                  
        graph, attn_weights = self.attn_graph(graph, mask)
        smiles_seq = self.rnn(data2.smiles_seq)
        smiles_seq = self.transformer_seq(smiles_seq)
        PAD_IDX = 0  # 或 tokenizer.pad_index，如果有定义
        seq_mask = (data2.smiles_seq != PAD_IDX)  # [B, L], bool 类型
        smiles_seq = smiles_seq.float()
        smiles_seq, seq_weights = self.attn_seq(smiles_seq, mask=seq_mask)
        img = self.resnet18(data2.smiles_img)
        #Early Fusion
        Early = self.graph(graph) + smiles_seq + img
        Early = self.layernorm_cla(Early)
        classification1 = self.class_fc_bbb(Early)
        return classification1

    def neu1(self, data3):
        graph = self.encoder(data3)
        graph, mask = to_dense_batch(graph, data3.batch)                   
        graph = self.transformer_graph(graph)
        graph, attn_weights = self.attn_graph(graph, mask)
        smiles_seq = self.rnn(data3.smiles_seq)
        smiles_seq = self.transformer_seq(smiles_seq)
        PAD_IDX = 0  # 或 tokenizer.pad_index，如果有定义
        seq_mask = (data3.smiles_seq != PAD_IDX)  # [B, L], bool 类型
        smiles_seq = smiles_seq.float()
        smiles_seq, seq_weights = self.attn_seq(smiles_seq, mask=seq_mask)
        img = self.resnet18(data3.smiles_img)
        #Early Fusion
        Early = self.graph(graph) + smiles_seq + img
        Early = self.layernorm_cla(Early)
        classification2 = self.class_fc_neu1(Early)
        return classification2

    def neu2(self, data4):
        graph = self.encoder(data4)
        graph, mask = to_dense_batch(graph, data4.batch)                   
        graph = self.transformer_graph(graph)
        graph, attn_weights = self.attn_graph(graph, mask)
        smiles_seq = self.rnn(data4.smiles_seq)
        smiles_seq = self.transformer_seq(smiles_seq)
        PAD_IDX = 0  # 或 tokenizer.pad_index，如果有定义
        seq_mask = (data4.smiles_seq != PAD_IDX)  # [B, L], bool 类型
        smiles_seq = smiles_seq.float()
        smiles_seq, seq_weights = self.attn_seq(smiles_seq, mask=seq_mask)
        img = self.resnet18(data4.smiles_img)
        #Early Fusion
        Early = self.graph(graph) + smiles_seq + img
        Early = self.layernorm_cla(Early)
        classification3 = self.class_fc_neu2(Early)
        return classification3

    def neu3(self, data5):
        graph = self.encoder(data5)
        graph, mask = to_dense_batch(graph, data5.batch)                 
        graph = self.transformer_graph(graph)
        graph, attn_weights = self.attn_graph(graph, mask)
        smiles_seq = self.rnn(data5.smiles_seq)
        smiles_seq = self.transformer_seq(smiles_seq)
        PAD_IDX = 0  # 或 tokenizer.pad_index，如果有定义
        seq_mask = (data5.smiles_seq != PAD_IDX)  # [B, L], bool 类型
        smiles_seq = smiles_seq.float()
        smiles_seq, seq_weights = self.attn_seq(smiles_seq, mask=seq_mask)
        img = self.resnet18(data5.smiles_img)
        #Early Fusion
        Early = self.graph(graph) + smiles_seq + img
        Early = self.layernorm_cla(Early)
        classification4 = self.class_fc_neu3(Early)
        return classification4

     

    def forward(self, data, data2, data3, data4, data5):

        Pridection =  self.regression(data)
        classification1 = self.BBB(data2)
        classification2 = self.neu1(data3)
        classification3 = self.neu2(data4)
        classification4 = self.neu3(data5)
        return Pridection, classification1, classification2, classification3, classification4


        
    def shared_modules(self):
        return [self.encoder,
                self.transformer_graph,
                self.attn_graph,
                self.rnn,
                self.transformer_seq,
                self.attn_seq,
                self.resnet18,
                self.class_fc_bbb] 

    def zero_grad_shared_modules(self):
        for mm in self.shared_modules():
            mm.zero_grad()


