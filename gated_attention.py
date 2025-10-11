import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops import rearrange

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=658, D=512, dropout=0.25, out=658):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, out)
        self.layer_norm = nn.LayerNorm(out)



    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        A = self.layer_norm(A)
        if x.shape[-1] == A.shape[-1]:
            A = A + x
        return A
