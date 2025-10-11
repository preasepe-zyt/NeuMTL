import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttnPooling(nn.Module):
    def __init__(self, input_dim):
        super(AttnPooling, self).__init__()
        self.attn_proj = nn.Linear(input_dim, 1)  # 将每个原子的特征映射到一个注意力分数

    def forward(self, h, mask=None):
        """
        h:     [batch_size, max_nodes, input_dim]
        mask:  [batch_size, max_nodes] — 可选，表示哪些节点有效（1）哪些是padding（0）
        """
        # 1. 计算每个节点的注意力打分
        scores = self.attn_proj(h)  # [batch_size, max_nodes, 1]
        scores = torch.softmax(scores, dim=1)  # 对每个图内部归一化

        if mask is not None:
            scores = scores * mask.unsqueeze(-1)  # mask 无效节点注意力为 0
            scores = scores / (scores.sum(dim=1, keepdim=True) + 1e-8)  # 重新归一化


        # 2. 加权求和得到图表示
        final =  (h * scores).sum(dim=1) 
        return final, scores  # 返回 attention 权重以供解释性可视化