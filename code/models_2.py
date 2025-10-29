import numpy as np
np.random.seed(1)
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import Adam
# import math
# import scipy.sparse as sp
class MultiHeadAttention(nn.Module):
    """修改后的多头注意力模块，适配2D特征图输入"""
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x):
        """输入x: [batch, seq_len, embed_dim]"""
        batch_size, seq_len, _ = x.shape
        # 线性变换并分头
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 加权求和
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        # 输出投影
        return self.out_proj(attn_output)
class ProteinModel(nn.Module):
    def __init__(self, shape6):
        super(ProteinModel, self).__init__()
        #####   DDE
        self.dde_conv = nn.Conv2d(shape6, 32, kernel_size=3, padding='same')
        self.dde_bn = nn.BatchNorm2d(32)
        self.dde_pool = nn.MaxPool2d(kernel_size=(3, 3))
        # self.dde_pool = nn.AdaptiveMaxPool2d((16, 16))
        self.dde_dropout = nn.Dropout(0.5)
        self.dde_attention = MultiHeadAttention(embed_dim=32, num_heads=4)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(32, 64)
        self.bn = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(64, 1)
    def forward(self, x_dde):
        x_dde = x_dde.permute(0, 3, 1, 2).float()

        x_p = F.relu(self.dde_conv(x_dde))
        x_p = self.dde_bn(x_p)
        x_p = self.dde_pool(x_p)
        x_p = self.dde_dropout(x_p)
        def apply_attention(x,y):
            batch, c, h, w = x.shape
            x = x.view(batch, c, -1).permute(0, 2, 1)  # [batch, h*w, c]
            x = self.flatten(y(x).mean(dim=1) ) # [batch, c]
            return x
        att1 = apply_attention(x_p, self.dde_attention)  # [batch, 32]
        merged = att1
        x = F.relu(self.dense1(merged))
        x = self.bn(x)
        penultimate_output = self.dropout(x)
        final_output = self.dense2(penultimate_output)
        # x = torch.sigmoid()
        return final_output, penultimate_output
def get_model_dna_pro_att_torch(INIT_LR, EPOCHS, shape6):
    model = ProteinModel(shape6)
    optimizer = Adam(model.parameters(), lr=INIT_LR, weight_decay=INIT_LR / EPOCHS)
    return model, optimizer
