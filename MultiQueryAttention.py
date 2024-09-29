import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, device:None):
        super().__init__(MultiQueryAttention, self)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        '''
        只创建 查询 的头向量，所以只有1个d_model
        而 键 和 值 则共享各自的一个 head_dim 的向量
        '''
        self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim)
        self.attn_fn = scaled_multihead_dot_product_attention
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)
        self.out_proj._is_residual = True

    def forward(self, x):
        qkv = self.Wqkv(x)  #(1, 512, 960)
        '''
        [self.d_model, self.head_dim, self.head_dim]：这是一个列表，表示要将 qkv 张量在 dim=2 维度上分割成的大小。
        这里 self.d_model 通常代表模型的维度大小，而 self.head_dim 则是每个头的维度大小。
        在这里，可以理解为将 qkv 张量分割成三个部分：查询、键和值。
        query -> (1, 512, 768)
        key -> (1, 512, 96)
        value -> (1, 512, 96)
        '''
        query, key, value = qkv.split([self.d_model, self.head_dim, self.head_dim], dim=2)
        context, attn_weights, past_key_value = self.attn_fn(query, key, value, self.heads, multiquery=True)
        return self.out_proj(context), attn_weights, past_key_value


