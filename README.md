# Multi-Query-Attention

"""
qkv.chunk(3, dim=-1) 将 qkv 张量沿最后一个维度拆分成三个部分。
每个部分的大小将为 (batch_size, seq_length, d_model // 3 + head_dim)，其中每个部分对应于查询、键和值。
"""
#Multi Head Attention
self.Wqkv = nn.Linear(
#Multi-Head Attention 的创建方法
self.d_model,
3 * self.d_model,
# 查询、键和值 3 个矩阵, 所以是 3* d_model
device=device
)
query, key, value = qkv.chunk(
# 每个 tensor 都是 (1, 512, 768)
3,
dim=2
)


#Multi Query Attention
self.Wqkv = nn.Linear(
#Multi-Query Attention 的创建方法
d_model,
d_model + 2 * self.head_dim,
# 只创建查询的头向量，所以是 1* d_model
device=device,
# 而键和值不再具备单独的头向量
)
query, key, value = qkv.split(
#query -> (1, 512, 768)
[self.d_model, self.head_dim, self.head_dim],
#key -> (1, 512, 96)
dim=2
#value -> (1, 512, 96)
)
