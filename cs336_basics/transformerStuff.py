import torch
import einx
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        w = torch.empty(out_features, in_features)
        std = (2 / (in_features + out_features)) ** (1/2)
        nn.init.trunc_normal_(w, mean = 0, std = std, a = -3 * std, b = 3 * std)
        self.mat = nn.Parameter(w)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return einx.dot("out [in], ... [in] -> ... out", self.mat, x)


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim)
        nn.init.trunc_normal_(w, mean = 0, std = 1, a = -3 , b = 3)
        self.mat = nn.Parameter(w)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.mat[x]

class MyRMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        gain = torch.ones(d_model)
        self.gain = nn.Parameter(gain)
        self.d_model = d_model
        self.eps = eps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        RMS_A = x ** 2
        RMS_A = RMS_A.sum(axis=-1)
        RMS_A /= self.d_model
        RMS_A += self.eps
        RMS_A = RMS_A ** (1/2)
        RMS_A = RMS_A.unsqueeze(-1)
        x = x / RMS_A * self.gain
        return x.to(in_dtype)

class MySwiGLU(nn.Module):
    def __init__(self, d_model, d_ff = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(round(8 / 3 * d_model / 64) * 64)
        self.d_ff = d_ff

        self.w1 = MyLinear(d_model, d_ff)
        self.w2 = MyLinear(d_ff, d_model)
        self.w3 = MyLinear(d_model, d_ff)

        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        w3x = self.w3(x)
        elwiseProd = self.SILU(w1x) * w3x
        out = self.w2(elwiseProd)
        return out

    def SILU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

import numpy as np

class MyRope(nn.Module):
    def __init__(self, d_key, theta, max_seq_length, device=None, dtype=None):
        super().__init__()
        seqOfThetaArr = []
        for i in range(max_seq_length):
            thetaArr = []
            k = 1
            for idx in range(d_key):
                tik = i / (theta ** ((2 * k - 2)/d_key) )
                if idx % 2 == 0:
                    thetaArr.append(np.array([np.cos(tik), -np.sin(tik)]))
                else:
                    thetaArr.append(np.array([np.sin(tik), np.cos(tik)]))
                    k += 1
            seqOfThetaArr.append(np.stack(thetaArr))
        precompThetaArr = np.stack(seqOfThetaArr)
        self.register_buffer("rotaryTable", torch.tensor(precompThetaArr), persistent=False)
        self.d_key = d_key
        self.evenIndices = [x if x % 2 == 0 else x - 1 for x in range(d_key)]
        self.oddIndices = [x if x % 2 == 1 else x + 1 for x in range(d_key)]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        token_positions = token_positions.unsqueeze(-1)
        tablesOfInterest = einx.get_at("[i] k z, ... [idx] -> ... k z", self.rotaryTable, token_positions)
        evenX = x[..., self.evenIndices]
        oddX = x[..., self.oddIndices]
        evenTables = tablesOfInterest[..., 0]
        oddTables = tablesOfInterest[..., 1]

        return (evenX * evenTables + oddX * oddTables).to(x.dtype)

def softmax(x: torch.Tensor, dim):
    x = x - x.max(dim=dim, keepdim=True).values
    denom = torch.sum(torch.exp(x), dim=dim, keepdim=True)
    x = torch.exp(x) / denom
    return x

def scaledDotProdAttention(queries, keys, values, mask=None):

    presoftAttention = einx.dot("b ... key [dim], b ... quer [dim] -> b ... quer key", keys, queries) / keys.shape[-1] ** 0.5
    if mask is not None:
        presoftAttention[~mask.expand(presoftAttention.shape)] = - torch.inf
    
    softAttention = softmax(presoftAttention, dim = -1)
    return einx.dot("b ... quer [key], b ... [key] d_v -> b ... quer d_v", softAttention, values)
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.Wq = MyLinear(d_model, d_model)
        self.Wk = MyLinear(d_model, d_model)
        self.Wv = MyLinear(d_model, d_model)
        self.Wo = MyLinear(d_model, d_model)
    def forward(self, x):
        seq = x.shape[-2]
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        queries = einx.rearrange("... seq (heads dk) -> ... heads seq dk", queries, dk=self.dk)
        keys = einx.rearrange("... seq (heads dk) -> ... heads seq dk", keys, dk=self.dk)
        values = einx.rearrange("... seq (heads dk) -> ... heads seq dk", values, dk=self.dk)
        mask = torch.tril(torch.ones((seq, seq))).bool()
        attended = scaledDotProdAttention(queries, keys, values, mask)
        multiHead = einx.rearrange("... heads seq dk -> ... seq (heads dk)", attended)
        return self.Wo(multiHead)
class MultiHeadSelfAttentionRope(nn.Module):
    def __init__(self, d_model, num_heads, max_seq_len, theta, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dk = d_model // num_heads

        self.rope = MyRope(self.dk, theta, max_seq_len)

        self.Wq = MyLinear(d_model, d_model)
        self.Wk = MyLinear(d_model, d_model)
        self.Wv = MyLinear(d_model, d_model)
        self.Wo = MyLinear(d_model, d_model)
    def forward(self, x, token_positions):
        seq = x.shape[-2]
        queries = self.Wq(x)
        keys = self.Wk(x)
        values = self.Wv(x)
        queries = einx.rearrange("... seq (heads dk) -> ... heads seq dk", queries, dk=self.dk)
        keys = einx.rearrange("... seq (heads dk) -> ... heads seq dk", keys, dk=self.dk)
        values = einx.rearrange("... seq (heads dk) -> ... heads seq dk", values, dk=self.dk)

        queries = self.rope(queries, token_positions.unsqueeze(-2).expand(keys.shape[:-1]))
        keys = self.rope(keys, token_positions.unsqueeze(-2).expand(keys.shape[:-1]))
        
        mask = torch.tril(torch.ones((seq, seq))).bool()

        attended = scaledDotProdAttention(queries, keys, values, mask)
        multiHead = einx.rearrange("... heads seq dk -> ... seq (heads dk)", attended)
        return self.Wo(multiHead)