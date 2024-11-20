import torch
import torch.nn as nn 



class MultiHeadAttention_V2(nn.Module):
    def __init__(self, d_in , d_out , context_length  , dropout ,num_heads,qkv_bias = False):
        super().__init__()
        assert d_out % num_heads  == 0,'d_out must be divisible by the num_heads'
        self.w_query = nn.Linear(d_in , d_out ,bias=qkv_bias)
        self.w_key = nn.Linear(d_in , d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in  , d_out,bias=qkv_bias)
        self.d_in =d_in
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.num_heads  = num_heads
        self.head_dim = d_out // num_heads
        self.out_proj  = nn.Linear(d_out , d_out)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length , context_length),diagonal=1)
        )

    def forward(self,x):
        b, num_tokens , d_in = x.shape
        keys = self.w_key(x)
        queries  = self.w_query(x)
        values = self.w_value(x)
        queries = queries.view(b, num_tokens , self.num_heads , self.head_dim)
        values = values.view(b , num_tokens , self.num_heads , self.head_dim)
        keys = keys.view( b, num_tokens , self.num_heads , self.head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool= self.mask.bool()[:num_tokens , :num_tokens]
        attn_scores.masked_fill(mask_bool , -torch.inf)
        attn_weights = torch.softmax(attn_scores /self.head_dim**0.5   , dim=-1 )
        attn_weights = self.dropout(attn_weights)
        context_vector = (attn_weights  @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(b , num_tokens , self.d_out)
        context_vector = self.out_proj(context_vector)
        return context_vector



