import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock 
from layernorm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):

        super().__init__()

        self.tok_emb = nn.Embedding(cfg['vocab_size'] , cfg['emb_dim'])

        self.pos_emb = nn.Embedding(cfg['context_length'] , cfg['emb_dim'])

        self.drop_rate = nn.Dropout(cfg['drop_rate'])

        self.transformer_block  = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg['n_layers'])])

        self.final_norm  =  LayerNorm(emb_dim=cfg['emb_dim'])

        self.out_head = nn.Linear(cfg['emb_dim'] , cfg['vocab_size'] , bias = False)

    def forward(self, idx):

        batch_size , seq_len = idx.shape

        token_embs = self.tok_emb(idx)

        pos_embs = self.pos_emb(torch.arange(seq_len , device= idx.device))

        x = token_embs + pos_embs

        x  =  self.drop_rate(x)

        x = self.transformer_block(x)

        x = self.final_norm(x)

        x = self.out_head(x)

        return x 
