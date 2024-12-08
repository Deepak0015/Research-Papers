import torch 
import torch.nn as nn 
from GPT2model.multiHeadAttention import MultiHeadAttention_V2




class LanguageModelAudio(nn.Module):
    def __init__(self , codebook_size , emb_dim   ,vocab_size ,  SSL_encoder , model  , cfg  ):
        super().__init__()
        self.audio_embedding = nn.Embedding(codebook_size , emb_dim)
        self.ssl_encoder = SSL_encoder
        self.text_embedding = nn.Embedding(vocab_size  , emb_dim )
        self.model = model 
        self.att = MultiHeadAttention_V2(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        context_length=cfg["context_length"],
        num_heads=cfg["n_heads"],
        dropout=cfg["drop_rate"],
        qkv_bias=cfg["qkv_bias"])
        

    def __Listen_channel(self, audio_input):
        audio_feature = self.ssl_encoder(audio_input).logits
        audio_feature_token = torch.argmax(audio_feature , dim = -1)
        return audio_feature_token

    def __speaking_token_generation_channel(self  ,audio_feature_token ,previous_token = None):
        audio_emb = self.audio_embedding(audio_feature_token )
        audio_emb =  audio_emb[: , :102, :]
        if previous_token is not None :
            text_emb  = self.text_embedding(previous_token)
            combine_emb  = torch.cat((audio_emb  , text_emb) , dim =1 )
            # print(combine_emb.shape)
            combine_emb= combine_emb[: , :102,:]

            combine_emb = combine_emb + self.att(combine_emb)
        else:
            combine_emb = audio_emb 
        tf_output = self.model.transformer_block(combine_emb)
        fc_norm = self.model.final_norm(tf_output)
        out_head = self.model.out_head(fc_norm)
        return out_head
        
    def forward(self , audio_feature ,sum_input =None , response = None):
        audio_tokens = self.__Listen_channel(audio_feature)
        speaking_token = self.__speaking_token_generation_channel(audio_tokens ,response)
        return speaking_token 
    
