import torch 
import torch.nn as nn 
import math 

class InputEmbedding(nn.Module):

    def __init__(self, vocab_size: int , d_model:int):

        super().__init__()

        self.d_model  =  d_model 

        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(vocab_size , d_model)

    def forward(self ,x):

        return self.embeddings(x) * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model:int , seq_len:int , dropout:float):

        super().__init__()

        self.d_model =  d_model 

        self.seq_len = seq_len

        self.dropout = nn.Dropout(dropout)

        positional_encoding = torch.zeros((self.seq_len , self.d_model))

        possitions = torch.arange(0,self.seq_len).unsqueeze(1)

        dimentions =  torch.arange(0 , self.d_model).unsqueeze(0)

        dinaminator = torch.pow(10000 ,(2*dimentions//2 )/self.d_model)

        positional_encoding[:,0::2] = torch.sin(possitions / dinaminator[:,0::2])

        positional_encoding[:,1::2] =  torch.cos(possitions / dinaminator[:,1::2])

        positional_encoding  = positional_encoding.unsqueeze(0)

        self.register_buffer('pe', positional_encoding)

    def forward(self ,x):

        x = x + (self.pe[:,:x.shape[1],:]).requires_grad_(False)
    
        return  self.dropout(x)
    



class LayerNormalizations(nn.Module):
    
    def __init__(self, eps:float = 10**-6):

        super().__init__()

        self.eps = eps 

        self.alpha = torch.nn.Parameter(torch.ones(1)) # for multiply 

        self.bias = torch.nn.Parameter(torch.zeros(1)) # for add 


    def forward(self,x):

        mean  = x.mean(dim = -1, keepdim =True )

        std  = x.std(dim =-1 , keepdim =  True)

        return self.alpha * (x -mean ) / (std + self.eps) + self.bias
 

    


class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model:int  , d_ff:int,  dropout:float ):

        super().__init__()

        self.linear_1  = nn.Linear(d_model, d_ff )
        
        self.dropout = nn.Dropout(dropout)

        self.linear_2  = nn.Linear(d_ff ,d_model)

    def forward(self, x):

        #(batch  , seq_len , d_model)---> (batch , seq_len, d_ff) --->(batch  , seq_len , d_model ))
            
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))



                
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model :int ,h:int , dropout:float):
        
        super().__init__()
        
        self.d_model  = d_model 

        self.dropout = dropout 

        self.h = h 

        assert d_model  % h  ==0 ,"d_model is not divisible by h "

        d_k  =  d_model // h 

        self.w_q = nn.Linear(d_model ,d_model)

        self.w_k = nn.Linear(d_model , d_model)

        self.w_v = nn.Linear(d_model ,d_model)

        self.w_o = nn.Linear(d_model , d_model)


    @staticmethod
    def attention(self,query , key ,value , mask ,dropout:nn.Dropout):
        
        d_k = query.shape[-1]

        attention_score   = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)

        if  mask is not None:
            
            attention_score.masked_fill(mask == 0 , -1e9)

        attention_score = attention_score.softmax(dim =-1) # batch  ,   h , seq_len, seq_len 

        if dropout:
            attention_score = dropout(attention_score)

        
        context_vector = attention_score @ value 

        return context_vector , attention_score 


        
    def forward(self, query , key , value , mask ):

        query = self.w_q(query) #(batch , seq_len , d_model  ) ---> (batch , seq_len , d_model)

        key  = self.w_k(key) #(batch , seq_len , d_model  ) ---> (batch , seq_len , d_model)

        value = self.w_v(value) #(batch , seq_len , d_model  ) ---> (batch , seq_len , d_model)

        #(batch  , seq_len , d_mode ) --->(batch , seq_len, h , d_k )---> (batch , h , seq_len  , d_k )
        query = query.view(query.shape[0], query.shape[1],self.h , self.d_k ).transpose(1, 2) 

        key = key.view(key.shape[0] , key.shape[1] , self.h , self.d_k ).transpose(1 ,2 )

        value = value.view(value.shape[0] , value.shape[1] ,self.h , self.d_k).transpose(1,2)

        x , self.attention_score = MultiHeadAttentionBlock.attention(query , key , value , mask , self.dropout)

        # (batch , h , seq_len, d_k ) ---> (batch ,seq_len , h , d_k) --->(batch  , seq_len , d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0] , -1 , self.h * self.d_k ) 

        x = self.w_o(x)

        return x #(batch  , seq_len , d_model)
    

class ResidualConnection(nn.Module):
    
    def __init__(self ,dropout:float):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.norm = LayerNormalizations()

    def forward(self, x ,sub_layer):
        
        return x + self.dropout(sub_layer(self.norm(x)))


class EncoderBlock(nn.Module):

    def __init__(self, self_attention:MultiHeadAttentionBlock , feed_forward:FeedForwardBlock, dropout:float):
        
        super().__init__()

        self.self_attention = self_attention 

        self.feed_forward = feed_forward 
        
        self.dorpout = nn.Dropout(dropout)

        self.residual_connection  =nn.ModuleList(ResidualConnection(dropout)  for _ in range(2))


    def forward(self, x  , src_mask):

        x =  self.residual_connection[0](x , lambda x:self.self_attention(x,x,x,src_mask ))

        x =  self.residual_connection[1](x , self.feed_forward)

        return x 
    


class Encoder(nn.Module):

    def __init__(self,layers:nn.ModuleList):
        
        super().__init__()

        self.layers = layers

        self.layernorm = LayerNormalizations()


    def forward(self,x   , mask):
        
        for layer in self.layers:

            x = layer(x ,mask)

        return self.layernorm(x)


class DecoderBlock(nn.Module):

    def __init__(self,self_attention:MultiHeadAttentionBlock , feed_forward:FeedForwardBlock , cross_attention:MultiHeadAttentionBlock ,dropout:float):

        super().__init__()        

        self.self_atttention = self_attention 

        self.feed_forward = feed_forward  

        self.recidiual_connection = nn.ModuleList([ResidualConnection(dropout)  for _ in range(3)])       
         
        self.cross_attention  = cross_attention


    def forward(self, x, encoder_output , src_mask , tgt_mask):

        x = self.recidiual_connection[0](x , lambda x: self.self_atttention(x ,x ,x,tgt_mask)) 

        x =  self.recidiual_connection[0](x, lambda x :self.cross_attention(x , encoder_output ,encoder_output ,src_mask))

        x = self.recidiual_connection[2](x , self.feed_forward)

        return x 
         




class Decoder(nn.Module):

    def __init__(self, layers:nn.ModuleList):

        super().__init__()

        self.layers = layers
        
        self.norm  = LayerNormalizations()


    def forward(self, x , encoder_output , src_mask  , tgt_mask ):

            for layer in self.layers:

                x  = layer(x , encoder_output , src_mask  ,tgt_mask) 

            return self.norm(x)
    




                    

class ProjectionLayer(nn.Module):

    def __init__(self,d_model:int , vocab_size:int ):

        super().__init__()

        self.projection = nn.Linear(d_model , vocab_size)


    def forward(self, x):

        return torch.log_softmax(self.projection(x) , dim= -1) # (batch ,seq_len  , d_model )--->(batch , seq_len , vocab_size )
    



class TransformerBlock(nn.Module):

    def __init__(self,encoder:Encoder , decoder:Decoder , projectionlayer:ProjectionLayer  , 
                                                     src_embed:InputEmbedding, tgt_embed:InputEmbedding ,
                                                                      src_pos:PositionalEncoding , tgt_pos:PositionalEncoding ):

        super().__init__()

        self.encoder = encoder

        self.decoder = decoder 

        self.proj = projectionlayer 

        self.src_embed = src_embed 

        self.tgt_embed = tgt_embed 

        self.src_pos = src_pos 

        self.tgt_pos = tgt_pos


    def encode(self , src, src_mack ):

        src = self.src_embed(src)

        src = self.src_pos(src)

        src = self.encoder(src , src_mack)


    def decode(self, encoder_output , src_mask , tgt , tgt_mask ):

        tgt = self.tgt_embed(tgt)

        tgt = self.tgt_pos(tgt)

        tgt = self.decoder(tgt , encoder_output , src_mask  , tgt_mask )

        return tgt  


    def projection(self, x ):

            return self.proj(x)



def build_transformer(src_vocab:int , tgt_vocab:int , src_seq_len:int , tgt_seq_len:int ,
                       d_model:int = 512 , N:int = 6, h= 8 , dropout= 0.01,d_ff:int= 2014):
    
    # token embedding 
    src_emb  = InputEmbedding(src_vocab ,d_model)

    tgt_emb = InputEmbedding(tgt_vocab , d_model)
 
    #pos embeding 

    src_pos = PositionalEncoding(d_model ,  src_seq_len , dropout )

    tgt_pos = PositionalEncoding(d_model , tgt_seq_len  , dropout)

    # Create the encoder block 

    encoder_blocks = []

    for _ in range(N):
        encoder_self_attention_block  = MultiHeadAttentionBlock(d_model , h , dropout )

        feed_forwardblock  = FeedForwardBlock(d_model  , d_ff , dropout )

        encoder_block= EncoderBlock(encoder_self_attention_block , feed_forwardblock ,dropout    )

        encoder_blocks.append(encoder_block)


    # create  the decoder block 

    decoder_blocks =  []

    for _ in range(N):

        decoder_self_attention  = MultiHeadAttentionBlock(d_model , h, dropout)

        decoder_cross_attention  = MultiHeadAttentionBlock(d_model , h  ,dropout)

        decoder_ffn  = FeedForwardBlock(d_model , d_ff , dropout)

        decoder_block = DecoderBlock(decoder_self_attention ,decoder_ffn ,  decoder_cross_attention ,dropout  )
        
        decoder_blocks.append(decoder_block)


    # create the encder and decoder 

    encoder  = Encoder(nn.ModuleList(encoder_blocks ))

    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Projection layer 

    projection_layer = ProjectionLayer(d_model  ,tgt_vocab)


    # create transformer 

    transformer = TransformerBlock(encoder= encoder , decoder= decoder , src_embed= src_emb , src_pos= src_pos 
                                   , tgt_embed= tgt_emb , tgt_pos= tgt_pos  , projectionlayer= projection_layer)
    

    for p in transformer.parameters():
        if p.dim() > 1 :
            nn.init.xavier_uniform_(p)


    return transformer 


