import torch 
from engine import train_step , test_step  , train_model
from GptModel import GPTModel
from generate_text import generate_text
from loss import calc_loss_loader
from dataloader import create_dataloader_v1 , GPTDatasetV1 
from simple_train import train_model 


torch.manual_seed(123)
GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024,
# Context length
"emb_dim": 768,
# Embedding dimension
"n_heads": 12,
# Number of attention heads
"n_layers": 12,
# Number of layers
"drop_rate": 0.1,
# Dropout rate
"qkv_bias": False
# Query-Key-Value bias
}

model = GPTModel(GPT_CONFIG_124M)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
optimizer  =  torch.optim.AdamW(model.parameters() , lr = 0.0004  , weight_decay= 0.01)
num_epochs = 1
train_ratio = 0.90
filename = 'combined_qa.txt'

with open(filename , 'r') as f:
    text_data = f.read()

text_data = text_data[:20000]

split = int(train_ratio * len(text_data))
print(split)
train_data= text_data[:split]
val_data = text_data[split:]
train_dataloader = create_dataloader_v1(txt= train_data , batch_size= 2 , max_length=GPT_CONFIG_124M['context_length'] , shuffle =  True , drop_last=True , stride=GPT_CONFIG_124M['context_length'])
val_dataloader = create_dataloader_v1(txt= val_data , batch_size= 2 , max_length=GPT_CONFIG_124M['context_length'] , shuffle =  False , drop_last=False , stride=GPT_CONFIG_124M['context_length'])


train_losses , val_losses  , token_seen = train_model(
    model= model , train_dataloader= train_dataloader , 
    eval_dataloaer= val_dataloader , optimizer= optimizer , eval_freq=5 , device= device,
    eval_iter=1 , start_context="Every effort moves you"
)