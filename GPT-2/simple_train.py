import torch
import torch.utils
import torch.utils.data
from loss import cal_loss_batch , calc_loss_loader
import torch.nn as nn
from tqdm.auto import tqdm
from text_fuc import text_to_token_ids , token_ids_to_text , generate_and_sample 




def evaluate_model(model , train_dataloader , eval_dataloaer , device , eval_iter ):
    model.eval()
    with torch.no_grad():
        train_loss =  calc_loss_loader(train_dataloader , model , device , num_batches= eval_iter)
        val_loss = calc_loss_loader(eval_dataloaer , model , device , num_batches=eval_iter)
    
    model.train()
    return train_loss , val_loss 



def generate_and_print_sample(model , tokenizer , device , start_context ):
    model.eval()
    context_size  = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context , tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_and_sample(
            model = model , idx  = encoded, max_new_tokens = 50 , context_size = context_size
        )
        decoded_text = token_ids_to_text(token_ids , tokenizer)
        print(decoded_text.replace("\n" , " "))
        model.train()

     
def train_model(model:nn.Module , train_dataloader:torch.utils.data.DataLoader ,  device:torch.device  ,
                eval_dataloaer:torch.utils.data.DataLoader , optimizer:torch.optim.Optimizer , 
                eval_freq , eval_iter , start_context, num_epochs:int = 1):
    train_losses , val_losses , track_tokens_seen = [] , [] ,[]
    tokens_seen  , global_step = 0, -1 
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for inputs_batch  , target_batch in train_dataloader:
            optimizer.zero_grad()
            loss = cal_loss_batch(input_batch=inputs_batch , target_batch=target_batch ,device=device  , model=model)
            optimizer.step()
            tokens_seen += inputs_batch.numel()
            global_step +=1 
            if global_step % eval_freq == 0:
                train_loss , val_loss = evaluate_model(
                    model , train_dataloader , eval_dataloaer , device , eval_iter 
                )

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Epoch:{epoch+1}(step {global_step:06d}):",
                    f"Train Loss {train_loss:3f} , Val loss {val_loss:.3f}" 

                )

        generate_and_print_sample(
            model , train_dataloader.dataset.tokenzier , device , start_context
        )
    return train_losses , val_losses , track_tokens_seen 



