
import torch
import torch.nn as nn 




def text_to_token_ids(text,  tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded = torch.tensor(encoded).unsqueeze(0)
    return encoded


def token_ids_to_text(tokens , tokenizer):
    flat  = tokens.squeeze(0)
    decode = tokenizer.decode(flat.tolist())
    return decode

    
def generate_and_sample(model  , idx , context_size ,max_new_tokens ):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
            print(logits.shape)
        logits  = logits[:, -1  , :]
        print(logits.shape)
        probs  = torch.softmax(logits  , dim=-1)
        print(probs)
        idx_next = torch.argmax(probs, dim=-1 , keepdim= True)
        print(idx_next)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx 