import torch 



def generate_text(model  , idx , context_size ,max_new_tokens ):
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