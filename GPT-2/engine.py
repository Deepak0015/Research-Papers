import torch 
import torch.nn as nn
import torch.utils
import torch.utils.data
from typing import List , Tuple ,Dict
from tqdm.auto import tqdm

def train_step(model:nn.Module , loss_fn , optimizer:torch.optim.Optimizer ,
                data_loader:torch.utils.data.DataLoader ,
                  device:torch.device ) -> Tuple[float , float]:
    model.train()
    train_loss = 0 
    for batch , (inputs , target ) in enumerate(data_loader):
        inputs , targets  = inputs.to(device) , targets.to(device)
        logits =  model(inputs)
        loss = loss_fn(logits , targets) 
        train_loss +=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    return train_loss


def test_step(model:nn.Module,loss_fn  , data_loader:torch.utils.data.DataLoader ,
               device:torch.device)->Tuple[float , float]:
    test_loss = 0
    model.eval()
    with torch.inference_mode():
        for X , y in data_loader:
            X  , y = X.to(device) , y.to(device)
            predictions = model(X)
            loss = loss_fn(predictions , y )
            test_loss += loss.item()

    test_loss /= len(data_loader)
    return test_loss


def train_model(model:nn.Module ,train_dataloader:torch.utils.data.DataLoader ,
           test_dataloader:torch.utils.data.DataLoader , optimizer:torch.optim.Optimizer ,
            loss_fn:nn.Module  , device:torch.device = None , epochs:int = 1  ):
    results = {
        "train_loss":[],
        "test_loss":[]
    }

    for epoch in tqdm(range(epochs)):
        train_loss = train_step(model= model , 
                                loss_fn=loss_fn , 
                                optimizer= optimizer , data_loader=train_dataloader ,device=device)
        test_loss = test_step(model=model , loss_fn= loss_fn , data_loader= test_dataloader   ,device= device)
        print(
            f"Epoch:{epoch}|"
            f"train_loss:{train_loss}|"
            f"test_loss:{test_loss}|"

        )
        results['train_loss'].append(train_loss.item() if isinstance(train_loss , torch.Tensor)  else train_loss)
        results['test_loss'].append(test_loss.item() if isinstance(test_loss , torch.Tensor)  else test_loss)


    return results