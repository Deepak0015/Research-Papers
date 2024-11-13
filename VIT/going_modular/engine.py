import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model:nn.Module ,
                loss_fn:nn.Module ,
                optimizer:torch.optim.Optimizer , 
               data_loader:torch.utils.data.DataLoader ,
                device :torch.device )-> Tuple[float, float]:

        model.train()
        train_loss , train_acc = 0 , 0 
        for batch , (X, y ) in enumerate(data_loader):
            X , y = X.to(device)  , y.to(device)
            y_pred = model(X)
            loss  = loss_fn(y_pred , y )
            train_loss += loss.item() 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_class  = torch.argmax(torch.softmax(y_pred, dim =1 ) , dim=1 )

            train_acc +=  (y_pred_class == y).sum().item()/len(y_pred)
            
        train_loss /= len(data_loader)
        train_acc  /= len(data_loader)
        return train_loss , train_acc

def test_step(model:nn.Module , loss_fn:nn.Module , data_loader:torch.utils.data.DataLoader,device:torch.device )-> Tuple[float , float]:
    test_loss , test_acc  = 0 ,0 
    model.eval()
    with torch.inference_mode():
        for X , y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred  , y)
            test_loss +=loss.item()
            test_pred_labels = torch.argmax(y_pred , dim= 1 )

            test_acc +=((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    test_loss /= len(data_loader)
    test_acc  /= len(data_loader)
    
    return test_loss , test_acc
    
def plot_loss_curves(results :Dict[str, List[float]]):
    loss = results['train_loss']
    accuracy = results['train_acc']
    test_loss = results['test_loss']
    test_accuracy = results['test_acc']
    epochs = range(len(results['train_loss']))
    plt.figure(figsize = (15, 7 ))
    plt.subplot(1 , 2, 1)
    plt.plot(epochs , loss , label = 'train_loss')
    plt.plot(epochs , test_loss, label = 'test_loss')
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(epochs , accuracy , label = 'train accuracy')
    plt.plot(epochs  ,test_accuracy , label = 'test accuracy')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

def train(model:nn.Module , train_dataloader:torch.utils.data.DataLoader,test_dataloader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer , loss_fn:nn.Module= nn.CrossEntropyLoss(),device:torch.device = None ,epochs :int = 5):
    results = {"train_loss":[],
               "train_acc":[],
                "test_loss":[],
               "test_acc":[]
              }
    
    
    for epoch in tqdm(range(epochs)):
            train_loss , train_acc = train_step(model , loss_fn , optimizer , train_dataloader , device= device 
                                                 )
            test_loss, test_acc = test_step(model=model,
                    data_loader=test_dataloader,
                    loss_fn=loss_fn , device= device 
                    )
            print(
                    f"Epoch: {epoch+1} | "
                    f"train_loss: {train_loss:.4f} | "
                    f"train_acc: {train_acc:.4f} | "
                    f"test_loss: {test_loss:.4f} | "
                    f"test_acc: {test_acc:.4f}"
                )

            results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
            results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
            results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
            results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
    return results
