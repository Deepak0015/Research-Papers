import torch
from torch import nn
class TinyVGG(nn.Module):
    def __init__(self , input_shape:int  , output_shape :int , hidden_units:int):
        super().__init__()
        self.conv_block1  = nn.Sequential(
            nn.Conv2d(in_channels = input_shape , out_channels = hidden_units , kernel_size = 3 , stride = 1 , padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels  = hidden_units  , out_channels = hidden_units , kernel_size = 3 , stride = 1 , padding = 1 ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size  = 2 , stride = 2 )
            
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_units , hidden_units , kernel_size = 3 , padding = 1 ),
            nn.ReLU(),
            nn.Conv2d(hidden_units  , hidden_units , kernel_size = 3 , padding  =1 )
            ,
            nn.ReLU()
            ,
            nn.MaxPool2d(2)
            
        )
            
        self.classifier   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_units * 16 * 16  , out_features = output_shape )
            
            
        )
        
    def forward(self , x ):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x 
    
    

