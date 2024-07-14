import torch
from decoder import VAE_attentionBlock, VAE_ResidualBlock
from torch import nn
from torch.nn import functional as F


class VAE_encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # Batch_size, Channel, Height, Width) --> (Batch_size, 128, height, width)
            nn.Conv2d(3, 128, padding=1),
            
            #(Batch_size, 128, height, width) --> (Batch_size, 128, height  , width)
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            #(Batch_size, 128, height, width) --> (Batch_size, 128, height/2, width/2)
            nn.Conv2d(128, 128, kernel_size= 3, stride= 2, padding= 0),
            
            #(Batch_size, 128, height/2, width/2) --> (Batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(128, 256),
            
            #(Batch_size, 256, height/2, width/2) --> (Batch_size, 256, height/2, width/2)
            VAE_ResidualBlock(256, 256),
            
            # (Batch_size, 256, height/2, width/2) --> (Batch_size, 256, height/4, width/4)
            nn.Conv2d(256, 256, kernel_size= 3, stride= 2, padding= 0),
            
            #(Batch_size, 256, height/4, width/4) --> (Batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(256, 512),
            
            #(Batch_size, 512, height/4, width/4) --> (Batch_size, 512, height/4, width/4)
            VAE_ResidualBlock(512, 512),
            
            #(Batch_size, 512, height/4, width/4) --> (Batch_size, 512, height/8, width/8)
            nn.Conv2d(512, 512, kernel_size= 3, stride= 2, padding= 0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            #(Batch_size, 512, height/8, width/8) --> (Batch_size, 512, height/8, width/8)
            VAE_ResidualBlock(512, 512),
            
            #(Batch_size, 512, height/8, width/8) --> (Batch_size, 512, height/8, width/8)
            VAE_attentionBlock(512, 512),
            
            #(Batch_size, 512, height/8, width/8) --> (Batch_size, 512, height/8, width/8)
            nn.Group(32, 512),
            nn.SiLU(),
            
            #(Batch_size, 512, height/8, width/8) --> (Batch_size, 8, height/8, width/8)
            nn.Conv2d(512, 8, kernel_size= 1, padding= 1),
            
            #(Batch_size, 8, height/8, width/8) --> (Batch_size, 8, height/8, width/8)
            nn.Conv2d(8, 8, kernel_size= 1, padding= 1)
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        #X = (Batch_size, 3, height, width) 
        #noise =  (Batch_size, Out_Channel, height/8, width/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                
                #(padding_left, padding_right, padding_top, padding_bottom) (0, 1, 0, 1) asimetrical padding
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
    
        #(Batch_size, 8, height/8, width/8) --> Two tensor of shape (Batch_size, 4, height/8, width/8)
        mean, log_variance = torch.chunk(x, 2, dim = 1)
        
        log_variance = torch.clamp(log_variance, min = -30, max = 20)
        
        variance = torch.exp(log_variance)
        
        std = torch.sqrt(variance)
        
        #Z(0, 1) = N(mean, std) = x?
        # X = mean + std * noise
        
        x = mean + std * noise
        
        #Scale by constant (taken from the paper)
        x *= 0.18215
        
        return x