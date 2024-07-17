import torch
from attention import CrossAttention, SelfAttention
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, n_embd: int):
        super().__init__()
        self.linear1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear2 = nn.Linear(4 * n_embd, 4 * n_embd)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(1, 320)
        x = self.linear1(x)
        x = F.silu(x)
        x = self.linear2(x)
        
        # (1, 1280)
        
        return x
        
class SwitchSequential(nn.Sequential):
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer,UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
                
        return x
class UNET(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.encoders = nn.Module([
            #Bartch_size, 4, height/8, width/8
            SwitchSequential(nn.Conv2d(4, 320, kenrl_size = 3, padding = 1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            #Bartch_size, 320, height/8, width/8 -> Bartch_size, 320, height/16, width/16
            SwitchSequential(nn.Conv2d(320, 320, kenrl_size = 3, stride = 2,padding = 1)),
            
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # Bartch_size, 640, height/16, width/16 -> Bartch_size, 640, height/32, width/32
            SwitchSequential(nn.Conv2d(640, 640, kenrl_size = 3, stride = 2,padding = 1)),
            
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # Bartch_size, 1280, height/32, width/32 -> Bartch_size, 1280, height/64, width/64
            SwitchSequential(nn.Conv2d(1280, 1280, kenrl_size = 3, stride = 2,padding = 1)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),
            UNIT_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )
class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
        
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        #latent : (Batch_size, 4, height/8, width/8)
        #context: (Batch_size, seq_len, dim)
        #time (1, 320)
        
        
        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        #(Batch_size, 4, height/8, width/8) -> (Batch_size, 320, height/8, width/8)
        output = self.unet(latent, context, time)
        
        #(Batch_size, 320, height/8, width/8) -> #(Batch_size, 4, height/8, width/8)
        output = self.final(output)
        
        return