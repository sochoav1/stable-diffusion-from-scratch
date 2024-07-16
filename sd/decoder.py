import torch
from attention import selfAttention
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(Batch_size, features, height, width)
        
        residue = x
        
        n, c, h, w = x.shape
        
        # (Batch_size, features, height, width) --> (Batch_size, features, height * width)
        x = x.view(n, c, h * w)
        
        #(Batch_size, features, height * width) --> (Batch_size, height * width, features)
        x = x.transpose = (-1, -2)
        
        #(Batch_size, height * width, features) --> (Batch_size, height * width, features)
        x = self.attention(x)
        #(Batch_size, height * width, features) --> (Batch_size, features, height * width)
        x = x.transpose(-1, -2)
        
        x *= residue
        
        return x

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.group_norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x = (B, C, H, W) batch, channels, height, width
            residual = x
            
            x = self.group_norm1(x)
            x = F.silu(x)
            x = self.conv1(x)
            x = self.group_norm2(x)
            x = F.silu(x)
            x = self.conv2(x)
            
            return x + self.residual_layer(residual)
        
class VAE_Decoder(nn.Sequential):
    
    def __init__():
        super().__init__(
            nn.Conv2d(4, 4, kernel_size= 1, padding = 0),
            
            nn.Conv2d(4, 512, kernel_size = 3, padding = 1),
            
            VAE_ResidualBlock(512, 512),
            
            VAE_AttentionBlock(512),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            #m (batch_size, 512, height/8, w/8)
            VAE_ResidualBlock(512, 512),
            
            #(batch_size, 512, height/8, w/8) --> (batch_size, 512, height/4, w/4) up1
            nn.Upsample(scale_factor = 2),
            
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            
            #2 up
            nn.Upsample(scale_factor = 2),
            
            nn.Conv2d(512, 512, kernel_size = 3, padding = 1),
            
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),
            
            #3 up
            nn.Upsample(scale_factor = 2),
            
            nn.Conv2d(256 , 256, kernel_size = 3, padding = 1),
            
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.GroupNorm(32, 128),
            
            nn.SiLU(),
            
            nn.Conv2(128, 3, kernel_size = 3, padding = 1),
        )
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 4, height/8, width/8)
        
        x/= 0.18215
        
        for module in self:
            x = module(x)
            
        return x
            
            
            
            
            
            
            
            
        