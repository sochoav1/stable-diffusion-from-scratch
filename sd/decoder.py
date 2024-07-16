import torch 
from torch import nn
from torch.nn import functional as F
from attention import selfAttention

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
        
