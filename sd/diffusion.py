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
class UNET_ResidualBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, n_time = 1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        self.linear_time = nn.Linear(n_time, out_channels)
        
        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
            
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size = 1)
    
    def forward(self, feature, time):
        #feature : batch_size, in_channels, height, width
        #time (1, 1280)
        
        residue = feature
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)
        time = F.silu(time)
        time = self.linear_time(time)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        return merged + self.residual_layer(residue)
class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embd: int, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        self.gorupnorm = nn.GroupNorm(32, channels)
        self.conv_intput = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        self.layernom_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias =False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.Attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias = False)
        self.layernom_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels	* 2) 
        self.linear_geglu_2 = nn.Linear(4*channels, channels)
        
    def forward(self, x, context):
        #x: (Batch_size, feature, height, width)
        #contxt : batch_size, seq_len, dim

        residue_long = x
        x = self.groupnorm(x)
    
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        #(Batch_size, feature, height, width) -> (Batch_size,  feature, height * width)
        x = x.view(n, c, h * w)
        #(Batch_size, height * width, feature) -> (Batch_size, height * width, 4 * feature)
        x = x.transpose(-1, -2)
        #normalization + self.attention with skip connection
        residue_short = x
        
        x = self.layernorm1(x)
        self.attention_1(x)
        x = x + residue_short
        
        residue_short = x
        
        #normalization + cross attention with skip connection
        
        x = self.layernorm2(x)
        #cross attention
        self.Attention_2(x, context)
        
        x += residue_short

        residue_short = x
        #normalization + feed forward with skip connection
        
        x = self.layernorm3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim = -1)
        x = x * F.silu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short
        
        #batch_size, width * height, feature -> batch_size, feature, height, width
        x = x.transpose(-1, -2)
        
        x.view((n, c, h, w))
        
        return self.conv_output(x) + residue_long
class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size = 3, padding = 1)
        
    def forward(self) -> torch.Tensor:
        #(Batch_size, feature, heigth, width) --> (Batch_size, feature, 2 * heigth, 2 * width)
        x = F.interpolate(x, scale_factor = 2, mode = 'nearest')
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
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 32, Width / 32) 
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 2560, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1920, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
            
            # (Batch_Size, 1920, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 1280, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16)
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 960, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),
            
            # (Batch_Size, 960, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 640, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.groupNorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1)
        
    def forward(self, x):
        #batch_size, 320, height/8, width/8
        x = self.groupNorm(x)
        x = F.silu(x)
        x = self.conv(x)
        
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
        
        return output