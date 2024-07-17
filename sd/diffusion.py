import torch
from attention import SelfAttention, CrossAttention
from torch import nn
from torch.nn import functional as F

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
        
    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor):
        #latent : (Batch_size, 4, height/8, width/8)
        #context: (Batch_size, seq_len, dim)
        #time (1, 320)
        
        time = self.time_embedding(time)
        
    