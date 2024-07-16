import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    
    def __init__(self, n_vocab: int, n_embed: int, n_token: int):
        super().__init__()
        self.token_embedding = nn.Embedding(n_vocab, n_embed)
        self.position_embedding = nn.Parameter(n_token, n_embed)
        
    def forward(self, tokens):
        #(Batch_size, seq_len) --> (Batch_size, seq_len, dim
        x = self.token_embedding(tokens)
        x += self.position_embedding

class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.Module([
            CLIPLayer(12, 768) for _ in range(12)
        ])
        
        self.layers_norm = nn.LayerNorm(768)
        
    def forward(self, token: torch.LongTensor) -> torch.FloatTensor:
        
        tokens = tokens.type(torch.long)
        
        #(Batch_size, seq_len) --> (Batch_size, seq_len, dim)
        state = self.embedding(tokens)
        
        for layer in self.layers:
            state = layer(state)
        
        
        #(Batch size, seq_len, dim)
        output = self.layers_norm(state)
        
        return output