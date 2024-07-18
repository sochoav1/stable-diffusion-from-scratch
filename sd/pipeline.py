import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENS_WIDTH = WIDTH // 8
LATENS_HEIGHT = HEIGHT // 8

def generate(prompt: str, uncond_prompt: str, input_image=None, strength=0.8, 
             do_cfg=True, cfg_scale=7.5, sampler_name="ddpm", inference_steps=50,
             seed=None, idle_device=None, tokenizer=None):
    
    with torch.no_grad():
        if not (0 <= strength <= 1):
            raise ValueError("Strength must be between 0 and 1")
        
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x
        
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)
        
        else:
            generator.manual_seed()
            
        clip = models['clip']
        clip.to(device)
        
        if do_cfg:
            #convert the promt in tokens
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length = 77).input_ids
            #(batch_size, seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device = device)

            