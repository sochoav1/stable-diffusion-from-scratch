import numpy as np
import torch
from ddpm import DDPMSampler
from tqdm import tqdm

WIDTH = 512
HEIGHT = 512
LATENS_WIDTH = WIDTH // 8
LATENS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
            context = torch.cat([cond_context, uncond_context])
        else:
            # Convert into a list of length Seq_Len=77
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            context = clip(tokens)
        to_idle(clip)
        
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            raise ValueError("Sampler not recognized")
        
        latens_shape = (1, 4, LATENS_HEIGHT, LATENS_WIDTH)
        
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)
            
            input_image_tensor = input_image.resize(WIDTH, HEIGHT)
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)
            
            encoder_noise = torch.randn(latens_shape, generator = generator, device = device)
            
            #pass imagen through the VAE
            latens = encoder(input_image_tensor, encoder_noise)
            
            sampler.set_strength(strength = strength)
            latents = sampler.add_noise(latents,sampler.timesteps[0])
            
            to_idle(encoder)
        
        else:
            latens = torch.randn(latens_shape, generator = generator, device = device)
            sampler.set_strength(strength = strength)
            latens = sampler.add_noise(latens, sampler.timesteps[0])
            
        diffusion = models["diffusion"]
        diffusion.to(device)
        
        timesteps = tqdm(sampler.timesteps)
        
        for i, timestep in enumerate(timesteps):
            time_embbeding = get_time_embedding(timestep).to(device)
            
        model_input = latents
        
        if do_cfg:
            #batch_size, 4, latents_height, latents_width) -> 2 * batch_size, 4, latents_height, latents_width)
            model_input = model_input.repeat(2, 1, 1, 1)
            
            model_output = diffusion(model_input, context, time_embbeding)
            
        if do_cfg:
            output_cond, output_uncond = model_output.chunk(2)
            model_output = cfg_scale *(output_cond - output_uncond) * output_uncond
        #remove noice predicted    
        latents = sampler.step(timestep, latents, model_output)
    to_idle(diffusion)
    
    decoder = models["decoder"]
    decoder.to(device)
    
    images = decoder(latents)
    
    to_idle(decoder)
    
    images = rescale(images, (-1, 1), (0, 255), clamp = True)
    
    images = images.permute(0, 2, 3, 1)
    images = images.to("cpu", torch.uint8).numpy()
    return images[0] 

def rescale(x, old_range, new_range, clamp = False):
    old_min, old_range = old_range
    new_min,new_max = new_range
    x -= old_min
    x *=(new_max-new_min) / (old_range - old_min)
    x += new_min
    if clamp:
        x = x,clamp(new_min, new_max)
        
    return x

def get_time_embedding(timestep):
    freqs = torch.pow(10000, -torch.arange(start=0,end=100,dtype=torch.float32)/160)
    x = torch.tensor(timestep, dtype=torch.float32)[:,None]*freqs[None]
    return torch.cat([torch.cos(x), torch.sin])