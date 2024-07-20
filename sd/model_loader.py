from clip import CLIP
from encoder import VAE_encoder
from decoder import VAE_decoder
from diffusion import Diffusion

def preload_model_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)
    
    encoder = VAE_encoder()
    encoder.load_state_dict(state_dict['encoder'], strict = True)
    
    decoder = VAE_decoder()
    decoder.load_state_dict(state_dict['decoder'], strict = True)
    
    