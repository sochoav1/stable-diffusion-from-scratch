import numpy as np
import torch


class DDPMSampler:
    def __init__(
        self,
        generator: torch.Generator,
        num_train_steps=1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.0120,
    ):
        self.betas = (
            torch.linspace(
                beta_start**0.5, beta_end**0.5, num_train_steps, dtype=torch.float32
            )
            ** 2
        )
        self.alphas = 1 - self.betas
        self.alpha_acum_prod = torch.cumprod(self.alphas, dim=0)  # acumulative product
        self.one = torch.tensor(1.0)
        self.generator = generator
        self.num_train_steps = num_train_steps
        self.timesteps = torch.from_numpy(np.arange(0, num_train_steps)[::-1].copy())

    def set_inference_timesteps(self, num_inference_steps=50):
        self.nump_inference_steps = num_inference_steps
        step_ratio = self.num_train_steps // self.num_inference_steps
        timesteps = (
            (np.arange(0, num_inference_steps) * step_ratio)[::-1]
            .copy()
            .astype(np.int64)
        )
        self.timesteps = torch.from_numpy(timesteps)

    def _get_previous_timestep(self, timestep: int) -> int:
        prev_t = self._get_previous_timestep(
            self.num_training_steps // self.num_inference_steps
        )

        return prev_t

    def _get_variance(self, timestep: int) -> torch.Tensor:
        prev_t = self._get_previous_timestep(timestep)

        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.one
        current_beta_t = 1 - alpha_prod_t / alpha_prod_t_prev

        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * current_beta_t

        variance = torch.clamp(variance, min=1e-20)

        return variance

    def set_strength(self, strength=1):

        start_step = self.num_inference_steps - int(self.num_inference_steps * strength)
        self.timesteps = self.timesteps[start_step:]
        self.start_step = start_step

    def step(self, timestep: int, latents: torch.Tensor, model_output: torch.Tensor):
        t = timestep
        prev_t = self._get_previous_timestep(t)

        alpha_prod_t = self.alpha_acum_prod[timestep]
        alpha_prod_t_prev = self.alpha_cump_prod[prev_t] if prev_t >= 0 else self.one
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        curr_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        pred_original_sample = (
            latents - beta_prod_t ** (0.5) * model_output
        ) / alpha_prod_t ** (0.5)

        pred_original_sample_coeff = (
            alpha_prod_t_prev ** (0.5) * current_beta_t
        ) / beta_prod_t
        current_sample_coeff = current_alpha_t ** (0.5) * beta_prod_t_prev / beta_prod_t

        pred_prev_sample = (
            pred_original_sample_coeff * pred_original_sample
            + current_sample_coeff * latents
        )

        variance = 0
        if t > 0:
            device = model_output.device
            noise = torch.randn(
                model_output.shape,
                generator=self.generator,
                device=device,
                dtype=model_output.dtype,
            )
            variance = (self._get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(
        self, original_sample: torch.FloatTensor, timesteps: torch.IntTensor
    ) -> torch.FloatTensor:
        alpha_acum_prod = torch.cumprod.to(device=original_sample.device.dtype)
        timesteps = timesteps.to(original_sample.device)

        sqrt_alpha_prod = alpha_acum_prod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original_sample.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alpha_acum_prod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original_sample.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        # taken from paper (eq 4 DDPM paper)
        noise = torch.randn(
            original_sample.shape,
            generator=self.generator,
            device=original_sample.device,
            dtype=original_sample.dtype,
        )
        noisy_sample = (sqrt_alpha_prod * original_sample) + (
            sqrt_one_minus_alpha_prod
        ) * noise
        return noisy_sample
