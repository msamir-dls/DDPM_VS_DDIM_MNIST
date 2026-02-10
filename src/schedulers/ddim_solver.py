# src/schedulers/ddim_solver.py
import torch
import numpy as np
from tqdm import tqdm
from .gaussian import GaussianDiffusion

class DDIM(GaussianDiffusion):
    """
    DDIM (Denoising Diffusion Implicit Models) Scheduler
    Uses the same trained model as DDPM but with faster sampling.
    
    Paper: https://arxiv.org/abs/2010.02502
    """
    
    def __init__(self, config):
        # Make sure config has 'diffusion' key
        if 'diffusion' not in config:
            # Create default diffusion config
            config['diffusion'] = {
                'timesteps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'schedule': 'linear'
            }
        
        # Make sure config has 'ddim' key
        if 'ddim' not in config:
            config['ddim'] = {'sampling_steps': 50, 'eta': 0.0}
        
        super().__init__(config)
        self.ddim_timesteps = config['ddim']['sampling_steps']
        self.eta = config['ddim'].get('eta', 0.0)
        
        # Create DDIM sampling sequence
        self.ddim_timestep_sequence = self.make_ddim_timesteps()

        
    def make_ddim_timesteps(self):
        """Create a subsequence of timesteps for DDIM sampling"""
        # Take equally spaced steps from the original 1000 timesteps
        step_ratio = self.timesteps // self.ddim_timesteps
        ddim_timesteps = np.asarray(
            list(range(0, self.timesteps, step_ratio))
        ) + 1
        ddim_timesteps = ddim_timesteps[::-1].copy()  # Reverse for sampling
        
        # Add the last timestep if not included
        if ddim_timesteps[-1] != 0:
            ddim_timesteps = np.append(ddim_timesteps, 0)
            
        return ddim_timesteps.astype(np.int64)
    
    def ddim_step(self, model_output, timestep, sample, prev_timestep, eta=0.0):
        """
        DDIM reverse step (Equation 12 in DDIM paper)
        
        Args:
            model_output: Predicted noise from the model
            timestep: Current timestep
            sample: Current noisy sample x_t
            prev_timestep: Previous timestep
            eta: Controls stochasticity (0 = deterministic DDIM)
        """
        # Get alpha and sigma values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)
        
        # Current prediction for x_0
        pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)
        
        # Direction pointing to x_t
        direction_to_xt = torch.sqrt(1 - alpha_prod_t_prev - eta**2) * model_output
        
        # Random noise
        noise = eta * torch.randn_like(model_output)
        
        # DDIM update
        prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample +
            direction_to_xt +
            noise
        )
        
        return prev_sample, pred_original_sample
    
    @torch.no_grad()
    def sample(self, model, shape, return_intermediates=False):
        """
        DDIM Sampling: 10-50x faster than DDPM
        
        Args:
            model: Trained UNet (same as DDPM)
            shape: (batch_size, channels, height, width)
            return_intermediates: Return intermediate denoising steps
        """
        device = next(model.parameters()).device
        batch_size = shape[0]
        
        # Start from pure noise
        sample = torch.randn(shape, device=device)
        
        intermediates = []
        if return_intermediates:
            intermediates.append(sample.cpu())
        
        # DDIM sampling loop (much fewer steps!)
        timesteps = self.ddim_timestep_sequence
        print(f" DDIM Sampling: {len(timesteps)-1} steps (vs {self.timesteps} for DDPM)")
        
        for i, (timestep, prev_timestep) in tqdm(
            enumerate(zip(timesteps[:-1], timesteps[1:])),
            total=len(timesteps)-1,
            desc="DDIM Sampling"
        ):
            # Create timestep tensor
            t = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
            
            # Predict noise
            model_output = model(sample, t)
            
            # DDIM update
            sample, pred_x0 = self.ddim_step(
                model_output=model_output,
                timestep=timestep,
                sample=sample,
                prev_timestep=prev_timestep,
                eta=self.eta
            )
            
            if return_intermediates and i % (len(timesteps) // 10) == 0:
                intermediates.append(sample.cpu())
        
        if return_intermediates:
            return sample, intermediates
        return sample
    
    def ddim_inversion(self, model, x0, num_inference_steps=50):
        """
        DDIM Inversion: Convert image to latent noise
        Useful for image editing applications
        """
        device = next(model.parameters()).device
        timesteps = self.ddim_timestep_sequence
        
        # Forward process (deterministic)
        x = x0.clone()
        trajectory = [x.cpu()]
        
        for timestep in tqdm(timesteps, desc="DDIM Inversion"):
            t = torch.full((x.shape[0],), timestep, device=device, dtype=torch.long)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = model(x, t)
            
            # Inversion step
            alpha_prod_t = self.alphas_cumprod[timestep]
            if timestep > 0:
                alpha_prod_t_prev = self.alphas_cumprod[timestep-1]
                noise = (x - torch.sqrt(alpha_prod_t) * x0) / torch.sqrt(1 - alpha_prod_t)
                x = torch.sqrt(alpha_prod_t_prev) * x0 + torch.sqrt(1 - alpha_prod_t_prev) * noise
            else:
                x = x0
            
            trajectory.append(x.cpu())
        
        return x, trajectory
    
    def interpolate(self, model, x1, x2, alpha=0.5, steps=50):
        """
        Latent space interpolation between two images
        """
        # Invert both images to noise space
        z1, _ = self.ddim_inversion(model, x1, steps)
        z2, _ = self.ddim_inversion(model, x2, steps)
        
        # Interpolate in noise space
        z_interp = (1 - alpha) * z1 + alpha * z2
        
        # Sample from interpolated noise
        samples = self.sample(model, z_interp.shape)
        
        return samples