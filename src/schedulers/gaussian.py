import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class GaussianDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.timesteps = config['diffusion']['timesteps']
        beta_start = config['diffusion']['beta_start']
        beta_end = config['diffusion']['beta_end']

        # 1. Define beta schedule (linear as per DDPM)
        betas = torch.linspace(beta_start, beta_end, self.timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # 2. Register buffers for the forward process (q)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        
        # 3. Register buffers for the reverse process (p)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        
        # 4. Calculations for posterior q(x_{t-1} | x_t, x_0)
        # This is used for the variance in the reverse step
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))

    def _extract(self, a, t, x_shape):
        """Helper to extract values for a specific batch of timesteps."""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def q_sample(self, x_start, t, noise=None):
        """Forward process: adds noise to the original image (q(x_t | x_0))."""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, model, x_start, t, noise=None, loss_type="l2"):
        """Calculate loss for training (simpler interface).
        
        Args:
            model: The denoising UNet
            x_start: Original clean images
            t: Batch of timesteps
            noise: Optional pre-generated noise
            loss_type: 'l1', 'l2', or 'huber' loss
            
        Returns:
            Loss value
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = model(x_noisy, t)
        
        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError(f"Unknown loss type: {loss_type}")
        
        return loss

    @torch.no_grad()
    def p_sample(self, model, x, t, t_index):
        """Reverse process: one step of DDPM sampling (p(x_{t-1} | x_t))."""
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in DDPM paper: Predicted mean of x_{t-1}
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            # Add noise for every step except the last one (Langevin dynamics)
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def sample(self, model, shape, return_intermediates=False, interval=100):
        """Full DDPM Sampling Loop.
        
        Args:
            model: Trained denoising UNet
            shape: Tuple of (batch_size, channels, height, width)
            return_intermediates: If True, return intermediate denoising steps
            interval: How often to save intermediates (in timesteps)
            
        Returns:
            Generated samples (and intermediates if requested)
        """
        device = next(model.parameters()).device
        b = shape[0]
        
        # Validate shape has 4 dimensions
        if len(shape) != 4:
            raise ValueError(f"Shape should be 4D (B, C, H, W), got {shape}")
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        intermediates = []
        if return_intermediates:
            intermediates.append(img.cpu())
        
        # Iterate backwards from T to 0
        for i in tqdm(reversed(range(0, self.timesteps)), desc='DDPM Sampling', total=self.timesteps):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(model, img, t, i)
            
            # Save intermediate steps if requested
            if return_intermediates and i % interval == 0:
                intermediates.append(img.cpu())
        
        if return_intermediates:
            return img, intermediates
        return img

    def get_noise_schedule_info(self):
        """Returns information about the noise schedule for analysis."""
        return {
            'timesteps': self.timesteps,
            'betas': self.betas.cpu().numpy(),
            'alphas': self.alphas.cpu().numpy(),
            'alphas_cumprod': self.alphas_cumprod.cpu().numpy(),
            'sqrt_one_minus_alphas_cumprod': self.sqrt_one_minus_alphas_cumprod.cpu().numpy()
        }

    @torch.no_grad()
    def denoise_step(self, model, x_t, t):
        """Single denoising step (useful for visualization)."""
        # This is essentially the noise prediction part
        return model(x_t, t)

    def compute_snr(self, t):
        """Compute Signal-to-Noise Ratio at timestep t.
        
        Useful for analyzing the noise schedule.
        """
        alpha_bar_t = self._extract(self.alphas_cumprod, t, (1, 1, 1, 1))
        snr = alpha_bar_t / (1 - alpha_bar_t)
        return snr


# Optional: Utility function for testing
def test_gaussian_diffusion():
    """Quick test function to verify the GaussianDiffusion class."""
    # Mock config
    config = {
        'diffusion': {
            'timesteps': 1000,
            'beta_start': 0.0001,
            'beta_end': 0.02
        }
    }
    
    # Create instance
    diffusion = GaussianDiffusion(config)
    
    print("Testing buffers...")
    print(f"  - betas shape: {diffusion.betas.shape}")
    print(f"  - alphas_cumprod shape: {diffusion.alphas_cumprod.shape}")
    print(f"  - posterior_variance shape: {diffusion.posterior_variance.shape}")
    
    # Test forward process
    batch_size = 4
    x_start = torch.randn(batch_size, 1, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    x_t = diffusion.q_sample(x_start, t)
    print(f"  - q_sample works: input {x_start.shape} -> output {x_t.shape}")
    
    # Test reverse process (mock model)
    class MockModel(nn.Module):
        def forward(self, x, t):
            return torch.randn_like(x)
    
    model = MockModel()
    
    # Test sampling
    shape = (2, 1, 32, 32)
    samples = diffusion.sample(model, shape)
    print(f"  - sampling works: output shape {samples.shape}")
    
    print("\nGaussianDiffusion test passed!")
    
    return diffusion


if __name__ == "__main__":
    # Run test if file is executed directly
    test_gaussian_diffusion()