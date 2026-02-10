import os
import torch
import torchvision
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import after adding path
from src.models.unet import UNet
from src.schedulers.ddim_solver import DDIM
from src.utils import load_config

def load_trained_model(checkpoint_path, config):
    """Load trained DDPM model for DDIM sampling"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = UNet(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load weights with strict=False to handle architecture differences
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print(f" Loaded model from {checkpoint_path}")
    return model, device

def main():
    """Simple DDIM inference"""
    # Load config or create default
    try:
        config = load_config("configs/ddim_inference.yaml")
    except:
        print("  Using default config")
        config = {
            'dataset': {'img_size': 32, 'channels': 1},
            'model': {
                'base_channels': 32,
                'channel_mult': [1, 2, 4],
                'num_res_blocks': 2,
                'attention': [False, False, False],
                'dropout': 0.1
            },
            'diffusion': {
                'timesteps': 1000,
                'beta_start': 0.0001,
                'beta_end': 0.02,
                'schedule': 'linear'
            },
            'ddim': {
                'sampling_steps': 50,
                'eta': 0.0
            }
        }
    
    # Ensure config has required keys
    if 'diffusion' not in config:
        config['diffusion'] = {
            'timesteps': 1000,
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'schedule': 'linear'
        }
    
    if 'ddim' not in config:
        config['ddim'] = {'sampling_steps': 50, 'eta': 0.0}
    
    print(f" DDIM Configuration:")
    print(f"   - Sampling steps: {config['ddim']['sampling_steps']}")
    print(f"   - Timesteps: {config['diffusion']['timesteps']}")
    
    # Load model
    model, device = load_trained_model('checkpoints/ddpm_final.pth', config)
    
    # Initialize DDIM
    ddim = DDIM(config).to(device)
    
    # Generate samples
    num_samples = 16
    print(f"\n Generating {num_samples} samples...")
    
    with torch.no_grad():
        sample_shape = (num_samples, 
                       config['dataset']['channels'], 
                       config['dataset']['img_size'], 
                       config['dataset']['img_size'])
        
        samples = ddim.sample(model, sample_shape)
        samples = (samples.clamp(-1, 1) + 1) / 2  # Convert to [0, 1]
    
    # Save grid
    os.makedirs('ddim_outputs', exist_ok=True)
    grid_path = 'ddim_outputs/ddim_samples.png'
    
    grid = torchvision.utils.make_grid(samples, nrow=4, padding=2)
    torchvision.utils.save_image(grid, grid_path, normalize=True)
    
    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy().squeeze(), cmap='gray')
    plt.title(f'DDIM Samples ({config["ddim"]["sampling_steps"]} steps)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('ddim_outputs/ddim_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n DDIM generation complete!")
    print(f"   - Generated {num_samples} samples")
    print(f"   - Saved to: {grid_path}")
    print(f"   - Check 'ddim_outputs/' folder")

if __name__ == "__main__":
    main()