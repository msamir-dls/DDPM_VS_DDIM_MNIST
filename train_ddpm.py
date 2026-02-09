import os
import torch
import torch.nn.functional as F
import mlflow
from tqdm import tqdm
import sys
import torchvision  # For saving images

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_mlflow, time_execution, save_checkpoint, get_device
from src.dataset import get_dataloader
from src.models.unet import UNet
from src.schedulers.gaussian import GaussianDiffusion

@time_execution
def train_one_epoch(model, dataloader, diffusion, optimizer, device, config, epoch):
    model.train()
    pbar = tqdm(dataloader)
    total_loss = 0
    
    # Get loss type from config (default to "l2" if not specified)
    loss_type = config.get('train', {}).get('loss_type', 'l2')
    
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        # Sample random timesteps for each image in the batch
        t = torch.randint(0, config['diffusion']['timesteps'], (images.shape[0],), device=device).long()
        
        # Use the new p_losses method for cleaner code
        optimizer.zero_grad()
        loss = diffusion.p_losses(model, images, t, loss_type=loss_type)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_description(f"Epoch {epoch} | Loss: {loss.item():.4f}")
        
        # Log step-level loss to MLflow
        if batch_idx % 10 == 0:
            mlflow.log_metric("train_loss_step", loss.item(), step=epoch * len(dataloader) + batch_idx)
            
            # Optional: Log gradient norms for debugging
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            mlflow.log_metric("grad_norm", total_norm, step=epoch * len(dataloader) + batch_idx)

    return total_loss / len(dataloader)

def sample_and_log_images(model, diffusion, config, device, epoch, num_samples=8):
    """Helper function to generate and log sample images."""
    model.eval()
    with torch.no_grad():
        sample_shape = (num_samples, 
                       config['dataset']['channels'], 
                       config['dataset']['img_size'], 
                       config['dataset']['img_size'])
        
        # Generate samples with intermediate steps for visualization
        final_samples, intermediate_samples = diffusion.sample(
            model, 
            sample_shape, 
            return_intermediates=True,
            interval=200  # Save every 200 steps for 1000-step diffusion
        )
        
        # Unnormalize [-1, 1] to [0, 1]
        final_samples = (final_samples.clamp(-1, 1) + 1) / 2
        
        # Save final samples
        samples_path = f"checkpoints/samples_epoch_{epoch}.png"
        torchvision.utils.save_image(
            final_samples, 
            samples_path,
            nrow=4,
            normalize=True
        )
        mlflow.log_artifact(samples_path)
        
        # Optional: Save a grid of intermediate steps
        if len(intermediate_samples) > 0:
            # Create a grid of intermediate samples at different timesteps
            selected_indices = [0, len(intermediate_samples)//4, len(intermediate_samples)//2, -1]
            selected_samples = [intermediate_samples[i] for i in selected_indices]
            
            # Process each intermediate sample
            processed_samples = []
            for sample_batch in selected_samples:
                sample_batch = (sample_batch.clamp(-1, 1) + 1) / 2
                # Take first 4 samples from each batch
                processed_samples.append(sample_batch[:4])
            
            # Create a grid
            intermediate_grid = torch.cat(processed_samples, dim=0)
            intermediate_path = f"checkpoints/intermediates_epoch_{epoch}.png"
            torchvision.utils.save_image(
                intermediate_grid,
                intermediate_path,
                nrow=4,
                normalize=True
            )
            mlflow.log_artifact(intermediate_path)
        
    model.train()
    return final_samples

def validate_model(model, dataloader, diffusion, device, config, epoch):
    """Validation step (optional)."""
    model.eval()
    total_val_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            t = torch.randint(0, config['diffusion']['timesteps'], (images.shape[0],), device=device).long()
            
            loss = diffusion.p_losses(model, images, t, loss_type="l2")
            total_val_loss += loss.item()
            num_batches += 1
            
            # Break early for faster validation
            if num_batches >= 10:  # Validate on 10 batches max
                break
    
    avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0
    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
    return avg_val_loss

def main():
    # 1. Load configuration
    config_path = "configs/ddpm_train.yaml"
    config = load_config(config_path)
    print(f"[*] Loaded configuration from {config_path}")
    
    # Debug info
    print(f"DEBUG: num_workers = {config['train'].get('num_workers')}")
    print(f"DEBUG: batch_size = {config['train']['batch_size']}")
    
    device = get_device()
    print(f"[*] Using device: {device}")
    
    # 2. Setup MLflow
    mlflow.set_tracking_uri(config['mlflow'].get('tracking_uri', 'http://localhost:5000'))
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=config['run_name']):
        # Log the config file itself
        mlflow.log_artifact(config_path)
        
        # Log all hyperparameters
        setup_mlflow(config)
        
        # 3. Initialize Data, Model, and Diffusion
        dataloader = get_dataloader(config)
        model = UNet(config).to(device)
        diffusion = GaussianDiffusion(config).to(device)
        
        # Log model architecture info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[*] Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")
        mlflow.log_metric("total_params", total_params)
        mlflow.log_metric("trainable_params", trainable_params)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
        
        # Learning rate scheduler with optional warmup
        if config['train'].get('use_warmup', False):
            from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
            warmup_epochs = config['train'].get('warmup_epochs', 2)
            scheduler1 = LinearLR(
                optimizer, 
                start_factor=0.01, 
                end_factor=1.0, 
                total_iters=warmup_epochs
            )
            scheduler2 = CosineAnnealingLR(
                optimizer, 
                T_max=config['train']['epochs'] - warmup_epochs,
                eta_min=1e-6
            )
            scheduler = SequentialLR(
                optimizer, 
                schedulers=[scheduler1, scheduler2],
                milestones=[warmup_epochs]
            )
            print(f"[*] Using warmup scheduler: {warmup_epochs} epochs warmup")
        else:
            # Simple cosine annealing
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config['train']['epochs'], 
                eta_min=1e-6
            )
        
        print(f"[*] Starting DDPM Training on {device}...")
        print(f"[*] Training for {config['train']['epochs']} epochs")
        print(f"[*] Batch size: {config['train']['batch_size']}")
        print(f"[*] Learning rate: {config['train']['lr']}")
        
        # Create checkpoints directory if it doesn't exist
        os.makedirs("checkpoints", exist_ok=True)
        
        # 4. Training Loop
        for epoch in range(1, config['train']['epochs'] + 1):
            avg_loss = train_one_epoch(model, dataloader, diffusion, optimizer, device, config, epoch)
            
            # Step the scheduler ONCE PER EPOCH
            scheduler.step()
            
            # Log learning rate
            current_lr = scheduler.get_last_lr()[0]
            mlflow.log_metric("learning_rate", current_lr, step=epoch)
            
            # Log epoch metrics
            mlflow.log_metric("avg_loss_epoch", avg_loss, step=epoch)
            print(f"[+] Epoch {epoch:03d}/{config['train']['epochs']:03d} | "
                  f"Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
            
            # 5. Save Checkpoint and generate samples
            if epoch % config['train']['save_every'] == 0 or epoch == config['train']['epochs']:
                ckpt_path = f"checkpoints/ddpm_mnist_epoch_{epoch}.pth"
                
                # Enhanced checkpoint with scheduler state
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': config,
                    'loss': avg_loss,
                }, ckpt_path)
                
                print(f"[+] Checkpoint saved at {ckpt_path}")
                mlflow.log_artifact(ckpt_path)
                
                # Generate and log sample images
                print(f"[*] Generating samples for epoch {epoch}...")
                samples = sample_and_log_images(model, diffusion, config, device, epoch)
                
                # Optional: Log noise schedule info for analysis
                if epoch == 1 or epoch % 10 == 0:
                    schedule_info = diffusion.get_noise_schedule_info()
                    # Log some key metrics about noise schedule
                    mlflow.log_metric("beta_start", schedule_info['betas'][0], step=epoch)
                    mlflow.log_metric("beta_end", schedule_info['betas'][-1], step=epoch)
                    mlflow.log_metric("alpha_cumprod_start", schedule_info['alphas_cumprod'][0], step=epoch)
                    mlflow.log_metric("alpha_cumprod_end", schedule_info['alphas_cumprod'][-1], step=epoch)

        # Save final model
        final_path = "checkpoints/ddpm_mnist_final.pth"
        torch.save({
            'epoch': config['train']['epochs'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'config': config,
            'final_loss': avg_loss,
        }, final_path)
        
        mlflow.log_artifact(final_path)
        print(f"[+] Final model saved at {final_path}")
        
        # Final sample generation
        print("[*] Generating final samples...")
        sample_and_log_images(model, diffusion, config, device, "final")

if __name__ == "__main__":
    main()