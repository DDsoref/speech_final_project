"""
Training Utilities and Trainer Class

Generic training loop with checkpointing, logging, and early stopping.

Paper Reference: Section IV.A - Training specifications
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import json
import time

from ..utils.audio import si_snr_loss
from ..evaluation.metrics import MetricsCalculator


class Trainer:
    """
    Generic trainer for speech enhancement models
    
    Paper: Section IV.A
    - Optimizer: Adam with lr=15e-5
    - Loss: SI-SNR
    - Training monitoring with validation
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        device: str = 'cuda',
        scheduler: Optional[any] = None,
        use_amp: bool = False,
        log_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None
    ):
        """
        Args:
            model: Model to train
            optimizer: Optimizer
            device: Device to use ('cuda', 'cpu', 'mps')
            scheduler: Learning rate scheduler
            use_amp: Use automatic mixed precision
            log_dir: Directory for logs
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.use_amp = use_amp
        
        # Directories
        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator(
            sample_rate=8000,
            metrics=['si_snr', 'sdr']
        )
        
        # AMP scaler
        if use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        max_grad_norm: float = 5.0,
        log_every_n_steps: int = 100
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            max_grad_norm: Maximum gradient norm for clipping
            log_every_n_steps: Log frequency
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (noisy, clean, lengths, info) in enumerate(pbar):
            # Move to device
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass (with or without AMP)
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    enhanced = self.model(noisy)
                    loss = si_snr_loss(enhanced, clean)
                
                # Backward with scaling
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                enhanced = self.model(noisy)
                loss = si_snr_loss(enhanced, clean)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
            
            # Logging
            if self.global_step % log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                print(f"  Step {self.global_step}: Loss = {avg_loss:.4f}")
        
        # Epoch metrics
        avg_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_loss,
            'epoch': self.current_epoch
        }
    
    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        compute_metrics: bool = True
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            compute_metrics: Whether to compute detailed metrics
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        # For detailed metrics
        all_metrics = [] if compute_metrics else None
        
        pbar = tqdm(val_loader, desc="Validation")
        
        for noisy, clean, lengths, info in pbar:
            noisy = noisy.to(self.device)
            clean = clean.to(self.device)
            
            # Forward pass
            enhanced = self.model(noisy)
            
            # Loss
            loss = si_snr_loss(enhanced, clean)
            total_loss += loss.item()
            num_batches += 1
            
            # Compute detailed metrics
            if compute_metrics:
                batch_metrics = self.metrics_calc.calculate_batch(
                    enhanced, clean, lengths
                )
                
                # Convert to list of dicts
                for i in range(len(noisy)):
                    sample_metrics = {
                        metric: values[i]
                        for metric, values in batch_metrics.items()
                    }
                    all_metrics.append(sample_metrics)
        
        # Aggregate metrics
        avg_loss = total_loss / num_batches
        results = {'val_loss': avg_loss}
        
        if compute_metrics and all_metrics:
            aggregated = self.metrics_calc.aggregate_metrics(all_metrics)
            results.update(aggregated)
        
        return results
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        early_stopping_patience: int = 10,
        save_every_n_epochs: int = 5,
        validate_every_n_epochs: int = 1
    ) -> Dict[str, list]:
        """
        Complete training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            early_stopping_patience: Patience for early stopping
            save_every_n_epochs: Checkpoint save frequency
            validate_every_n_epochs: Validation frequency
        
        Returns:
            Training history
        """
        print(f"\n{'='*70}")
        print(f"Starting Training".center(70))
        print(f"{'='*70}\n")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            
            # Get learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            history['learning_rate'].append(current_lr)
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Validation
            if (epoch + 1) % validate_every_n_epochs == 0:
                val_metrics = self.validate(val_loader, compute_metrics=True)
                history['val_loss'].append(val_metrics['val_loss'])
                
                print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
                if 'si_snr_mean' in val_metrics:
                    print(f"  Val SI-SNR: {val_metrics['si_snr_mean']:.2f} dB")
                
                # Learning rate scheduler
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Check for improvement
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_model.pt",
                        val_metrics
                    )
                    print("  ✓ New best model saved!")
                else:
                    self.epochs_without_improvement += 1
                    print(f"  No improvement for {self.epochs_without_improvement} epochs")
                
                # Early stopping
                if self.epochs_without_improvement >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            
            # Regular checkpoint
            if (epoch + 1) % save_every_n_epochs == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
                )
            
            # Epoch time
            epoch_time = time.time() - epoch_start_time
            print(f"  Epoch time: {epoch_time:.2f}s")
            print()
        
        print(f"\n{'='*70}")
        print(f"Training Complete".center(70))
        print(f"{'='*70}\n")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint(
            self.checkpoint_dir / "final_model.pt"
        )
        
        # Save training history
        history_path = self.log_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        
        return history
    
    def save_checkpoint(
        self,
        path: Path,
        metrics: Optional[Dict] = None
    ):
        """
        Save training checkpoint
        
        Args:
            path: Path to save checkpoint
            metrics: Optional metrics to save
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'epochs_without_improvement': self.epochs_without_improvement
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if metrics:
            checkpoint['metrics'] = metrics
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: Path):
        """
        Load training checkpoint
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.epochs_without_improvement = checkpoint['epochs_without_improvement']
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Checkpoint loaded: {path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")


# ============================================================================
# Utility Functions
# ============================================================================

def setup_optimizer(
    model: nn.Module,
    learning_rate: float = 15e-5,
    weight_decay: float = 0.0,
    optimizer_type: str = 'adam'
) -> Optimizer:
    """
    Setup optimizer
    
    Paper: Uses Adam with lr=15e-5
    
    Args:
        model: Model
        learning_rate: Learning rate (paper: 15e-5)
        weight_decay: Weight decay
        optimizer_type: Optimizer type
    
    Returns:
        Optimizer
    """
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer


def setup_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = 'plateau',
    patience: int = 3,
    factor: float = 0.5
) -> Optional[any]:
    """
    Setup learning rate scheduler
    
    Args:
        optimizer: Optimizer
        scheduler_type: Scheduler type
        patience: Patience for ReduceLROnPlateau
        factor: Factor to reduce LR
    
    Returns:
        Scheduler
    """
    if scheduler_type == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
        )
    elif scheduler_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )
    elif scheduler_type is None or scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    return scheduler


# ============================================================================
# Demo and Testing
# ============================================================================

if __name__ == "__main__":
    print("Testing Trainer...")
    
    # This is just a structure test
    # Actual training would use real model and data
    
    print("\n✓ Trainer utilities ready!")