"""
Incremental Training Script for Sessions 1-5

Trains adapters and decoders for new noise domains while keeping
the backbone frozen.

Paper Reference: Section III - Incremental Learning
"""

import torch
from pathlib import Path
import argparse
import numpy as np

from ..models.lna_model import LNAModel
from ..data.dataset import get_session_dataloaders
from ..training.trainer import Trainer, setup_optimizer, setup_scheduler
from ..selectors.noise_selector import create_selector
from ..utils.config import ProjectConfig, get_default_config
from ..utils.audio import extract_features_for_clustering


def train_incremental_session(
    config: ProjectConfig,
    session_id: int,
    pretrained_model_path: str,
    data_root: str = "data/final_data",
    selector_path: str = None
):
    """
    Train incremental session
    
    Paper: "When faced with new noise domains, LNAs dynamically train 
    noise adapters tailored to adapt to the specific domain"
    
    Args:
        config: Project configuration
        session_id: Session ID (1, 2, 3, 4, 5)
        pretrained_model_path: Path to pre-trained LNA model
        data_root: Root directory with session data
        selector_path: Path to existing selector (if continuing from previous session)
    """
    if session_id == 0:
        raise ValueError("Use pretrain.py for session 0")
    
    print("\n" + "="*80)
    print(f"SESSION {session_id}: INCREMENTAL TRAINING".center(80))
    print("="*80 + "\n")
    
    # Device
    device = config.training.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Device: {device}")
    print(f"Session ID: {session_id}")
    print(f"Data root: {data_root}")
    print()
    
    # Load pre-trained model
    print("Loading pre-trained LNA model...")
    model = LNAModel(
        n_basis=config.sepformer.N,
        kernel_size=config.sepformer.L,
        num_layers=config.sepformer.num_layers,
        nhead=config.sepformer.nhead,
        dim_feedforward=config.sepformer.d_ffn,
        dropout=config.sepformer.dropout,
        adapter_bottleneck_dim=config.adapter.bottleneck_dim,
        max_sessions=6
    )
    
    checkpoint = model.load_checkpoint(pretrained_model_path)
    print(f"  Loaded from: {pretrained_model_path}")
    
    # Add new session adapters and decoder
    print(f"\nAdding adapters and decoder for session {session_id}...")
    model.add_new_session(
        session_id=session_id,
        bottleneck_dim=config.adapter.bottleneck_dim
    )
    
    # Set training mode (freeze backbone, train new adapters/decoder)
    print("\nConfiguring training mode...")
    model.set_training_mode(
        session_id=session_id,
        freeze_backbone=config.incremental.freeze_backbone,
        freeze_previous_adapters=config.incremental.freeze_previous_adapters,
        freeze_previous_decoders=config.incremental.freeze_previous_decoders
    )
    
    adapter_info = model.get_adapter_info()
    print(f"  Adapter parameters: {adapter_info['adapter_parameters']:,}")
    print(f"  Adapter percentage: {adapter_info['adapter_percentage']:.2f}%")
    
    # Create dataloaders
    print(f"\nLoading Session {session_id} data...")
    train_loader, val_loader, test_loader = get_session_dataloaders(
        data_root=data_root,
        session_id=session_id,
        batch_size_train=config.data.train_batch_size,
        batch_size_val=config.data.val_batch_size,
        batch_size_test=config.data.test_batch_size,
        num_workers=config.data.num_workers,
        sample_rate=config.data.sample_rate
    )
    
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Setup optimizer (only for trainable parameters)
    print("\nSetting up optimizer...")
    optimizer = setup_optimizer(
        model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        optimizer_type=config.training.optimizer
    )
    
    print(f"  Trainable parameters: {model.get_num_parameters(trainable_only=True):,}")
    
    # Setup scheduler
    scheduler = None
    if config.training.use_scheduler:
        scheduler = setup_scheduler(
            optimizer,
            scheduler_type=config.training.scheduler_type,
            patience=config.training.patience,
            factor=config.training.factor
        )
    
    # Create trainer
    checkpoint_dir = Path(config.checkpoint_dir) / f"session{session_id}_incremental"
    log_dir = Path(config.log_dir) / f"session{session_id}_incremental"
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        use_amp=config.training.use_amp,
        log_dir=str(log_dir),
        checkpoint_dir=str(checkpoint_dir)
    )
    
    # Train
    print("\n" + "="*80)
    print("STARTING INCREMENTAL TRAINING".center(80))
    print("="*80 + "\n")
    
    print(f"Training for {config.training.incremental_epochs} epochs")
    print(f"Frozen: Backbone (encoder + masking network)")
    print(f"Frozen: Previous adapters and decoders")
    print(f"Training: New adapters and decoder for session {session_id}")
    print()
    
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.training.incremental_epochs,
        early_stopping_patience=config.training.early_stopping_patience,
        save_every_n_epochs=config.training.save_every_n_epochs,
        validate_every_n_epochs=config.training.validate_every_n_epochs
    )
    
    # Save model
    model_checkpoint_path = checkpoint_dir / f"lna_session{session_id}.pt"
    model.save_checkpoint(
        str(model_checkpoint_path),
        session_id=session_id
    )
    
    # ========================================================================
    # Train Noise Selector
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING NOISE SELECTOR".center(80))
    print("="*80 + "\n")
    
    print("Paper Section III.D: Unsupervised Noise Selector")
    print("Uses K-Means clustering on encoder features\n")
    
    # Load or create selector
    if selector_path and Path(selector_path).exists():
        print(f"Loading existing selector from {selector_path}")
        selector = create_selector(
            selector_type=config.selector.selector_type,
            feature_dim=config.sepformer.N
        )
        selector.load(selector_path)
    else:
        print(f"Creating new {config.selector.selector_type} selector")
        selector = create_selector(
            selector_type=config.selector.selector_type,
            feature_dim=config.sepformer.N,
            n_clusters=config.selector.n_clusters if config.selector.selector_type == 'kmeans' else None,
            bandwidth=config.selector.bandwidth if config.selector.selector_type == 'meanshift' else None
        )
    
    # Extract features from training data
    print(f"\nExtracting features for session {session_id}...")
    model.eval()
    
    all_features = []
    with torch.no_grad():
        for noisy, clean, lengths, info in train_loader:
            noisy = noisy.to(device)
            
            # Extract encoder features
            features = model.get_encoder_features(noisy)  # [B, N, L]
            
            # Apply mean pooling: [B, N, L] -> [B, N]
            features_pooled = torch.mean(features, dim=2)  # Paper: MeanP(E(x))
            
            all_features.append(features_pooled.cpu().numpy())
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)  # [N_samples, N]
    print(f"  Extracted {all_features.shape[0]} feature vectors")
    print(f"  Feature dimension: {all_features.shape[1]}")
    
    # Fit selector for this session
    print(f"\nFitting selector for session {session_id}...")
    selector.fit_session(all_features, session_id=session_id)
    
    # Save selector
    selector_save_path = checkpoint_dir / f"selector_upto_session{session_id}.pkl"
    selector.save(str(selector_save_path))
    print(f"✓ Selector saved: {selector_save_path}")
    
    # Test selector accuracy on validation set
    print("\nTesting selector on validation set...")
    correct = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for noisy, clean, lengths, info in val_loader:
            noisy = noisy.to(device)
            features = model.get_encoder_features(noisy)
            features_pooled = torch.mean(features, dim=2).cpu().numpy()
            
            for i in range(len(noisy)):
                predicted = selector.predict(features_pooled[i])
                # For this session, all samples should be predicted as this session
                if predicted == session_id:
                    correct += 1
                total += 1
    
    accuracy = 100 * correct / total
    print(f"  Selector accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    print(f"\n✓ Session {session_id} training complete!")
    
    return history, model_checkpoint_path, selector_save_path


def train_all_incremental_sessions(
    config: ProjectConfig,
    pretrained_model_path: str,
    data_root: str = "data/final_data",
    session_ids: list = [1, 2, 3, 4, 5]
):
    """
    Train all incremental sessions sequentially
    
    Args:
        config: Project configuration
        pretrained_model_path: Path to pre-trained model
        data_root: Data root directory
        session_ids: List of session IDs to train
    """
    current_model_path = pretrained_model_path
    current_selector_path = None
    
    results = {}
    
    for session_id in session_ids:
        print(f"\n\n{'#'*80}")
        print(f"# TRAINING SESSION {session_id}".center(78, ' ') + " #")
        print(f"{'#'*80}\n")
        
        history, model_path, selector_path = train_incremental_session(
            config=config,
            session_id=session_id,
            pretrained_model_path=current_model_path,
            data_root=data_root,
            selector_path=current_selector_path
        )
        
        # Update paths for next session
        current_model_path = model_path
        current_selector_path = selector_path
        
        results[f'session_{session_id}'] = {
            'model_path': str(model_path),
            'selector_path': str(selector_path),
            'history': history
        }
    
    print("\n\n" + "="*80)
    print("ALL INCREMENTAL SESSIONS COMPLETE!".center(80))
    print("="*80 + "\n")
    
    print("Trained sessions:")
    for session_id in session_ids:
        info = results[f'session_{session_id}']
        print(f"  Session {session_id}:")
        print(f"    Model: {info['model_path']}")
        print(f"    Selector: {info['selector_path']}")
    
    return results


def main():
    """Main function for CLI"""
    parser = argparse.ArgumentParser(description="Incremental Training (Sessions 1-5)")
    parser.add_argument(
        "--session_id",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5],
        help="Session ID to train"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Path to pre-trained LNA model"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/final_data",
        help="Root directory with session data"
    )
    parser.add_argument(
        "--selector",
        type=str,
        default=None,
        help="Path to existing selector (for continuing from previous session)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device to use"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train all incremental sessions (1-5) sequentially"
    )
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ProjectConfig.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Override device
    if args.device:
        config.training.device = args.device
    
    # Train
    if args.all:
        results = train_all_incremental_sessions(
            config=config,
            pretrained_model_path=args.pretrained_model,
            data_root=args.data_root
        )
    else:
        history, model_path, selector_path = train_incremental_session(
            config=config,
            session_id=args.session_id,
            pretrained_model_path=args.pretrained_model,
            data_root=args.data_root,
            selector_path=args.selector
        )
    
    print("\n✓ Incremental training complete!")


if __name__ == "__main__":
    main()