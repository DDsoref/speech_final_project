"""
Run LNA pipeline on 10% subset of data for quick testing

This script creates temporary subset datasets and runs the full pipeline on them.
Original data is untouched - subsets are created in /tmp or local Colab disk.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def create_subset_metadata(original_metadata_path, output_metadata_path, fraction=0.1):
    """
    Create a subset of metadata by randomly sampling
    
    Args:
        original_metadata_path: Path to original metadata.json
        output_metadata_path: Path to save subset metadata.json
        fraction: Fraction of data to keep (0.1 = 10%)
    """
    with open(original_metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Randomly sample
    subset_size = max(1, int(len(metadata) * fraction))
    subset = random.sample(metadata, subset_size)
    
    # Save subset
    with open(output_metadata_path, 'w') as f:
        json.dump(subset, f, indent=2)
    
    return len(subset)


def create_subset_session(original_session_dir, subset_session_dir, fraction=0.1):
    """
    Create a subset session by copying structure and sampling metadata
    
    Args:
        original_session_dir: Path to original session
        subset_session_dir: Path to create subset session
        fraction: Fraction of data to keep
    """
    original_session_dir = Path(original_session_dir)
    subset_session_dir = Path(subset_session_dir)
    
    # Create directory structure
    subset_session_dir.mkdir(parents=True, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        split_dir = subset_session_dir / split
        split_dir.mkdir(exist_ok=True)
        
        # Create symlinks to audio folders (don't copy audio files - just reference them)
        original_clean = original_session_dir / split / 'clean'
        original_noisy = original_session_dir / split / 'noisy'
        subset_clean = split_dir / 'clean'
        subset_noisy = split_dir / 'noisy'
        
        # Remove if exists
        if subset_clean.exists():
            subset_clean.unlink()
        if subset_noisy.exists():
            subset_noisy.unlink()
        
        # Create symlinks (works on Linux/Colab, not Windows)
        try:
            subset_clean.symlink_to(original_clean.resolve())
            subset_noisy.symlink_to(original_noisy.resolve())
        except:
            # If symlinks fail, just copy the directories
            shutil.copytree(original_clean, subset_clean)
            shutil.copytree(original_noisy, subset_noisy)
        
        # Create subset metadata
        original_metadata = original_session_dir / split / 'metadata.json'
        subset_metadata = split_dir / 'metadata.json'
        
        num_samples = create_subset_metadata(original_metadata, subset_metadata, fraction)
        print(f'  {split}: {num_samples} samples')


def create_subset_data(original_data_root, subset_data_root, fraction=0.1):
    """
    Create subset of entire dataset
    
    Args:
        original_data_root: Path to data/final_data
        subset_data_root: Path to create subset data
        fraction: Fraction of data to keep
    """
    original_data_root = Path(original_data_root)
    subset_data_root = Path(subset_data_root)
    
    print(f'\nCreating {int(fraction*100)}% subset of data...')
    print(f'Original: {original_data_root}')
    print(f'Subset: {subset_data_root}')
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process Session 0
    print('\nSession 0 (pretrain):')
    create_subset_session(
        original_data_root / 'session0_pretrain',
        subset_data_root / 'session0_pretrain',
        fraction
    )
    
    # Process Sessions 1-5
    for i in range(1, 6):
        # Find the session directory
        session_dirs = list(original_data_root.glob(f'session{i}_incremental_*'))
        if session_dirs:
            session_dir = session_dirs[0]
            session_name = session_dir.name
            
            print(f'\nSession {i} ({session_name}):')
            create_subset_session(
                session_dir,
                subset_data_root / session_name,
                fraction
            )
    
    # Copy dataset_info.json if exists
    if (original_data_root / 'dataset_info.json').exists():
        shutil.copy(
            original_data_root / 'dataset_info.json',
            subset_data_root / 'dataset_info.json'
        )
    
    print('\n✓ Subset data created!')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LNA pipeline on subset of data')
    parser.add_argument(
        '--original-data',
        type=str,
        default='/content/drive/MyDrive/speech_final_project/data/final_data',
        help='Path to original data'
    )
    parser.add_argument(
        '--subset-data',
        type=str,
        default='/tmp/subset_data',
        help='Path to create subset data (temporary location)'
    )
    parser.add_argument(
        '--fraction',
        type=float,
        default=0.1,
        help='Fraction of data to use (0.1 = 10%%)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to use'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='all',
        choices=['all', 'pretrain', 'incremental', 'evaluate'],
        help='Pipeline mode'
    )
    
    args = parser.parse_args()
    
    # Create subset data
    create_subset_data(args.original_data, args.subset_data, args.fraction)
    
    # Run pipeline on subset
    print('\n' + '='*80)
    print('Running pipeline on subset data...')
    print('='*80 + '\n')
    
    cmd = f'python run_pipeline.py --mode {args.mode} --device {args.device} --data_root {args.subset_data}'
    print(f'Command: {cmd}\n')
    
    os.system(cmd)
    
    print('\n' + '='*80)
    print('Pipeline complete!')
    print('='*80)