"""
Evaluation Script

Evaluate trained models on test sets and compare methods.

Paper Reference: Section IV - Experiments
Evaluates on cumulative test sets Z^{1,...,t}
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import json
from typing import Dict, List
from tqdm import tqdm

from ..models.lna_model import LNAModel
from ..data.dataset import MultiSessionDataset, create_dataloader
from ..evaluation.metrics import MetricsCalculator
from ..selectors.noise_selector import create_selector
from ..utils.config import ProjectConfig, get_default_config


def evaluate_model_on_session(
    model: LNAModel,
    test_loader,
    session_id: int,
    device: str = 'cuda',
    selector = None
) -> Dict[str, float]:
    """
    Evaluate model on one session's test set
    
    Args:
        model: Trained model
        test_loader: Test data loader
        session_id: Session ID
        device: Device to use
        selector: Noise selector (if using)
    
    Returns:
        Dictionary with metrics
    """
    model.eval()
    model.to(device)
    
    metrics_calc = MetricsCalculator(
        sample_rate=8000,
        metrics=['si_snr', 'sdr', 'pesq']
    )
    
    all_metrics = []
    selector_correct = 0
    selector_total = 0
    
    with torch.no_grad():
        for noisy, clean, lengths, info in tqdm(test_loader, desc=f"Session {session_id}"):
            noisy = noisy.to(device)
            clean = clean.to(device)
            
            # If using selector, predict domain
            if selector:
                features = model.get_encoder_features(noisy)
                features_pooled = torch.mean(features, dim=2).cpu().numpy()
                
                for i in range(len(noisy)):
                    predicted_session = selector.predict(features_pooled[i])
                    
                    # Track selector accuracy
                    if predicted_session == session_id:
                        selector_correct += 1
                    selector_total += 1
                    
                    # Use predicted session's decoder
                    enhanced = model(noisy[i:i+1], session_id=predicted_session)
                    
                    # Calculate metrics
                    sample_metrics = metrics_calc.calculate_all(
                        enhanced.squeeze(), clean[i, :lengths[i]]
                    )
                    all_metrics.append(sample_metrics)
            else:
                # Use specified session decoder
                enhanced = model(noisy, session_id=session_id)
                
                # Calculate metrics for batch
                for i in range(len(noisy)):
                    sample_metrics = metrics_calc.calculate_all(
                        enhanced[i, :lengths[i]], clean[i, :lengths[i]]
                    )
                    all_metrics.append(sample_metrics)
    
    # Aggregate metrics
    aggregated = metrics_calc.aggregate_metrics(all_metrics)
    
    if selector:
        aggregated['selector_accuracy'] = 100 * selector_correct / selector_total
    
    return aggregated


def evaluate_cumulative(
    model_path: str,
    selector_path: str,
    data_root: str,
    session_ids: List[int],
    config: ProjectConfig,
    output_path: str = None
) -> Dict:
    """
    Evaluate on cumulative test sets
    
    Paper: "the resulting model must be evaluated using the 
    aggregated test sets Z^{1,...,t}"
    
    Args:
        model_path: Path to trained model
        selector_path: Path to trained selector
        data_root: Data root directory
        session_ids: List of session IDs to evaluate
        config: Configuration
        output_path: Path to save results
    
    Returns:
        Dictionary with all results
    """
    device = config.training.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    print("\n" + "="*80)
    print("CUMULATIVE EVALUATION".center(80))
    print("="*80 + "\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
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
    model.load_checkpoint(model_path)
    
    # Load selector
    print(f"Loading selector from: {selector_path}")
    selector = create_selector(
        selector_type=config.selector.selector_type,
        feature_dim=config.sepformer.N
    )
    selector.load(selector_path)
    
    # Evaluate on each session
    results = {}
    
    for session_id in session_ids:
        print(f"\n{'='*80}")
        print(f"Evaluating Session {session_id}".center(80))
        print(f"{'='*80}\n")
        
        # Create test loader
        from ..data.dataset import SpeechEnhancementDataset
        
        # Find session directory
        if session_id == 0:
            session_dir = Path(data_root) / "session0_pretrain"
        else:
            session_dirs = list(Path(data_root).glob(f"session{session_id}_incremental_*"))
            if not session_dirs:
                print(f"Warning: Session {session_id} not found, skipping")
                continue
            session_dir = session_dirs[0]
        
        # Create dataset
        test_dataset = SpeechEnhancementDataset(
            data_dir=str(session_dir),
            split="test",
            sample_rate=8000
        )
        
        test_loader = create_dataloader(
            test_dataset,
            batch_size=config.data.test_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
        
        # Evaluate
        session_metrics = evaluate_model_on_session(
            model=model,
            test_loader=test_loader,
            session_id=session_id,
            device=device,
            selector=selector
        )
        
        results[f'session_{session_id}'] = session_metrics
        
        # Print results
        print(f"\nSession {session_id} Results:")
        for metric, value in session_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")
    
    return results


def compare_methods(
    results_dir: str,
    methods: List[str] = ['baseline', 'lna']
) -> Dict:
    """
    Compare different methods
    
    Args:
        results_dir: Directory containing results from different methods
        methods: List of method names to compare
    
    Returns:
        Comparison dictionary
    """
    print("\n" + "="*80)
    print("METHOD COMPARISON".center(80))
    print("="*80 + "\n")
    
    results_dir = Path(results_dir)
    comparison = {}
    
    for method in methods:
        method_file = results_dir / f"{method}_results.json"
        if method_file.exists():
            with open(method_file, 'r') as f:
                comparison[method] = json.load(f)
        else:
            print(f"Warning: {method_file} not found")
    
    # Print comparison table
    if comparison:
        print("\nComparison (SI-SNR in dB):")
        print("-" * 60)
        print(f"{'Session':<15} {'Baseline':<15} {'LNA':<15} {'Improvement':<15}")
        print("-" * 60)
        
        for session_key in comparison[methods[0]].keys():
            session_id = session_key.split('_')[1]
            
            baseline_snr = comparison.get(methods[0], {}).get(session_key, {}).get('si_snr_mean', 0)
            lna_snr = comparison.get(methods[1], {}).get(session_key, {}).get('si_snr_mean', 0)
            improvement = lna_snr - baseline_snr
            
            print(f"{session_id:<15} {baseline_snr:<15.2f} {lna_snr:<15.2f} {improvement:<15.2f}")
        
        print("-" * 60)
    
    return comparison


def main():
    """Main function for CLI"""
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--selector",
        type=str,
        required=True,
        help="Path to trained selector"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="data/final_data",
        help="Root directory with session data"
    )
    parser.add_argument(
        "--sessions",
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4, 5],
        help="Session IDs to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save results"
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
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = ProjectConfig.from_yaml(args.config)
    else:
        config = get_default_config()
    
    # Override device
    if args.device:
        config.training.device = args.device
    
    # Evaluate
    results = evaluate_cumulative(
        model_path=args.model,
        selector_path=args.selector,
        data_root=args.data_root,
        session_ids=args.sessions,
        config=config,
        output_path=args.output
    )
    
    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()