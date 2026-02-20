"""
Plot Results for LNA Paper
Recreates plots from "Learning Noise Adapters for Incremental Speech Enhancement"

Usage:
    python plot_results.py

Reads from:
    results/evaluation_results.json
    logs/session*/training_history.json

Creates:
    figures/session_comparison.png - SI-SNR comparison across sessions (like Fig 3)
    figures/training_curves.png - Training loss curves
    figures/metrics_heatmap.png - All metrics heatmap
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================================
# Load Results
# ============================================================================

def load_evaluation_results():
    """Load evaluation results from JSON"""
    with open('results/evaluation_results.json', 'r') as f:
        return json.load(f)

def load_training_history(session_name):
    """Load training history for a session"""
    history_path = Path(f'logs/{session_name}/training_history.json')
    if history_path.exists():
        with open(history_path, 'r') as f:
            return json.load(f)
    return None

# ============================================================================
# Plot 1: Session Performance Comparison (Like Paper Fig 3)
# ============================================================================

def plot_session_comparison(results):
    """
    Plot SI-SNR improvement across sessions
    Similar to Figure 3 in the paper
    """
    # Extract session data
    sessions = []
    si_snr_values = []
    pesq_values = []
    stoi_values = []
    
    for session_key in sorted(results.keys()):
        if 'session' in session_key:
            session_num = int(session_key.split('_')[0].replace('session', ''))
            sessions.append(f"S{session_num}")
            
            metrics = results[session_key]
            si_snr_values.append(metrics.get('si_snr', 0))
            pesq_values.append(metrics.get('pesq', 0))
            stoi_values.append(metrics.get('stoi', 0))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    x = np.arange(len(sessions))
    width = 0.6
    
    # SI-SNR
    axes[0].bar(x, si_snr_values, width, color='#3498db', alpha=0.8)
    axes[0].set_xlabel('Session', fontsize=12)
    axes[0].set_ylabel('SI-SNR (dB)', fontsize=12)
    axes[0].set_title('SI-SNR Improvement per Session', fontsize=13, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sessions)
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    
    # Add values on bars
    for i, v in enumerate(si_snr_values):
        axes[0].text(i, v + 0.3, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # PESQ
    axes[1].bar(x, pesq_values, width, color='#2ecc71', alpha=0.8)
    axes[1].set_xlabel('Session', fontsize=12)
    axes[1].set_ylabel('PESQ', fontsize=12)
    axes[1].set_title('PESQ Score per Session', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(sessions)
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([1, 5])  # PESQ range
    
    for i, v in enumerate(pesq_values):
        axes[1].text(i, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
    
    # STOI
    axes[2].bar(x, stoi_values, width, color='#e74c3c', alpha=0.8)
    axes[2].set_xlabel('Session', fontsize=12)
    axes[2].set_ylabel('STOI', fontsize=12)
    axes[2].set_title('STOI Score per Session', fontsize=13, fontweight='bold')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(sessions)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_ylim([0, 1])  # STOI range
    
    for i, v in enumerate(stoi_values):
        axes[2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "session_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# Plot 2: Training Curves
# ============================================================================

def plot_training_curves():
    """Plot training and validation loss curves for all sessions"""
    
    sessions = [
        ('session0_pretrain', 'Session 0 (Pretrain)'),
        ('session1_incremental_volvo', 'Session 1 (Volvo)'),
        ('session2_incremental_f16', 'Session 2 (F16)'),
        ('session3_incremental_m109', 'Session 3 (M109)'),
        ('session4_incremental_destroyerops', 'Session 4 (Destroyer)'),
        ('session5_incremental_machinegun', 'Session 5 (Machinegun)'),
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (session_name, title) in enumerate(sessions):
        history = load_training_history(session_name)
        
        if history is None:
            axes[idx].text(0.5, 0.5, 'No data', ha='center', va='center')
            axes[idx].set_title(title)
            continue
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot loss curves
        axes[idx].plot(epochs, history['train_loss'], 'o-', label='Train Loss', 
                      linewidth=2, markersize=4)
        axes[idx].plot(epochs, history['val_loss'], 's-', label='Val Loss', 
                      linewidth=2, markersize=4)
        
        axes[idx].set_xlabel('Epoch', fontsize=10)
        axes[idx].set_ylabel('Loss (SI-SNR)', fontsize=10)
        axes[idx].set_title(title, fontsize=11, fontweight='bold')
        axes[idx].legend(fontsize=9)
        axes[idx].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = np.argmin(history['val_loss']) + 1
        best_val = min(history['val_loss'])
        axes[idx].axvline(x=best_epoch, color='red', linestyle='--', 
                         alpha=0.5, linewidth=1)
        axes[idx].text(best_epoch, best_val, f'  Best: {best_val:.2f}', 
                      fontsize=8, color='red')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# Plot 3: Metrics Heatmap
# ============================================================================

def plot_metrics_heatmap(results):
    """Create heatmap of all metrics across sessions"""
    
    # Extract data
    sessions = []
    metrics_data = []
    metric_names = ['SI-SNR', 'SDR', 'PESQ', 'STOI']
    
    for session_key in sorted(results.keys()):
        if 'session' in session_key:
            session_num = int(session_key.split('_')[0].replace('session', ''))
            noise_type = session_key.split('_')[-1] if 'incremental' in session_key else 'pretrain'
            sessions.append(f"S{session_num}\n{noise_type[:8]}")
            
            metrics = results[session_key]
            metrics_data.append([
                metrics.get('si_snr', 0),
                metrics.get('sdr', 0),
                metrics.get('pesq', 0),
                metrics.get('stoi', 0) * 10  # Scale STOI for visibility
            ])
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = np.array(metrics_data).T
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    
    # Set ticks
    ax.set_xticks(np.arange(len(sessions)))
    ax.set_yticks(np.arange(len(metric_names)))
    ax.set_xticklabels(sessions, fontsize=10)
    ax.set_yticklabels(metric_names, fontsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Score (STOI ×10)', fontsize=11)
    
    # Add text annotations
    for i in range(len(metric_names)):
        for j in range(len(sessions)):
            text = ax.text(j, i, f'{data[i, j]:.2f}',
                          ha="center", va="center", color="black", 
                          fontsize=9, fontweight='bold')
    
    ax.set_title('Metrics Heatmap Across Sessions', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Session', fontsize=12)
    ax.set_ylabel('Metric', fontsize=12)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "metrics_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# Plot 4: Incremental Learning Performance
# ============================================================================

def plot_incremental_performance(results):
    """
    Plot how performance changes with incremental learning
    Shows the benefit of LNA method
    """
    
    # Extract incremental sessions only
    incremental_sessions = []
    si_snr_improvements = []
    noise_types = []
    
    for session_key in sorted(results.keys()):
        if 'incremental' in session_key:
            session_num = int(session_key.split('_')[0].replace('session', ''))
            noise_type = session_key.split('_')[-1]
            
            incremental_sessions.append(session_num)
            noise_types.append(noise_type.capitalize())
            
            metrics = results[session_key]
            si_snr_improvements.append(metrics.get('si_snr', 0))
    
    if not incremental_sessions:
        print("⚠️  No incremental sessions found")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    bars = ax.bar(range(len(incremental_sessions)), si_snr_improvements, 
                  color=colors[:len(incremental_sessions)], alpha=0.8, width=0.6)
    
    # Customize
    ax.set_xlabel('Incremental Session', fontsize=13)
    ax.set_ylabel('SI-SNR Improvement (dB)', fontsize=13)
    ax.set_title('Incremental Learning Performance\n(LNA Adapter Method)', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(incremental_sessions)))
    ax.set_xticklabels([f"S{s}\n({n})" for s, n in zip(incremental_sessions, noise_types)],
                       fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, si_snr_improvements)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.2f} dB',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add average line
    avg_improvement = np.mean(si_snr_improvements)
    ax.axhline(y=avg_improvement, color='green', linestyle=':', linewidth=2, 
              label=f'Average: {avg_improvement:.2f} dB')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "incremental_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# Main
# ============================================================================

def main():
    print("\n" + "="*80)
    print("PLOTTING LNA RESULTS".center(80))
    print("="*80 + "\n")
    
    # Load results
    print("Loading results...")
    results = load_evaluation_results()
    print(f"✓ Loaded results for {len(results)} sessions\n")
    
    # Generate plots
    print("Generating plots...")
    plot_session_comparison(results)
    plot_training_curves()
    plot_metrics_heatmap(results)
    plot_incremental_performance(results)
    
    print("\n" + "="*80)
    print(f"✅ ALL PLOTS SAVED TO: {OUTPUT_DIR}/")
    print("="*80 + "\n")
    
    print("Generated files:")
    for plot_file in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  - {plot_file}")
    print()

if __name__ == "__main__":
    main()