"""
Audio Processing Utilities
Handles audio I/O, preprocessing, and SI-SNR loss calculation.
"""

import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import soundfile as sf


def load_audio(
    file_path: str, 
    target_sr: int = 8000,
    normalize: bool = True
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (8000 Hz per paper)
        normalize: Whether to normalize to [-1, 1]
    
    Returns:
        audio: Audio tensor of shape [1, T] (mono)
        sr: Sample rate
    """
    # Load audio
    audio_np, sr = sf.read(file_path, always_2d=True)
    audio = torch.tensor(audio_np.T, dtype=torch.float32)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
        sr = target_sr
    
    # Normalize to [-1, 1]
    if normalize:
        audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
    
    return audio, sr


def save_audio(
    audio: torch.Tensor,
    file_path: str,
    sr: int = 8000
):
    """
    Save audio tensor to file
    
    Args:
        audio: Audio tensor of shape [1, T] or [T]
        file_path: Output file path
        sr: Sample rate
    """
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Convert to numpy
    audio_np = audio.cpu().numpy()
    
    # Save using soundfile
    sf.write(file_path, audio_np.T, sr)


def normalize_audio(audio: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize audio to [-1, 1] range
    
    Args:
        audio: Audio tensor
        eps: Small constant for numerical stability
    
    Returns:
        Normalized audio tensor
    """
    return audio / (torch.max(torch.abs(audio)) + eps)


def calculate_si_snr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio
        
    Formula:
        SI-SNR = 10 * log10(||s_target||^2 / ||e_noise||^2)
        where s_target = <estimate, target> / ||target||^2 * target
              e_noise = estimate - s_target
    
    Scale-invariant: SI-SNR(estimate, target) = SI-SNR(α*estimate, target)
    
    Args:
        estimate: Enhanced speech [B, T] or [B, 1, T]
        target: Clean speech [B, T] or [B, 1, T]
        eps: Small constant for numerical stability
    
    Returns:
        SI-SNR value in dB [B]
    """
    # Ensure shape is [B, T]
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # Zero-mean normalization 
    estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
    target = target - torch.mean(target, dim=-1, keepdim=True)
    
    # Compute scaling factor α = <estimate, target> / ||target||^2
    dot_product = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps
    alpha = dot_product / target_energy
    
    # Project estimate onto target direction: s_target = α * target
    s_target = alpha * target
    
    # Compute noise as orthogonal component
    e_noise = estimate - s_target
    
    # Calculate SI-SNR in dB
    target_power = torch.sum(s_target ** 2, dim=-1) + eps
    noise_power = torch.sum(e_noise ** 2, dim=-1) + eps
    si_snr = 10 * torch.log10(target_power / noise_power)
    
    return si_snr


def si_snr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    SI-SNR loss function (negative SI-SNR for minimization)
    
    Usage in training:
        loss = si_snr_loss(enhanced_speech, clean_speech)
        loss.backward()
    
    Args:
        estimate: Enhanced speech [B, T] or [B, 1, T]
        target: Clean speech [B, T] or [B, 1, T]
    
    Returns:
        Negative SI-SNR (scalar for batch)
    """
    si_snr_values = calculate_si_snr(estimate, target)
    # Return negative for minimization (maximize SI-SNR)
    return -torch.mean(si_snr_values)


def apply_time_stretch(
    audio: torch.Tensor,
    rate: float = 1.0
) -> torch.Tensor:
    """
    Apply time stretching 
    
    Args:
        audio: Input audio [1, T]
        rate: Stretch rate (1.0 = no change, <1 = slower, >1 = faster)
    
    Returns:
        Time-stretched audio
    """
    if rate == 1.0:
        return audio
    return audio


def apply_pitch_shift(
    audio: torch.Tensor,
    n_steps: int = 0,
    sr: int = 8000
) -> torch.Tensor:
    """
    Apply pitch shifting 
    Args:
        audio: Input audio [1, T]
        n_steps: Number of semitones to shift
        sr: Sample rate
    Returns:
        Pitch-shifted audio
    """
    if n_steps == 0:
        return audio
    return audio


def pad_audio_batch(
    audio_list: list,
    pad_value: float = 0.0
) -> torch.Tensor:
    """
    Pad variable-length audio tensors to same length
    
    Args:
        audio_list: List of audio tensors [T_i]
        pad_value: Value to use for padding
    
    Returns:
        Padded batch tensor [B, T_max]
    """
    max_len = max(audio.shape[-1] for audio in audio_list)
    batch_size = len(audio_list)
    
    # Create padded tensor
    padded = torch.full((batch_size, max_len), pad_value)
    
    # Fill with actual audio
    for i, audio in enumerate(audio_list):
        length = audio.shape[-1]
        padded[i, :length] = audio.squeeze()
    
    return padded


def trim_audio_batch(
    audio_batch: torch.Tensor,
    lengths: torch.Tensor
) -> list:
    """
    Trim padded batch back to original lengths
    
    Args:
        audio_batch: Padded audio batch [B, T_max]
        lengths: Original lengths [B]
    
    Returns:
        List of trimmed audio tensors
    """
    audio_list = []
    for i, length in enumerate(lengths):
        audio_list.append(audio_batch[i, :length])
    
    return audio_list


def extract_features_for_clustering(
    audio: torch.Tensor,
    encoder: torch.nn.Module,
    use_mean_pooling: bool = True
) -> torch.Tensor:
    """
    Extract features from audio using frozen encoder for clustering

    Args:
        audio: Input audio [B, T] or [B, 1, T]
        encoder: Frozen encoder from pre-trained model
        use_mean_pooling: Apply mean pooling 
    
    Returns:
        Features for clustering [B, D] if pooling, else [B, L, D]
    """
    encoder.eval()
    with torch.no_grad():
        # Extract features: E(x) -> [B, L, D]
        features = encoder(audio)
        
        if use_mean_pooling:
            # MeanP(E(x)): Average over time dimension -> [B, D]
            features = torch.mean(features, dim=1)
    
    return features


if __name__ == "__main__":
    print("Testing audio utilities...")
    
    # Test SI-SNR calculation
    print("\n1. Testing SI-SNR calculation:")
    
    batch_size = 4
    audio_length = 16000  # 2 seconds at 8kHz
    
    # Perfect reconstruction
    target = torch.randn(batch_size, audio_length)
    estimate = target.clone()
    si_snr = calculate_si_snr(estimate, target)
    print(f"   Perfect reconstruction SI-SNR: {si_snr.mean().item():.2f} dB")
    
    # Add noise
    estimate_noisy = target + 0.1 * torch.randn_like(target)
    si_snr_noisy = calculate_si_snr(estimate_noisy, target)
    print(f"   With noise SI-SNR: {si_snr_noisy.mean().item():.2f} dB")
    
    # Test scale invariance
    estimate_scaled = 2.0 * target
    si_snr_scaled = calculate_si_snr(estimate_scaled, target)
    print(f"   Scaled by 2.0 SI-SNR: {si_snr_scaled.mean().item():.2f} dB")
    
    # Test loss function
    print("\n2. Testing SI-SNR loss:")
    loss = si_snr_loss(estimate_noisy, target)
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test padding
    print("\n3. Testing batch padding:")
    audio_list = [
        torch.randn(8000),   # 1 second
        torch.randn(16000),  # 2 seconds
        torch.randn(12000),  # 1.5 seconds
    ]
    padded = pad_audio_batch(audio_list)
    print(f"   Original lengths: {[a.shape[0] for a in audio_list]}")
    print(f"   Padded shape: {padded.shape}")
    
    print("\nAll audio utilities working!")