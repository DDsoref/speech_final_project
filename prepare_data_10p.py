"""
Data Preparation Script for Incremental Speech Enhancement - 10% SUBSET VERSION
Paper: "Learning Noise Adapters for Incremental Speech Enhancement"

Creates organized dataset with 10% of samples:
- Session 0: Pre-training with 10 noises (~4,040 training samples instead of 40,400)
- Sessions 1-5: Incremental learning with 1 noise each (~121 training samples instead of 1,212)

Author: Daniel's Final Project
Date: 2026
"""

import os
import numpy as np
import soundfile as sf
import librosa
import json
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Tuple, Dict

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for data preparation - 10% SUBSET VERSION"""
    
    # ========== PATHS ==========
    LIBRISPEECH_ROOT = "data/LibriSpeech"
    PRETRAIN_NOISE_DIR = "data/pretrain_noises"
    INCREMENTAL_NOISE_DIR = "data/incremental_noises"
    OUTPUT_ROOT = "data/final_data"  # Clean organized output
    
    # ========== AUDIO PARAMETERS ==========
    TARGET_SR = 8000  # 8 kHz as per paper
    
    # ========== SNR CONFIGURATION ==========
    TRAIN_SNR_LEVELS = [-5, 0, 5, 10]  # 4 levels for training
    VAL_TEST_SNR_RANGE = (-5, 10)      # Random SNR for val/test
    
    # ========== SESSION 0 (PRE-TRAIN) SPECIFICATIONS - 10% SUBSET ==========
    SESSION0_TRAIN_UTTERANCES = 101      # 10% of 1010
    SESSION0_TRAIN_SPEAKERS = 101        # Keep same (speaker diversity important)
    SESSION0_VAL_UTTERANCES = 121        # 10% of 1206
    SESSION0_VAL_SPEAKERS = 8            # Keep same
    SESSION0_TEST_UTTERANCES = 65        # 10% of 651
    SESSION0_TEST_SPEAKERS = 8           # Keep same
    
    # ========== INCREMENTAL SESSIONS SPECIFICATIONS - 10% SUBSET ==========
    INCREMENTAL_TRAIN_UTTERANCES = 30    # 10% of 303
    INCREMENTAL_TRAIN_SPEAKERS = 101     # Keep same (speaker diversity important)
    INCREMENTAL_VAL_UTTERANCES = 121     # 10% of 1206
    INCREMENTAL_VAL_SPEAKERS = 10        # Keep same
    INCREMENTAL_TEST_UTTERANCES = 65     # 10% of 651
    INCREMENTAL_TEST_SPEAKERS = 8        # Keep same
    
    # ========== NOISE ASSIGNMENTS ==========
    SESSION0_NOISES = [
        "babble.wav", "buccaneer1.wav", "buccaneer2.wav",
        "destroyerengine.wav", "factory1.wav", "factory2.wav",
        "hfchannel.wav", "leopard.wav", "pink.wav", "white.wav"
    ]
    
    INCREMENTAL_SESSIONS = [
        {"id": 1, "noise": "volvo.wav", "name": "volvo"},
        {"id": 2, "noise": "f16.wav", "name": "f16"},
        {"id": 3, "noise": "m109.wav", "name": "m109"},
        {"id": 4, "noise": "destroyerops.wav", "name": "destroyerops"},
        {"id": 5, "noise": "machinegun.wav", "name": "machinegun"},
    ]
    
    # ========== REPRODUCIBILITY ==========
    RANDOM_SEED = 42

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)

def print_header(text: str):
    """Print formatted section header"""
    print(f"\n{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}\n")

def load_audio(file_path: str, target_sr: int = 8000) -> np.ndarray:
    """Load and resample audio file to target sample rate"""
    audio, sr = librosa.load(file_path, sr=None)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    return audio

def save_audio(audio: np.ndarray, file_path: str, sr: int = 8000):
    """Save audio file"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio, sr)

def calculate_rms(audio: np.ndarray) -> float:
    """Calculate RMS energy of audio signal"""
    return np.sqrt(np.mean(audio ** 2))

def add_noise_at_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add noise to clean speech at specified SNR level
    
    Args:
        clean: Clean speech signal
        noise: Noise signal (will be adjusted to match clean length)
        snr_db: Target SNR in dB
    
    Returns:
        Noisy speech signal (clean + scaled noise)
    """
    # Match noise length to clean speech
    if len(noise) < len(clean):
        # Repeat noise if shorter
        repeats = int(np.ceil(len(clean) / len(noise)))
        noise = np.tile(noise, repeats)[:len(clean)]
    else:
        # Randomly crop noise if longer
        start_idx = random.randint(0, len(noise) - len(clean))
        noise = noise[start_idx:start_idx + len(clean)]
    
    # Calculate RMS values
    clean_rms = calculate_rms(clean)
    noise_rms = calculate_rms(noise)
    
    # Scale noise to achieve target SNR
    # SNR(dB) = 20*log10(clean_rms / noise_rms)
    # noise_rms_target = clean_rms / (10^(SNR/20))
    if noise_rms > 0:
        target_noise_rms = clean_rms / (10 ** (snr_db / 20))
        scaling_factor = target_noise_rms / noise_rms
        noise_scaled = noise * scaling_factor
    else:
        noise_scaled = noise
    
    # Mix clean and scaled noise
    noisy = clean + noise_scaled
    
    return noisy

# ============================================================================
# LIBRISPEECH DATA LOADING
# ============================================================================

def get_librispeech_speakers(librispeech_root: str) -> Dict[str, List[str]]:
    """
    Scan LibriSpeech directory and organize files by speaker
    
    Returns:
        Dictionary: {speaker_id: [list of .flac file paths]}
    """
    print("📚 Scanning LibriSpeech dataset...")
    speakers_files = {}
    
    for subset in ["train-clean-100", "dev-clean", "test-clean"]:
        subset_path = os.path.join(librispeech_root, subset)
        
        if not os.path.exists(subset_path):
            print(f"  ⚠️  Warning: {subset} not found, skipping...")
            continue
        
        print(f"  Scanning {subset}...")
        
        for speaker_id in os.listdir(subset_path):
            speaker_path = os.path.join(subset_path, speaker_id)
            
            if not os.path.isdir(speaker_path):
                continue
            
            if speaker_id not in speakers_files:
                speakers_files[speaker_id] = []
            
            # Recursively find all .flac files
            for root, dirs, files in os.walk(speaker_path):
                for file in files:
                    if file.endswith('.flac'):
                        speakers_files[speaker_id].append(os.path.join(root, file))
    
    # Filter out speakers with no files
    speakers_files = {k: v for k, v in speakers_files.items() if len(v) > 0}
    
    total_files = sum(len(files) for files in speakers_files.values())
    print(f"  ✓ Found {len(speakers_files)} speakers with {total_files} total files\n")
    
    return speakers_files

def select_speakers_and_files(
    speakers_dict: Dict[str, List[str]],
    num_speakers: int,
    num_utterances: int,
    exclude_speakers: List[str] = None
) -> Tuple[List[str], List[str]]:
    """
    Randomly select speakers and their utterances
    
    Strategy:
    1. Select num_speakers random speakers (excluding already used ones)
    2. Sample ~equal number of utterances from each speaker
    3. Ensure exact total count matches num_utterances
    
    Args:
        speakers_dict: {speaker_id: [file_paths]}
        num_speakers: Number of speakers to select
        num_utterances: Total utterances needed
        exclude_speakers: Speaker IDs to exclude (avoid overlap)
    
    Returns:
        (selected_speaker_ids, selected_file_paths)
    """
    if exclude_speakers is None:
        exclude_speakers = []
    
    # Filter available speakers
    available = {k: v for k, v in speakers_dict.items() 
                 if k not in exclude_speakers and len(v) > 0}
    
    if len(available) < num_speakers:
        raise ValueError(
            f"Not enough speakers available. Need {num_speakers}, have {len(available)} "
            f"(after excluding {len(exclude_speakers)} used speakers)"
        )
    
    # Randomly select speakers
    selected_speaker_ids = random.sample(list(available.keys()), num_speakers)
    
    # Calculate files per speaker
    base_files_per_speaker = num_utterances // num_speakers
    extra_files = num_utterances % num_speakers
    
    selected_files = []
    
    for i, speaker_id in enumerate(selected_speaker_ids):
        files_for_this_speaker = base_files_per_speaker + (1 if i < extra_files else 0)
        
        speaker_files = available[speaker_id]
        
        if len(speaker_files) < files_for_this_speaker:
            # If speaker doesn't have enough files, take all
            selected_files.extend(speaker_files)
        else:
            # Randomly sample required number
            selected_files.extend(random.sample(speaker_files, files_for_this_speaker))
    
    # Verify we got exact count
    if len(selected_files) != num_utterances:
        # Adjust if needed (rare edge case)
        if len(selected_files) > num_utterances:
            selected_files = selected_files[:num_utterances]
        else:
            # Need more files - sample from any available speaker
            deficit = num_utterances - len(selected_files)
            all_remaining = []
            for spk_id in selected_speaker_ids:
                all_remaining.extend([f for f in available[spk_id] if f not in selected_files])
            selected_files.extend(random.sample(all_remaining, deficit))
    
    return selected_speaker_ids, selected_files

# ============================================================================
# DATASET CREATION
# ============================================================================

def create_mixed_dataset(
    clean_files: List[str],
    noise_files: List[str],
    output_dir: str,
    split_name: str,
    config: Config,
    mix_all_noises: bool = False
):
    """
    Create mixed noisy dataset from clean files and noise
    
    Args:
        clean_files: List of clean speech file paths
        noise_files: List of noise file paths
        output_dir: Output directory (will create split_name subdirectory)
        split_name: "train", "val", or "test"
        config: Configuration object
        mix_all_noises: If True (Session 0), mix each clean file with ALL noises
                       If False (Incremental), mix with only first noise
    """
    # Load noise files
    print(f"  Loading {len(noise_files)} noise files...")
    noises = {}
    for noise_path in noise_files:
        noise_name = os.path.basename(noise_path).replace('.wav', '')
        noises[noise_name] = load_audio(noise_path, config.TARGET_SR)
    
    # Create output directories
    noisy_dir = os.path.join(output_dir, split_name, "noisy")
    clean_dir = os.path.join(output_dir, split_name, "clean")
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(clean_dir, exist_ok=True)
    
    # Determine SNR levels
    if split_name == "train":
        snr_levels = config.TRAIN_SNR_LEVELS
    else:
        # Val/Test use random SNR from range
        snr_levels = [None]  # Will be randomly selected
    
    metadata = []
    
    print(f"  Processing {len(clean_files)} clean files...")
    
    for idx, clean_path in enumerate(tqdm(clean_files, desc=f"  Creating {split_name}")):
        # Load clean audio
        clean_audio = load_audio(clean_path, config.TARGET_SR)
        
        # Extract speaker_id from LibriSpeech path structure
        # Path format: .../LibriSpeech/train-clean-100/SPEAKER_ID/CHAPTER_ID/filename.flac
        speaker_id = Path(clean_path).parts[-3]
        
        # Save clean file
        clean_filename = f"clean_{idx:05d}.wav"
        save_audio(clean_audio, os.path.join(clean_dir, clean_filename), config.TARGET_SR)
        
        # Decide which noises to use
        if mix_all_noises:
            noises_to_use = list(noises.items())
        else:
            # Use only first noise (for incremental sessions)
            noises_to_use = [list(noises.items())[0]]
        
        # Mix with each noise type and SNR level
        for noise_name, noise_audio in noises_to_use:
            for snr_idx, snr_value in enumerate(snr_levels):
                # Random SNR for val/test
                if snr_value is None:
                    snr_value = np.random.uniform(*config.VAL_TEST_SNR_RANGE)
                
                # Add noise
                noisy_audio = add_noise_at_snr(clean_audio, noise_audio, snr_value)
                
                # Save noisy file
                if mix_all_noises:
                    noisy_filename = f"noisy_{idx:05d}_{noise_name}_snr{int(snr_value):+03d}.wav"
                else:
                    noisy_filename = f"noisy_{idx:05d}_snr{int(snr_value):+03d}.wav"
                
                save_audio(noisy_audio, os.path.join(noisy_dir, noisy_filename), config.TARGET_SR)
                
                # Record metadata
                metadata.append({
                    "noisy_file": noisy_filename,
                    "clean_file": clean_filename,
                    "speaker_id": speaker_id,
                    "noise_type": noise_name,
                    "snr_db": float(snr_value),
                    "duration_sec": float(len(clean_audio) / config.TARGET_SR)
                })
    
    # Save metadata
    metadata_path = os.path.join(output_dir, split_name, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ {split_name} complete: {len(metadata)} noisy files + {len(clean_files)} clean files\n")

def create_session(
    session_id: int,
    session_name: str,
    noise_files: List[str],
    speakers_dict: Dict[str, List[str]],
    config: Config,
    exclude_speakers: List[str] = None
) -> List[str]:
    """
    Create complete session dataset (train + val + test)
    
    Returns:
        List of all speaker IDs used in this session
    """
    if exclude_speakers is None:
        exclude_speakers = []
    
    output_dir = os.path.join(config.OUTPUT_ROOT, session_name)
    
    print_header(f"SESSION {session_id}: {session_name.upper()}")
    
    # Get specifications
    is_pretrain = (session_id == 0)
    if is_pretrain:
        train_utt, train_spk = config.SESSION0_TRAIN_UTTERANCES, config.SESSION0_TRAIN_SPEAKERS
        val_utt, val_spk = config.SESSION0_VAL_UTTERANCES, config.SESSION0_VAL_SPEAKERS
        test_utt, test_spk = config.SESSION0_TEST_UTTERANCES, config.SESSION0_TEST_SPEAKERS
    else:
        train_utt, train_spk = config.INCREMENTAL_TRAIN_UTTERANCES, config.INCREMENTAL_TRAIN_SPEAKERS
        val_utt, val_spk = config.INCREMENTAL_VAL_UTTERANCES, config.INCREMENTAL_VAL_SPEAKERS
        test_utt, test_spk = config.INCREMENTAL_TEST_UTTERANCES, config.INCREMENTAL_TEST_SPEAKERS
    
    print(f"Noise files: {[os.path.basename(f) for f in noise_files]}")
    print(f"  Train: {train_utt} utterances, {train_spk} speakers")
    print(f"  Val:   {val_utt} utterances, {val_spk} speakers")
    print(f"  Test:  {test_utt} utterances, {test_spk} speakers\n")
    
    # Select speakers and files
    print("📊 Selecting speakers and utterances...")
    
    train_speakers, train_files = select_speakers_and_files(
        speakers_dict, train_spk, train_utt, exclude_speakers
    )
    print(f"  Train: {len(train_speakers)} speakers, {len(train_files)} files")
    
    val_speakers, val_files = select_speakers_and_files(
        speakers_dict, val_spk, val_utt, exclude_speakers + train_speakers
    )
    print(f"  Val:   {len(val_speakers)} speakers, {len(val_files)} files")
    
    test_speakers, test_files = select_speakers_and_files(
        speakers_dict, test_spk, test_utt, exclude_speakers + train_speakers + val_speakers
    )
    print(f"  Test:  {len(test_speakers)} speakers, {len(test_files)} files\n")
    
    # Create datasets
    print("🎵 Creating training dataset...")
    create_mixed_dataset(
        train_files, noise_files, output_dir, "train", config,
        mix_all_noises=is_pretrain  # Only Session 0 mixes with all noises
    )
    
    print("🎵 Creating validation dataset...")
    create_mixed_dataset(val_files, noise_files, output_dir, "val", config)
    
    print("🎵 Creating test dataset...")
    create_mixed_dataset(test_files, noise_files, output_dir, "test", config)
    
    # Save session summary
    summary = {
        "session_id": session_id,
        "session_name": session_name,
        "noise_types": [os.path.basename(f).replace('.wav', '') for f in noise_files],
        "train": {
            "num_speakers": len(train_speakers),
            "num_utterances": len(train_files),
            "num_noisy_files": len(train_files) * len(config.TRAIN_SNR_LEVELS) * (len(noise_files) if is_pretrain else 1),
            "speakers": train_speakers
        },
        "val": {
            "num_speakers": len(val_speakers),
            "num_utterances": len(val_files),
            "speakers": val_speakers
        },
        "test": {
            "num_speakers": len(test_speakers),
            "num_utterances": len(test_files),
            "speakers": test_speakers
        }
    }
    
    with open(os.path.join(output_dir, "session_info.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✅ {session_name} complete!\n")
    
    return train_speakers + val_speakers + test_speakers

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    config = Config()
    set_seed(config.RANDOM_SEED)
    
    print_header("INCREMENTAL SPEECH ENHANCEMENT - DATA PREPARATION (10% SUBSET)")
    
    print("📋 Configuration:")
    print(f"  Target sample rate: {config.TARGET_SR} Hz")
    print(f"  Training SNRs: {config.TRAIN_SNR_LEVELS} dB")
    print(f"  Val/Test SNR range: {config.VAL_TEST_SNR_RANGE} dB")
    print(f"  Random seed: {config.RANDOM_SEED}")
    print(f"  Data fraction: 10% (for quick validation)")
    print(f"  Output: {config.OUTPUT_ROOT}\n")
    
    # Create output directory
    os.makedirs(config.OUTPUT_ROOT, exist_ok=True)
    
    # Load LibriSpeech data
    speakers_dict = get_librispeech_speakers(config.LIBRISPEECH_ROOT)
    
    # Track used speakers to avoid overlap
    session0_speakers = []
    
    # ========================================================================
    # SESSION 0: Pre-training with 10 noises
    # ========================================================================
    
    session0_noises = [
        os.path.join(config.PRETRAIN_NOISE_DIR, n) for n in config.SESSION0_NOISES
    ]
    
    # Verify files exist
    missing = [f for f in session0_noises if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing noise files: {missing}")
    
    session0_speakers = create_session(
        session_id=0,
        session_name="session0_pretrain",
        noise_files=session0_noises,
        speakers_dict=speakers_dict,
        config=config
    )
    
    # ========================================================================
    # SESSIONS 1-5: Incremental learning
    # ========================================================================
    
    for sess_info in config.INCREMENTAL_SESSIONS:
        noise_path = os.path.join(config.INCREMENTAL_NOISE_DIR, sess_info["noise"])
        
        if not os.path.exists(noise_path):
            raise FileNotFoundError(f"Missing noise file: {noise_path}")
        
        # Incremental sessions use DIFFERENT speakers than Session 0
        create_session(
            session_id=sess_info["id"],
            session_name=f"session{sess_info['id']}_incremental_{sess_info['name']}",
            noise_files=[noise_path],
            speakers_dict=speakers_dict,
            config=config,
            exclude_speakers=session0_speakers  # Exclude Session 0 speakers
        )
    
    # ========================================================================
    # Create overall summary
    # ========================================================================
    
    print_header("DATA PREPARATION COMPLETE!")
    
    summary = {
        "project": "Incremental Speech Enhancement",
        "paper": "Learning Noise Adapters for Incremental Speech Enhancement",
        "data_fraction": 0.1,
        "note": "10% subset for quick validation",
        "config": {
            "sample_rate_hz": config.TARGET_SR,
            "train_snr_levels_db": config.TRAIN_SNR_LEVELS,
            "val_test_snr_range_db": list(config.VAL_TEST_SNR_RANGE),
            "random_seed": config.RANDOM_SEED
        },
        "sessions": {
            "session0": {
                "type": "pretrain",
                "num_noises": len(config.SESSION0_NOISES),
                "noise_types": [n.replace('.wav', '') for n in config.SESSION0_NOISES],
                "expected_train_samples": 4040  # 10% of 40,400
            }
        }
    }
    
    for sess_info in config.INCREMENTAL_SESSIONS:
        summary["sessions"][f"session{sess_info['id']}"] = {
            "type": "incremental",
            "noise_type": sess_info["name"],
            "expected_train_samples": 121  # ~10% of 1,212
        }
    
    summary_path = os.path.join(config.OUTPUT_ROOT, "dataset_info.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Dataset summary: {summary_path}")
    print(f"📁 All data saved to: {config.OUTPUT_ROOT}")
    
    print("\n" + "="*80)
    print("🎉 SUCCESS! Ready for training!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()