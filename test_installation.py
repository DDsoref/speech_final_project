#!/usr/bin/env python3
"""
Quick Test Script

Verifies that all components are working correctly.

Usage:
    python test_installation.py
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        # Core modules
        from src.utils.config import get_default_config
        from src.utils.audio import calculate_si_snr, si_snr_loss
        from src.data.dataset import SpeechEnhancementDataset
        from src.models.adapters import Adapter, TransformerBlockWithAdapters
        from src.models.sepformer import SepFormer
        from src.models.lna_model import LNAModel
        from src.selectors.noise_selector import KMeansSelector, MeanShiftSelector
        from src.evaluation.metrics import MetricsCalculator
        from src.training.trainer import Trainer
        
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_config():
    """Test configuration system"""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import get_default_config, get_experiment_config
        
        # Test default config
        config = get_default_config()
        assert config.adapter.bottleneck_dim == 1
        assert config.sepformer.num_layers == 8
        assert config.training.pretrain_epochs == 40
        
        # Test experiment configs
        config_exp = get_experiment_config("larger_adapter")
        assert config_exp.adapter.bottleneck_dim == 16
        
        print("  ✓ Configuration system working")
        return True
    except Exception as e:
        print(f"  ✗ Configuration error: {e}")
        return False


def test_audio_utils():
    """Test audio utilities"""
    print("\nTesting audio utilities...")
    
    try:
        from src.utils.audio import calculate_si_snr, si_snr_loss
        
        # Create test signals
        clean = torch.randn(2, 16000)
        noisy = clean + 0.1 * torch.randn_like(clean)
        
        # Test SI-SNR calculation
        si_snr = calculate_si_snr(noisy, clean)
        assert si_snr.shape == (2,)
        assert torch.all(si_snr < 30)  # Should be reasonable
        
        # Test loss
        loss = si_snr_loss(noisy, clean)
        assert loss.item() < 0  # Negative SI-SNR
        
        print(f"  ✓ SI-SNR calculation working (mean: {si_snr.mean():.2f} dB)")
        return True
    except Exception as e:
        print(f"  ✗ Audio utils error: {e}")
        return False


def test_adapter():
    """Test adapter module"""
    print("\nTesting adapter modules...")
    
    try:
        from src.models.adapters import Adapter, TransformerBlockWithAdapters
        
        # Test basic adapter
        adapter = Adapter(input_dim=256, bottleneck_dim=1)
        x = torch.randn(2, 100, 256)
        out = adapter(x)
        
        assert out.shape == x.shape
        params = adapter.get_num_parameters()
        assert params < 600  # Should be small
        
        # Test transformer block with adapters
        block = TransformerBlockWithAdapters(d_model=256, bottleneck_dim=1)
        block.add_new_session_adapters()
        out = block(x)
        
        assert out.shape == x.shape
        
        print(f"  ✓ Adapters working (params: {params})")
        return True
    except Exception as e:
        print(f"  ✗ Adapter error: {e}")
        return False


def test_sepformer():
    """Test SepFormer model"""
    print("\nTesting SepFormer...")
    
    try:
        from src.models.sepformer import SepFormer
        
        model = SepFormer(
            n_basis=256,
            num_layers=2,  # Small for testing
            use_speechbrain=False
        )
        
        noisy = torch.randn(2, 1, 16000)
        enhanced = model(noisy)
        
        assert enhanced.shape == noisy.shape
        
        # Test encoder output
        encoded = model.get_encoder_output(noisy)
        assert encoded.shape[0] == 2
        assert encoded.shape[1] == 256
        
        print(f"  ✓ SepFormer working ({model.get_num_parameters():,} params)")
        return True
    except Exception as e:
        print(f"  ✗ SepFormer error: {e}")
        return False


def test_lna_model():
    """Test complete LNA model"""
    print("\nTesting LNA model...")
    
    try:
        from src.models.lna_model import LNAModel
        
        model = LNAModel(
            n_basis=256,
            num_layers=2,
            adapter_bottleneck_dim=1
        )
        
        # Test session 0 (pre-training)
        model.set_training_mode(session_id=0)
        noisy = torch.randn(2, 1, 16000)
        enhanced = model(noisy, session_id=0)
        assert enhanced.shape == noisy.shape
        
        # Test adding incremental session
        model.add_new_session(session_id=1)
        model.set_training_mode(session_id=1)
        enhanced = model(noisy, session_id=1)
        assert enhanced.shape == noisy.shape
        
        info = model.get_adapter_info()
        
        print(f"  ✓ LNA model working")
        print(f"    Total params: {info['total_parameters']:,}")
        print(f"    Adapter params: {info['adapter_parameters']:,}")
        print(f"    Adapter %: {info['adapter_percentage']:.2f}%")
        return True
    except Exception as e:
        print(f"  ✗ LNA model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_selector():
    """Test noise selector"""
    print("\nTesting noise selectors...")
    
    try:
        from src.selectors.noise_selector import KMeansSelector, MeanShiftSelector
        
        # Generate test features
        features_s0 = np.random.randn(100, 256) * 0.5
        features_s1 = np.random.randn(100, 256) * 0.5 + 2
        
        # Test K-Means
        selector = KMeansSelector(feature_dim=256, n_clusters=10)
        selector.fit_session(features_s0, session_id=0)
        selector.fit_session(features_s1, session_id=1)
        
        # Test prediction
        test_feature = np.random.randn(256) * 0.5 + 2
        pred = selector.predict(test_feature)
        assert pred in [0, 1]
        
        # Test Mean-Shift
        ms_selector = MeanShiftSelector(feature_dim=256)
        ms_selector.fit_session(features_s0, session_id=0)
        
        print(f"  ✓ Selectors working")
        print(f"    K-Means clusters: {len(selector.cluster_centers[0])}")
        return True
    except Exception as e:
        print(f"  ✗ Selector error: {e}")
        return False


def test_metrics():
    """Test evaluation metrics"""
    print("\nTesting metrics...")
    
    try:
        from src.evaluation.metrics import MetricsCalculator
        
        calc = MetricsCalculator(sample_rate=8000, metrics=['si_snr', 'sdr'])
        
        clean = torch.randn(16000)
        enhanced = clean + 0.05 * torch.randn_like(clean)
        
        metrics = calc.calculate_all(enhanced, clean)
        
        assert 'si_snr' in metrics
        assert 'sdr' in metrics
        assert metrics['si_snr'] > 0  # Should be positive
        
        print(f"  ✓ Metrics working")
        print(f"    SI-SNR: {metrics['si_snr']:.2f} dB")
        print(f"    SDR: {metrics['sdr']:.2f} dB")
        return True
    except Exception as e:
        print(f"  ✗ Metrics error: {e}")
        return False


def check_data():
    """Check if data is prepared"""
    print("\nChecking data...")
    
    data_root = Path("data/final_data")
    
    if not data_root.exists():
        print(f"  ⚠ Data not prepared yet")
        print(f"    Run: python prepare_data_final.py")
        return False
    
    # Check Session 0
    session0 = data_root / "session0_pretrain"
    if session0.exists():
        train_noisy = len(list((session0 / "train" / "noisy").glob("*.wav")))
        print(f"  ✓ Session 0 found: {train_noisy} training samples")
    else:
        print(f"  ✗ Session 0 not found")
        return False
    
    # Check incremental sessions
    for i in range(1, 6):
        session_dirs = list(data_root.glob(f"session{i}_incremental_*"))
        if session_dirs:
            print(f"  ✓ Session {i} found")
        else:
            print(f"  ✗ Session {i} not found")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("LNA INSTALLATION TEST".center(80))
    print("="*80)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['config'] = test_config()
    results['audio'] = test_audio_utils()
    results['adapter'] = test_adapter()
    results['sepformer'] = test_sepformer()
    results['lna_model'] = test_lna_model()
    results['selector'] = test_selector()
    results['metrics'] = test_metrics()
    results['data'] = check_data()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY".center(80))
    print("="*80 + "\n")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:15} {status}")
    
    print(f"\n  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Ready to train!")
    else:
        print("\n  ⚠ Some tests failed. Please check the errors above.")
    
    print("\n" + "="*80 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)