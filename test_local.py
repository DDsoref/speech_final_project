"""
Quick Test Script - Runs in ~5 minutes on CPU
Tests that everything works before remote GPU run

This uses TINY amounts of data just to verify:
- Data pipeline works
- Training starts
- Model can train for 1 epoch
- Evaluation runs
- Results are saved
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd):
    print(f"\n▶ Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Failed!")
        sys.exit(1)
    print("✓ Success")

def main():
    print("="*80)
    print("QUICK TEST - CPU VERSION")
    print("="*80)
    
    # Step 1: Check if data exists
    if not Path('data/LibriSpeech').exists():
        print("\n⚠️  No data found. Please run: bash set_up_data.sh")
        print("(This downloads ~10GB and takes ~30 min)")
        response = input("\nRun data download now? (y/N): ")
        if response.lower() == 'y':
            run_cmd("bash set_up_data.sh")
        else:
            print("Exiting. Run data download first.")
            sys.exit(1)
    
    # Step 2: Create tiny test data prep script
    print("\n▶ Creating tiny test dataset script...")
    
    test_prep = """
import os
import sys
sys.path.insert(0, os.getcwd())

# Modify prepare_data_10p.py to use even smaller amounts
import prepare_data_10p as prep

# Override config
prep.Config.SESSION0_TRAIN_UTTERANCES = 10
prep.Config.SESSION0_VAL_UTTERANCES = 5
prep.Config.SESSION0_TEST_UTTERANCES = 3
prep.Config.INCREMENTAL_TRAIN_UTTERANCES = 3
prep.Config.INCREMENTAL_VAL_UTTERANCES = 5
prep.Config.INCREMENTAL_TEST_UTTERANCES = 3

# Run
if __name__ == '__main__':
    prep.main()
"""
    
    with open('test_prepare.py', 'w') as f:
        f.write(test_prep)
    
    print("✓ Test script created")
    
    # Step 3: Prepare tiny dataset
    print("\n▶ Preparing tiny test dataset (~1 min)...")
    run_cmd("python test_prepare.py")
    
    # Step 4: Run training (1 epoch only)
    print("\n▶ Running 1 epoch training test (~3 min)...")
    run_cmd("python run_pipeline.py --mode pretrain --device cpu")
    
    # Step 5: Check outputs
    print("\n" + "="*80)
    print("CHECKING OUTPUTS")
    print("="*80)
    
    expected_files = [
        'checkpoints/session0_pretrain/best_model.pt',
        'logs/session0_pretrain/training_history.json',
    ]
    
    all_good = True
    for file in expected_files:
        if Path(file).exists():
            print(f"✓ {file}")
        else:
            print(f"❌ Missing: {file}")
            all_good = False
    
    if all_good:
        print("\n" + "="*80)
        print("✅ TEST PASSED!")
        print("="*80)
        print("\nYour code works! Ready for remote GPU training.")
        print("\nNext steps:")
        print("1. Push all code to GitHub")
        print("2. Tell your friend to use REMOTE_GPU_GUIDE.md")
        print("3. She runs: python run_pipeline.py --mode all --device cuda")
    else:
        print("\n❌ TEST FAILED - Some outputs missing")
        sys.exit(1)

if __name__ == "__main__":
    main()