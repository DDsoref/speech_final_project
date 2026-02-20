
# Test version - uses 0.1% of data for quick CPU test

from src.utils.config import Config as OrigConfig

class TestConfig(OrigConfig):
    # Tiny data amounts
    SESSION0_TRAIN_UTTERANCES = 10       # Just 10 samples
    SESSION0_VAL_UTTERANCES = 5
    SESSION0_TEST_UTTERANCES = 3
    
    INCREMENTAL_TRAIN_UTTERANCES = 3
    INCREMENTAL_VAL_UTTERANCES = 5
    INCREMENTAL_TEST_UTTERANCES = 3
    
    # Fast training
    pretrain_epochs = 1
    incremental_epochs = 1
    train_batch_size = 2
    num_workers = 0  # Mac compatibility
