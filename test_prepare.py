
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
