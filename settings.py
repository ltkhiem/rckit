from pathlib import Path

"""
This part is for putting some general configuration for interacting with the dataset
"""

# General settings
SUBJECTS = ['0000', '0001', '0002', '0003', '0004', '0005', '0006', '0007', '0008']
NUM_BLOCKS = 4  # Number of sessions
N_SAMPLES = 96  # 24 samples per block, totaling 96

# Data paths:
DATA_PATH = Path('/mnt/DATA/ltkhiem/rcir/dataset/') 
SUBJECT_PATHS = [DATA_PATH / sbj for sbj in SUBJECTS]
ET_PATHS = [path / 'fixed_et' for path in SUBJECT_PATHS]
EOG_PATHS = [path / 'fixed_eog' for path in SUBJECT_PATHS]
INFO_PATHS = [path / 'fixed_info' for path in SUBJECT_PATHS]
META_INFO_PATHS = [[path / f'metadata_log_{bid}.tsv') for bid in range(NUM_BLOCKS)] for path in INFO_PATHS]
SCREENSHOT_PATHS = [path / 'screenshot' for path in SUBJECT_PATHS]

# Settings for Eye Tracking
ET_CALIB_PATHS = [path / 'tracker_calib_log.tsv' for path in ET_PATHS]
ET_EVENT_PATHS = [path / 'tracker_event_log.tsv') for path in ET_PATHS]
ET_DATA_PATHS = [[path / f'tracker_data_log_{bid}.tsv') for bid in range(NUM_BLOCKS)] for path in ET_PATHS]

# Reading behaviours
CONDITION_TEXT = ['reading', 'scanning', 'skimming', 'proof_reading']



"""
This part is for setting up the parameter for ocular movement detection algorithms
"""
_TH_FIXA_DUR = 200 # milliseconds
