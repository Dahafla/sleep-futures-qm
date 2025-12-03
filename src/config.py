import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = BASE_DIR / "data" / "raw" / "sleep-export.csv"
DATA_PROCESSED = BASE_DIR / "data" / "processed" / "sleep_daily.csv"

# Sleep / contract settings
TARGET_SLEEP_HOURS = 7.5
CONTRACT_MULTIPLIER = 10.0  # $ per SleepIndex point

# Modeling
TEST_SIZE_FRACTION = 0.1   # last 30% days as test
RANDOM_STATE = 42
