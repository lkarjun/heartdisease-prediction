import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(__file__).resolve().parent.parent.parent/'Dataset'
TRAIN = "heartdisease_indicator_train.csv"
VALIDATION = "heartdisease_indicator_t.csv"

with open(MODEL_DIR/'VERSION.json', 'r') as version:
    __version__ = json.load(version)['Version']