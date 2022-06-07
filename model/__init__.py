import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(__file__).resolve().parent.parent.parent/'dataset'
TRACKING_URI = Path(__file__).resolve().parent/'model_tracking'
TRAIN = "heartdisease_indicator_train.csv"
VALIDATION = "heartdisease_indicator_t.csv"
__experiment_id__ = "0"

def get_infos():
    with open(MODEL_DIR/'info.json', 'r') as version:
        data = json.load(version)
    return data