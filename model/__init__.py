import json
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(__file__).resolve().parent.parent.parent/'dataset'
TRACKING_URI = Path(__file__).resolve().parent/'model_tracking'
TRAIN = "heartdisease_indicator_train.csv"
VALIDATION = "heartdisease_indicator_t.csv"

with open(MODEL_DIR/'info.json', 'r') as version:
    __meta__ = json.load(version)
    __version__ = __meta__['version']
    __tag__ = __meta__['tag']
    __latest_run_id__ = __meta__['run_id']
    __experiment_id__ = __meta__['experiment_id']