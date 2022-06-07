from . import __version__, MODEL_DIR, __latest_run_id__, __experiment_id__, TRACKING_URI, __updated_time__
import mlflow
import pickle
from pandas import DataFrame
from .preprocess import PreProcess
import numpy as np
from pathlib import Path

mlflow.set_tracking_uri(TRACKING_URI.absolute())
experiment = mlflow.set_experiment(experiment_id=__experiment_id__)

class Inference:

    def __init__(self):
        self.version = __version__
        self.model_run_id = __latest_run_id__
        self.experiment_id = __experiment_id__
        self.updated_time = __updated_time__
        self.retry_state = 0
        self.c = {0: 'No', 1: 'Yes'}
        self.model = self._load_model()
        self.preprocess = PreProcess()
        self.failed = False

    def predict(self, data: DataFrame):
        if self.model == None:
            return {'Error': 'Failed to load model'}
        data = self.preprocess.transform(data)
        pred = self.model.predict_proba(data)
        pred = pred.flatten()
        accurate_prediciton = self.c[np.argmax(pred)]
        prediction = {"version": self.version,
                      "run_id": self.model_run_id,
                      "predictions": {'No': pred[0], 'Yes': pred[1]},
                      "predicted": accurate_prediciton}
        return prediction

    def _load_model(self):
        try:
            self.model_path = Path(f"model/model_tracking/{self.experiment_id}/{self.model_run_id}/artifacts/model/")
        
            with open(self.model_path/'model.pkl', 'rb') as fb:
                model = pickle.load(fb)
            return model
        except Exception:
            status = self._pull_model()
            if status and (self.retry_state >= 5):
                print(f"Downloading model -> Retrying {self.retry_state}")
                self.failed = True
                return
            self._load_model()
            self.retry_state += 1

    def _pull_model(self):
        from os import system
        status = system(f"dvc pull -r models {self.model_path}.dvc")
        return status
