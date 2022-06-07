from . import TRACKING_URI, get_infos, __experiment_id__
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
        self.infos = get_infos()
        self.version = self.infos['version']
        self.model_run_id = self.infos['run_id']
        self.experiment_id = self.infos['experiment_id']
        self.updated_time = self.infos['updated_time']
        self.retry_state = 0
        self.c = {0: 'No', 1: 'Yes'}
        self.model = self._load_model()
        self.preprocess = PreProcess()
        self.failed = False

    def predict(self, data: DataFrame):

        self._pre_predict()

        data = self.preprocess.transform(data)
        pred = self.model.predict_proba(data)
        pred = pred.flatten()
        accurate_prediciton = self.c[np.argmax(pred)]
        prediction = {"version": self.version,
                      "run_id": self.model_run_id,
                      "predictions": {'No': pred[0], 'Yes': pred[1]},
                      "predicted": accurate_prediciton}
        return prediction
    
    def _pre_predict(self, **kwargs):
        if self.model == None:
            raise f'Error failed to load model {self.model_run_id}'

    def _load_model(self):
        try:
            self.model_path = Path(f"model/model_tracking/{self.experiment_id}/{self.model_run_id}/artifacts/model/")
        
            with open(self.model_path/'model.pkl', 'rb') as fb:
                model = pickle.load(fb)
            return model
        except Exception:
            status = self._pull_model()
            if status and (self.retry_state <= 5):
                print(f"Downloading model -> Retrying {self.retry_state}")
                self.failed = True
                return
            self._load_model()
            self.retry_state += 1

    def _pull_model(self):
        from os import system
        status = system(f"dvc pull -r models {self.model_path}.dvc")
        return status
