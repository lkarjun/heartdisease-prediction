from . import TRACKING_URI, get_infos, __experiment_id__
import mlflow
import pickle
from pandas import DataFrame
import numpy as np
from pathlib import Path
from .utils import experiment, get_detail
from .preprocess import PreProcess, COLUMNS

ASSET_BASE = Path(f'mlruns/{experiment.experiment_id}')

class Inference:

    def __init__(self):
        self.retry_state = 0
        self.c = {0: 'No', 1: 'Yes'}
        self.failed = False
        self.preprocess = PreProcess()
        self.model = self._load_model()

    def predict(self, data: DataFrame):

        data = self._pre_predict(data = data)

        data = self.preprocess.transform(data)
        pred = self.model.predict_proba(data)
        pred = pred.flatten()

        prediction = self._post_predict(pred)
        return prediction
    
    def _pre_predict(self, data, **kwargs):
        if self.failed:
            raise f'Error failed to load model {self.model_run_id}'
        return data[COLUMNS]

    def _post_predict(self, pred, **kwargs):
        accurate_prediciton = self.c[np.argmax(pred)]
        prediction = {
                      "tag": self.tag,
                      "version": self.version,
                      "run_id": self.model_run_id,
                      "predictions": {'No': pred[0], 'Yes': pred[1]},
                      "predicted": accurate_prediciton
                    }
        
        return prediction

    def _load_model(self):
        try:
            self.latest_run = get_detail().latest_versions[-1]
            self.model_path = Path(ASSET_BASE/f"{self.latest_run.run_id}/artifacts/model/")
            
            with open(self.model_path/'model.pkl', 'rb') as fb:
                model = pickle.load(fb)
                
            self.model_run_id = self.latest_run.run_id
            self.version = self.latest_run.version
            self.tag = self.latest_run.name
            return model
        except Exception as e:
            print(e)
            status = self._pull_model()
            if status == 0:
                return self._load_model()
            elif self.retry_state <= 5 and status == 1:
                return self._load_model()
            else:
                self.retry_state += 1
                self.failed = True
                return


    def _pull_model(self):
        from os import system
        status = system(f"dvc pull -r models {self.model_path}.dvc")
        return status
