from . import __version__, MODEL_DIR
import joblib
from pandas import DataFrame
from .preprocess import PreProcess
import numpy as np

class Inference:

    def __init__(self, model_name: str = "classifier.pkl"):
        self.version = __version__
        self.model_name = model_name
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
        prediction = {"__version__": self.version,
                      "predictions": {'No': pred[0], 'Yes': pred[1]},
                      "label": accurate_prediciton}
        return prediction

    def _load_model(self):
        try:
            model = joblib.load(MODEL_DIR/self.model_name)
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
        status = system("dvc pull Models/model/classifier.pkl.dvc")
        return status
