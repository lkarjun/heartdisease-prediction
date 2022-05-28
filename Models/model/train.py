from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import PreProcess
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import joblib

## STATIC PATHS
TARGET = "HeartDisease"
ASSEST_DIR = Path(__file__).resolve().parent/'Assests'
MODEL_DIR = Path(__file__).resolve().parent
DATASET_DIR = Path(__file__).resolve().parent.parent.parent/'Dataset'

## DATALOADING
TRAIN_TEST = pd.read_csv(DATASET_DIR/'heartdisease_indicator_train.csv')
VALID = pd.read_csv(DATASET_DIR/'heartdisease_indicator_test.csv')
print("\n\nLoad data...✅")

## DATA PREPROCESS
TRAIN_TEST['GenHealth'] = pd.Categorical(TRAIN_TEST.GenHealth, 
                                         categories=["Poor","Fair","Good",
                                                     "Very good","Excellent"],
                                         ordered=True)

X_train, Y_train = TRAIN_TEST.drop(TARGET, axis = 1), TRAIN_TEST[TARGET]

X_test, Y_test = VALID.drop(TARGET, axis = 1), VALID[TARGET]


preprocess = PreProcess()

(X_train, Y_trian), (X_test, Y_test) = preprocess.transform(X_train, Y_train), preprocess.transform(X_test, Y_test)

print("Data Preprocess...✅")


## MODELING
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)

model.fit(X_train, Y_train)

print("Model Fit...✅")

## PREDICTION
y_pred = model.predict(X_test)

## VALIDATION
ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)

## SAVE FIG AND METRICS
plt.savefig(ASSEST_DIR/'confusion_matrix.png')

## SAVE MODEL
joblib.dump(model, MODEL_DIR/'classifier.pkl')
print("Save Model...✅\n\n")