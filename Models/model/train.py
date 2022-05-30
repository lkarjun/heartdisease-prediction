from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess import PreProcess
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import joblib

SEED = 232
METRICS = {'Dataset': ['Training', 'Testing']}

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
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=SEED)

model.fit(X_train, Y_train)

print("Model Fit...✅")

## PREDICTION
Y_train_preds = model.predict(X_train)
Y_test_preds = model.predict(X_test)

METRICS['Accuracy'] = [round(accuracy_score(Y_train, Y_train_preds), 2), 
                       round(accuracy_score(Y_test, Y_test_preds), 2)]

METRICS['Precision'] = [round(precision_score(Y_train, Y_train_preds), 2), 
                       round(precision_score(Y_test, Y_test_preds), 2)]

METRICS['Recall'] = [round(recall_score(Y_train, Y_train_preds), 2), 
                       round(recall_score(Y_test, Y_test_preds), 2)]

METRICS['F1-Score'] = [round(f1_score(Y_train, Y_train_preds), 2), 
                       round(f1_score(Y_test, Y_test_preds), 2)]

METRICS['ROU-AUC-SCORE'] = [round(roc_auc_score(Y_train, Y_train_preds), 2), 
                            round(roc_auc_score(Y_test, Y_test_preds), 2)]

## SAVE METRICS

with open(ASSEST_DIR/'metrics.md', "w") as file:
    scores = pd.DataFrame(METRICS).to_markdown()
    file.write(scores)

print("Save Metrics...✅")

## VALIDATION
ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)

## SAVE FIG AND METRICS
plt.savefig(ASSEST_DIR/'confusion_matrix.png')

## SAVE MODEL
joblib.dump(model, MODEL_DIR/'classifier.pkl')
print("Save Model...✅\n\n")
