from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from preprocess import PreProcess
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

METRICS = {'Dataset': ['Training', 'Testing']}

## STATIC PATHS
TARGET = "HeartDisease"
ASSEST_DIR = Path(__file__).resolve().parent/'resources'
DATASET_DIR = Path(__file__).resolve().parent.parent/'dataset'

ASSEST_DIR.mkdir(exist_ok=True)

def get_preprocessed_data():

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

    (X_train, Y_train), (X_test, Y_test) = preprocess.transform(X_train, Y_train), preprocess.transform(X_test, Y_test)

    print("Data Preprocess...✅")
    return X_train, Y_train, X_test, Y_test

def train():
    X_train, Y_train, X_test, Y_test = get_preprocessed_data()

    ## MODELING
    model = RandomForestClassifier(n_estimators=70, max_depth=5, random_state=2390, max_features='log2', bootstrap = False)
    model.fit(X_train, Y_train)
    print("Model Fit...✅")
    
    ## PREDICTION
    Y_train_preds = model.predict(X_train)
    Y_test_preds = model.predict(X_test)
    
    ## METRICS

    METRICS['Accuracy']= accuracy_train_test = [round(accuracy_score(Y_train, Y_train_preds), 2), 
                                                round(accuracy_score(Y_test, Y_test_preds), 2)]

    METRICS['Precision'] = precision_train_test = [round(precision_score(Y_train, Y_train_preds), 2), 
                                                  round(precision_score(Y_test, Y_test_preds), 2)]

    METRICS['Recall'] = recall_train_test = [round(recall_score(Y_train, Y_train_preds), 2), 
                                               round(recall_score(Y_test, Y_test_preds), 2)]

    METRICS['F1-Score'] = f1_score_train_test = [round(f1_score(Y_train, Y_train_preds), 2), 
                                                 round(f1_score(Y_test, Y_test_preds), 2)]

    METRICS['ROU-AUC-SCORE'] = rocauc_train_test = [round(roc_auc_score(Y_train, Y_train_preds), 2), 
                                                    round(roc_auc_score(Y_test, Y_test_preds), 2)]


    ## VALIDATION
    ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test)

    ## SAVE FIG AND METRICS
    plt.savefig(ASSEST_DIR/'Confusion-Matrix.png')


    pd.DataFrame(METRICS).to_html(ASSEST_DIR/'metrics.md', index = False)
 

    print("Save Metrics...✅")

    return_data = {'metrics': {'accuracy': accuracy_train_test[1],
                               'precision': precision_train_test[1],
                               'recall': recall_train_test[1],
                               'f1_score': f1_score_train_test[1],
                               'roc_auc': rocauc_train_test[1]},
                   'params': model.get_params(),
                   'model': model,
                   'tags': {'estimator_name': type(model).__name__,
                            'model_tag': "BaseLine",
                            'model_version': "0.2.3",
                            'model_type': "Ensemble",
                            'model': 'Tree',
                            'trained_from': 'Github CI CD'
                            }
                   }

    return return_data


if __name__ == "__main__":
    train()
