from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, \
                            precision_score, recall_score, \
                            f1_score
from preprocess import PreProcess
from utils import load_data, save_fig, save_metrics, save_preprocess, ASSEST_DIR
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parent/'configs'

# Change if metrics need to be updated
def cal_matrics(Y_train, Y_train_preds, Y_test, Y_test_preds):
    METRICS = {'Dataset': ['Training', 'Testing']}

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

    metrics = {'accuracy': accuracy_train_test[1],
               'precision': precision_train_test[1],
               'recall': recall_train_test[1],
               'f1_score': f1_score_train_test[1],
               'roc_auc': rocauc_train_test[1]}
    return metrics, METRICS

def run_training(model_parmas: dict, tags: dict, reg_model_name: str, compare_latest:bool):
    X_train, Y_train, X_test, Y_test = load_data()
    preprocess = PreProcess()
    X_train, Y_train = preprocess.transform(X_train, Y_train)
    X_test, Y_test = preprocess.transform(X_test, Y_test)
    print("Data Preprocess...✅")

    ## MODELING
    model = RandomForestClassifier(**model_parmas)
    model.fit(X_train, Y_train)
    print("Model Fit...✅")
    
    ## PREDICTION
    Y_train_preds = model.predict(X_train)
    Y_test_preds = model.predict(X_test)
    
    metrics, METRICS = cal_matrics(Y_train, Y_train_preds, Y_test, Y_test_preds)

    ## SAVE FIGs, METRICS & Preprocessor
    save_fig(model, X_test, Y_test, reg_model_name, compare_latest)
    save_preprocess(preprocess)
    save_metrics(METRICS)

    data = {'metrics': metrics, 'params': model.get_params(), 'model': model,
            'tags': tags | {'estimator_name': type(model).__name__}}

    return data

def train(reg_model_name, compare_latest = True):
    with open(CONFIG_PATH/'training_config.yaml') as file:
        data = yaml.load(file, yaml.FullLoader)
    return run_training(**data, reg_model_name=reg_model_name, compare_latest=compare_latest)

if __name__ == "__main__":

    with open(CONFIG_PATH/'track_training_config.yaml') as file:
        data = yaml.load(file, yaml.FullLoader)

    train(data['registered_model_name'], True)