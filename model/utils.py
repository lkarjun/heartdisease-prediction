from sklearn.metrics import roc_auc_score, accuracy_score, \
                            precision_score, recall_score, \
                            f1_score, precision_recall_curve, \
                            roc_curve, ConfusionMatrixDisplay
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
import pickle

ASSEST_DIR = Path(__file__).resolve().parent/'resources'
ASSEST_DIR.mkdir(exist_ok=True)
DATASET_DIR = Path(__file__).resolve().parent.parent/'dataset'

TRACKING_URI =  Path(__file__).resolve().parent/"tracking/records.db"
mlflow.set_tracking_uri(f"sqlite:///{TRACKING_URI.absolute()}")
experiment = mlflow.set_experiment("CI/CD")


ARTIFACT_PATH = Path(__file__).resolve().parent.parent/"mlruns"/experiment.experiment_id

def get_run_infos(latest: bool = False, second_last: bool = False):
    runs = mlflow.list_run_infos(experiment.experiment_id)
    if latest: return runs[0]
    if second_last: return runs[1]
    return runs

def get_run_id(latest = True):
    run = get_run_infos(return_latest=True) if latest else get_run_infos(second_last=True)
    return run.run_id

def get_model(reg_model_name = "BASELINE", latest: bool = True, second_last: bool = True):
    client = MlflowClient()
    mvs = client.search_registered_models(f"name='{reg_model_name}'")
    
    if len(mvs) and latest:
        mvs = mvs[0].latest_versions
        return get_model_helper(mvs[-1])
    if len(mvs) and second_last and len(mvs) >= 2:
        mvs = mvs[0].latest_versions
        return get_model_helper(mvs[-2])
    return False

def get_model_helper(mv):
    run_id = mv.run_id
    with open(ARTIFACT_PATH/f'{run_id}/artifacts/model/model.pkl', 'rb') as fb:
        model = pickle.load(fb)
    return model, mv.version

def get_regmodel(reg_model_name = "BASELINE"):
    client = MlflowClient()
    mvs = client.search_registered_models(f"name='{reg_model_name}'")[0]
    return mvs.latest_versions[::-1]

def load_data():
    TARGET = "HeartDisease"
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
    return X_train, Y_train, X_test, Y_test

def save_metrics(METRICS):
    pd.DataFrame(METRICS).to_html(ASSEST_DIR/'metrics.md', index = False)
    print("Save Metrics...✅")

def save_preprocess(preprocess):
    with open(ASSEST_DIR/"preprocessor.pkl", "wb") as file:
        pickle.dump(preprocess, file)
        print("Save Preprocessor...✅")

def save_fig(model, X_test, Y_test, reg_model_name, compare_latest):
    ConfusionMatrixDisplay.from_estimator(model, X_test, Y_test, cmap='PuRd')
    plt.savefig(ASSEST_DIR/'Confusion-Matrix.png')

    comparing_model = get_model(reg_model_name, latest=compare_latest)
    if not comparing_model:
        return 
    fig, axs = plt.subplots(ncols=2)
    fig.set_size_inches(14, 5)
    label = [f"{reg_model_name} Model", f"{reg_model_name} Model Version: {comparing_model[1]}"]
    cmodel_prob = comparing_model[0].predict_proba(X_test)
    model1_prob = model.predict_proba(X_test)

    fpr, tpr, thresholds = roc_curve(Y_test, model1_prob[:,1])
    fpr2, tpr2, thresholds2 = roc_curve(Y_test, cmodel_prob[:,1])
    axs[0].plot(fpr, tpr, color='#FAC213', linewidth=2, linestyle='--')
    axs[0].plot(fpr2, tpr2, color="#5FD068", linewidth=2, linestyle='--')
    axs[0].legend(label)
    axs[0].set(xlabel='False Positive Rate',
                ylabel='True Positive Rate',
                xlim=[-.01, 1.01], ylim=[-.01, 1.01],
                title='ROC curve')
    axs[0].grid(True)
    
    precision, recall, _ = precision_recall_curve(Y_test, model1_prob[:,1])
    precision2, recall2, _ = precision_recall_curve(Y_test, cmodel_prob[:,1])
    axs[1].plot(recall, precision, color='#FAC213', linewidth=2, linestyle='--')
    axs[1].plot(recall2, precision2, color='#5FD068', linewidth=2, linestyle='--')
    axs[1].legend(label)


    axs[1].set(xlabel='Recall', ylabel='Precision',
        xlim=[-.01, 1.01], ylim=[-.01, 1.01],
        title='Precision-Recall curve')
    axs[1].grid(True)
    plt.suptitle(f'Comparing {reg_model_name} Models')
    plt.savefig(ASSEST_DIR/'Comparing-Version.png')



if __name__ == "__main__":
    if get_model("BASELINE-2"):
        print("ok")