from pathlib import Path
from train import train, ASSEST_DIR
from mlflow.tracking import MlflowClient
import mlflow
import argparse
import warnings
import os
import yaml


warnings.filterwarnings("ignore")

TRACKING_URI =  Path(__file__).resolve().parent/"tracking/records.db"

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_URI.absolute()}")

experiment = mlflow.set_experiment("CI/CD")

def set_reg_model_tag(registered_model_tag: dict, registered_model_name:str):
    client = MlflowClient()
    for k, v in registered_model_tag.items():
        client.set_registered_model_tag(
                    name=registered_model_name,
                    key=k, 
                    value=v)

def change_model_stage(registered_model_name:str):
    
    client = MlflowClient()
    mvs = client.search_model_versions(f"name='{registered_model_name}'")
    for mv in mvs[:-1]:
        client.transition_model_version_stage(
                            name=registered_model_name,
                            version=mv.version,
                            stage='Archived')
    client.transition_model_version_stage(
                            name=registered_model_name,
                            version=mvs[-1].version,
                            stage='Production'
    )
    print("Model stage changes...✅")
    return

def track(run_name:str, 
          registered_model_name: str, 
          registered_model_tag:dict,
          ):
    with mlflow.start_run(run_name=run_name) as run:
        return_data = train()
        mlflow.log_params(return_data['params'])
        print("\nLog Params...✅")
        mlflow.log_metrics(return_data['metrics'])
        print("Log Metrics...✅")
        mlflow.set_tags(return_data['tags'])
        print("Set Tags...✅")
        mlflow.sklearn.log_model(sk_model = return_data['model'], 
                                 artifact_path = 'model',
                                 registered_model_name = registered_model_name)
        print("Log Model...✅")
        for file in ASSEST_DIR.iterdir():
            mlflow.log_artifact(file.absolute(), artifact_path = 'resources')
            print(f"Log Resourse: {file.name}...✅")
    mlflow.end_run()
    
    change_model_stage(registered_model_name = registered_model_name)
    set_reg_model_tag(registered_model_name = registered_model_name,
                      registered_model_tag = registered_model_tag)
    push_latest_model_to_azure()
    print("Finshed...✔️\n")

def get_run_info(return_latest: bool = False):
    runs = mlflow.list_run_infos(experiment.experiment_id)
    return runs[0] if return_latest else runs

def push_latest_model_to_azure():
    experiment_id = experiment.experiment_id
    run = get_run_info(return_latest=True)
    artifact_path = f"mlruns/{experiment_id}/{run.run_id}"
    add_command = f"dvc add {artifact_path}"
    push_command = f"dvc push -r models {artifact_path}.dvc"
    add_status = os.system(add_command)
    if add_status == 0:
        push_status = os.system(push_command)
        if push_status:
            print("Failed to push model...❌")
        else: print("Successfuly pushed model...✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=False, help = "Return Run Latest Id", type=bool)
    args = parser.parse_args()

    if args.run_id:
        print(get_run_info(return_latest=True).run_id)
    else:
        with open(TRACKING_URI.parent.parent/'configs/track_training_config.yaml') as file:
            data = yaml.load(file, yaml.FullLoader)
        
        track(**data)