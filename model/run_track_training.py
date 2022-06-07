from pathlib import Path
from train import train, ASSEST_DIR
from os import system
import argparse
import warnings
import mlflow
import json
import os

file_prefix = "file:///" if os.name == 'nt' else "file://"

warnings.filterwarnings("ignore")

TRACKING_URI = Path(__file__).resolve().parent/'model_tracking'
mlflow.set_tracking_uri(f"{file_prefix}{TRACKING_URI.absolute()}")
experiment = mlflow.set_experiment("CI/CD")

def track():
    with mlflow.start_run(run_name="CI/CD-AUTO") as run:
        return_data = train()
        mlflow.log_params(return_data['params'])
        print("\nLog Params...✅")
        mlflow.log_metrics(return_data['metrics'])
        print("Log Metrics...✅")
        mlflow.set_tags(return_data['tags'])
        print("Set Tags...✅")
        mlflow.sklearn.log_model(return_data['model'], 'model')
        print("Log Model...✅")
        for file in ASSEST_DIR.iterdir():
            mlflow.log_artifact(file.absolute(), artifact_path = 'resources')
            print(f"Log Resourse: {file.name}...✅")
    save_info(run, experiment.experiment_id)
    push_latest_model_to_azure()
    mlflow.end_run()
    print("Finshed...✔️\n")

def save_info(run, experiment_id):
    run_id = run.info.run_id
    run = mlflow.get_run(run_id)
    tags = run.data.tags
    updated_time = json.loads(tags['mlflow.log-model.history'])[0]['utc_time_created']
    data = {'run_id': run_id, 
            'tag': tags['model_tag'], 
            'version': tags['model_version'],
            'experiment_id': experiment_id,
            'updated_time': updated_time}
    with open('model/info.json', 'w') as fp:
        json.dump(data, fp)
    print(f"Save Info...✅")

def get_run_info(return_latest: bool = False):
    runs = mlflow.list_run_infos(experiment.experiment_id)
    return runs[0] if return_latest else runs

def push_latest_model_to_azure():
    experiment_id = experiment.experiment_id
    run = get_run_info(return_latest=True)
    artifact_path = f"model/model_tracking/{experiment_id}/{run.run_id}/artifacts/model"
    add_command = f"dvc add {artifact_path}"
    push_command = f"dvc push -r models {artifact_path}.dvc"
    add_status = system(add_command)
    if add_status == 0:
        push_status = system(push_command)
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
        track()