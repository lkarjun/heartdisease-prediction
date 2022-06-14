import streamlit as st
from model.utils import get_regmodel_names, experiment, Path
import mlflow
import json
from datetime import datetime
from .static_html import render, track_model_html
from time import sleep
import pandas as pd

ASSET_BASE = Path(__file__).resolve().parent.parent/'mlruns'

def format_string(uri, expid, runid):
    return uri/f"{expid}/{runid}/artifacts/resources"

def get_run_info(return_latest: bool = False):
    runs = mlflow.list_run_infos(__experiment_id__)
    return runs[0] if return_latest else runs

def get_info(run_id, version):
    run = mlflow.get_run(run_id)
    tags = run.data.tags
    user_defined_tags = {k:v for k, v in tags.items() if not k.startswith("mlflow") and k !='model_version'}
    update_date = json.loads(tags['mlflow.log-model.history'])[0]['utc_time_created']
    update_date = datetime.strptime(update_date, '%Y-%m-%d %H:%M:%S.%f').replace(microsecond=0)
    data = {'run_id': run_id, 
            'tag': user_defined_tags, 
            'version': f"Version {version}",
            'experiment_id': experiment.experiment_id,
            'updated_time': update_date.date()}
    return data

def write_time_line(registeredmodels: list):
    c = 1
    for model in registeredmodels[::-1]:
        with st.spinner("Loading"): 
            sleep(1)
            run = model.latest_versions[-1]
            data = get_info(run.run_id, run.version)
            ASSESTS_DIR = ASSET_BASE/f"{experiment.experiment_id}/{run.run_id}/artifacts/resources"

            latest = True if c == 1 else False
            render(track_model_html, data=data, latest = latest, name=run.name, 
                    height=155, version=run.version, status= run.status, 
                    stage = run.current_stage, notdetails = True)

            st.markdown(f"<h5 style='text-align: left;'>Performance Metrics</h5>", True)
            metrics = pd.read_html(ASSESTS_DIR/'metrics.md')[0]
            st.dataframe(metrics)
            st.markdown("<br>", True)
            st.markdown(f"<h5 style='text-align: left;'>Confusion Matrix</h5>", True)
            image = (ASSESTS_DIR/'Confusion-Matrix.png').read_bytes()
            st.image(image)
            st.markdown("<br>", True)
            if (ASSESTS_DIR/'Comparing-Version.png').exists():
                st.markdown(f"<h5 style='text-align: left;'>Comparison</h5>", True)
                image = (ASSESTS_DIR/'Comparing-Version.png').read_bytes()
                st.image(image)
            st.markdown("<br>", True)
            st.write('---')
            st.markdown("<br>", True)
        c += 1

def main():
    st.markdown(f"<h2 style='text-align: Center;'>Model Timeline 🫀</h2>", True)
    with st.spinner("Loading"):
        reg_models = get_regmodel_names()
    st.write('---')
    write_time_line(reg_models)