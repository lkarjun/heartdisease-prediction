import streamlit as st
from model import __experiment_id__, TRACKING_URI, __latest_run_id__
import mlflow
import json
from datetime import datetime
from .static_html import render, track_model_html
from time import sleep

mlflow.set_tracking_uri(TRACKING_URI.absolute())
experiment = mlflow.set_experiment(experiment_id=__experiment_id__)

def format_string(uri, expid, runid):
    return uri/f"{expid}/{runid}/artifacts/resources"

def get_run_info(return_latest: bool = False):
    runs = mlflow.list_run_infos(__experiment_id__)
    return runs[0] if return_latest else runs

def get_info(run_id):
    run = mlflow.get_run(run_id)
    tags = run.data.tags
    user_defined_tags = {k:v for k, v in tags.items() if not k.startswith("mlflow") and k !='model_version'}
    update_date = json.loads(tags['mlflow.log-model.history'])[0]['utc_time_created']
    update_date = datetime.strptime(update_date, '%Y-%m-%d %H:%M:%S.%f').replace(microsecond=0)
    data = {'run_id': run_id, 
            'tag': user_defined_tags, 
            'version': tags['model_version'],
            'experiment_id': __experiment_id__,
            'updated_time': update_date.date()}
    return data

def write_time_line(runs: list, end: int = 0):
    for run in runs:
        with st.spinner("Loading"): 
            sleep(1)
            data = get_info(run.run_id)
            ASSESTS_DIR = TRACKING_URI/f"{__experiment_id__}/{run.run_id}/artifacts/resources"

            st.markdown(f"<h6 style='text-align: center; color: green'>ID: {data['run_id']}</h6>", True)

            latest = True if __latest_run_id__ == run.run_id else False
            render(track_model_html, data=data, latest = latest)

            st.markdown(f"<h5 style='text-align: left;'>Performance Metrics</h5>", True)
            metrics = (ASSESTS_DIR/'metrics.md').read_text()
            st.markdown(metrics, True)
            st.markdown("<br>", True)
            st.markdown(f"<h5 style='text-align: left;'>Confusion Matrix</h5>", True)
            image = (ASSESTS_DIR/'Confusion-Matrix.png').read_bytes()
            st.image(image, width = 500)
            st.write('---')
            st.write('---')

def main():
    st.markdown(f"<h2 style='text-align: Center;'>Model Timeline 🫀</h2>", True)
    
    with st.spinner("Loading..."):
        full_runs = get_run_info()
    st.write('---')
    write_time_line(full_runs)
    # track_model_render(runs = full_runs, 
    #                    get_info = get_info, 
    #                    TRACKING_URI=TRACKING_URI,
    #                    __experiment_id__ = __experiment_id__,
    #                    format_string = format_string,
    #                    st = st)