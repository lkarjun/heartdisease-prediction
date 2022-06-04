import streamlit as st
from model import __experiment_id__, TRACKING_URI
import mlflow
import json
from datetime import datetime
from annotated_text import annotated_text, annotation

mlflow.set_tracking_uri(f"file:/{TRACKING_URI.absolute()}")
experiment = mlflow.set_experiment(experiment_id=__experiment_id__)


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
            'updated_time': update_date}
    return data

def write_time_line(runs: list, end: int = 0):
    for run in runs:
        data = get_info(run.run_id)
        ASSESTS_DIR = TRACKING_URI/f"{__experiment_id__}/{run.run_id}/artifacts/resources"

        st.markdown(f"<h6 style='text-align: left; color: green'>Run ID: {data['run_id']}</h6>", True)
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Date: {data['updated_time'].date()}")
        with col2:
            st.write(f"Model Version: {data['version']}")

        st.write(f'Tags: &emsp;', '&emsp;'.join(v for _, v in data['tag'].items()))
        st.markdown("<br>", True)

        st.markdown(f"<h5 style='text-align: left;'>Performance Metrics</h5>", True)
        metrics = (ASSESTS_DIR/'metrics.md').read_text()
        st.markdown(metrics)
        st.markdown("<br>", True)
        st.markdown(f"<h5 style='text-align: left;'>Confusion Matrix</h5>", True)
        image = (ASSESTS_DIR/'Confusion-Matrix.png').read_bytes()
        st.image(image, width = 500)

        st.write('---')

def main():
    st.markdown(f"<h2 style='text-align: Center;'>Model Timeline 🫀</h2>", True)
    
    with st.spinner("Loading..."):
        full_runs = get_run_info()
    st.write('---')
    write_time_line(full_runs)