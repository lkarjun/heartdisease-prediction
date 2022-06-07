import streamlit as st
from model.inference import Inference
from model import TRACKING_URI, __experiment_id__, TRAIN, get_infos
from datetime import datetime
from .static_html import render, track_model_html
from pathlib import Path
import pandas as pd

DATASET = Path(__file__).parent.parent/"dataset"/TRAIN


def display_about_data():
    if not DATASET.exists():
        from os import system
        with st.spinner("Loading Dataset..."):
            system(f'dvc pull dataset/{TRAIN}.dvc')
    st.markdown(f"<h2 style='text-align: center;'>Training Dataset</h2>", True)
    st.write('---')
    df = pd.read_csv(DATASET)
    st.dataframe(df.head(5))
    st.markdown("<br>", True)
    st.markdown(f"<h4 style='text-align: left;'>Target Count</h4>", True)
    st.bar_chart(df['HeartDisease'].value_counts())

def about_model():
    with st.spinner("Loading"):
        infos = get_infos()
        __updated_time__ = infos['updated_time']
        __latest_run_id__ = infos['run_id']
        __tag__ = infos['tag']
        __version__ = infos['version']
        ASSESTS_DIR = TRACKING_URI/f"{__experiment_id__}/{__latest_run_id__}/artifacts/resources"
        update_date_time = datetime.strptime(__updated_time__, '%Y-%m-%d %H:%M:%S.%f').replace(microsecond=0)
        
    st.markdown(f"<h2 style='text-align: center;'>Model Version: '{__version__} : {__tag__}'</h2>", True)
    st.write('---')
    st.markdown(f"<h6 style='text-align: center; color: green'>ID: {__latest_run_id__}</h6>", True)
    data = {'version': __version__, 'updated_time': update_date_time, 'tag': {'_': __tag__}}
    render(track_model_html, 30, data=data, latest = True)
    
    metrics = pd.read_html(ASSESTS_DIR/'metrics.md')[0]
    st.subheader(f"Performance Metrics")
    st.dataframe(metrics)

    st.markdown("<br>", True)
    st.subheader(f"Confusion Metrics of Model")

    image = (ASSESTS_DIR/'Confusion-Matrix.png').read_bytes()
    st.image(image)
    st.markdown("<br><br>", True)

def main():
    about_model()
    display_about_data()