import streamlit as st
from model import TRACKING_URI, __experiment_id__, TRAIN, get_infos
from model.utils import get_detail, experiment
from datetime import datetime
from .static_html import render, track_model_html
from .track_model import get_info
from pathlib import Path
import pandas as pd

DATASET = Path(__file__).parent.parent/"dataset"/TRAIN
ASSET_BASE = Path(__file__).resolve().parent.parent/'mlruns'

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
        reg_model = get_detail()
        latest_model = reg_model.latest_versions[-1]
        version = latest_model.version
        name = latest_model.name
        data = get_info(latest_model.run_id, version)
        ASSESTS_DIR = ASSET_BASE/f"{experiment.experiment_id}/{latest_model.run_id}/artifacts/resources"
        
        st.markdown(f"<h2 style='text-align: center;'>Model Version: '{name} : {version}.0'</h2>", True)
        st.write('---')
        render(track_model_html, data=data, latest = True, name=name, 
                        height=100, version=version, status=latest_model.status, 
                        stage = latest_model.current_stage,
                        notdetails = False)
        
        metrics = pd.read_html(ASSESTS_DIR/'metrics.md')[0]
        st.subheader(f"Performance Metrics")
        st.dataframe(metrics)

        st.markdown("<br>", True)
        st.subheader(f"Confusion Metrics of Model")

        image = (ASSESTS_DIR/'Confusion-Matrix.png').read_bytes()
        st.image(image)
        st.markdown("<br>", True)
        if (ASSESTS_DIR/'Comparing-Version.png').exists():
            st.markdown(f"<h5 style='text-align: left;'>Comparison</h5>", True)
            image = (ASSESTS_DIR/'Comparing-Version.png').read_bytes()
            st.image(image)
        st.markdown("<br>", True)

def main():
    about_model()
    display_about_data()