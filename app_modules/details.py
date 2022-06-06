import streamlit as st
from model.inference import Inference
from model import  __version__, __tag__, TRACKING_URI, __experiment_id__, __latest_run_id__, __updated_time__
from datetime import datetime
from .static_html import render, track_model_html

update_date_time = datetime.strptime(__updated_time__, '%Y-%m-%d %H:%M:%S.%f').replace(microsecond=0)


ASSESTS_DIR = TRACKING_URI/f"{__experiment_id__}/{__latest_run_id__}/artifacts/resources"

def main():

    st.markdown(f"<h2 style='text-align: center;'>Model Version: '{__version__} : {__tag__}'</h2>", True)
    st.write('---')
    st.markdown(f"<h6 style='text-align: center; color: green'>ID: {__latest_run_id__}</h6>", True)
    data = {'version': __version__, 'updated_time': update_date_time, 'tag': {'_': __tag__}}
    render(track_model_html, 30, data=data, latest = True)
    
    col1, col2 = st.columns(2)
    metrics = (ASSESTS_DIR/'metrics.md').read_text()
    st.subheader(f"Performance Metrics")
    st.markdown(metrics, True)

    st.markdown("<br>", True)
    st.subheader(f"Confusion Metrics of Model")

    image = (ASSESTS_DIR/'Confusion-Matrix.png').read_bytes()
    st.image(image, use_column_width ='auto')

