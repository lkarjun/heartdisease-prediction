import streamlit as st
from model.inference import Inference
from model import  __version__,__tag__, TRACKING_URI, __experiment_id__, __latest_run_id__, __updated_time__
from datetime import datetime

update_date_time = datetime.strptime(__updated_time__, '%Y-%m-%d %H:%M:%S.%f').replace(microsecond=0)


ASSESTS_DIR = TRACKING_URI/f"{__experiment_id__}/{__latest_run_id__}/artifacts/resources"

def main():

    st.markdown(f"<h2 style='text-align: left;'>Model Version: '{__version__}:{__tag__}'</h2>", True)
    st.write(f"Model Run ID: {__latest_run_id__}")
    st.write(f"Last Updated: {update_date_time}")
    # rslt = st.empty()
    # if rslt.button("Load Latest Model"):
    #     rslt.empty()
    #     with st.spinner("Updating Latest Model..."):
    #         infr = Inference()
    #         infr._pull_model()
    #         infr._load_model()
    #     rslt.success('Updated Model')

    col1, col2 = st.columns(2)
    metrics = (ASSESTS_DIR/'metrics.md').read_text()
    st.subheader(f"Performance Metrics")
    st.markdown(metrics)

    st.markdown("<br>", True)
    st.subheader(f"Confusion Metrics of Model")

    image = (ASSESTS_DIR/'Confusion-Matrix.png').read_bytes()
    st.image(image, use_column_width ='auto')

