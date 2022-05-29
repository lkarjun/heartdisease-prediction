import streamlit as st
from Models.model.inference import Inference, __version__, MODEL_DIR

ASSESTS_DIR = MODEL_DIR/'Assests'

def main():

    st.markdown(f"<h2 style='text-align: left;'>Model Version: '{__version__}'</h2>", True)

    rslt = st.empty()
    if rslt.button("Load Latest Model"):
        infr = Inference()
        infr._pull_model()
        infr._load_model()
        rslt.success('Updated Model')

    col1, col2 = st.columns(2)
    metrics = (ASSESTS_DIR/'metrics.md').read_text()
    st.subheader(f"Performance Metrics")
    st.write(metrics)

    st.markdown("<br>", True)
    st.subheader(f"Confusion Metrics of Model")

    image = (ASSESTS_DIR/'confusion_matrix.png').read_bytes()
    st.image(image, width=500)

