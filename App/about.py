import requests
import streamlit as st
import requests

def main():
    link = "https://raw.githubusercontent.com/lkarjun/heartdisease-prediction/master/readme.md?token=GHSAT0AAAAAABUAR3NCOCNVL2E3YG4OMPI6YUTRZUA"
    data = requests.get(link).text
    st.markdown(data, True)


