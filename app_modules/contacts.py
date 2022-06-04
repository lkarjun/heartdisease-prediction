import streamlit as st
import requests

def main():
    data = requests.get("https://raw.githubusercontent.com/lkarjun/lkarjun/main/README.md").text
    st.markdown(data, True)
