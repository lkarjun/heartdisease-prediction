import requests
import streamlit as st
import requests
from pathlib import Path

readme_file = Path(__file__).resolve().parent.parent/'readme.md'

def main():
    st.markdown(readme_file.read_text('utf-8'), True)


