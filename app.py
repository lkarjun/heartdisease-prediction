import streamlit as st
import app_modules
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
import os
from app_modules import about, contacts, home, details, track_model

### Env
os.environ["AZURE_STORAGE_ACCOUNT"] = st.secrets["AZURE_STORAGE_ACCOUNT"]
os.environ["AZURE_STORAGE_KEY"] = st.secrets["AZURE_STORAGE_KEY"]

st.set_page_config(page_title="HeartDisease", page_icon="💕", 
                    menu_items={'About': "🫀 **Heart Disease** Prediction App"})

with st.sidebar:
    choose = option_menu("Application", 
                        ["Home", "Details", "Track Development", "About Project", "About Me"],
                         icons = ['house', 'journal-code', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                                "container": {"padding": "5!important", "background-color": "#F5DF99"},
                                "icon": {"color": "#a83f39", "font-size": "30px"}, 
                                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"10px", "--hover-color": "orange"},
                                "nav-link-selected": {"background-color": "#02ab21"}},
    )

match choose:
    case "Home":
        home.home()
    case "Details": 
        details.main()
    case "Track Development": 
        track_model.main()
    case "About Project": 
        about.main()
    case "About Me": 
        contacts.main()
