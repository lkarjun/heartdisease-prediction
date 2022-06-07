import streamlit as st
import app_modules
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from st_aggrid import AgGrid
import os
from app_modules import about, contacts

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
                                "container": {"padding": "5!important", "background-color": "#fafafa"},
                                "icon": {"color": "#a83f39", "font-size": "35px"}, 
                                "nav-link": {"font-size": "14px", "text-align": "left", "margin":"13px", "--hover-color": "orange"},
                                "nav-link-selected": {"background-color": "#02ab21"}},
    )

match choose:
    case "Home": 
        with st.spinner("Loading"):
            from app_modules import home
        home.home()
    case "Details": 
        with st.spinner("Loading"):
            from app_modules import details
        details.main()
    case "Track Development": 
        with st.spinner("Loading"):
            from app_modules import track_model
        track_model.main()
    case "About Project": 
        about.main()
    case "About Me": 
        contacts.main()
