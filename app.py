import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from st_aggrid import AgGrid
from App import home, details, contacts, about
import os

### Env
_ = os.environ["AZURE_STORAGE_ACCOUNT"] == st.secrets["AZURE_STORAGE_ACCOUNT"]
_ = os.environ["AZURE_STORAGE_KEY"] == st.secrets["AZURE_STORAGE_KEY"]

st.set_page_config(page_title="HeartDisease", page_icon="💕", 
                    menu_items={'About': "🫀 **Heart Disease** Prediction App"})

with st.sidebar:
    choose = option_menu("Application", 
                        ["Home", "Details", "About Project", "About Me"],
                         icons = ['house', 'journal-code', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
                                "container": {"padding": "5!important", "background-color": "#fafafa"},
                                "icon": {"color": "orange", "font-size": "25px"}, 
                                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                "nav-link-selected": {"background-color": "#02ab21"}},
    )

match choose:
    case "Home": home.home()
    case "Details": details.main()
    case "About Project": about.main()
    case "About Me": contacts.main()
