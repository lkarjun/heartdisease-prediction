import streamlit as st
from model.inference import Inference
from model.preprocess import COLUMNS
import pandas as pd
from time import sleep
from .static_html import render, home_version_display

def main_title():
    st.markdown(f"<h1 style='text-align: Center;'>🫀HeartDisease Prediction</h1>", True)

    with st.spinner('Loading Model...'):
        infr = Inference()
        render(home_version_display, 29, v = infr.version)
    st.markdown("<br>", True)
    return infr

def form_section(form):
    data = {}
    with form.form("section1"):
        st.markdown(f"<h4 style='text-align: left;'>Check HeartDisease📝</h4>", True)
        with st.expander("Tell Me About Your Details", True):
        
            col1, col2 = st.columns(2)

            with col1:
                name_ = st.text_input("Enter You're Name 🦸‍♂️", value="User")
                age = st.selectbox("Choose Age Group 👩‍🦲", ('18-24','25-29','30-34','35-39', '40-44','45-49',
                                                '50-54','55-59','60-64','65-69','70-74', '75-79',
                                                '80 or older'))
            with col2:
                height = st.number_input("Your Height(cm)🧍")
                weight = st.number_input("Your Weight(kg)🚶‍♂️")
                bmi_ = 0 if height == 0 or weight == 0 else round(weight / (height/100)**2, 2)
        
        with st.expander("Tell Me About Your Bad Habits"):
            drink = st.selectbox("Do You Drink 🍾",('No', 'Yes'))
            smoke = st.selectbox("Do You Smoke 🚭",('Yes', 'No'))
        
        with st.expander("Tell Me About Your Medical Conditions"):
            col1, col2 = st.columns(2)
            with col1:
                stroke = st.selectbox("Do You Have Stroke💔",('No', 'Yes'))
                asthma = st.selectbox("Do You Have Asthma🫁",('No', 'Yes'))
                diabetics = st.selectbox("Are You Diabetic 🍬", ('Yes', 'No'))
            with col2:
                skin_cancer = st.selectbox("Do You Have Skin Cancer🤚",('No', 'Yes'))
                diffwalking = st.selectbox("Are You DiffWalker 🚶", ('Yes', 'No'))
                kidney_disease = st.selectbox("Do You Have Kindney Diseases🤚",('No', 'Yes'))

        with st.expander("Tell Me About Your Health", False):   
            col1, col2 = st.columns(2)
            with col1:
                sleeptime = st.number_input("Your Average sleeping time 😴", min_value=3)
                mental_health = st.number_input("Your Mental Health In Last 30 days❤️‍🩹", min_value=0, max_value=30)
                
            with col2:
                physical_health = st.number_input("Your Physical Health In Last 30 days❤️‍🩹🏃‍♀️", max_value=30)
                physical_activity = st.selectbox("Are You Physically Active 🏃‍♀️",('Yes', 'No'))

            health_status = st.selectbox("Your Health Status 🧑‍⚕️", 
                                    ('Excellent','Very good',
                                    'Good', 'Fair', 'Poor'))
        warning = st.empty()
        if st.form_submit_button("Check ✅"):
                if bmi_ != 0:
                    data["Name"] = name_
                    data["BMI"] = [bmi_]
                    data["AgeCategory"] = [age]
                    data['AlcoholDrinking'] = [drink]
                    data['Asthma'] = [asthma]
                    data['Diabetic'] = [diabetics]
                    data['Smoking'] = [smoke]
                    data['Stroke'] = [stroke]
                    data['DiffWalking'] = [diffwalking]
                    data['SleepTime'] = [sleeptime]
                    data['MentalHealth'] = [mental_health]
                    data['PhysicalHealth'] = [physical_health]
                    data['SkinCancer'] = [skin_cancer]
                    data['KidneyDisease'] = [kidney_disease]
                    data['PhysicalActivity'] = [physical_activity]
                    data['GenHealth'] = [health_status]
                    return True, data
                else: warning.warning("**Please expand & fill all details**")

    return False, False     


def collect_data():
    data = {}
    status = False
    form = st.empty()
    status, data = form_section(form)
    if status:
        form.empty()
    return status, data

def write_prediction(rslt, data, infr):
    rslt.warning("Getting Predition")
    sleep(1)
    username = data['Name']
    data.pop('Name')
    df = pd.DataFrame(data)
    pred = infr.predict(df[COLUMNS])
    st.write(pred)
    predictions = pred['predictions']
    if pred['predicted'] == 'Yes':
        st.snow()
        rslt.subheader(f"{username}! There is an {round(predictions['Yes'] * 100)}% Chance you have Heart Disease😔 Please ping with doctor🧑‍⚕️")
    else:
        rslt.subheader(f"Hurray💕, {round(predictions['No'] * 100)}% Sure You Don't Have HeartDiseases😊Stay Healthy🏃‍♀️")
        st.balloons()

def home():
    infr = main_title()
    prediciton_rslt = st.empty()

    status, data = collect_data()
        
    if status:
        write_prediction(prediciton_rslt, data, infr)
        if st.button("Show Form"):
            home()
        
    
