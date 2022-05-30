import streamlit as st
from Models.model.inference import Inference
from Models.model.preprocess import COLUMNS
import pandas as pd
from time import sleep

def main_title():
    title = st.title("🫀HeartDisease Prediction", 'title')

    with st.spinner('Loading Model...'):
        infr = Inference()

    st.markdown(f"<h6 style='text-align: left;'>&emsp;v {infr.version}</h6>", True)

    rslt = st.empty()
    if rslt.button("Load Latest Model"):
        infr._pull_model()
        infr._load_model()
        rslt.success('Updated Model')
    st.write('---')
    return infr

def form_section(form):
    data = {}
    with form.form("section1"):
            col1, col2 = st.columns(2)

            with col1:
                name_ = st.text_input("Enter You're Name 🦸‍♂️", value="User Name")
                height = st.number_input("Your Height(cm)🧍")
            with col2:
                age = st.selectbox("Choose Age Group 👩‍🦲", ('18-24','25-29','30-34','35-39', '40-44','45-49',
                                                '50-54','55-59','60-64','65-69','70-74', '75-79',
                                                '80 or older'))
                weight = st.number_input("Your Weight(kg)🚶‍♂️")
                bmi_ = 0 if height == 0 or weight == 0 else round(weight / (height/100)**2, 2)
            
            col1_, col2_ = st.columns(2)

            with col1_:
                drink = st.selectbox("Do You Drink 🍾",('Yes', 'No'))
                asthma = st.selectbox("Do You Have Asthma🫁",('Yes', 'No'))
                diabetics = st.selectbox("Are You Diabetic 🍬", ('Yes', 'No'))
                sleeptime = st.number_input("Your avg sleep time 😴", min_value=3)
                mental_health = st.number_input("You're Mental Health❤️‍🩹", min_value=0, max_value=30)
                skin_cancer = st.selectbox("Do You Have Skin Cancer🤚",('Yes', 'No'))
            with col2_:
                smoke = st.selectbox("Do You Smoke 🚭",('Yes', 'No'))
                stroke = st.selectbox("Do You Have Stroke💔",('Yes', 'No'))
                diffwalking = st.selectbox("Do You Have DiffWalking 🚶", ('Yes', 'No'))
                kidney_disease = st.selectbox("Do You Have Kindney Diseases🤚",('Yes', 'No'))
                physical_health = st.number_input("You're Physical Health ❤️‍🩹🏃‍♀️", max_value=30)
                physical_activity = st.selectbox("Are You PhysicalActivity 🏃‍♀️",('Yes', 'No'))

            health_status = st.selectbox("You're Health Status 🧑‍⚕️❤️‍🩹", 
                                    ('Excellent','Very good',
                                    'Good', 'Fair', 'Poor'))
            warning = st.empty()
            if st.form_submit_button("Get Result"):
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
                else: warning.warning("Please fill all details...")

    return False, False     


def collect_data():
    data = {}
    status = False
    form_head = st.empty()
    form_head.markdown(f"<h4 style='text-align: center;'>Enter Details to Check HeartDisease</h4>", True)
    form = st.empty()
    status, data = form_section(form)
    if status:
        form_head.empty()
        form.empty()
    return status, data

def write_prediction(rslt, data, infr):
    rslt.warning("Getting Predition")
    sleep(1)
    username = data['Name']
    data.pop('Name')
    df = pd.DataFrame(data)
    pred = infr.predict(df[COLUMNS])
    predictions = pred['predictions']
    if pred['label'] == 'Yes':
        st.snow()
        rslt.subheader(f"{username}! There is an {round(predictions['Yes'] * 100)}% Chance you have Heart Disease😔 Please ping with doctor🧑‍⚕️")
    else:
        rslt.subheader(f"Hurray💕, Your You Don't Have HeartDiseases😊Stay Healthy🏃‍♀️")
        st.balloons()

def home():
    infr = main_title()
    prediciton_rslt = st.empty()
    status, data = collect_data()
    if status:write_prediction(prediciton_rslt, data, infr)
        
    
