import streamlit as st
from Models.model.inference import Inference
import pandas as pd
import os


st.set_page_config(page_title="HeartDisease", page_icon="💕", 
                    menu_items={'About': "🫀 **Heart Disease** Prediction App"})

### Env
_ = os.environ["AZURE_STORAGE_ACCOUNT"] == st.secrets["AZURE_STORAGE_ACCOUNT"]
_ = os.environ["AZURE_STORAGE_KEY"] == st.secrets["AZURE_STORAGE_KEY"]

title = st.title("🫀HeartDisease Prediction", 'title')

with st.spinner('Loading Model...'):
    infr = Inference()


st.markdown(f"<h6 style='text-align: left;'>v {infr.version}</h6>", True)

rslt = st.empty()

if rslt.button("Load Latest Model"):
    infr._pull_model()
    infr._load_model()
    rslt.empty()

if not infr.failed:

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Enter You're Name 🦸‍♂️", placeholder="Arjun")

    with col2:
        age = st.selectbox("Choose Age Group 👩‍🦲", ('18-24','25-29','30-34','35-39', '40-44','45-49',
                                                '50-54','55-59','60-64','65-69','70-74', '75-79',
                                                '80 or older'))


    col1, col2, col3 = st.columns(3)

    with col1:

        smoke = st.selectbox("Do You Smoke 🚭",('Yes', 'No'))
        stroke = st.selectbox("Do You Have Stroke💔",('Yes', 'No'))
        diffwalking = st.selectbox("Do You Have DiffWalking 🚶", ('Yes', 'No'))
        sleeptime = st.number_input("Your avg sleep time 😴", min_value=3)

    with col2:
    
        drink = st.selectbox("Do You Drink 🍾",('Yes', 'No'))
        asthma = st.selectbox("Do You Have Asthma🫁",('Yes', 'No'))
        diabetics = st.selectbox("Are You Diabetic 🍬", ('Yes', 'No'))
        mental_health = st.number_input("You're Mental Health❤️‍🩹", min_value=0, max_value=30)

    with col3:

        skin_cancer = st.selectbox("Do You Have Skin Cancer🤚",('Yes', 'No'))
        kidney_disease = st.selectbox("Do You Have Kindney Diseases🤚",('Yes', 'No'))
        physical_activity = st.selectbox("Are You PhysicalActivity 🏃‍♀️",('Yes', 'No'))
        physical_health = st.number_input("You're Physical Health ❤️‍🩹🏃‍♀️", max_value=30)


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        health_status = st.selectbox("You're Health Status 🧑‍⚕️❤️‍🩹", 
                                ('Excellent','Very good',
                                 'Good', 'Fair', 'Poor'))

    with col2:
        height = st.number_input("Your Height(cm)🧍")

    with col3:
        weight = st.number_input("Your Weight(kg)🚶‍♂️")

    with col4:
        bmi_ = 0 if height == 0 or weight == 0 else round(weight / (height/100)**2, 2)
        bmi = st.text_input("Body Mass Index 🧑‍⚕️", value = bmi_, disabled=True)

    if st.button("Get Result"):
        rslt.warning("Getting Predition")
        data = {"BMI": [bmi_], "Smoking": [smoke], "AlcoholDrinking": [drink], 
            "Stroke": [stroke], "PhysicalHealth": [physical_health],
            "MentalHealth": [mental_health], "DiffWalking": [diffwalking],
            "AgeCategory": [age], "Diabetic": [diabetics], 
            "PhysicalActivity": [physical_activity], "GenHealth": [health_status],
            "SleepTime": [sleeptime], "Asthma": [asthma], "KidneyDisease": [kidney_disease],
            "SkinCancer": [skin_cancer]
            }

        pred = infr.predict(pd.DataFrame(data))
        predictions = pred['predictions']
        if pred['label'] == 'Yes':
            rslt.subheader(f"{name}! There is {predictions['Yes'] * 100}% Chances you have Heart Disease😔 Please ping with doctor🧑‍⚕️")
        else:
            rslt.subheader(f"Hurray💕, {name} There is {predictions['No'] * 100}% Chances you don't have Heart Disease😊 Please Stay Healthy🏃‍♀️")
            st.balloons()

else:
    st.warning("Failed to load model. Please try again🛑")