import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import json
import numpy as np


with open('model_files/model_lin_reg.pkl', 'rb') as file_1:
  model_lin_reg = pickle.load(file_1)

with open('model_files/model_scaler.pkl', 'rb') as file_2:
  model_scaler = pickle.load(file_2)

with open('model_files/model_encoder.pkl','rb') as file_3:
  model_encoder = pickle.load(file_3)

with open('model_files/list_num_cols.txt', 'r') as file_4:
  list_num_cols = json.load(file_4)

with open('model_files/list_cat_cols.txt', 'r') as file_5:
  list_cat_cols = json.load(file_5)

def model():
    # Title
    st.title("Predict player Rating")

    # User Input
    with st.form(key = "Player"):
    # Name Input
        name = st.text_input("Nama Pemain:",
                        placeholder= "Luctfy Alkatiri")
        # Usia input
        usia = st.number_input("Usia Pemain:",
                        value= 20, min_value=0,max_value=100)
        tinggi = st.number_input("Tinggi Badan Pemain:",
                        value= 180, min_value=0,max_value=250, help="Tinggi dalam CM")
        berat = st.number_input("Berat Badan Pemain:",
                        value= 80, min_value=0,max_value=120, help="Tinggi dalam KG")
        harga = st.number_input("Usia Pemain:",
                        value= 500000, help="Harga dalam EUR")
        atk = st.selectbox("Attacking Work Rate:",["Low", "Medium", "High"])
        defen = st.selectbox("Defensive Work Rate:",["Low", "Medium", "High"])

        # Total Columns
        pace = st.slider("Pace Total", min_value=0, max_value=100)
        shooting = st.slider("Shooting Total", min_value=0, max_value=100)
        passing = st.slider("Passing Total", min_value=0, max_value=100)
        dribbling = st.slider("Dribbling Total", min_value=0, max_value=100)
        defending = st.slider("Defending Total", min_value=0, max_value=100)
        physicality = st.slider("Physicality Total", min_value=0, max_value=100)

        submit = st.form_submit_button("Predict")

    if submit:

        data_inf = {
            'Name': name,
            'Age': usia,
            'Height': tinggi,
            'Weight': berat,
            'Price': harga,
            'AttackingWorkRate': atk,
            'DefensiveWorkRate': defen,
            'PaceTotal': pace,
            'ShootingTotal': shooting,
            'PassingTotal': passing,
            'DribblingTotal': dribbling,
            'DefendingTotal': defending,
            'PhysicalityTotal':physicality
        }

        data_inf = pd.DataFrame([data_inf])
        st.dataframe(data_inf)

        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]

        # Feature Scaling and Feature Encoding

        ## Feature Scaling
        data_inf_num_scaled = model_scaler.transform(data_inf_num)

        ## Feature Encoding
        data_inf_cat_encoded = model_encoder.transform(data_inf_cat)

        ## Concate
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis=1)

        # Predict using Linear Regression
        y_pred_inf = model_lin_reg.predict(data_inf_final)

        st.write("# Prediction: ", int(y_pred_inf[0]))

if __name__ == '__main__':
    model()
