import streamlit as st
import eda, predict

st.markdown(
    """
    <style>
    .stApp {
        background-color: #D9E9CF; /* Light gray background */
        color: blackç;
    }
    [data-testid="stSidebar"] {
        background-color: #B6CEB4;
        color: black;
    }
    [data-testid="stHeader"] {
        background-color: #96A78D;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("Page Navigation")
    page = st.radio("Page: ",("EDA", "Model Inference"))

if page == "EDA":
    eda.eda()
else:
    predict.model()
