import streamlit as st
import eda, predict

st.markdown(
    """
    <style>
    .stApp {
        background-color: #96A78D; /* Light gray background */
        color: black;
    }
    [data-testid="stSidebar"] {
        background-color: #B6CEB4;
        color:
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
