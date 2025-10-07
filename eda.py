import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def eda():
    # Element Title
    st.title("FIFA Data Exploration")

    # Header
    st.header("Latar Belakang")

    # Image
    st.image("fc26-4229870419.jpg.webp", caption="Source: Google Image")

    # Markdown
    st.markdown('''
                Menurut laporan [FIFA 2022](https://publications.fifa.com/en/annual-report-2021/around-fifa/professional-football-2021/), 
                jumlah pemain sepakbola pada tahun 2021 kurang lebih sebanyak 130.000 pemain. Namun, dalam dataset yang digunakan pada kali ini, 
                hanya mencakup 20.000 pemain saja.

                Project kali ini bertujuan untuk memprediksi rating pemain FIFA 2022 sehingga semua pemain sepak bola profesional dapat diketahui 
                ratingnya dan tidak menutup kemungkinan untuk lahirnya talenta/wonderkid baru.

                Project ini akan dibuat menggunakan algoritma Linear Regresison dan akan dievaluasi dengan menggunakan metrics MAE (Mean Absolute Error).
                ''')

    st.header("Dataset")
    st.markdown("Ini adalah dataset yang kita gunakan sebagai bahan pembelajaran model machine learning.")

    # Load data dengan pandas
    data = pd.read_csv('https://raw.githubusercontent.com/FTDS-learning-materials/phase-1/refs/heads/v2.3/w1/P1W1D1PM%20-%20Machine%20Learning%20Problem%20Framing.csv')
    data.rename(columns={'ValueEUR': 'Price', 'Overall': 'Rating'}, inplace=True)

    st.dataframe(data)

    chart = plt.figure(figsize=(16, 5))
    sns.histplot(data['Rating'], kde=True, bins=30)
    plt.title('Histogram of Rating')

    # Header
    st.header("Exploratory Data Analysis")

    # Sub Header
    st.subheader("Player Rating Distribution")

    # Menampilkan matplotlib chart
    st.pyplot(chart)

    # Weight VS Height
    st.subheader("Weight vs Height Distribution")

    # Markdown
    st.markdown('''
        Terlihat dari Histogram Plot diatas bahwa `Rating` memiliki distribusi normal dengan mayoritas data berada pada rentang `60` hingga `70`.

    `Height` dan `Weight` mempunyai relasi yang searah. Artinya, semakin besar nilai `Height` maka nilai `Weight` juga akan semakin besar. Dapat disimpulkan bahwa mayoritas pemain sepak bola pada dataset ini memiliki kondisi tubuh yang proporsional.
    ''')

    # plotly chart
    fig = px.scatter(data, x="Weight", y="Height", hover_name="Name")
    st.plotly_chart(fig)

    # Header

    st.subheader("Player Stats")

    # User iNPUT
    nama_kolom = data.columns
    total_cols = [col for col in nama_kolom if "Total" in col]

    # User Input
    input = st.selectbox(
        "Pilih Kolom untuk Divisualisasikan",
        options = total_cols
    )

    # Pakai .write()
    st.write("You selected:", input)
    chart_2 = plt.figure(figsize=(16, 5))
    sns.histplot(data[input], kde=True, bins=30)
    plt.title(f'History Of {input}')
    st.pyplot(chart_2)

    # Pakai looping
    # for i in total_cols:
    #     if i == input:
    #         chart_2 = plt.figure(figsize=(16, 5))
    #         sns.histplot(data[i], kde=True, bins=30)
    #         plt.title(f'{i}')
    #         st.pyplot(chart_2)

if __name__ == '__main__':
    eda()