import streamlit as st
import pandas as pd
import numpy as np

# st.set_page_config(page_title="Dataset", page_icon="📂")
# st.subheader("Dataset BPJS Kesehatan")

# """ **Data Hasil Scrapping**"""

# data_5000 = pd.read_csv('C:/Users/Lenovo/hello_ds/DataHasilLabelingFix.csv', encoding='latin-1')
# st.dataframe(data_5000)
st.set_page_config(page_title="Dataset", page_icon="📂")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Data", "1050")
col2.metric("Data Training", "840",delta_color="off")
col3.metric("Data Testing", "210",delta_color="off")

tab1, tab2 = st.tabs(["📂Dataset Hasil Scraping dari Twitter & Instagram", "📂Dataset Hasil Pre-Processing"])

tab1.subheader("Dataset Hasil Scrapping dari Twitter dan Instagram")
# tab1.write(
#     """Pengambilan data dari Twitter dilakukan dengan cara _scraping_.
#     Kemudian dilakukan pelabelan manual dan pelabelan dengan VADER. 
#     Berikut ini merupakan dataset yang telah dilabeli:"""
# )
df = pd.read_csv('C:/Users/Lenovo/hello_ds/DataHasilLabelingFix.csv', encoding='latin-1')
tab1.dataframe(df)

tab2.subheader("Dataset Hasil Pre-Processing")
# tab2.write("""Dilakukan _pre-processing_ pada data yang sudah dilabeli.
#     Berikut ini merupakan dataset yang telah melalui _pre-processing_:"""
# )
df2 = pd.read_csv('C:/Users/Lenovo/hello_ds/DataHasilPreprocessing.csv', encoding='latin-1')
tab2.dataframe(df2) 