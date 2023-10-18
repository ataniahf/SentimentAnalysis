import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
# from streamlit_option_menu import option_menu

# # #navigasi sidebar
st.set_page_config(
    page_title="Dashboard",
    page_icon="📜",
)

st.subheader("Perbandingan Metode Naïve Bayes Classifier Dan K-Nearest Neighbor Untuk Analisis Sentimen Komentar Masyarakat Terhadap BPJS Kesehatan")

image = Image.open('bpjs.png')
st.image(image)




# with st.sidebar :
#     selected = option_menu('Perbandingan Metode Naïve Bayes Classifier Dan K-Nearest Neighbor Untuk Analisis Sentimen Komentar Masyarakat Terhadap Bpjs Kesehatan',
#     ['Home','Dataset','Klasifikasi','Pengujian'],
#     default_index=0)
# # if(selected == 'Home'):
# #     st.title('Analisis Sentimen BPJS Kesehatan')
# # if(selected == 'Dataset'):
    
#     st.title('Dataset')
# # option = st.selectbox('NLP Service',('Sentiment Analysis', 'Entity Extraction', 'Text Summarization'))
# df = pd.read_csv('DataHasilPrepro1.csv')
# st.dataframe(df)


    

# if(selected == 'Klasifikasi'):
#     st.title('Klasifikasi')

# if(selected == 'Pengujian'):
#     st.title('Pengujian')