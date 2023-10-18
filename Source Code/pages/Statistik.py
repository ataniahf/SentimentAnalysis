import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Hasil Pengujian", page_icon="📊")

tab1, tab2 = st.tabs(["📊Presentase Hasil Pengujian", "📊Grafik Hasil Pengujian"])

tab1.subheader("Hasil Pengujian Naive Bayes Classifier")
tab1.text("Menampilkan perbandingan hasil pengujian dari skenario 1 dan skenario 2.")
col1, col2 = tab1.columns(2)
col1.metric("Normalization", "87,14%", "Akurasi", delta_color="off")
col2.metric("No Normalization", "86,67%", "Akurasi")

col3, col4 = tab1.columns(2)
col3.metric("", "87,18%", "Presisi", delta_color="off")
col4.metric("", "86,64%", "Presisi")

col5, col6 = tab1.columns(2)
col5.metric("", "87,14%", "Recall", delta_color="off")
col6.metric("", "86,67%", "Recall")

col7, col8 = tab1.columns(2)
col7.metric("", "87,03%", "F1-Score", delta_color="off")
col8.metric("", "86,57%", "F1-Score")

tab1.subheader("Hasil Pengujian K- Nearest Neighbor")
tab1.text("Menampilkan perbandingan hasil pengujian dari skenario 1 & skenario 2.")
col1, col2 = tab1.columns(2)
col1.metric("Normalization", "80,48%", "Akurasi", delta_color="off")
col2.metric("No Normalization", "77,14%", "Akurasi")

col3, col4 = tab1.columns(2)
col3.metric("", "80,93%", "Presisi", delta_color="off")
col4.metric("", "78,12%", "Presisi")

col5, col6 = tab1.columns(2)
col5.metric("", "80,47%", "Recall", delta_color="off")
col6.metric("", "77,14%", "Recall")

col7, col8 = tab1.columns(2)
col7.metric("", "80,43%", "F1-Score", delta_color="off")
col8.metric("", "77,15%", "F1-Score")

tab2.subheader("Grafik Hasil Pengujian")
tab2.text("Menampilkan hasil pengujian dalam bentuk grafik.")

categories = ['Akurasi', 'Presisi', 'Recall', 'F1-Score']
group1 = [87.14, 87.18, 87.14, 87.03]
group2 = [86.67, 86.64, 86.67, 86.57]
group3 = [80.48, 80.93, 80.47, 80.43]
group4 = [77.14, 78.12, 77.14, 77.15]

# Create a figure and axis
fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figsize for wider bars

# Set positions for the bars with some separation
x = np.arange(len(categories))
spacing = 0.2  # Separation between bars
width = (1 - spacing) / 4  # Width of the bars

# Plotting the bars with adjusted positions
bars1 = ax.bar(x - 1.5*width, group1, width, label='Naive Bayes Skenario I')
bars2 = ax.bar(x - 0.5*width, group2, width, label='Naive Bayes Skenario II')
bars3 = ax.bar(x + 0.5*width, group3, width, label='K-Nearest Neighbor Skenario I')
bars4 = ax.bar(x + 1.5*width, group4, width, label='K-Nearest Neighbor Skenario II')

# Adding labels and title
ax.set_ylabel('Presentase')
ax.set_title('Hasil Pengujian Confusion Matrix')
ax.set_xticks(x)
ax.set_xticklabels(categories)

# Adding values as annotations on bars with adjusted font size
def add_annotations(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', fontsize=9)

add_annotations(bars1)
add_annotations(bars2)
add_annotations(bars3)
add_annotations(bars4)

# Moving the legend outside of the chart
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Extend y-axis limits for annotations
ax.set_ylim(0, max(max(group1), max(group2), max(group3), max(group4)) * 1.1) 

# Display the plot using Streamlit
st.pyplot(fig)