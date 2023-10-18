import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter


#library prepro
import nltk
import string
import re
import requests
import gensim
from nltk.tokenize import word_tokenize
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import ast  # Tambahkan ini untuk mengolah kamus dalam format teks


# st.subheader("Perbandingan Metode Naïve Bayes Classifier Dan K-Nearest Neighbor Untuk Analisis Sentimen Komentar Masyarakat Terhadap Bpjs Kesehatan")

# load Naive Bayes
model_NBC1 = pickle.load(open('NB1_full.sav','rb'))
model_NBC2 = pickle.load(open('NB1_no.sav','rb'))

#Load KNN
model_KNN1 = pickle.load(open('KNN3_full.sav','rb'))
model_KNN2 = pickle.load(open('KNN3_no.sav','rb'))

tfidf = TfidfVectorizer

loaded_voc1 = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("tfidfBPJS1.sav", "rb"))))
loaded_voc2 = TfidfVectorizer(decode_error="replace", vocabulary=set(pickle.load(open("tfidfBPJS2.sav", "rb"))))

original_data = pd.read_csv('C:/Users/Lenovo/hello_ds/data_train_prepro1.csv') 
#PREPROCESSING

# lowercase
def case_folding(text):
    text = text.lower()
    return text

#remove emoji
def emoji(text):
    text = re.sub(r'[^\x00-\x7f]', r'', text) # Remove non ASCII chars
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    return text

# cleaning text
def cleaning_text(text):
    text = re.sub(r'@[\w]*', ' ', text) # Remove mention handle user (@)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r'\\u\w\w\w\w', '', text) # Remove link web
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#([^\s]+)', '', text) # Remove #tagger
    text = re.sub(r"[.,:;+!\-_<^/=?\"'\(\)\d\*]", " ", text) # Remove simbol, angka dan karakter aneh
    return text

def replaceThreeOrMore(text):
    # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gol).
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    # return pattern.sub(r"\1", text) #2 atau lebih
    return pattern.sub(r"\1\1", text) #3 atau lebih

def tokenize(text):
    return word_tokenize(text)

def wordnormalization(text):
    kamus_slangword = eval(open("kamus_normalisasi_baru.txt").read()) # Membuka dictionary slangword
    pattern = re.compile(r'\b( ' + '|'.join (kamus_slangword.keys())+r')\b') # Search pola kata (contoh kpn -> kapan)
    content = [] #variabel conten untuk menyimpan kata baru
    for kata in text:
        filteredSlang = pattern.sub(lambda x: kamus_slangword[x.group()],kata) # Replace (mengganti) slangword berdasarkan pola review yg telah ditentukan (kata)
        content.append(filteredSlang.lower())#menambahkan nilai array slangword baru pada urutan akhir.
    text = content
    return text
# Nama file kamus
# file_path = 'kamus_normalisasi_baru.txt'

# # Membaca kamus dari file
# kamus = baca_kamus(file_path)

# fungsi yang digunakan untuk membaca kamus dari file teks
def baca_kamus(file_path): #parameter yang berisi jalur ke file teks yang berisi kamus.
   kamus = {} #inisialisasi kamus sebagai kamus kosong
   with open(file_path, 'r', encoding='utf-8') as file: #membuka file teks dalam mode baca ('read') dengan pengkodean UTF-8. Kode ini akan membuka file di jalur yang diberikan dan mengkaitkannya dengan variabel file, penggunaan with ini memastikan bahwa file akan ditutup dengan benar setelah digunakan.
       kamus = ast.literal_eval(file.read())  # Gunakan ast.literal_eval untuk membaca kamus  mengubah teks tersebut menjadi objek Python dan untuk menghindari risiko eksekusi kode berbahaya dari file teks, sehingga ini aman digunakan untuk membaca kamus dari file.
   return kamus

# fungsi yang digunakan untuk menulis kamus kembali ke file teks.
def tulis_kamus(file_path, kamus): # parameter yang berisi jalur ke file teks yang akan ditulis.
   with open(file_path, 'w', encoding='utf-8') as file: #cara membuka file teks dalam mode tulis ('w') dengan pengkodean UTF-8. Kode ini akan membuka file di jalur yang diberikan dan mengkaitkannya dengan variabel file
       file.write(str(kamus))  # Tulis kamus ke file sebagai string

#kamus stopwords
factory = StopWordRemoverFactory()
more_stopword = ['di',"ke",'ber','mah','nya','pas','in','an','se','yang','dan']
sastrawi_stopword = factory.get_stop_words()+more_stopword
# sastrawi_stopword.remove('tidak')
# create path url for each stopword
path_stopwords = []

# combine stopwords
stopwords_l = sastrawi_stopword
for path in path_stopwords:
    response = requests.get(path)
    stopwords_l += response.text.split('\n')

# create dictionary with unique stopword
st_words = set(stopwords_l)

# result stopwords
stop_words = st_words

def remove_stopword(text, stop_words=stop_words):
    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return ' '.join(filtered_sentence)

def stemming_and_lemmatization(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

def prepro1(text):
    text = str(text)
    casefolding = text.lower()

    cleaned_text = emoji(casefolding)

    cleaned_text = cleaning_text(cleaned_text)

    replace = replaceThreeOrMore(cleaned_text)

    token = tokenize(replace)

    slang = wordnormalization(token)
    # st.write("Hasil Word Normalization:",slang)

    joined = " ".join(slang)
    # stopwords = remove_stopword(joined)
    # st.write("Hasil Remove Stopwords:",stopwords)

    stemming = stemming_and_lemmatization(joined)
   
    stopwords = remove_stopword(stemming)
    
    st.dataframe(
        {
            # "Preprocessing": ["Casefolding", "Remove Emoji", "Cleaning", "Remove Emoji", "Remove Character", "Tokenization", "Stemming", "Remove Stopwords"],
            # "Hasil" :[casefolding], 
            # "Hasil" :[cleaned_text], 
            "Casefolding": [casefolding],
            "Remove Emoji": [cleaned_text],
            "Cleaning": [cleaned_text],
            "Remove Character": [replace],
            "Tokenization": [token],
            "Word Normalization": [slang],
            "Stemming": [stemming],
            "Remove Stopwords": [stopwords],
        }
        )
    return stopwords

def prepro2(text):
    text = str(text)
    casefolding = text.lower()

    cleaned_text = emoji(casefolding)

    cleaned_text = cleaning_text(cleaned_text)

    replace = replaceThreeOrMore(cleaned_text)

    token = tokenize(replace)

    joined = " ".join(token)

    stemming = stemming_and_lemmatization(joined)

    stopwords = remove_stopword(stemming)

    st.dataframe(
        {
            "Casefolding": [casefolding],
            "Remove Emoji": [cleaned_text],
            "Cleaning": [cleaned_text],
            "Remove Character": [replace],
            "Tokenization": [token],
            "Stemming": [stemming],
            "Remove Stopwords": [stopwords],
        }
        )
    return stopwords

# Nama file kamus
file_path = 'kamus_normalisasi_baru.txt'

# Membaca kamus dari file
kamus = baca_kamus(file_path) #Membaca kamus dari file menggunakan baca_kamus(file_path) dan simpan dalam variabel kamus

# Input kata slang baru dan kata normalisasi baru
kata_slang_baru = st.text_input("Masukkan Kata Tidak Normal (Slang) / Tidak Baku:")
kata_normalisasi_baru = st.text_input("Masukkan Kata Hasil Normalisasi Baru:")

# Tombol untuk menambahkan kamus
if st.button("Tambahkan Kamus"):
   kamus[kata_slang_baru] = kata_normalisasi_baru #Tambahkan kata slang baru dan kata normalisasi baru ke kamus
   #  Tulis kamus kembali ke file menggunakan tulis_kamus(file_path, kamus)
   tulis_kamus(file_path, kamus)
   #tampilkan pesan sukses 
   st.success(f"Kata slang '{kata_slang_baru}' dan kata normalisasi '{kata_normalisasi_baru}' telah ditambahkan ke kamus.")



st.title ('Klasifikasi Naive Bayes Classifier & K-Nearest Neighbor')

clean_teks = st.text_input('Masukkan Teks')


if st.button('Hasil Deteksi'):

#SKENARIO 1 DENGAN WORD NORMALIZATION

    st.subheader("Hasil Klasifikasi Naive Bayes Classifier:")
    """ **Hasil Preprocessing Skenario 1:**"""
    preprotext1 = prepro1(clean_teks)

    predict_NBC1 = model_NBC1.predict(loaded_voc1.fit_transform([preprotext1]))
    probabilitas1 = model_NBC1.predict_proba(loaded_voc1.fit_transform([preprotext1]))[0]
    probabilitas_new1 = probabilitas1
    
    probabilitas_hasil1 =  "[negatif : {:.6f}] | [positif : {:.6f}] | [netral : {:.6f}]".format(
        probabilitas1[0], probabilitas1[-1], probabilitas1[1])
    probabilitas_new1 = probabilitas_hasil1

     # Output kelas Skenario I
    if(predict_NBC1 == 0):
        st.warning("Netral :|")
    elif (predict_NBC1 == 1):
        st.success("Positif :)")
    elif (predict_NBC1 == -1):
        st.error("Negatif :(")
    
    st.write("Probabilitas :", probabilitas_new1)
    
    """ **Hasil Preprocessing Skenario 2:**"""
    preprotext2 = prepro2(clean_teks)

    predict_NBC2 = model_NBC2.predict(loaded_voc2.fit_transform([preprotext2]))

    probabilitas2 = model_NBC2.predict_proba(loaded_voc2.fit_transform([preprotext2]))[0]
    probabilitas_new2 = probabilitas2
    
    probabilitas_hasil2 =  "[negatif : {:.6f}] | [positif : {:.6f}] | [netral : {:.6f}]".format(
        probabilitas2[0], probabilitas2[-1], probabilitas2[1])
    probabilitas_new2 = probabilitas_hasil2

    if(predict_NBC2 == 0):
        st.warning("Netral :|")
    elif (predict_NBC2 == 1):
        st.success("Positif :)")
    elif (predict_NBC2 == -1):
        st.error("Negatif :(")
    st.write("Probabilitas :", probabilitas_new2)
 

#SKENARIO K-Nearest Neighbor

    st.subheader("Hasil Klasifikasi K-Nearest Neighbor:")
    """ **Hasil Preprocessing Skenario 1:**"""
    preprotext1 = prepro1(clean_teks)
    predict_KNN1 = model_KNN1.predict(loaded_voc1.fit_transform([preprotext1]))

    if(predict_KNN1 == 0):
        st.warning("Netral :|")
    elif (predict_KNN1 == 1):
        st.success("Positif :)")
    else :
        st.error("Negatif :(")

    jarak1 = model_KNN1.kneighbors(loaded_voc1.fit_transform([preprotext1]))

    # st.write("Jarak :", jarak2)
    jarak_1a, jarak_1b = jarak1
    # st.write("Jarak Eucledean Distace (ascending):", jarak2)
    
    # Get the corresponding labels from the original data using the indices obtained
    closest_indices1 = jarak_1b[0]
    closest_labels1 = original_data.iloc[closest_indices1]['Sentiments']  # Replace 'label_column_name' with the actual column name of the labels

    # st.write("Jarak Eucledean Distace (ascending):", jarak_2a, closest_labels2)
    
    # st.write("Corresponding Labels:", closest_labels2)
    st.write("Jarak Eucledean Distace (ascending):")
    st.write(pd.DataFrame(data=jarak_1a.reshape(-1, 1), columns=['Jarak Eucledean']))

    closest_labels_table1 = pd.DataFrame(data={'Label Kelas': closest_labels1})
    st.write(closest_labels_table1)
#     
    """ **Hasil Preprocessing Skenario 2:**"""
    # Output kelas Skenario I
    preprotext2 = prepro2(clean_teks)

    predict_KNN2 = model_KNN2.predict(loaded_voc2.fit_transform([preprotext2]))

    if(predict_KNN2 == 0):
        st.warning("Netral :|")
    elif (predict_KNN2 == 1):
        st.success("Positif :)")
    else :
        st.error("Negatif :(")
    
    jarak2 = model_KNN2.kneighbors(loaded_voc2.fit_transform([preprotext2]))
    # st.write("Jarak :", jarak2)
    jarak_2a, jarak_2b = jarak2
    # st.write("Jarak Eucledean Distace (ascending):", jarak2)
    
    # Get the corresponding labels from the original data using the indices obtained
    closest_indices2 = jarak_2b[0]
    closest_labels2 = original_data.iloc[closest_indices2]['Sentiments']  # Replace 'label_column_name' with the actual column name of the labels

    # st.write("Jarak Eucledean Distace (ascending):", jarak_2a, closest_labels2)
    
    # st.write("Corresponding Labels:", closest_labels2)
    st.write("Jarak Eucledean Distace (ascending):")
    st.write(pd.DataFrame(data=jarak_2a.reshape(-1, 1), columns=['Jarak Eucledean']))

    closest_labels_table2 = pd.DataFrame(data={'Label Kelas': closest_labels2})
    st.write(closest_labels_table2)

    











    
    