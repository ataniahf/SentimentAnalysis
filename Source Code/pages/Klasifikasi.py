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
            "Stemming": [stemming],
            "Remove Stopwords": [stopwords],
        }
        )
    return stopwords

# judul halaman
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

    # vectorizer = TfidfVectorizer()
    # test_tfidf = vectorizer.fit_transform([preprotext2])
    # hasil = vectorizer.get_feature_names_out()
    # hasil2 = test_tfidf.shape

    # prior2 = model_NBC2.predict_log_proba(loaded_voc2.fit_transform([preprotext2]))[0]
    # prior_new2 = prior2

    # prior3 = model_NBC2.predict_joint_log_proba(loaded_voc2.fit_transform([preprotext2]))[0]
    # prior_new3 = prior3

    # # Use ComplementNB to predict the class
    # predict_ComplementNB1 = model_NBC2.predict(loaded_voc2.fit_transform([preprotext2]))

    # # Get prior probabilities for each class
    # prior_probs1 = model_NBC2.class_log_prior_

    # # Get conditional probabilities for each class and each feature
    # conditional_probs1 = model_NBC2.feature_log_prob_

    # # Transpose the conditional_probs1 array to make it compatible for the dot product
    # conditional_probs1 = conditional_probs1.T

    # # Calculate the log likelihood
    # log_likelihood1 = np.dot(loaded_voc2.transform([preprotext2]), conditional_probs1)

    # # Calculate the log posterior probabilities (unnormalized)
    # log_posterior_probs1 = prior_probs1 + log_likelihood1.flatten()

    # # Normalize the log posterior probabilities to get the probability distribution
    # normalized_log_posterior_probs1 = log_posterior_probs1 - np.logaddexp.reduce(log_posterior_probs1)

    # st.write("Prior Probabilities:", prior_probs1)
    # st.write("Conditional Probabilities:", conditional_probs1)
    # st.write("Log Likelihood:", log_likelihood1)
    # st.write("Log Posterior Probabilities (Unnormalized):", log_posterior_probs1)
    # st.write("Normalized Log Posterior Probabilities:", normalized_log_posterior_probs1)

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
    
    # st.write("Nilai prior :", prior_new2)
    # st.write("Nilai log prior + log likelihood :", prior_new3)
    
    # st.write("Nilai TFIDF :", hasil2)
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

    











    
    