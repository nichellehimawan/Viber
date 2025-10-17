from transformers import pipeline
import torch
import streamlit as st
import re
from collections import defaultdict
import statistics
import random
import pandas as pd
import numpy as np
from annoy import AnnoyIndex
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(
    page_title="Viber: Music Recommender",
    page_icon="🎵", 
    layout="centered"
)

st.markdown(
    """
    <meta name="google-site-verification" content="cPmg8dYJmadfZd-RCHneygSYxIJ2zeg4-IxmLsF-sYI" />
    """,
    unsafe_allow_html=True
)

df = pd.read_csv("spotify_data.csv")
df = df.drop_duplicates(subset=['track_name'])
df['tempo_norm'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())
feature_cols = ['acousticness', 'danceability', 'energy', 'mode', 'tempo_norm', 'valence']
features = df[feature_cols].values
scaler = joblib.load('scaler.pkl')
features_scaled = scaler.transform(features)
features_shape = features_scaled.shape[1]
annoy_index = AnnoyIndex(features_shape, metric='angular')
annoy_index.load('songs.ann')

jh = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
bs = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)
lexicon = defaultdict(lambda: defaultdict(int))
with open("nrc_lexicon.txt", 'r', encoding='utf-8') as f:
    for line in f:
        word, emotion, value = line.strip().split('\t')
        lexicon[word][emotion] = int(value)
emotions = {"anger": [0.3, 0.7, 0.8, 0, 70, 0.3], 
            "anticipation": [0.4, 0.7, 0.7, 1, 50, 0.6], 
            "disgust": [0.6, 0.4, 0.4, 0, 20, 0.3], 
            "fear": [0.5, 0.6, 0.8, 0, 50, 0.2], 
            "joy": [0.5, 0.9, 0.8, 1, 70, 0.9], 
            "negative": [0.5, 0.3, 0.5, 0, 0.5, 0.2], 
            "positive": [0.5, 0.7, 0.6, 1, 0.5, 0.8], 
            "sadness": [0.8, 0.1, 0.2, 0, -30, 0.2], 
            "surprise": [0.7, 0.8, 0.7, 1, 40, 0.7], 
            "trust": [0.8, 0.4, 0.6, 1, 10, 0.8]}

st.title("Viber: Text to Music Recommender for Writers and Readers")
text = st.text_area("Enter text (can be a word, phrase, sentence, or even paragraphs): ")

def extract_words(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words_f = [word for word in words if word in lexicon]
    return words_f

if text:
    jhr = jh(text)[0]
    bsr = bs(text)[0]

    jh_acoustic = (jhr[0]['score']*0.3 + jhr[1]['score']*0.6 + jhr[2]['score']*0.5 + jhr[3]['score']*0.5 + jhr[4]['score']*0.5 + jhr[5]['score']*0.8 + jhr[6]['score']*0.7)/3.9
    jh_dance = (jhr[0]['score']*0.7 + jhr[1]['score']*0.4 + jhr[2]['score']*0.6 + jhr[3]['score']*0.9 + jhr[4]['score']*0.5 + jhr[5]['score']*0.1 + jhr[6]['score']*0.8)/4.0
    jh_energy = (jhr[0]['score']*0.8 + jhr[1]['score']*0.4 + jhr[2]['score']*0.8 + jhr[3]['score']*0.8 + jhr[4]['score']*0.5 + jhr[5]['score']*0.2 + jhr[6]['score']*0.7)/4.2
    jh_mode = (jhr[3]['score']*1 + jhr[4]['score']*1 + jhr[6]['score']*1)/3.0
    if jh_mode <= 0.5:
        jh_mode = 0
    else:
        jh_mode = 1
    jh_tempo_c = (jhr[0]['score']*70 + jhr[1]['score']*20 + jhr[2]['score']*50 + jhr[3]['score']*70 - jhr[5]['score']*30 + jhr[6]['score']*40)/220
    jh_val = (jhr[0]['score']*0.3 + jhr[1]['score']*0.3 + jhr[2]['score']*0.2 + jhr[3]['score']*0.9 + jhr[4]['score']*0.5 + jhr[5]['score']*0.2 + jhr[6]['score']*0.7)/3.1

    bs_acoustic = (bsr[0]['score']*0.8 + bsr[1]['score']*0.5 + bsr[2]['score']*0.9 + bsr[3]['score']*0.3 + bsr[4]['score']*0.5 + bsr[5]['score']*0.7)/3.7
    bs_dance = (bsr[0]['score']*0.1 + bsr[1]['score']*0.9 + bsr[2]['score']*0.7 + bsr[3]['score']*0.7 + bsr[4]['score']*0.6 + bsr[5]['score']*0.8)/3.8
    bs_energy = (bsr[0]['score']*0.2 + bsr[1]['score']*0.8 + bsr[2]['score']*0.7 + bsr[3]['score']*0.8 + bsr[4]['score']*0.8 + bsr[5]['score']*0.7)/4.0
    bs_mode = (bsr[1]['score']*1 + bsr[2]['score']*1 + bsr[5]['score']*1)/3.0
    if bs_mode <= 0.5:
        bs_mode = 0
    else:
        bs_mode = 1
    bs_tempo_c = (bsr[0]['score']*-30 + bsr[1]['score']*70 + bsr[2]['score']*10 + bsr[3]['score']*70 + bsr[4]['score']*50 + bsr[5]['score']*40)/210
    bs_val = (bsr[0]['score']*0.2 + bsr[1]['score']*0.9 + bsr[2]['score']*0.9 + bsr[3]['score']*0.3 + bsr[4]['score']*0.2 + bsr[5]['score']*0.7)/3.2

    words = extract_words(text)
    counts = defaultdict(lambda: defaultdict(int))
    for emotion in emotions:
        counter = 0
        for word in words:
            counter += lexicon[word][emotion]
        counts[emotion] = counter
    n = len(words)
    if n == 0:
        for emotion in counts:
            counts[emotion] = 0.5
    else:
        for emotion in counts:
            counts[emotion] = counts[emotion]/n
    
    lex_features = [0] * 6
    for i in range(6):
        score = 0
        sum = 0
        for emotion in emotions:
            score += counts[emotion]*emotions[emotion][i]
            sum += emotions[emotion][i]
        lex_features[i] = score/sum
    if lex_features[3] <= 0.5:
        lex_features[3] = 0
    else:
        lex_features[3] = 1

    acoustic = 0.4*jh_acoustic + 0.4*bs_acoustic + 0.2*lex_features[0] if n > 6 else 0.2*jh_acoustic + 0.2*bs_acoustic + 0.6*lex_features[0]
    dance = 0.4*jh_dance + 0.4*bs_dance + 0.2*lex_features[1] if n > 6 else 0.2*jh_dance + 0.2*bs_dance + 0.6*lex_features[1]
    energy = 0.4*jh_energy + 0.4*bs_energy + 0.2*lex_features[2] if n > 6 else 0.2*jh_energy + 0.2*bs_energy + 0.6*lex_features[2]
    modes = [jh_mode, bs_mode, lex_features[3]]
    mode = statistics.mode(modes)
    tempo = 0.4*jh_tempo_c + 0.4*bs_tempo_c + 0.2*lex_features[4] + 100 if n > 6 else 0.2*jh_tempo_c + 0.2*bs_tempo_c + 0.6*lex_features[4] + 100
    tempo_norm = (tempo - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())
    valence = 0.4*jh_val + 0.4*bs_val + 0.2*lex_features[5] if n > 6 else 0.2*jh_val + 0.2*bs_val + 0.6*lex_features[5]
    targets = np.array([acoustic, dance, energy, mode, tempo_norm, valence]).reshape(1, -1)
    targets_scaled = scaler.transform(targets)[0]
    neighbor_indices = annoy_index.get_nns_by_vector(targets_scaled, 200, search_k=10000)
    indices = random.sample(neighbor_indices, 10)
    recs = df.iloc[indices]
    results = recs.to_dict(orient='records')
    
    st.header("Results: ")
    labelcols = st.columns([1, 1, 1])
    with labelcols[0]:
        st.subheader("Title")
    with labelcols[1]:
        st.subheader("Artists")
    with labelcols[2]:
        st.subheader("Album")
    st.markdown("---")
    for song in results:
        cols = st.columns([1, 1, 1])
        with cols[0]:
            st.markdown(f"**{song['track_name']}**")
        with cols[1]:
            st.markdown(f"*{song['artists']}*")
        with cols[2]:
            st.markdown(f"{song['album_name']}")
        st.markdown("---")