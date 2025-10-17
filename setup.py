import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from annoy import AnnoyIndex
import joblib

df = pd.read_csv("spotify_data.csv")
df = df.drop_duplicates(subset=['track_name'])
df['tempo_norm'] = (df['tempo'] - df['tempo'].min()) / (df['tempo'].max() - df['tempo'].min())
feature_cols = ['acousticness', 'danceability', 'energy', 'mode', 'tempo_norm', 'valence']
features = df[feature_cols].values
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
joblib.dump(scaler, "scaler.pkl")
features_shape = features_scaled.shape[1]
annoy_index = AnnoyIndex(features_shape, metric='angular')
for i, vector in enumerate(features_scaled):
    annoy_index.add_item(i, vector)
annoy_index.build(1000)
annoy_index.save('songs.ann')