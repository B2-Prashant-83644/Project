from collections import defaultdict
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from scipy.spatial.distance import cdist

app = Flask(__name__)

# Load the serialized models
with open('cluster_pipeline.pkl', 'rb') as f:
    song_cluster_pipeline = pickle.load(f)

# Initialize Spotify client
os.environ["SPOTIPY_CLIENT_ID"] = "d91340eac1614b02bbe5cae4d2496dcd"
os.environ["SPOTIPY_CLIENT_SECRET"] = "6690b132476741a7833a20f8ab9cb3f3"
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials())

number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy',
               'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode',
               'popularity', 'speechiness', 'tempo']

def find_song(name, year):
    song_data = defaultdict()
    results = sp.search(q=f'track: {name} year: {year}', limit=1)
    if not results['tracks']['items']:
        return None

    results = results['tracks']['items'][0]
    track_id = results['id']
    audio_features = sp.audio_features(track_id)[0]

    song_data['name'] = [name]
    song_data['year'] = [year]
    song_data['explicit'] = [int(results['explicit'])]
    song_data['duration_ms'] = [results['duration_ms']]
    song_data['popularity'] = [results['popularity']]

    for key, value in audio_features.items():
        song_data[key] = value

    return pd.DataFrame(song_data)

def get_song_data(song, spotify_data):
    try:
        song_data = spotify_data[(spotify_data['name'] == song['name']) & (spotify_data['year'] == song['year'])].iloc[0]
        return song_data
    except IndexError:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f'Warning: {song["name"]} does not exist in Spotify or in the database')
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)

    song_matrix = np.array(song_vectors)
    return np.mean(song_matrix, axis=0)

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)

    song_center = get_mean_vector(song_list, spotify_data)
    scaler = song_cluster_pipeline.steps[0][1]
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, 'cosine')
    index = list(np.argsort(distances)[:, :n_songs][0])

    rec_songs = spotify_data.iloc[index]
    rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
    return rec_songs[metadata_cols].to_dict(orient='records')

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    if request.method == 'POST':
        song_name = request.form['musicName']
        song_year = int(request.form['musicYear'])
        print(f'Received song: {song_name}, year: {song_year}')  # Debug statement
        song_list = [{'name': song_name, 'year': song_year}]

        # Check if the PKL file exists
        pkl_file_path = 'models/spotify_data.pkl'
        if not os.path.exists(pkl_file_path):
            return "Error: The required data file does not exist."

        # Load Spotify data from the PKL file
        with open(pkl_file_path, 'rb') as f:
            spotify_data = pickle.load(f)

        # Debug: Check if the data is loaded correctly
        print(f'Loaded spotify_data with shape: {spotify_data.shape}')  # Debug statement

        recommendations = recommend_songs(song_list, spotify_data, n_songs=10)

        # Debug: Check if recommendations are generated
        print(f'Generated recommendations: {recommendations}')  # Debug statement

    return render_template('prashant.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000, debug=True)
