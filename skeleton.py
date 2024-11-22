import threading
import tkinter as tk
from tkinter import messagebox
from flask import Flask, request
import requests
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import webbrowser

from transformers import pipeline

# --- Spotify API Setup ---
SPOTIFY_CLIENT_ID = "f3ca21df362544ecb3cef0fdf2e9eb91"
SPOTIFY_CLIENT_SECRET = "d378370c59564391b72b894ae8d89f44"
SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"
SPOTIFY_SCOPE = "user-top-read playlist-read-private"

# Placeholder for Spotify access token
SPOTIFY_ACCESS_TOKEN = None

# --- Flask App for Authentication Callback ---
app = Flask(__name__)

@app.route("/callback")
def callback():
    """
    Handles the Spotify redirect and authorization code exchange.
    """
    global SPOTIFY_ACCESS_TOKEN
    auth_code = request.args.get("code")
    token_url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    }
    response = requests.post(token_url, data=payload)
    if response.status_code == 200:
        SPOTIFY_ACCESS_TOKEN = response.json()["access_token"]
        return "Spotify login successful! You can close this window."
    else:
        return "Error: Failed to authenticate with Spotify."

def run_flask():
    app.run(port=8888)

# --- Functions for Spotify Authentication and Data Fetching ---
def spotify_login():
    auth_url = (
        f"https://accounts.spotify.com/authorize?"
        f"client_id={SPOTIFY_CLIENT_ID}&response_type=code&redirect_uri={SPOTIFY_REDIRECT_URI}&scope={SPOTIFY_SCOPE}"
    )
    webbrowser.open(auth_url)

def fetch_spotify_user_data():
    if not SPOTIFY_ACCESS_TOKEN:
        messagebox.showerror("Error", "Please log in to Spotify first.")
        return None

    headers = {"Authorization": f"Bearer {SPOTIFY_ACCESS_TOKEN}"}
    user_data = {}

    # Fetch top tracks
    response = requests.get("https://api.spotify.com/v1/me/top/tracks?limit=50", headers=headers)
    if response.status_code == 200:
        tracks = response.json()["items"]
        user_data["top_tracks"] = [{"name": track["name"], "id": track["id"], "artist": track["artists"][0]["name"]} for track in tracks]

    # Fetch audio features
    track_ids = [track["id"] for track in user_data["top_tracks"]]
    if track_ids:
        response = requests.get(f"https://api.spotify.com/v1/audio-features?ids={','.join(track_ids)}", headers=headers)
        if response.status_code == 200:
            audio_features = response.json()["audio_features"]
            user_data["audio_features"] = pd.DataFrame(audio_features)

    # Fetch playlist tracks for collaborative filtering
    response = requests.get("https://api.spotify.com/v1/me/playlists", headers=headers)
    if response.status_code == 200:
        playlists = response.json()["items"]
        user_data["playlists"] = []
        for playlist in playlists:
            playlist_id = playlist["id"]
            tracks_response = requests.get(f"https://api.spotify.com/v1/playlists/{playlist_id}/tracks", headers=headers)
            if tracks_response.status_code == 200:
                playlist_tracks = tracks_response.json()["items"]
                user_data["playlists"].extend([track["track"]["id"] for track in playlist_tracks if track["track"]])

    return user_data

# --- Recommendation System ---
def generate_recommendations(user_data, query):
    """
    Combines collaborative filtering, content-based filtering, and SVD for recommendations.
    Includes a sarcastic response based on the user's preferences.
    """
    if not user_data:
        return ["No recommendations available."]

    audio_features = user_data.get("audio_features", pd.DataFrame())
    if audio_features.empty:
        return ["No recommendations available."]

    # Content-based filtering using cosine similarity
    feature_vector = audio_features[["danceability", "energy", "valence"]]
    query_vector = pd.DataFrame([feature_vector.mean()], columns=feature_vector.columns)
    similarities = cosine_similarity(feature_vector, query_vector)
    content_recommendations = sorted(zip(user_data["top_tracks"], similarities.flatten()), key=lambda x: x[1], reverse=True)[:5]

    # Collaborative filtering using playlists
    collaborative_recommendations = []
    if "playlists" in user_data and user_data["playlists"]:
        playlist_track_ids = user_data["playlists"]
        headers = {"Authorization": f"Bearer {SPOTIFY_ACCESS_TOKEN}"}
        response = requests.get(f"https://api.spotify.com/v1/audio-features?ids={','.join(playlist_track_ids[:100])}", headers=headers)
        if response.status_code == 200:
            playlist_audio_features = pd.DataFrame(response.json()["audio_features"])
            if not playlist_audio_features.empty:
                playlist_feature_vector = playlist_audio_features[["danceability", "energy", "valence"]].fillna(0)
                collaborative_model = NearestNeighbors(metric='cosine', algorithm='brute')
                collaborative_model.fit(playlist_feature_vector)
                distances, indices = collaborative_model.kneighbors(query_vector, n_neighbors=5)
                # Fetch track names and artists for collaborative recommendations
                for i in indices[0]:
                    track_id = playlist_track_ids[i]
                    track_info_response = requests.get(f"https://api.spotify.com/v1/tracks/{track_id}", headers=headers)
                    if track_info_response.status_code == 200:
                        track_info = track_info_response.json()
                        collaborative_recommendations.append(f"{track_info['name']} by {track_info['artists'][0]['name']}")
            else:
                collaborative_recommendations = []
        else:
            collaborative_recommendations = []

    # Matrix factorization using SVD
    svd = TruncatedSVD(n_components=3)
    svd_features = svd.fit_transform(feature_vector)
    svd_recommendations = [user_data["top_tracks"][i] for i in svd_features.argsort(axis=0)[:, 0][:5]]

    # Combine recommendations
    recommendations = list({
        f"{rec[0]['name']} by {rec[0]['artist']}" for rec in content_recommendations
    }.union(collaborative_recommendations).union({
        f"{rec['name']} by {rec['artist']}" for rec in svd_recommendations
    }))

    # Generate sarcasm
    sarcasm = generate_sarcasm(user_data, recommendations, query)

    # Format output
    output = [sarcasm] + recommendations[:5]
    return output


def generate_sarcasm(user_data, recommendations, query):
    """
    Generates a sarcastic response using the SarcasMLL-1B model based on user profile and recommendations.
    Removes the input prompt from the generated response.
    """
    # Use Hugging Face Transformers pipeline with the SarcasMLL-1B model
    generator = pipeline("text-generation", model="AlexandrosChariton/SarcasMLL-1B", device=0, truncation=False)

    # Refine the prompt to focus on sarcasm
    profile_summary = f"Top artist: {user_data['top_tracks'][0]['artist']}, Average energy: {user_data['audio_features']['energy'].mean():.2f}"
    prompt = (
        f"User listens to {profile_summary}.\n"
        f"Recommendations: {', '.join(recommendations[:3])}.\n"
        f"Generate a short sarcastic remark about their taste in music."
    )

    # Generate the sarcastic response
    try:
        response = generator(prompt, max_new_tokens=100, num_return_sequences=1, pad_token_id=50256)
        generated_text = response[0]["generated_text"].strip()

        # Remove the prompt from the generated text
        if generated_text.startswith(prompt):
            sarcasm_only = generated_text[len(prompt):].strip()
        else:
            sarcasm_only = generated_text

        # Truncate after the last period
        clean_response = truncate_after_last_period(sarcasm_only)
        return clean_response
    except Exception as e:
        # Handle model errors gracefully
        return f"Oops, something went wrong while generating sarcasm: {str(e)}"


def truncate_after_last_period(response):
    """
    Truncates the response string after the last period ('.').
    If no period is found, returns the full response.
    """
    if '.' in response:
        last_period_index = response.rfind('.')
        return response[:last_period_index + 1]  # Include the last period
    return response.strip()  # Return the full response if no period is found



# --- Tkinter GUI ---
def handle_query():
    """
    Handles the user's query input, generates sarcasm, and provides recommendations.
    """
    global SPOTIFY_ACCESS_TOKEN

    if not SPOTIFY_ACCESS_TOKEN:
        messagebox.showerror("Error", "Please log in to Spotify first.")
        return

    user_data = fetch_spotify_user_data()
    if not user_data:
        return

    query = user_input.get()
    if not query.strip():
        messagebox.showwarning("Input Error", "Please enter a query!")
        return

    recommendations = generate_recommendations(user_data, query)

    # Display sarcasm and recommendations cleanly
    result_text.set("\n".join(recommendations))


def build_gui():
    global user_input, result_text

    root = tk.Tk()
    root.title("RoastedRecs")
    root.geometry("600x500")

    login_button = tk.Button(root, text="Log in to Spotify", command=spotify_login)
    login_button.pack(pady=10)

    input_label = tk.Label(root, text="What are you looking for?")
    input_label.pack(pady=10)

    user_input = tk.Entry(root, width=60)
    user_input.pack(pady=5)

    submit_button = tk.Button(root, text="Submit", command=handle_query)
    submit_button.pack(pady=10)

    result_text = tk.StringVar()
    result_label = tk.Label(root, textvariable=result_text, justify="left", wraplength=500)
    result_label.pack(pady=20)

    root.mainloop()

# --- Main Function ---
if __name__ == "__main__":
    threading.Thread(target=run_flask, daemon=True).start()
    build_gui()
