import requests
import pandas as pd
from tkinter import messagebox
from flask_app import get_access_token

def fetch_audio_features_in_batches(track_ids, headers):
    """Fetch audio features in batches of 100."""
    audio_features = []
    batch_size = 100
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        try:
            response = requests.get(
                f"https://api.spotify.com/v1/audio-features?ids={','.join(batch)}",
                headers=headers
            )
            response.raise_for_status()
            batch_features = response.json().get("audio_features", [])
            audio_features.extend(batch_features)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching batch {i // batch_size + 1}: {e}")
    return audio_features

def fetch_spotify_user_data():
    """Fetch Spotify user data, including top tracks and playlist information."""
    try:
        access_token = get_access_token()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None

    headers = {"Authorization": f"Bearer {access_token}"}
    user_data = {}

    # Fetch top tracks
    try:
        response = requests.get("https://api.spotify.com/v1/me/top/tracks?limit=50", headers=headers)
        response.raise_for_status()
        tracks = response.json()["items"]
        user_data["top_tracks"] = [{"name": track["name"], "id": track["id"], "artist": track["artists"][0]["name"]} for track in tracks]
    except requests.exceptions.RequestException as e:
        print(f"Error fetching top tracks: {e}")
        return None

    # Fetch audio features for top tracks
    try:
        track_ids = [track["id"] for track in user_data["top_tracks"] if track.get("id")]
        audio_features = fetch_audio_features_in_batches(track_ids, headers)
        user_data["audio_features"] = pd.DataFrame(audio_features)
    except Exception as e:
        print(f"Error fetching audio features: {e}")
        user_data["audio_features"] = pd.DataFrame()

    return user_data
