from flask import Flask, request, jsonify, redirect
import requests
import json
import time

SPOTIFY_CLIENT_ID = "f3ca21df362544ecb3cef0fdf2e9eb91"
SPOTIFY_CLIENT_SECRET = "f4cec6145f5e4878a0a1ab41ce515f75"
SPOTIFY_REDIRECT_URI = "http://127.0.0.1:60000"
TOKEN_FILE = "access_token.json"

app = Flask(__name__)

def save_token_data(token_data):
    """Save token data (access token, refresh token, and expiration time) to a file."""
    data = {
        "access_token": token_data["access_token"],
        "expires_at": time.time() + token_data["expires_in"],  # Current time + token lifespan
    }
    if "refresh_token" in token_data:
        data["refresh_token"] = token_data["refresh_token"]
    with open(TOKEN_FILE, "w") as f:
        json.dump(data, f)

def load_token_data():
    """Load token data from the file."""
    try:
        with open(TOKEN_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def refresh_access_token(refresh_token):
    """Use the refresh token to get a new access token."""
    token_url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    }
    response = requests.post(token_url, data=payload)
    response.raise_for_status()
    token_data = response.json()
    save_token_data(token_data)  # Save the new token data
    return token_data["access_token"]

def get_access_token():
    """Retrieve a valid access token, refreshing it if necessary."""
    token_data = load_token_data()
    if not token_data:
        raise Exception("No token data found. Please log in to Spotify first.")
    if time.time() >= token_data["expires_at"]:
        print("Access token expired, refreshing...")
        return refresh_access_token(token_data["refresh_token"])
    return token_data["access_token"]

@app.route("/login")
def login():
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?response_type=code&client_id={SPOTIFY_CLIENT_ID}"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}&scope=user-read-private"
    )
    return redirect(auth_url)

@app.route("/callback")
def callback():
    """Handle Spotify redirect and authorization code exchange."""
    auth_code = request.args.get("code")
    if not auth_code:
        return "Error: Missing authorization code.", 400

    token_url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "authorization_code",
        "code": auth_code,
        "redirect_uri": SPOTIFY_REDIRECT_URI,
        "client_id": SPOTIFY_CLIENT_ID,
        "client_secret": SPOTIFY_CLIENT_SECRET,
    }

    try:
        response = requests.post(token_url, data=payload)
        response.raise_for_status()
        token_data = response.json()
        save_token_data(token_data)
        return "Spotify login successful! You can close this window."
    except requests.exceptions.RequestException as e:
        return f"Error during Spotify token exchange: {e}", 500

if __name__ == "__main__":
    """Run the Flask app on a fixed port."""
    app.run(debug=True, host="0.0.0.0", port=60000)