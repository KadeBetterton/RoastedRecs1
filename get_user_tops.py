import os

import spotipy
from spotipy.oauth2 import SpotifyOAuth


class get_user_tops():
    # Set your credentials
    SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "your ID here")
    SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "<your secret here")
    SPOTIFY_REDIRECT_URI = "http://localhost:8888/callback"

    # Define the required scope
    SCOPE = "user-top-read"


    def authenticate_spotify(self):
        """
        Authenticates the user with Spotify and returns a Spotipy client instance.
        """
        sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=self.SPOTIFY_CLIENT_ID,
            client_secret=self.SPOTIFY_CLIENT_SECRET,
            redirect_uri=self.SPOTIFY_REDIRECT_URI,
            scope=self.SCOPE
        ))
        return sp


    def get_top_tracks(self, sp, limit=10):
        """
        Fetches the user's top tracks.
        """
        top_tracks = sp.current_user_top_tracks(limit=limit, time_range='medium_term')
        return [{"name": track["name"], "artist": track["artists"][0]["name"], "popularity": track["popularity"]} for track
                in top_tracks["items"]]


    def get_top_artists(self, sp, limit=10):
        """
        Fetches the user's top artists.
        """
        top_artists = sp.current_user_top_artists(limit=limit, time_range='medium_term')
        return [{"name": artist["name"], "genres": artist["genres"], "popularity": artist["popularity"]} for artist in
                top_artists["items"]]


    def main(self):
        sp = self.authenticate_spotify()

        #print("Fetching top tracks...")
        top_tracks = self.get_top_tracks(sp)
        #print("\nYour Top Tracks:")
        #for idx, track in enumerate(top_tracks, start=1):
            #print(f"{idx}. {track['name']} by {track['artist']} (Popularity: {track['popularity']})")

        #print("\nFetching top artists...")
        top_artists = self.get_top_artists(sp)
        #print("\nYour Top Artists:")
        #for idx, artist in enumerate(top_artists, start=1):
            #print(f"{idx}. {artist['name']} (Genres: {', '.join(artist['genres'])}, Popularity: {artist['popularity']})")

        return top_tracks, top_artists
