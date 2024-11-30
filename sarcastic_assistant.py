import google.generativeai as genai
import pandas as pd
from recommendations import get_user_data_and_recommendations
import re


class SarcasticAssistant:
    def __init__(self):
        # Configure the Gemini API key
        genai.configure(api_key="<your API key here>")  # Replace with your actual API key

    def create_prompt(self, user_top_tracks, user_top_artists, recommendations):
        """
        Creates a prompt for the sarcastic assistant using user data and recommendations.
        """
        top_tracks_str = "\n".join([
            f"{i + 1}. {track['name']} by {track['artist']} (Popularity: {track['popularity']})"
            for i, track in enumerate(user_top_tracks)
        ])
        top_artists_str = "\n".join([
            f"{i + 1}. {artist['name']} (Popularity: {artist['popularity']})"
            for i, artist in enumerate(user_top_artists)
        ])

        def format_artists(artists):
            # Ensure artists is a list
            if isinstance(artists, str):
                try:
                    artists = eval(artists)
                except Exception:
                    artists = [artists]
            elif not isinstance(artists, list):
                artists = [str(artists)]
            return ", ".join(artists)

        recommendations_str = "\n".join([
            f"{i + 1}. {row['name']} by {format_artists(row['artists'])})"
            for i, row in recommendations.iterrows()
        ])

        # Regular expression to find numbers immediately before periods
        recommendations_str = re.sub(r'\b\d+\.', '', recommendations_str)

        return f"""
        You are a sarcastic music recommendation assistant. Based on the user's listening habits, top tracks, 
        and artists, you will make rude assumptions about the user's personality and musical tastes before delivering 
        the recommendations.

        Here's the user's data:
        - Top Tracks:
        {top_tracks_str}

        - Top Artists:
        {top_artists_str}

        Now, make some rude remarks about the user and their tastes, then deliver the following 
        recommendations sarcastically, but do not say '(with sarcasm)' or '(delivered sarcastically)' etc - just get to
        the point. Incorporate digs at a couple of their top artists and tracks.
        The response must be single-spaced, NOT double-spaced.
        Here are the recommendations, deliver them sarcastically:
        {recommendations_str}
        """

    def generate_response(self, prompt):
        """
        Sends the prompt to the Gemini API and retrieves the sarcastic response.
        """
        model = genai.GenerativeModel("gemini-1.5-flash-8b")
        response = model.generate_content(prompt)
        return response.text

    def run(self):
        """
        Orchestrates the fetching of user data, prompt creation, and response generation.
        """
        # Fetch user data, recommendations, and average popularity
        user_top_tracks, user_top_artists, recommendations = get_user_data_and_recommendations()

        # Create the prompt
        prompt = self.create_prompt(user_top_tracks, user_top_artists, recommendations)

        # Generate and display the sarcastic response
        response = self.generate_response(prompt)
        print(response)


if __name__ == "__main__":
    assistant = SarcasticAssistant()
    assistant.run()
