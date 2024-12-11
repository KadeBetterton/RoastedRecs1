import re
import time

import google.generativeai as genai
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
from unidecode import unidecode
import tkinter as tk
from tkinter import scrolledtext


def normalize_text(text):
    text = unidecode(text)  # convert accented characters to ASCII
    text = text.lower()  # convert to lowercase
    text = re.sub(r'\(.*?\)', '', text)  # remove text in parentheses
    text = re.sub(r'[^a-z0-9\s]', '', text)  # remove non-alphanumeric characters (except spaces)
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    return text

# suppress non-critical warnings:
warnings.filterwarnings("ignore")
'''
# suppress gRPC logging:
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_LOG_SEVERITY"] = "ERROR"
'''

# configure Gemini:
genai.configure(api_key="AIzaSyDh37esnX-KiysJhqx9P1OeWGUw67xjNh8")


# to extract artist name using LLM:
def extract_artist_name(user_input):
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content(f"Extract the artist's name from this user input: '{user_input}'. "
                                      f"If it looks a lot like a popular musician you know of, please extract the "
                                      f"corrected version. DO NOT say anything but the artist name (but if it looks "
                                      f"misspelled, please correct it).")

    # clean response:
    clean_artist_name = response.text.strip()
    return clean_artist_name


# to generate natural language examples using LLM:
def generate_song_examples(artist, song_list):
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content(
        f"Here are some songs by {artist}: {', '.join(song_list[:5])}. Format this nicely to display to the user in a "
        f"numbered list. Just the list of songs.")
    return response.text


# generate recommendations using LLM:
def generate_explanation(selected_song, recommendations):
    # convert recommendations to a formatted string:
    recommendation_list = "\n".join(
        [f"{i+1}. {row['track_name']} by {row['artists']}" for i, row in recommendations.iterrows()]
    )

    # LLM prompt:
    prompt = (
        f"The user selected the song '{selected_song['track_name']}' by {selected_song['artists']}. "
        f"Here are some recommended songs based on their choice:\n{recommendation_list}\n"
        "Write a paragraph explaining how each song is similar musically to the user's selected song (just a paragraph "
        "- do not list them). "
        "Use outside knowledge about the songs when possible."
    )

    # generate response:
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content(prompt)

    '''
    # use textwrap to format the output:
    wrapped_response = textwrap.fill(response.text, width=80)

    return wrapped_response
    '''
    return response.text


# load & preprocess data:
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Remove duplicate tracks by name & artist
    data = data.drop_duplicates(subset=['track_name', 'artists'])

    # Exclude specific genres
    excluded_genres = ['brazil', 'turkish', 'malay', 'anime', 'iranian', 'sleep', 'kids', 'latin', 'french', 'tango',
                       'study', 'indian', 'children', 'pop-film', 'j-pop', 'j-dance', 'cantopop', 'mandopop', 'disney']
    data = data[~data['track_genre'].isin(excluded_genres)]

    # Keep only relevant features and drop rows with missing values
    features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    data = data.dropna(subset=features)

    data.reset_index(drop=True, inplace=True)
    return data, features


# to scale features:
def scale_features(data, feature_columns):
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])
    scaled_data = pd.DataFrame(scaled_features, columns=feature_columns)
    return scaled_data, scaler


# K-means recommendations:
def kmeans_recommend_songs(selected_title, user_input_features, data, scaled_data, scaler, pca, kmeans, n_recommendations=5):
    user_input_df = pd.DataFrame([user_input_features], columns=scaled_data.columns)
    user_scaled = scaler.transform(user_input_df)
    user_pca = pca.transform(user_scaled)
    cluster_label = kmeans.predict(user_pca)[0]

    # Filter songs from the same cluster
    cluster_songs = data[data['Cluster'] == cluster_label]

    # Compute similarity within the cluster
    cluster_songs['Similarity'] = cosine_similarity(user_input_df, cluster_songs[scaled_data.columns])[0]

    # Ensure selected_title is normalized and filter songs
    if selected_title:
        selected_title_normalized = normalize_text(selected_title)  # Normalize for consistency
        cluster_songs = cluster_songs[
            ~cluster_songs['track_name'].fillna('').apply(normalize_text).str.contains(selected_title_normalized, case=False)
        ]

    # Sort by similarity and return top recommendations
    recommendations = cluster_songs.sort_values(by='Similarity', ascending=False).head(n_recommendations)
    return recommendations



# Content-based recommendations:
def content_based_recommend_songs(user_input_features, data, feature_columns, selected_song, n_recommendations=5):
    user_input_df = pd.DataFrame([user_input_features], columns=feature_columns)
    similarities = cosine_similarity(user_input_df, data[feature_columns])
    data['Similarity'] = similarities[0]

    # Normalize the selected song title for filtering
    selected_title_normalized = normalize_text(selected_song['track_name'])

    # Filter out songs that have identical or overlapping titles
    data = data[
        ~data['track_name'].apply(lambda x: selected_title_normalized in normalize_text(x))
    ]

    # Exclude the user-selected song
    recommendations = data[data['track_name'] != selected_song['track_name']].sort_values(by='Similarity', ascending=False).head(n_recommendations)
    return recommendations


# Evaluate MRR:
def evaluate_mrr(recommendations, selected_song, feature_columns):
    selected_features = selected_song[feature_columns].values.reshape(1, -1)
    similarities = cosine_similarity(selected_features, recommendations[feature_columns])
    recommendations['Similarity'] = similarities[0]
    recommendations = recommendations.sort_values(by='Similarity', ascending=False)
    for rank, (_, row) in enumerate(recommendations.iterrows(), start=1):
        if row['Similarity'] > 0.8:  # Example threshold for relevance
            return 1 / rank
    return 0  # No relevant recommendations


# Compute diversity:
def compute_diversity(recommendations, feature_columns):
    features = recommendations[feature_columns].values
    pairwise_distances = cosine_similarity(features)
    diversity = 1 - pairwise_distances.mean()
    return diversity


class ChatInterface:
    def __init__(self, root, data, scaled_data, scaler, pca, kmeans, feature_columns):
        self.root = root
        self.data = data
        self.scaled_data = scaled_data
        self.scaler = scaler
        self.pca = pca
        self.kmeans = kmeans
        self.feature_columns = feature_columns

        # Set up the window
        self.root.title("Music Recommendation System")
        self.root.geometry("600x600")

        # Chat display
        self.chat_display = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled', height=25, width=70)
        self.chat_display.pack(pady=10)

        # User input area
        self.user_input = tk.Entry(self.root, width=60)
        self.user_input.pack(pady=5)
        self.user_input.bind("<Return>", self.process_input)

        # Send button
        self.send_button = tk.Button(self.root, text="Send", command=self.send_input)
        self.send_button.pack()

        # Start the conversation
        self.add_message("Welcome! I'm here to help you find new music similar to what you already enjoy. Tell me an artist you like, or you can type 'quit' to exit.", sender="System")
        self.awaiting_response = None  # Tracks context for user input

    def add_message(self, message, sender="User"):
        self.chat_display.configure(state='normal')
        if sender == "User":
            self.chat_display.insert(tk.END, f"\nYou: {message}\n")
        else:
            self.chat_display.insert(tk.END, f"\n{message}\n")
        self.chat_display.configure(state='disabled')
        self.chat_display.yview(tk.END)

    def process_input(self, event=None):
        user_message = self.user_input.get().strip()
        if user_message:
            self.add_message(user_message, sender="User")
            self.user_input.delete(0, tk.END)
            if self.awaiting_response:
                self.awaiting_response(user_message)
            else:
                self.handle_user_message(user_message)

    def send_input(self):
        self.process_input()

    def handle_user_message(self, message):
        if message.lower() == "quit":
            self.add_message("Goodbye!", sender="System")
            time.sleep(1)
            self.root.quit()
            return

        corrected_artist = extract_artist_name(message)
        artist_songs = self.data[self.data['artists'].str.contains(corrected_artist, case=False, na=False)]

        if artist_songs.empty:
            self.add_message(f"Sorry, no songs found for '{corrected_artist}' in the dataset.", sender="System")
            return

        self.add_message(f"Okay, here are some songs by {corrected_artist}:", sender="System")
        song_list = artist_songs['track_name'].tolist()
        response = generate_song_examples(corrected_artist, song_list)
        self.add_message(response, sender="System")

        self.add_message("Enter the number of a song for recommendations:", sender="System")
        self.awaiting_response = lambda song_choice: self.handle_song_choice(song_choice, artist_songs, song_list)

    def handle_song_choice(self, song_choice, artist_songs, song_list):
        try:
            song_choice = int(song_choice)
            if song_choice < 1 or song_choice > len(song_list):
                self.add_message("Invalid selection. Try again.", sender="System")
                return
        except ValueError:
            self.add_message("Invalid input. Please enter a number.", sender="System")
            return

        selected_song = artist_songs.iloc[song_choice - 1]
        user_input_features = selected_song[self.feature_columns].values

        self.add_message(f"Okay, I'll recommend you songs musically similar to \"{selected_song['track_name']}.\"", sender="System")

        self.add_message("Please choose a recommendation method (1 for K-means, 2 for Content-based):", sender="System")
        self.awaiting_response = lambda method_choice: self.handle_method_choice(method_choice, selected_song, user_input_features)

    def handle_method_choice(self, method_choice, selected_song, user_input_features):
        if method_choice == "1":
            recommendations = kmeans_recommend_songs(
                selected_title=selected_song['track_name'],
                user_input_features=user_input_features,
                data=self.data,
                scaled_data=self.scaled_data,
                scaler=self.scaler,
                pca=self.pca,
                kmeans=self.kmeans,
                n_recommendations=5
            )
        elif method_choice == "2":
            recommendations = content_based_recommend_songs(
                user_input_features, self.data, self.feature_columns, selected_song
            )
        else:
            self.add_message("Invalid method choice. Please choose again.", sender="System")
            return

        # Format and display recommendations
        self.add_message("\nMy recommendations:", sender="System")
        for _, row in recommendations.iterrows():
            # Replace ";" with "and" in the 'artists' column for display
            artists_cleaned = row['artists'].replace(';', ' and ')
            formatted_recommendation = f"- {row['track_name']} by {artists_cleaned} ({row['track_genre']})"
            self.add_message(formatted_recommendation, sender="System")

        # Generate and display explanation
        explanation = generate_explanation(selected_song, recommendations)
        self.add_message("\nWhy you'll like these:", sender="System")
        self.add_message(explanation, sender="System")

        # Compute and display MRR and diversity
        mrr = evaluate_mrr(recommendations, selected_song, self.feature_columns)
        diversity = compute_diversity(recommendations, self.feature_columns)

        if mrr >= 0.7:
            self.add_message(
                f"\nThe MMR score of {mrr:.2f} means these songs are very musically similar to {selected_song['track_name']}",
                sender="System")
        if diversity <= 0.3:
            self.add_message(
                f"The diversity of these songs is {diversity:.2f}, meaning they are very similar to each other.",
                sender="System")

        # Offer the choice to restart or exit
        self.add_message("\nWould you like to start with a new artist or quit? Type 'new artist' or 'quit'.",
                         sender="System")
        self.awaiting_response = self.restart_or_exit

    def restart_or_exit(self, choice):
        if choice.lower() == "new artist":
            self.add_message("Sounds good! What other artist do you want to find music similar to?", sender="System")
            self.awaiting_response = None
        elif choice.lower() == "quit":
            self.add_message("Goodbye!", sender="System")
            time.sleep(1)
            self.root.quit()
        else:
            self.add_message("Invalid input. Please type 'new artist' or 'quit'.", sender="System")


# Main function to start the interface
def main():
    file_path = 'english_songs.csv'
    data, feature_columns = load_and_preprocess_data(file_path)
    scaled_data, scaler = scale_features(data, feature_columns)
    pca = PCA(n_components=2)
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    kmeans = KMeans(n_clusters=20, random_state=42)
    kmeans.fit(pca_data)
    data['Cluster'] = kmeans.labels_

    root = tk.Tk()
    ChatInterface(root, data, scaled_data, scaler, pca, kmeans, feature_columns)
    root.mainloop()

if __name__ == "__main__":
    main()
