import google.generativeai as genai
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import textwrap

# suppress non-critical warnings:
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# configure Gemini:
genai.configure(api_key="AIzaSyDh37esnX-KiysJhqx9P1OeWGUw67xjNh8")


# to extract artist name using LLM:
def extract_artist_name(user_input):
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content(f"Extract the artist's name from this user input: '{user_input}'. If it looks a lot like a popular musician you know of, please extract the corrected version. DO NOT say anything but the artist name (but if it looks misspelled, please correct it.")  # include user input in model prompt

    # clean response:
    clean_artist_name = response.text.strip()
    print(f"Extracted Artist Name: {clean_artist_name}")
    return clean_artist_name


# to generate natural language examples using LLM:
def generate_song_examples(artist, song_list):
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content(
        f"Here are some songs by {artist}: {', '.join(song_list[:5])}. Format this nicely to display to the user in a numbered list.")
    return response.text


# generate recommendations using LLM:
def generate_recommendations(selected_song, recommendations):
    # Convert recommendations to a formatted string
    recommendation_list = "\n".join(
        [f"{i+1}. {row['track_name']} by {row['artists']}" for i, row in recommendations.iterrows()]
    )

    # LLM prompt
    prompt = (
        f"The user selected the song '{selected_song['track_name']}' by {selected_song['artists']}. "
        f"Here are some recommended songs based on their choice:\n{recommendation_list}\n"
        "Write a paragraph explaining how each song is similar to the user's selected song (just a paragraph - do not list them). "
        "Use outside knowledge about the songs when possible."
    )

    # Generate response
    model = genai.GenerativeModel("gemini-1.5-flash-8b")
    response = model.generate_content(prompt)

    # Use textwrap to format the output
    wrapped_response = textwrap.fill(response.text, width=80)

    return wrapped_response


# load & preprocess data:
def load_and_preprocess_data(file_path):
    # load dataset:
    data = pd.read_csv(file_path)

    # remove duplicate tracks by name & artist:
    data = data.drop_duplicates(subset=['track_name', 'artists'])

    # exclude specific genres (example list provided):
    excluded_genres = ['brazil', 'turkish', 'malay', 'anime', 'iranian', 'sleep', 'kids', 'latin', 'french', 'tango',
                       'study', 'indian', 'children', 'pop-film', 'j-pop', 'cantopop', 'mandopop']
    data = data[~data['track_genre'].isin(excluded_genres)]

    # keep only relevant features & drop rows w/ missing values:
    features = [
        'danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
        'instrumentalness', 'liveness', 'valence', 'tempo'
    ]
    data = data.dropna(subset=features)

    # reset index after filtering:
    data.reset_index(drop=True, inplace=True)

    return data, features


# to scale features:
def scale_features(data, feature_columns):
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])
    scaled_data = pd.DataFrame(scaled_features, columns=feature_columns)
    return scaled_data, scaler


# the recommendation function using K-means:
def kmeans_recommend_songs(user_input_features, data, scaled_data, scaler, pca, kmeans, n_recommendations=5):
    user_input_df = pd.DataFrame([user_input_features], columns=scaled_data.columns)
    user_scaled = scaler.transform(user_input_df)
    user_pca = pca.transform(user_scaled)
    cluster_label = kmeans.predict(user_pca)[0]
    cluster_songs = data[data['Cluster'] == cluster_label]
    n_recommendations = min(n_recommendations, len(cluster_songs))
    recommendations = cluster_songs.sample(n=n_recommendations)
    return recommendations


# the recommendation function using content-based filtering:
def content_based_recommend_songs(user_input_features, data, feature_columns, selected_song, n_recommendations=5):
    user_input_df = pd.DataFrame([user_input_features], columns=feature_columns)
    similarities = cosine_similarity(user_input_df, data[feature_columns])
    data['Similarity'] = similarities[0]

    # exclude the user-selected song from the recommendations:
    recommendations = data[data['track_name'] != selected_song['track_name']].sort_values(
        'Similarity', ascending=False).head(n_recommendations)

    return recommendations


def user_interface(data, scaled_data, scaler, pca, kmeans, feature_columns):
    print("Welcome to the Enhanced Music Recommendation System!")

    while True:
        print("\nEnter an artist you like in natural language, or type 'exit' to quit:")
        user_input = input("Your input: ").strip()

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        # use LLM to extract & correct artist name:
        corrected_artist = extract_artist_name(user_input)

        if not corrected_artist:
            print(f"Sorry, no close matches found for '{corrected_artist}' in the dataset. Try another artist.")
            continue

        print(f"Matched Artist: {corrected_artist}")

        # find songs by the matched artist:
        artist_songs = data[data['artists'].str.contains(corrected_artist, case=False, na=False)]

        if artist_songs.empty:
            print(f"Sorry, no songs found for '{corrected_artist}' in the dataset. Try another artist.")
            continue

        print(f"\nSongs by {corrected_artist} in the dataset:")

        # generate natural language response for song examples:
        song_list = artist_songs['track_name'].tolist()
        response = generate_song_examples(corrected_artist, song_list)
        print("\n" + response)

        # user selects a song:
        try:
            song_choice = int(input("\nSelect a song number for recommendations: ")) - 1
            if 0 <= song_choice < len(song_list):
                selected_song = artist_songs.iloc[song_choice]
                print(f"You selected: {selected_song['track_name']} by {selected_song['artists']}")

                # use the song's features for recommendation:
                user_input_features = selected_song[feature_columns].values

                # let user choose a recommendation method:
                print("\nChoose a recommendation method:")
                print("1. K-means-based recommendations")
                print("2. Content-based recommendations")
                method = input("Enter 1 or 2: ").strip()

                if method == "1":
                    recommendations = kmeans_recommend_songs(user_input_features, data, scaled_data, scaler, pca,
                                                             kmeans)
                    print("\nK-means Based Recommendations:")
                elif method == "2":
                    recommendations = content_based_recommend_songs(
                        user_input_features, data, feature_columns, selected_song)
                    print("\nContent-Based Recommendations:")
                else:
                    print("Invalid choice. Returning to artist selection.")
                    continue

                # generate commentary for recommendations:
                print(recommendations[['track_name', 'artists', 'track_genre']].to_string(index=False))

                commentary = generate_recommendations(selected_song, recommendations)
                print("\nSimilarities with Your Chosen Song:")
                print(commentary)

            else:
                print("Invalid choice. Returning to artist selection.")
        except ValueError:
            print("Invalid input. Returning to artist selection.")


# main:
if __name__ == "__main__":
    file_path = 'english_songs.csv'

    # load & preprocess the data:
    data, feature_columns = load_and_preprocess_data(file_path)

    # scale the features:
    scaled_data, scaler = scale_features(data, feature_columns)

    # fit PCA:
    pca = PCA(n_components=2)
    pca.fit(scaled_data)

    # apply PCA & KMeans:
    pca_data = pca.transform(scaled_data)
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(pca_data)

    # add cluster labels to data:
    data['Cluster'] = kmeans.labels_

    # start user interface:
    user_interface(data, scaled_data, scaler, pca, kmeans, feature_columns)