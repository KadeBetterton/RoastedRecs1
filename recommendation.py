import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sarcasm import generate_sarcasm

def generate_recommendations(user_data, query):
    """
    Combines collaborative filtering, content-based filtering, and SVD for recommendations.
    Includes a sarcastic response based on the user's preferences.

    Parameters:
        user_data (dict): User's Spotify data, including top tracks and audio features.
        query (str): The user's query (e.g., mood or preference).

    Returns:
        list: A list of recommendations and a sarcastic remark.
    """
    if not user_data or user_data["audio_features"].empty:
        return ["No recommendations available."]

    recommendations = []

    # Step 1: Content-Based Filtering
    try:
        print("Performing content-based filtering...")
        features = user_data["audio_features"][["danceability", "energy", "valence"]].fillna(0)
        query_vector = pd.DataFrame([features.mean()], columns=features.columns)
        similarities = cosine_similarity(features, query_vector).flatten()
        top_indices = similarities.argsort()[-5:][::-1]
        content_recommendations = [user_data["top_tracks"][i] for i in top_indices]
        recommendations.extend(content_recommendations)
    except Exception as e:
        print(f"Error in content-based filtering: {e}")

    # Step 2: Collaborative Filtering (if playlist data is available)
    try:
        print("Performing collaborative filtering...")
        playlist_track_ids = user_data.get("playlists", [])
        if playlist_track_ids:
            collaborative_features = user_data["audio_features"][["danceability", "energy", "valence"]].fillna(0)
            collaborative_model = NearestNeighbors(metric="cosine", algorithm="brute")
            collaborative_model.fit(collaborative_features)
            distances, indices = collaborative_model.kneighbors(query_vector, n_neighbors=5)
            collaborative_recommendations = [user_data["top_tracks"][i] for i in indices[0]]
            recommendations.extend(collaborative_recommendations)
    except Exception as e:
        print(f"Error in collaborative filtering: {e}")

    # Step 3: SVD-Based Recommendations
    try:
        print("Performing SVD-based recommendations...")
        svd = TruncatedSVD(n_components=3)
        svd_features = svd.fit_transform(features)
        svd_indices = svd_features.argsort(axis=0)[:, 0][:5]
        svd_recommendations = [user_data["top_tracks"][i] for i in svd_indices]
        recommendations.extend(svd_recommendations)
    except Exception as e:
        print(f"Error in SVD-based recommendations: {e}")

    # Combine recommendations (ensure uniqueness)
    unique_recommendations = list({f"{rec['name']} by {rec['artist']}" for rec in recommendations})

    # Step 4: Generate a Sarcastic Response
    try:
        sarcasm = generate_sarcasm(user_data, unique_recommendations, query)
        unique_recommendations.insert(0, sarcasm)
    except Exception as e:
        print(f"Error generating sarcasm: {e}")
        unique_recommendations.insert(0, "Couldn't generate sarcasm, but here are your recommendations:")

    return unique_recommendations
