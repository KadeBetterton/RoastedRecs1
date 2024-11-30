import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from get_user_tops import get_user_tops

# Load datasets
tracks_df = pd.read_csv('tracks.csv')
artists_df = pd.read_csv('artists.csv')

# Debugging: Check columns of the loaded DataFrames
# print("Tracks DataFrame columns:", tracks_df.columns)
# print("Artists DataFrame columns:", artists_df.columns)

# Preprocessing Tracks Data
tracks_df = tracks_df.dropna(subset=['danceability', 'energy', 'valence', 'tempo', 'popularity', 'artists'])
tracks_df['artists'] = tracks_df['artists'].apply(eval)  # Convert string to list
tracks_features = tracks_df[['danceability', 'energy', 'valence', 'tempo', 'popularity']]

# Scaling the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(tracks_features)

# Building a nearest neighbors model
neighbors = NearestNeighbors(n_neighbors=5, metric='cosine')
neighbors.fit(scaled_features)


def recommend_from_user_top(user_top_tracks, user_top_artists):
    """
    Recommends tracks based on user's top tracks and artists.
    - Matches user top tracks with tracks.csv to get feature vectors.
    - Matches user top artists with artists.csv to find other associated tracks.
    """
    # Ensure all names are strings
    tracks_df['name'] = tracks_df['name'].astype(str)

    recommendations = set()

    # Match user top tracks with tracks.csv
    for track in user_top_tracks:
        try:
            match = tracks_df[tracks_df['name'].str.contains(track, case=False, na=False)]
            if not match.empty:
                track_index = match.index[0]
                distances, indices = neighbors.kneighbors([scaled_features[track_index]])
                recommendations.update(indices.flatten())
        except Exception as e:
            print(f"Error matching track '{track}': {e}")

    # Match user top artists with artists.csv and find their tracks
    for artist in user_top_artists:
        try:
            artist_match = artists_df[artists_df['name'].str.contains(artist, case=False, na=False)]
            if not artist_match.empty:
                artist_tracks = tracks_df[tracks_df['artists'].apply(lambda x: artist in x)]
                if not artist_tracks.empty:
                    artist_indices = artist_tracks.index
                    for idx in artist_indices:
                        distances, indices = neighbors.kneighbors([scaled_features[idx]])
                        recommendations.update(indices.flatten())
        except Exception as e:
            print(f"Error matching artist '{artist}': {e}")

    # Return the recommended track names
    return tracks_df.iloc[list(recommendations)][['name', 'artists']]


def get_user_data_and_recommendations():
    """
    Fetches user data (top tracks, top artists) and recommendations.
    Returns:
        tuple: (user_top_tracks, user_top_artists, recommendations, avg_popularity)
    """
    # Fetch user data
    tops_instance = get_user_tops()
    user_top_tracks, user_top_artists = tops_instance.main()

    # Convert user data to expected format
    user_top_tracks_names = [track['name'] for track in user_top_tracks]
    user_top_artists_names = [artist['name'] for artist in user_top_artists]

    # Generate recommendations
    recommendations = recommend_from_user_top(user_top_tracks_names, user_top_artists_names)

    # Return everything
    return user_top_tracks, user_top_artists, recommendations[:10]


# Get user info
tops_instance = get_user_tops()
user_top_tracks, user_top_artists = tops_instance.main()

# Extract track and artist names from the fetched data
user_top_tracks = [track['name'] for track in user_top_tracks]
user_top_artists = [artist['name'] for artist in user_top_artists]

# Get recommendations
recommendations = recommend_from_user_top(user_top_tracks, user_top_artists)

# Ensure popularity and artists are present in the recommendations DataFrame
if 'popularity' in tracks_df.columns and 'artists' in tracks_df.columns:
    # Merge recommendations with popularity and artists from tracks_df
    recommendations = recommendations.merge(
        tracks_df[['name', 'popularity', 'artists']], on='name', how='left'
    )

    # Debugging: Check columns after merge
    # print("Recommendations columns after merge:", recommendations.columns)

    # Drop duplicate columns if they exist
    if 'artists_x' in recommendations.columns:
        recommendations = recommendations.rename(columns={'artists_x': 'artists'})
    elif 'artists_y' in recommendations.columns:
        recommendations = recommendations.rename(columns={'artists_y': 'artists'})

    # Remove duplicates by track name
    recommendations = recommendations.drop_duplicates(subset=['name'])

    # Identify tracks associated with user's top artists
    if 'artists' in recommendations.columns:
        recommendations['is_user_top_artist'] = recommendations['artists'].apply(
            lambda x: bool(set(x) & set(user_top_artists)) if isinstance(x, list) else False
        )
    else:
        raise ValueError("'artists' column is missing in the recommendations DataFrame.")

    # Separate tracks by user's top artists and others
    artist_tracks = recommendations[recommendations['is_user_top_artist']]
    other_tracks = recommendations[~recommendations['is_user_top_artist']]

    # Limit to 2 tracks for all user top artists combined
    artist_tracks = artist_tracks.nlargest(2, 'popularity')

    # Combine limited artist tracks with other tracks
    recommendations = pd.concat([artist_tracks, other_tracks])

    # Sort by popularity in descending order and select the top 10
    recommendations = recommendations.sort_values(by='popularity', ascending=False).head(10)

'''
# Display the top 10 most popular recommendations
print("\nTop 10 Most Popular Recommendations:")
for index, row in recommendations.iterrows():
    print(f"{row['name']} by {', '.join(row['artists'])} (Popularity: {row['popularity']})")
'''