import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import os

def load_data(
    data_path="data/processed/tracks_artists_merged.csv",
    feature_path="data/processed/feature_matrix_embeddings.npz"
):
    """Loads processed dataset and precomputed feature matrix"""
    if not os.path.exists(data_path) or not os.path.exists(feature_path):
        raise FileNotFoundError("❌ Processed data or feature matrix not found. Run preprocessing and feature extraction first.")

    print("📂 Loading dataset and feature matrix...")
    df = pd.read_csv(data_path)
    X = sparse.load_npz(feature_path)
    print(f"✅ Data loaded: {df.shape[0]} tracks, feature matrix shape: {X.shape}")
    return df, X


def recommend_track(track_name, df, X, top_n=10):
    """
    Given a track name, find the most similar tracks based on cosine similarity.
    """
    # Try to find the track index
    matches = df[df['name_track'].str.lower() == track_name.lower()]
    if matches.empty:
        print(f"⚠️ Track '{track_name}' not found in dataset.")
        return []

    idx = matches.index[0]

    # Compute cosine similarity
    print(f"🎧 Finding recommendations for: {df.loc[idx, 'name_track']} by {df.loc[idx, 'artists']}")
    sim_scores = cosine_similarity(X[idx], X).flatten()

    # Get top N indices (excluding itself)
    similar_indices = sim_scores.argsort()[::-1][1:top_n+1]

    # Prepare results
    recommendations = df.iloc[similar_indices][['name_track', 'artists', 'genres', 'popularity_track']].copy()
    recommendations['similarity'] = sim_scores[similar_indices]
    return recommendations


if __name__ == "__main__":
    df, X = load_data()
    track_name = input("Enter a song name to get recommendations: ").strip()
    recs = recommend_track(track_name, df, X, top_n=10)

    if len(recs) > 0:
        print("\n🎵 Top Recommendations:")
        print(recs.to_string(index=False))
