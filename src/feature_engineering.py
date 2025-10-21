import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import os

def create_feature_matrix(merged_path="data/processed/tracks_artists_merged.csv",
                          save_path="data/processed/feature_matrix.npz"):
    # Load merged dataset
    df = pd.read_csv(merged_path)
    print(f"Loaded merged dataset with shape: {df.shape}")

    # Numeric features (track + artist)
    numeric_cols = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms"
    ]
    if 'followers' in df.columns:
        numeric_cols.append('followers')
    if 'popularity_artist' in df.columns:
        numeric_cols.append('popularity_artist')

    X_numeric = df[numeric_cols].values

    # Genres -> TF-IDF
    if 'genres' in df.columns:
        tfidf = TfidfVectorizer(max_features=200)  # limit to top 200 genres
        X_genres = tfidf.fit_transform(df['genres'].fillna(''))
        # Combine numeric + genres
        from scipy.sparse import hstack
        X = hstack([X_numeric, X_genres])
    else:
        X = X_numeric

    # Save feature matrix for later
    import scipy.sparse
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(X, pd.DataFrame):
        # If still dense DataFrame
        X.to_csv(save_path.replace('.npz', '.csv'), index=False)
    else:
        # If sparse matrix
        scipy.sparse.save_npz(save_path, X)

    print(f"✅ Feature matrix saved to {save_path}")
    return X, df

if __name__ == "__main__":
    X, df = create_feature_matrix()
