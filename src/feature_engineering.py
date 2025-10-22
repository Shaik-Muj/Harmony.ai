import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import ast

def create_feature_matrix(
    merged_path="data/processed/tracks_artists_merged.csv",
    save_path="data/processed/feature_matrix_embeddings.npz"
):
    # Load dataset
    df = pd.read_csv(merged_path)
    print(f"✅ Loaded merged dataset with shape: {df.shape}")

    # --- 1. Clean text fields ---
    def clean_text_field(val):
        """Handle cases like ['Pop', 'Rock'] or NaN."""
        if pd.isna(val):
            return ""
        try:
            # Convert string repr of list to actual list
            if isinstance(val, str) and val.startswith('['):
                val = ast.literal_eval(val)
            if isinstance(val, list):
                return " ".join(map(str, val))
            return str(val)
        except:
            return str(val)

    df["genres"] = df["genres"].apply(clean_text_field)
    df["artists"] = df["artists"].apply(clean_text_field)
    df["name_artist"] = df["name_artist"].apply(clean_text_field)
    df["name_track"] = df["name_track"].apply(clean_text_field)

    # --- 2. Prepare text input ---
    df["text_features"] = (
        df["name_track"] + " " + df["artists"] + " " + df["name_artist"] + " " + df["genres"]
    )

    # --- 3. Extract numerical features ---
    numeric_cols = [
        "danceability", "energy", "loudness", "speechiness", "acousticness",
        "instrumentalness", "liveness", "valence", "tempo", "duration_ms",
        "followers", "popularity_artist", "popularity_track"
    ]
    X_numeric = df[numeric_cols].fillna(0).values
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X_numeric)

    # --- 4. Generate text embeddings ---
    print("🔍 Generating text embeddings (this may take a few minutes)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # lightweight, accurate
    X_text = model.encode(df["text_features"].tolist(), batch_size=256, show_progress_bar=True)
    X_text = csr_matrix(X_text)

    # --- 5. Combine numeric + text embeddings ---
    X_numeric_sparse = csr_matrix(X_numeric)
    X_combined = hstack([X_numeric_sparse, X_text])

    # --- 6. Save the feature matrix ---
    from scipy import sparse
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    sparse.save_npz(save_path, X_combined)
    print(f"✅ Saved feature matrix with shape {X_combined.shape} to {save_path}")

    return X_combined, df


if __name__ == "__main__":
    X, df = create_feature_matrix()
