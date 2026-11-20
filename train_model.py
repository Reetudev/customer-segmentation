import pandas as pd
from sklearn.cluster import KMeans
import joblib

# Load dataset
df = pd.read_csv("dataset/Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Create KMeans model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

# Train model
kmeans.fit(X)

# Save model
joblib.dump(kmeans, "kmeans_model.pkl")

print("✅ KMeans Model Saved Successfully")