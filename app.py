import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("kmeans_model.pkl")

st.title("🛒 Customer Segmentation System")

st.write("Find which customer segment a customer belongs to.")

income = st.slider(
    "Annual Income (k$)",
    10,
    150,
    50
)

score = st.slider(
    "Spending Score (1-100)",
    1,
    100,
    50
)

if st.button("Find Customer Segment"):

    customer = pd.DataFrame(
        [[income, score]],
        columns=[
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    )

    cluster = model.predict(customer)

    segment_names = {
        0: "Premium Customers",
        1: "Budget Customers",
        2: "Regular Customers",
        3: "High Spending Customers",
        4: "New Customers"
    }

    st.success(
        f"Customer belongs to Cluster {cluster[0]}"
    )

    if cluster[0] in segment_names:
        st.info(segment_names[cluster[0]])