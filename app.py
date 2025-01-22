# from flask import Flask, request, render_template, jsonify
# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer
# import os

# # Initialize Flask app
# app = Flask(__name__)

# # Load dataset dynamically
# data_path = os.path.join(os.path.dirname(__file__), 'data', 'amazon.csv')
# df = pd.read_csv(data_path)

# # Data Cleaning and Preprocessing
# # Clean 'rating' column
# df['rating'] = df['rating'].replace(r'[^0-9.]', '', regex=True)
# df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
# df['rating'].fillna(df['rating'].mean(), inplace=True)

# # Clean 'discounted_price' column
# df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)

# # Clean 'actual_price' column
# df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)

# # Create text features for recommendation
# df['text_features'] = df['product_name'] + " " + df['about_product']

# # Vectorize text features using TF-IDF
# tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
# tfidf_matrix = tfidf.fit_transform(df['text_features'])

# # Compute cosine similarity
# cosine_sim = cosine_similarity(tfidf_matrix)

# # Content-based recommendation function
# def recommend_content_based(product_id, n_recommendations=5):
#     try:
#         product_idx = df[df['product_id'] == product_id].index[0]
#         similar_indices = np.argsort(-cosine_sim[product_idx])[1:]  # Exclude self
#         recommendations = df.iloc[similar_indices[:n_recommendations]]['product_name']
#         return recommendations.tolist()
#     except IndexError:
#         return []  # Return an empty list if product_id is invalid

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     try:
#         # Handle JSON requests from JavaScript or form submissions
#         if request.is_json:
#             product_id = request.json.get('product_id')
#         else:
#             product_id = request.form.get('product_id')

#         # Validate the product ID
#         if not product_id or product_id not in df['product_id'].values:
#             return jsonify({'recommendations': []})

#         # Get recommendations
#         recommendations = recommend_content_based(product_id)
#         return jsonify({'recommendations': recommendations})
#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({'error': 'Something went wrong'}), 500

# if __name__ == '__main__':
#     app.run(debug=True)


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Load dataset dynamically
@st.cache_data  # Cache the dataset for better performance
def load_data():
    try:
        data_path = os.path.join(os.path.dirname(__file__), 'data', 'amazon.csv')
        df = pd.read_csv(data_path)

        # Data Cleaning and Preprocessing
        df['rating'] = df['rating'].replace(r'[^0-9.]', '', regex=True)
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        df['rating'].fillna(df['rating'].mean(), inplace=True)
        df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
        df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)
        df['text_features'] = df['product_name'] + " " + df['about_product']
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

# Generate TF-IDF matrix and cosine similarity
@st.cache_data  # Cache the computation for faster reloads
def compute_similarity(data):
    try:
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = tfidf.fit_transform(data['text_features'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        return cosine_sim
    except Exception as e:
        st.error(f"Error computing similarity: {e}")
        return None

if df is not None:
    cosine_sim = compute_similarity(df)

# Recommendation Function
def recommend_content_based(product_id, n_recommendations=5):
    if df is None or cosine_sim is None:
        return []
    try:
        product_idx = df[df['product_id'] == product_id].index[0]
        similar_indices = np.argsort(-cosine_sim[product_idx])[1:]  # Exclude self
        recommendations = df.iloc[similar_indices[:n_recommendations]]['product_name']
        return recommendations.tolist()
    except IndexError:
        return []

# Streamlit Interface
st.title("Product Recommendation System")
st.write("This application provides recommendations based on content similarity.")

if df is None:
    st.error("Dataset not loaded. Please check the 'data/amazon.csv' file.")
else:
    # Input for Product ID
    product_id = st.text_input("Enter a Product ID", key="product_id")
    if st.button("Get Recommendations"):
        if product_id.strip() and product_id in df['product_id'].values:
            recommendations = recommend_content_based(product_id.strip())
            if recommendations:
                st.subheader("Top Recommendations:")
                for i, rec in enumerate(recommendations, 1):
                    st.write(f"{i}. {rec}")
            else:
                st.warning("No recommendations found for the given Product ID.")
        else:
            st.error("Invalid Product ID. Please enter a valid one from the dataset.")
