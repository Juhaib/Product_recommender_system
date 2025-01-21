from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv('./data/amazon.csv')

# Data Cleaning and Preprocessing
# Clean 'rating' column
df['rating'] = df['rating'].replace(r'[^0-9.]', '', regex=True)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['rating'].fillna(df['rating'].mean(), inplace=True)

# Clean 'discounted_price' column
df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)

# Clean 'actual_price' column
df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)

# Create text features for recommendation
df['text_features'] = df['product_name'] + " " + df['about_product']

# Vectorize text features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix)

# Content-based recommendation function
def recommend_content_based(product_id, n_recommendations=5):
    product_idx = df[df['product_id'] == product_id].index[0]
    similar_indices = np.argsort(-cosine_sim[product_idx])[1:]  # Exclude self
    recommendations = df.iloc[similar_indices[:n_recommendations]]['product_name']
    return recommendations.tolist()

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/recommend', methods=['POST'])
# def recommend():
#     product_id = request.form['product_id']
#     recommendations = recommend_content_based(product_id)
#     return jsonify({'recommendations': recommendations})

@app.route('/recommend', methods=['POST'])
def recommend():
    # Handle JSON requests from JavaScript or form submissions
    if request.is_json:
        product_id = request.json.get('product_id')
    else:
        product_id = request.form.get('product_id')

    # Validate the product ID
    if not product_id or product_id not in df['product_id'].values:
        return jsonify({'recommendations': []})

    # Get recommendations
    recommendations = recommend_content_based(product_id)
    return jsonify({'recommendations': recommendations})

print(df['product_id'].unique())

if __name__ == '__main__':
    app.run(debug=True)
