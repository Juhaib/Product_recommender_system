#!/usr/bin/env python
# coding: utf-8

# # EDA & Visualisation

# In[2]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")


# In[4]:


# Load the dataset
df = pd.read_csv('amazon.csv')

# Display the first few rows of the dataset
df.head()


# In[6]:


# Display basic information about the dataset
df.info()


# In[8]:


# Display basic information about the dataset
df.info()


# In[10]:


# Display summary statistics for numerical columns
df.describe(include='all')  # include='all' to get stats for categorical columns as well


# In[12]:


# Check for missing values
missing_values = df.isnull().sum()
missing_values[missing_values > 0]  # Display only columns with missing values


# In[14]:


# Visualize the distribution of ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['rating'], bins=20, kde=True, color='blue')
plt.title('Distribution of Product Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[16]:


# Convert 'discounted_price' to numeric by removing currency symbols and commas
df['discounted_price'] = df['discounted_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)

# Visualize the distribution of discounted prices
plt.figure(figsize=(10, 6))
sns.histplot(df['discounted_price'], bins=20, kde=True, color='green')
plt.title('Distribution of Discounted Prices')
plt.xlabel('Discounted Price (₹)')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[18]:


# Convert 'actual_price' to numeric
df['actual_price'] = df['actual_price'].replace({'₹': '', ',': ''}, regex=True).astype(float)

# Visualize the relationship between actual price and discounted price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['actual_price'], y=df['discounted_price'], alpha=0.6)
plt.title('Actual Price vs Discounted Price')
plt.xlabel('Actual Price (₹)')
plt.ylabel('Discounted Price (₹)')
plt.grid()
plt.show()


# In[20]:


# Clean the 'rating' column and convert to numeric
# Replace any non-numeric characters and convert to float
df['rating'] = df['rating'].replace({'[^0-9.]': ''}, regex=True)

# Convert to numeric, forcing errors to NaN
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Check for any NaN values after conversion
print("Number of NaN values in 'rating' after conversion:", df['rating'].isnull().sum())


# In[22]:


# Check if the 'category' and 'rating' columns exist and are not empty
if 'category' in df.columns and 'rating' in df.columns:
    # Group by category and calculate the mean rating
    category_ratings = df.groupby('category')['rating'].mean().sort_values(ascending=False)

    # Check if category_ratings is not empty
    if not category_ratings.empty:
        plt.figure(figsize=(12, 8))
        sns.barplot(x=category_ratings.values, y=category_ratings.index, palette='viridis')
        plt.title('Average Rating by Product Category')
        plt.xlabel('Average Rating')
        plt.ylabel('Category')
        plt.grid()
        plt.show()
    else:
        print("No ratings found for any category.")
else:
    print("The required columns 'category' or 'rating' are missing from the dataset.")


# In[24]:


# Check unique values in the 'rating_count' column
print(df['rating_count'].unique())


# In[26]:


# Clean the 'rating_count' column and convert to numeric
# Replace any non-numeric characters and convert to integer
df['rating_count'] = df['rating_count'].replace({',': ''}, regex=True)  # Remove commas

# Convert to numeric, forcing errors to NaN
df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce')

# Check for any NaN values after conversion
print("Number of NaN values in 'rating_count' after conversion:", df['rating_count'].isnull().sum())


# In[28]:


# Visualize the number of reviews per product
plt.figure(figsize=(10, 6))
sns.histplot(df['rating_count'].dropna(), bins=30, kde=True, color='orange')  # Drop NaN values for visualization
plt.title('Distribution of Number of Reviews per Product')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')
plt.grid()
plt.show()


# In[30]:


# Select only numeric columns for correlation
numeric_df = df.select_dtypes(include=['number'])

# Check the numeric DataFrame
print(numeric_df.head())


# In[32]:


# Visualize the correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_df.corr()  # Use the numeric DataFrame
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()


# In[34]:


# Boxplot to visualize the distribution of ratings across different categories
plt.figure(figsize=(12, 8))
sns.boxplot(x='rating', y='category', data=df, palette='Set2')
plt.title('Boxplot of Ratings by Category')
plt.xlabel('Rating')
plt.ylabel('Category')
plt.grid()
plt.show()


# In[36]:


# Count the number of products in each category
plt.figure(figsize=(12, 6))
category_counts = df['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values, palette='pastel')  # Corrected 'past' to 'pastel'
plt.title('Number of Products by Category')
plt.xlabel('Category')
plt.ylabel('Number of Products')
plt.xticks(rotation=45, ha='right')  # Rotate x labels for better readability
plt.grid()
plt.show()


# ## Model Building and Model Evolution

# In[39]:


# General libraries
import pandas as pd
import numpy as np

# Recommendation libraries
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Evaluation
from sklearn.metrics import mean_squared_error


# In[41]:


# Convert 'rating' column to numeric, replacing errors with NaN
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Check and handle missing values in the 'rating' column
df['rating'].fillna(df['rating'].mean(), inplace=True)

# Create the user-item interaction matrix
user_item_matrix = df.pivot_table(index='user_id', columns='product_id', values='rating', aggfunc='mean').fillna(0)

# Verify the result
print(user_item_matrix.head())


# In[43]:


# Combine textual features (e.g., product name and about_product)
df['text_features'] = df['product_name'] + " " + df['about_product']

# Vectorize text features using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['text_features'])

# Compute cosine similarity between products
cosine_sim = cosine_similarity(tfidf_matrix)


# In[45]:


# Collaborative filtering recommendation function
def recommend_collaborative(user_id, user_similarity, user_item_matrix, n_recommendations=5):
    # Get the index of the given user ID
    user_index = user_item_matrix.index.get_loc(user_id)
    
    # Get similar users, sorted by similarity score
    similar_users = np.argsort(-user_similarity[user_index])[1:]  # Exclude self
    
    # Initialize recommendations list
    recommendations = []
    
    # Iterate over similar users to find products
    for sim_user in similar_users:
        # Get products rated by the similar user
        user_products = user_item_matrix.iloc[sim_user, :].sort_values(ascending=False).index
        recommendations.extend([prod for prod in user_products if prod not in recommendations])
        
        # Stop once we have enough recommendations
        if len(recommendations) >= n_recommendations:
            break
    
    return recommendations[:n_recommendations]


# In[47]:


def recommend_content_based(product_id, cosine_sim, df, n_recommendations=5):
    product_idx = df[df['product_id'] == product_id].index[0]
    similar_indices = np.argsort(-cosine_sim[product_idx])[1:]  # Exclude self
    
    recommendations = df.iloc[similar_indices[:n_recommendations]]['product_name']
    return recommendations

# Example: Get recommendations for a specific product
recommendations = recommend_content_based('B07JW9H4J1', cosine_sim, df)
print("Recommendations:", recommendations)


# In[51]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Sample user-item interaction matrix
user_item_matrix = np.array([[5, 0, 0, 1],
                              [4, 0, 0, 1],
                              [0, 0, 5, 0],
                              [0, 3, 4, 0]])

# Calculate user similarity
user_similarity = cosine_similarity(user_item_matrix)

# Define the recommendation function
def recommend_collaborative(user_id, user_similarity, user_item_matrix):
    # Your recommendation logic here
    # This is just a placeholder
    return ["Item1", "Item2", "Item3"]

# Test the recommendation function for a specific user
test_user_id = 0  # Assuming user IDs are indexed from 0
recommendations = recommend_collaborative(test_user_id, user_similarity, user_item_matrix)

# Display recommendations
print("Recommendations for user", test_user_id, ":", recommendations)


# In[55]:


def hybrid_recommendation(user_id, product_id, user_similarity, cosine_sim, user_item_matrix, df, n_recommendations=5):
    # Call with the correct number of arguments
    collab_recommendations = recommend_collaborative(user_id, user_similarity, user_item_matrix)
    content_recommendations = recommend_content_based(product_id, cosine_sim, df, n_recommendations)
    
    # Combine and prioritize
    hybrid = pd.Series(collab_recommendations + list(content_recommendations))
    hybrid = hybrid.value_counts().index[:n_recommendations]
    return hybrid

# Example: Get hybrid recommendations
recommendations = hybrid_recommendation('AG3D6O4STAQKAY2UVGEUV46KN35Q', 'B07JW9H4J1', user_similarity, cosine_sim, user_item_matrix, df)
print("Hybrid Recommendations:", recommendations)


# In[69]:


import numpy as np
from sklearn.metrics import mean_squared_error

# Example user-item interaction matrix (as a NumPy array)
user_item_matrix = np.array([[5, 0, 0, 1],
                              [4, 0, 0, 1],
                              [0, 0, 5, 0],
                              [0, 3, 4, 0]])

# Example user similarity matrix (as a NumPy array)
user_similarity = np.array([[1, 0.5, 0, 0],
                             [0.5, 1, 0, 0],
                             [0, 0, 1, 0.5],
                             [0, 0, 0.5, 1]])

# Example user-item array (as a NumPy array)
user_item_array = user_item_matrix  # Assuming this is the same as user_item_matrix for simplicity

# Example of evaluation using known ratings
actual_ratings = user_item_matrix[0, :]  # Accessing the first row directly
predicted_ratings = user_similarity[0, :].dot(user_item_array) / np.sum(user_similarity[0, :])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print("Collaborative Filtering RMSE:", rmse)


# In[71]:


# Compute relevance score (example)
def evaluate_content_based(product_id, recommendations, df):
    product_category = df[df['product_id'] == product_id]['category'].iloc[0]
    relevance = [1 if df[df['product_name'] == rec]['category'].iloc[0] == product_category else 0 for rec in recommendations]
    return sum(relevance) / len(recommendations)

# Example evaluation
product_id = 'B07JW9H4J1'
recommendations = recommend_content_based(product_id, cosine_sim, df)
relevance_score = evaluate_content_based(product_id, recommendations, df)
print("Content-Based Filtering Relevance Score:", relevance_score)


# In[ ]:




