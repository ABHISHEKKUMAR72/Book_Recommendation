import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load datasets
books = pd.read_csv('books.csv', low_memory=False)
users = pd.read_csv('users.csv')
ratings = pd.read_csv('ratings.csv')

# Data inspection
print("Books shape:", books.shape)
print("Ratings shape:", ratings.shape)
print("Users shape:", users.shape)
print("Books missing values:\n", books.isnull().sum())
print("Users missing values:\n", users.isnull().sum())
print("Ratings missing values:\n", ratings.isnull().sum())
print("Books duplicated:", books.duplicated().sum())
print("Ratings duplicated:", ratings.duplicated().sum())
print("Users duplicated:", users.duplicated().sum())

# Convert 'Book-Rating' to numeric and handle errors
ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')
ratings = ratings.dropna(subset=['Book-Rating'])
ratings['Book-Rating'] = ratings['Book-Rating'].astype(int)

# Merge ratings with book details
ratings_with_books = ratings.merge(books, on='ISBN', how='inner')

# Calculate the number of ratings per book
num_rating_df = (
    ratings_with_books.groupby('Book-Title')
    .count()['Book-Rating']
    .reset_index()
    .rename(columns={'Book-Rating': 'num_ratings'})
)

# Calculate the average rating for each book
avg_rating_df = (
    ratings_with_books.groupby('Book-Title')
    .mean(numeric_only=True)['Book-Rating']
    .reset_index()
    .rename(columns={'Book-Rating': 'avg_rating'})
)

# Merge number of ratings and average ratings
popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title', how='inner')
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

print("Popular books sample:\n", popular_df.head())

# Filtering users with more than 200 ratings
x = ratings_with_books.groupby('User-ID').count()['Book-Rating'] > 200
padhe_likhe_users = x[x].index
filtered_rating = ratings_with_books[ratings_with_books['User-ID'].isin(padhe_likhe_users)]

# Filtering books with more than 50 ratings
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]

# Create a pivot table
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Compute cosine similarity
similarity_scores = cosine_similarity(pt)
print("Similarity scores shape:", similarity_scores.shape)

# Recommendation function
def recommend(book_name):
    if book_name not in pt.index:
        return f"Book '{book_name}' not found in dataset."
    
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    
    return data

# Example recommendation
print(recommend('1984'))

# Save required data using pickle
pickle.dump(popular_df, open('popular.pkl', 'wb'))
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))
