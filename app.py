import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load datasets
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    return movies, ratings

movies, ratings = load_data()

# Preprocessing
movies['genres'] = movies['genres'].fillna('')

# Generate TF-IDF matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Create a reverse map of movie titles to indices
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Function to recommend movies
def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in indices:
        return ["Movie not found. Please try another title."]
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit App
st.title("Movie Recommendation System 🎥")
st.subheader("Find movies similar to your favorites!")

movie_name = st.text_input("Enter a movie title:")
if st.button("Recommend"):
    if movie_name.strip():
        recommendations = get_recommendations(movie_name)
        st.write("Here are some recommendations:")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
    else:
        st.write("Please enter a valid movie title.")
