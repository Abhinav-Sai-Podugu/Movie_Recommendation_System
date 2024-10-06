import streamlit as st
import pickle
import pandas as pd
import numpy as np
import requests


def fetch_poster(movie_id):
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key=8265bd1679663a7ea12ac168da84d2e8&language=en-US'.format(movie_id))
    data = response.json()

    poster_path = data.get('poster_path')
    if poster_path:
        return "http://image.tmdb.org/t/p/w500/" + data['poster_path']
    else :
        return "https://upload.wikimedia.org/wikipedia/commons/6/65/No-Image-Placeholder.svg"


def recommend(movie):
    id = movies[movies['title'] == movie].index[0]
    distance = similarity1[id]
    movies_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:11]

    recommended_movies = []
    recommended_poster = []
    recommended_id = []
    for i in movies_list:
        movie_id = movies.iloc[i[0]].movie_id

        recommended_movies.append(movies.iloc[i[0]].title)
        recommended_poster.append(fetch_poster(movie_id))
        recommended_id.append(movies.iloc[i[0]].movie_id)
    return recommended_movies, recommended_poster, recommended_id

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
with open('similarity_part1.pkl', 'rb') as f1:
    similarity_part1 = pickle.load(f1)

with open('similarity_part2.pkl', 'rb') as f2:
    similarity_part2 = pickle.load(f2)

similarity1 = np.vstack((similarity_part1, similarity_part2))

st.title('Movie Recommender System')

option = st.selectbox(
    "What is your favorite movie?",
    movies['title'].values)


if st.button("Recommendations"):
    names, posters, ids = recommend(option)

    cols = st.columns(5, vertical_alignment="bottom")

    for i in range(2):
        for j in range(5):
            with cols[j]:
                name = names[(i*5)+j]
                id = ids[(i*5)+j]
                name1 = name.lower().replace(" ", "-")

                st.markdown(f"[{name}](https://www.themoviedb.org/movie/{id}-{name1})")
                st.image(posters[(i*5)+j])