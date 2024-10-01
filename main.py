import numpy as np
import pandas as pd
import ast
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
import pickle

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

movies = movies.merge(credits, on = 'title')
movies = movies[['movie_id', 'title', 'genres', 'keywords', 'overview', 'production_companies', 'vote_average', 'vote_count', 'crew', 'cast']]
print(movies.head())

movies.isnull().sum()

movies.dropna(inplace = True)
movies.duplicated().sum()

def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['production_companies'] = movies['production_companies'].apply(convert)

def convert1(obj):
    L = []
    c = 0
    for i in ast.literal_eval(obj):
        if c != 3:
            L.append(i['name'])
            c+=1
        else:
            break
    return L

movies['cast'] = movies['cast'].apply(convert1)
def convert2(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(convert2)
movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['overview'] = movies['overview'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ", "") for i in x])

movies['final'] = movies['overview'] + movies['keywords'] + movies['genres'] + movies['production_companies'] + movies['cast'] + movies['crew']
dataset = movies[['movie_id', 'title', 'final', 'vote_average', 'vote_count']]
dataset['final'] = dataset['final'].apply(lambda x:" ".join(x))

dataset['final'] = dataset['final'].apply(lambda x:x.lower())


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

dataset['final'] = dataset['final'].apply(stem)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')

vectors = cv.fit_transform(dataset['final']).toarray()

print(cv.get_feature_names_out())

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vectors)

def recommend(movie):
    if movie not in dataset['title'].values:
        return []

    id = dataset[dataset['title'] == movie].index[0]

    if id >= similarity.shape[0]:
        return []

    distance = similarity[id]
    movies_list = sorted(list(enumerate(distance)), reverse = True, key = lambda x:x[1])[1:6]

    for i in movies_list:
        print(dataset.iloc[i[0]].title)

recommend('My Date with Drew')


pickle.dump(dataset.to_dict(), open('movie_dict.pkl', 'wb'))

half_size = len(similarity) // 2

similarity_part1 = similarity[:half_size, :]
similarity_part2 = similarity[half_size:, :]

with open('similarity_part1.pkl', 'wb') as f1:
    pickle.dump(similarity_part1, f1)

with open('similarity_part2.pkl', 'wb') as f2:
    pickle.dump(similarity_part2, f2)
