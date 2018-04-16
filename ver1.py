# %matplotlib inline
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')
pd.options.display.encoding = sys.stdout.encoding
start = time.time()
md = pd.read_csv('movies_metadata1.csv',encoding='utf8') # Doc du lieu vao dataframe
# print(list(amd))
##############Clear Data
# rc50 = amd[(amd['vote_count'])<50]
# rc50 = rc50[(rc50['vote_average'])>=6]
# rc50 = rc50[(rc50['vote_count'])>=20]
# # rc50 = rc50['title']
# # rc50.to_csv('ratingcountless50andmore6.4ratemean.csv', sep=',', encoding='utf-8')
# print(rc50.shape)
# print(amd[amd['vote_count']>=50].count())
# print(amd.shape)

# #print(md.head(5))
# md = amd[(amd['vote_average'])>=5]
# md = md[(md['vote_count'])>=50]
# print(md.count())
# frames=[md,rc50]
# md=pd.concat(frames)
# print('Newdata')
# md = md.drop(md.columns[0], axis=1)
# md.loc[:, ~md.columns.str.contains('Unnamed')]
print(md.shape)
# md.to_csv('movies_metadata2.csv', sep=',', encoding='utf-8')
# print(md.head(4))
# quit()
##############convert du lieu tu string sang float
# md[['vote_count','vote_average']] = md[['vote_count','vote_average']].apply(pd.to_numeric)
# md['vote_count'] = md['vote_count'].astype(float)
md['vote_count'] = pd.to_numeric(md['vote_count'], errors='coerce')
md['vote_average'] = pd.to_numeric(md['vote_average'], errors='coerce')
md['id'] = pd.to_numeric(md['id'], errors='coerce')
# quit()
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
#tong so vote
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
# print(md['vote_count'].head(50))
D = vote_counts.mean()#so vote trung binh
# print(D)
#tong so luot vote
vote_averages = md[md['vote_average'].notnull()]['vote_average'].astype('int')
C = vote_averages.mean()#diem vote trung binh
print(C)
m = vote_counts.quantile(0.95)
print(m)
md['year'] = pd.to_datetime(md['release_date'], errors='coerce').apply(lambda x: str(x).split('-')[0] if x != np.nan else np.nan)
qualified = md[(md['vote_count'] >= m) & (md['vote_count'].notnull()) & (md['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity', 'genres']]

qualified['vote_count'] = qualified['vote_count'].astype('int') 

qualified['vote_average'] = qualified['vote_average'].astype('int')
print(qualified.shape)
#Weighted Rating (WR) =  (v/(v+m).R)+(m/(v+m).C)
#v is the number of vote for the movie
#m is the minimum votes required to be listed in the chart
#R is the average rating of the movie
#C is the mean vote across the whole report
def weighted_rating(x):
    v = x['vote_count']
    R = x['vote_average']
    return (v/(v+m) * R) + (m/(m+v) * C)
qualified['wr'] = qualified.apply(weighted_rating, axis=1)
qualified = qualified.sort_values('wr', ascending=False).head(250)
# print(qualified.head(20))

s = md.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_md = md.drop('genres', axis=1).join(s)

def build_chart(genre, percentile=0.85):
    df = gen_md[gen_md['genre'] == genre]
    vote_counts = df[df['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = df[df['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(percentile)
    
    qualified = df[(df['vote_count'] >= m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['title', 'year', 'vote_count', 'vote_average', 'popularity']]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average']
    
    qualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(250)
    
    return qualified
# print('top Comedy')
# print(build_chart('Comedy').head(15))
# print('Top romance')
# print(build_chart('Romance'))

#################### END of simple recommender
##################### begin content based recommender
############## Su dung do giong nhau ve noi dung trong doan mo ta ngan ve phim
links_small = pd.read_csv('links.csv')
links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
# print(md.head(5))
# md = md.drop([1,2,3])
# print(md.head(5))
# md = md.drop([19730, 29503, 35587])
#Check EDA Notebook for how and why I got these indices.
md['id'] = md['id'].astype('int')
# smd = md[md['id'].isin(links_small)]
smd = md
print(smd.shape)
smd['tagline'] = smd['tagline'].fillna('')
smd['description'] = smd['overview'] + smd['tagline']
smd['description'] = smd['description'].fillna('')
tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(smd['description']) #https://viblo.asia/p/trich-chon-thuoc-tinh-trong-doan-van-ban-voi-tf-idf-Az45bAOqlxY
print(tfidf_matrix.shape)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_sim[0])
########### lay ve 30 phim co diem gan nhat
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    return titles.iloc[movie_indices]
# print(get_recommendations('Grown Ups 2').head(15))
#### end of contentbased
########### begin metadata recommender
credits = pd.read_csv('credits.csv')
keywords = pd.read_csv('keywords.csv')
keywords['id'] = keywords['id'].astype('int')
credits['id'] = credits['id'].astype('int')
md['id'] = md['id'].astype('int')
md = md.merge(credits, on='id')
md = md.merge(keywords, on='id')
smd = md[md['id'].isin(links_small)]
print("day la shape cuar megre credit")
print(smd.shape)
smd['cast'] = smd['cast'].apply(literal_eval)
smd['crew'] = smd['crew'].apply(literal_eval)
smd['keywords'] = smd['keywords'].apply(literal_eval)
smd['cast_size'] = smd['cast'].apply(lambda x: len(x))
smd['crew_size'] = smd['crew'].apply(lambda x: len(x))

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan
smd['director'] = smd['crew'].apply(get_director)
smd['cast'] = smd['cast'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: x[:3] if len(x) >=3 else x)
smd['keywords'] = smd['keywords'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
smd['cast'] = smd['cast'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['director'] = smd['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))
smd['director'] = smd['director'].apply(lambda x: [x,x, x])
s = smd.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'
s = s.value_counts()
# print(s[:5])
s = s[s > 1]
# Loc cac tu khoa dong nghia
stemmer = SnowballStemmer('english') 
# print(stemmer.stem('dogs'))
# print(111111111111111)
def filter_keywords(x):
    words = []
    for i in x:
        if i in s:
            words.append(i)
    return words
smd['keywords'] = smd['keywords'].apply(filter_keywords)
smd['keywords'] = smd['keywords'].apply(lambda x: [stemmer.stem(i) for i in x])
smd['keywords'] = smd['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])
smd['soup'] = smd['keywords'] + smd['cast'] + smd['director'] + smd['genres']
smd['soup'] = smd['soup'].apply(lambda x: ' '.join(x))
count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(smd['soup'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)
smd = smd.reset_index()
titles = smd['title']
indices = pd.Series(smd.index, index=smd['title'])
# print(get_recommendations('Grown Ups 2').head(10))
def improved_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year']]
    vote_counts = movies[movies['vote_count'].notnull()]['vote_count'].astype('int')
    vote_averages = movies[movies['vote_average'].notnull()]['vote_average'].astype('int')
    C = vote_averages.mean()
    m = vote_counts.quantile(0.60)
    qualified = movies[(movies['vote_count'] >= m) & (movies['vote_count'].notnull()) & (movies['vote_average'].notnull())]
    qualified['vote_count'] = qualified['vote_count'].astype('int')
    qualified['vote_average'] = qualified['vote_average'].astype('int')
    qualified['wr'] = qualified.apply(weighted_rating, axis=1)
    qualified = qualified.sort_values('wr', ascending=False).head(10)
    return qualified
# print(improved_recommendations('Chungking Express'))
# print(list(improved_recommendations('Se7en')))
print("end of metadata")

#Collaborative Filtering
end = time.time()
elapsed = end - start
print(elapsed)
reader = Reader()
tmdbid = smd['id']
ratings = pd.read_csv('newratings.csv')
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
data.split(n_folds=10)
svd = SVD()
evaluate(svd, data, measures=['RMSE', 'MAE'])
trainset = data.build_full_trainset()
svd.train(trainset)
print(ratings[ratings['userId'] == 554])
print(svd.predict(554, 509, 4))
#end Collaborative Filtering
def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan
id_map = pd.read_csv('links_small.csv')[['movieId', 'tmdbId']]
id_map['tmdbId'] = id_map['tmdbId'].apply(convert_int)
id_map.columns = ['movieId', 'id']
id_map = id_map.merge(smd[['title', 'id']], on='id').set_index('title')
indices_map = id_map.set_index('id')
end = time.time()
elapsed = end - start
print(elapsed)
def hybrid(userId, title):
    idx = indices[title]
    tmdbId = id_map.loc[title]['id']
    #print(idx)
    movie_id = id_map.loc[title]['movieId']
    
    sim_scores = list(enumerate(cosine_sim[int(idx)]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:26]
    movie_indices = [i[0] for i in sim_scores]
    
    movies = smd.iloc[movie_indices][['title', 'vote_count', 'vote_average', 'year', 'id']]
    movies['est'] = movies['id'].apply(lambda x: svd.predict(userId, indices_map.loc[x]['movieId']).est)
    movies = movies.sort_values('est', ascending=False)
    return movies.head(10)
# print(hybrid(30395, 'American Pie'))
end = time.time()
elapsed = end - start
print(elapsed)
# print(type(svd))
print(hybrid(30403, 'The Thirteenth Floor'))
end = time.time()
elapsed = end - start
print(elapsed)
print(type(svd))