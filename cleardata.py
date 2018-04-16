# %matplotlib inline
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
amd = pd.read_csv('movies_metadata.csv',encoding='utf8') # Doc du lieu vao dataframe
# print(list(amd))
print(amd.shape)
#print(md.head(5))
md = amd[(amd['vote_average'])>=5]
md = md[(md['vote_count'])>=100]
print('Newdata')
print(md.shape)
md['genres'] = md['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
#tong so vote
vote_counts = md[md['vote_count'].notnull()]['vote_count'].astype('int')
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
print('top Comedy')
print(build_chart('Comedy').head(15))
# print('Top romance')
# print(build_chart('Romance'))

#################### END of simple recommender
##################### begin content based recommender
############## Su dung do giong nhau ve noi dung trong doan mo ta ngan ve phim
# links_small = pd.read_csv('links_small.csv')
# links_small = links_small[links_small['tmdbId'].notnull()]['tmdbId'].astype('int')
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
# print(cosine_sim[0])
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
print(get_recommendations('Apollo 13').head(20))
#### end of contentbased
#
reader = Reader()
ratings = pd.read_csv('ratings.csv')
sm_ratings = pd.read_csv('ratings_small.csv')
print(ratings.shape)
print(sm_ratings.shape)
ratings = ratings[(ratings['rating'])>=4]
sm_ratings = sm_ratings[(sm_ratings['rating'])>=3]
print(ratings.shape)
print(sm_ratings.shape)
# data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
# data.split(n_folds=5)
# trainset = data.build_full_trainset()
# svd.train(trainset)
# ratings[ratings['userId'] == 1]
