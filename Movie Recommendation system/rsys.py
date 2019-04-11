import pandas as pd
import numpy as np

items = pd.read_csv("../../Kaggle Data/Movie Lens/ml-100k/u.item",sep='|',names=['movieid','movie title','release date','video release date',
              'IMDb URL', 'unknown','Action','Adventure','Animation',
              'Children\'s', 'Comedy', 'Crime','Documentary','Drama','Fantasy',
              'Film-Noir','Horror','Musical','Mystery','Romance' ,'Sci-Fi','Thriller','War','Western'],encoding='latin-1')
test = pd.read_csv("../../Kaggle Data/Movie Lens/ml-100k/u1.test",sep='\t',names=['userid','movieid','ratings','timestamp'],encoding='latin-1')
user = pd.read_csv("../../Kaggle Data/Movie Lens/ml-100k/u1.base",sep='\t',names=['userid','movieid','ratings','timestamp'],encoding='latin-1')


user_items_merged = user.merge(items, on = 'movieid',how='inner')
user_items_subset = user_items_merged[['movie title','userid','ratings']]

pivoted_user_items = user_items_subset.pivot_table(values='ratings', index= 'userid', columns='movie title',fill_value = 0)

def distance_metrics(vec1, vec2 , metrics):
    if metrics == 'euclidean':
        result = np.sqrt(np.sum((vec1-vec2)**2))
    elif metrics == 'manhattan':
        result = np.sum(np.abs(vec1-vec2))
    return result

def find_similar_users(self,movie_not_seen,k):
    movies_rated_k_users = {}
    for movie in movie_not_seen:
        user_subset_seen_movie = pivoted_user_items[pivoted_user_items[movie] > 0]
        similarity_matrix = []
        for other in user_subset_seen_movie.index:
            similarity_matrix.append((distance_metrics(pivoted_user_items.loc[self],pivoted_user_items.loc[other],'euclidean'),other))
        movies_rated_k_users[movie] = np.mean([[pivoted_user_items.loc[users[1],movie] for users in sorted(similarity_matrix)][1:k+1]])
    return movies_rated_k_users

similar_users = {}
for self in pivoted_user_items.index:
    movie_not_seen = pivoted_user_items.columns[pivoted_user_items.loc[self,pivoted_user_items.columns] == 0]
    similar_users[self] = find_similar_users(self,movie_not_seen,10)
    print("Ratings for user {} predicted".format(self))


# k most similar users for every movie not seen by each user.
predictions = pd.DataFrame(similar_users).fillna(0).T

denom = 0
num = 0 
for test_user in predictions.index:
    for test_movie in predictions.columns:
        r = 1
        x = predictions.loc[test_user, test_movie]
        try:
            y = pivoted_user_test.loc[test_user,test_movie]
        except KeyError:
            r = 0
        denom += r
        num += r * np.abs(x-y)
print(num/denom)
