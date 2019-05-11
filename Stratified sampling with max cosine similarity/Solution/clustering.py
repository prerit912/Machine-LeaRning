import pandas as pd
import numpy as np
from scipy import spatial
from collections import Counter


def cosine(sample,population):
    return 1 - spatial.distance.cosine(sample,population)
    
def find_proportions(sample_size = 64):
    data = pd.read_csv('../store_cluster_assignments.csv')
    #data = read_data()
    unique_clusters = data['ClusterId'].unique()
    # Group by the ClusterId and count.
    
    data_group_clusters = data.groupby('ClusterId').size().sort_values()
    clusters = data_group_clusters.index
    proportions_of_stores = data_group_clusters.values/np.sum(data_group_clusters)
    proportions_of_stores = np.insert(proportions_of_stores,0,0,axis=0)
    bins = np.cumsum(proportions_of_stores)
    random_numbers = np.random.random(sample_size)
    random_samples = pd.cut(random_numbers,bins,include_lowest = True, right=False,labels=unique_clusters)
    sample_counts = Counter(random_samples)
    dict_sample_counts = dict(sample_counts)
    sample_dataframe = pd.DataFrame(list(dict_sample_counts.values()),index = dict_sample_counts.keys())
    data_group_clusters_df = pd.DataFrame(data_group_clusters)
    joined_dataframe = data_group_clusters_df.merge(sample_dataframe, how = 'left'
                                                    , left_index = True, right_index = True)
    joined_dataframe_rm_na = joined_dataframe.fillna(0)
    print(cosine(joined_dataframe_rm_na['0_x'],joined_dataframe_rm_na['0_y']))
    

def find_proportions2(sample_size = 64):
    data = pd.read_csv('../store_cluster_assignments.csv')
    #data = read_data()
    unique_clusters = data['ClusterId'].unique()
    # Group by the ClusterId and count.
    
    data_group_clusters = data.groupby('ClusterId').size().sort_values()
    clusters = data_group_clusters.index
    proportions_of_stores = data_group_clusters.values/np.sum(data_group_clusters)
    proportions_of_stores = np.insert(proportions_of_stores,0,0,axis=0)
    
find_proportions()
    

#x = [25,25,10,40]
#y = [5,5,2,8]
