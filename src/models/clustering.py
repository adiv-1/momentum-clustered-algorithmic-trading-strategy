import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def apply_kmeans_clustering(data):
    target_rsi_values = [40, 55, 65, 80]
    n_features = data.shape[1]
    rsi_col_idx = data.columns.tolist().index('rsi')

    initial_centroids = np.zeros((len(target_rsi_values), n_features))
    initial_centroids[:, rsi_col_idx] = target_rsi_values

    def get_clusters(df):
        df['cluster'] = KMeans(
            n_clusters=4, random_state=0, init=initial_centroids
        ).fit(df).labels_
        return df

    data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)
    return data


def select_momentum_cluster(data, cluster_id=3):
    filtered_df = data[data['cluster'] == cluster_id].copy()
    filtered_df = filtered_df.reset_index(level=1)
    filtered_df.index = filtered_df.index + pd.DateOffset(1)
    filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

    dates = filtered_df.index.get_level_values('date').unique().tolist()
    fixed_dates = {
        d.strftime('%Y-%m-%d'): filtered_df.xs(d, level=0).index.tolist()
        for d in dates
    }
    return fixed_dates
