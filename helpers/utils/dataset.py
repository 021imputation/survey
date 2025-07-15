import numpy as np
import pandas as pd
import streamlit as st
from helpers.utils.fuzzy import fuzzy_kmeans_fit_predict
from helpers.utils.imputation import impute_knn_mean
from helpers.utils.projections import projections_dict
from helpers.drive import api


@st.cache_data
def generate_incomplete_dataset(seed, name, incomplete_column, na_frac) -> pd.DataFrame:
    df = st.session_state[f'ds_{name}'].copy(deep=True)
    na_indexes = df.sample(frac=na_frac, random_state=seed).index
    df.loc[na_indexes, incomplete_column] = np.NaN
    return df


def add_projection_dimensions_to_df_incomplete(projection_key, X, df):
    X_projected = projections_dict[projection_key](X)
    df['x'] = [el[0] for el in X_projected]
    df['y'] = [el[1] for el in X_projected]
    return df

@st.cache_data
# TODO: adapt for other clustering method than k-means
def find_uncertain_y_indexes(X, n_clusters, fuzzy_certainty_thres=0.5):
    fuzzy_labels = fuzzy_kmeans_fit_predict(X, n_clusters)
    max_labels = fuzzy_labels.max(axis=1)
    filter_indexes = max_labels < fuzzy_certainty_thres
    indexes = [ind for ind, element in enumerate(filter_indexes) if element == True]
    return indexes

@st.cache_data
def find_uncertain_y_indexes(df_incomplete, dataset_settings, na_indexes):
    incomplete_column = dataset_settings['incomplete_column']
    knn = impute_knn_mean(df_incomplete, incomplete_column, na_indexes, dataset_settings['reference_columns'])
    real_values = st.session_state[f'ds_{dataset_settings["name"]}'][incomplete_column][na_indexes].tolist()
    indexes = abs(pd.Series(knn) - real_values).sort_values().index.tolist()[-15:]
    return indexes


def save_results(values, seed, dataset_name, na_fraction,projection_key,MAE,RMSE,
                 mean_preds, cluster_mean_preds, knn_preds, cluster_knn_preds,mice_preds,random_forest_preds,
                 true_values):
    api.save_results(
        st.session_state["gsheet"],
        values,
        st.session_state['id'],
        seed,
        dataset_name,
        na_fraction,
        projection_key,
        MAE,
        RMSE,
        mean_preds,
        cluster_mean_preds,
        knn_preds,
        cluster_knn_preds,
        mice_preds,
        random_forest_preds,
        true_values
    )
