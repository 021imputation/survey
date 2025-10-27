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
                 mean_preds, cluster_mean_preds, knn_preds, cluster_knn_preds,
                 #mice_preds,
                 #random_forest_preds,
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
        #mice_preds,
        #random_forest_preds,
        true_values
    )
import pandas as pd
import time

def build_results_df(
    values, true_values,
    mean_preds, cluster_mean_preds, knn_preds, cluster_knn_preds,
    annotator_id, seed, dataset_name, na_fraction, projection,
    MAE, RMSE
):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    idx = pd.Index(list(values.keys()), name="point no")
    df = pd.DataFrame({"point no": idx})

    df["true_values"] = pd.Series(true_values, index=idx).to_numpy()
    df["annotator predicted value"]   = idx.map(values)
    df["mean_preds"]                  = np.asarray(mean_preds)
    df["cluster_mean_preds"]          = np.asarray(cluster_mean_preds)
    df["knn_preds"]                   = np.asarray(knn_preds)
    df["cluster_knn_preds"]           = np.asarray(cluster_knn_preds)

    df["data"]          = ts
    df["annotator_id"]  = annotator_id
    df["seed"]          = seed
    df["na_fraction"]   = na_fraction
    df["projection"]    = projection

    for c in ["id_empty1","id_empty2","id_empty3",
              "method_empty1","method_empty2","method_empty3","method_empty4","method_empty5",
              "MAE_empty1","MAE_empty2","MAE_empty3","MAE_empty4","MAE_empty5",
              "RMSE_empty1","RMSE_empty2","RMSE_empty3","RMSE_empty4","RMSE_empty5"]:
        df[c] = ""

    df["MAE annotator predicted value"] = MAE[0]
    df["MAE mean_preds"]                = MAE[1]
    df["MAE cluster_mean_preds"]        = MAE[2]
    df["MAE knn_preds"]                 = MAE[3]
    df["MAE cluster_knn_preds"]         = MAE[4]

    df["RMSE annotator predicted value"] = RMSE[0]
    df["RMSE mean_preds"]                = RMSE[1]
    df["RMSE cluster_mean_preds"]        = RMSE[2]
    df["RMSE knn_preds"]                 = RMSE[3]
    df["RMSE cluster_knn_preds"]         = RMSE[4]
    cols = [
        'data','annotator_id','seed','na_fraction','projection','point no',
        'true_values','id_empty1','id_empty2','id_empty3',
        'annotator predicted value','mean_preds','cluster_mean_preds','knn_preds','cluster_knn_preds',
        'method_empty1','method_empty2','method_empty3','method_empty4','method_empty5',
        'MAE annotator predicted value','MAE mean_preds','MAE cluster_mean_preds','MAE knn_preds','MAE cluster_knn_preds',
        'MAE_empty1','MAE_empty2','MAE_empty3','MAE_empty4','MAE_empty5',
        'RMSE annotator predicted value','RMSE mean_preds','RMSE cluster_mean_preds','RMSE knn_preds','RMSE cluster_knn_preds',
        'RMSE_empty1','RMSE_empty2','RMSE_empty3','RMSE_empty4','RMSE_empty5'
    ]
    return df[cols]

