from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer


def weighted_average(distribution, weights):
    numerator = sum([distribution[i] * weights[i] for i in range(len(distribution))])
    denominator = sum(weights)
    return round(numerator / denominator, 2)


def impute_global_mean(df, incomplete_column, na_indexes):
    mean = df[incomplete_column].mean()
    return {idx: mean for idx in na_indexes}


def impute_knn_mean_helper(df, na_indexes, reference_columns, incomplete_column, n_neighbors=5, weight_func=lambda x: 1):
    not_na_mask = df[incomplete_column].notna()

    if len(na_indexes) > 0:
        # performing KNN for each NA record
        # neigh_distances, neigh_indexes are arays - their elements corresponds to NA records
        # each element of those arrays is also an array containing 'n_neighbors' nearest neighbors details (distances and indexes)
        # WARNING: neigh_indexes contains 'row numbers' relative to 'df_not_na' (which was fitted by 'neigh')
        # it means that neigh_indexes are numbers of rows in df_not_na, NOT indexes from original df
        # TODO: refactor this - use class not arrays in arrays to be more readable
        na_records = df[reference_columns].loc[na_indexes]
        df_not_na = df[not_na_mask][reference_columns]
        neigh = NearestNeighbors(n_neighbors=min(len(df_not_na), n_neighbors))
        neigh.fit(df_not_na)
        neigh_distances, neigh_indexes = neigh.kneighbors(na_records)

        # for ind, neigh_ind, neigh_dist in zip(na_indexes, neigh_indexes, neigh_distances):
        # getting df indexes based on neigh_indexes (see WARNING above) df_ind = df_not_na.index[neigh_ind]
        return {idx: weighted_average(list(df.loc[df_not_na.index[neigh_ind]][incomplete_column]),
                                      [weight_func(dist) for dist in neigh_dist])
                for idx, neigh_ind, neigh_dist in zip(na_indexes, neigh_indexes, neigh_distances)}


def impute_knn_mean(df, incomplete_column, na_indexes, reference_columns, n_neighbors=5, weight_func=lambda x: 1):
    return impute_knn_mean_helper(df, na_indexes, reference_columns, incomplete_column, n_neighbors, weight_func)


def impute_cluster_mean(df, incomplete_column, na_indexes):
    cluster_means = []
    number_of_clusters = df['y_pred'].max() + 1
    for label in range(number_of_clusters):
        curent_cluster_mask = df['y_pred'] == label
        cluster_means.append(df[curent_cluster_mask][incomplete_column].mean())

    return {idx: cluster_means[df['y_pred'][idx]] for idx in na_indexes}


def impute_cluster_knn_mean(df, incomplete_column, na_indexes, reference_columns, n_neighbors=5,
                            weight_func=lambda x: 1):
    values = {}
    number_of_clusters = df['y_pred'].max() + 1
    for label in range(number_of_clusters):
        current_cluster_mask = df['y_pred'] == label
        df_current_cluster = df[current_cluster_mask]

        res = impute_knn_mean_helper(df_current_cluster,
                                     list(set(na_indexes).intersection(df_current_cluster.index.tolist())),
                                     reference_columns, incomplete_column, n_neighbors, weight_func)

        if res:
            values.update(res)

    return {idx: values[idx] for idx in na_indexes}


def impute_mice(df_incomplete, incomplete_column, na_indexes, random_state, reference_columns=None, max_iter=10, precision=None):
    """
    IterativeImputer (MICE-ish) na macierzy [y|X], gdzie:
      y = incomplete_column
      X = reference_columns (tylko numeryczne)
    Zwraca dict {idx: pred} dla idx z na_indexes w tej samej kolejności.
    """
    if reference_columns is None:
        raise ValueError("impute_mice: musisz przekazać reference_columns (lista kolumn numerycznych).")

    pos = df_incomplete.index.get_indexer(na_indexes)
    if (pos < 0).any():
        raise KeyError("impute_mice: na_indexes zawiera indeksy nieobecne w df_incomplete.")

    y = pd.to_numeric(df_incomplete[incomplete_column], errors="coerce").to_numpy(dtype=float)
    X = df_incomplete[reference_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    y_missing = y.copy()
    y_missing[pos] = np.nan
    M = np.column_stack([y_missing, X])

    imp = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=int(max_iter),
        random_state=int(random_state),
        imputation_order="ascending",
        initial_strategy="mean",
        skip_complete=True,
    )
    M_imp = imp.fit_transform(M)
    preds = M_imp[pos, 0]

    if precision is not None and precision > 0:
        decimals = int(max(0, np.ceil(-np.log10(precision))))
        preds = np.round(np.round(preds / precision) * precision, decimals=decimals)

    return {idx: float(val) for idx, val in zip(na_indexes, preds)}

def impute_with_rf(df_incomplete, incomplete_column, na_indexes, random_state, reference_columns=None, n_estimators=300, n_jobs=1, precision=None):
    """
    RandomForestRegressor:
      - trenuje na wierszach gdzie y znane (poza na_indexes)
      - przewiduje dla na_indexes
    Zwraca dict {idx: pred} w tej samej kolejności co na_indexes.
    """
    if reference_columns is None:
        raise ValueError("impute_with_rf: musisz przekazać reference_columns (lista kolumn numerycznych).")

    pos = df_incomplete.index.get_indexer(na_indexes)
    if (pos < 0).any():
        raise KeyError("impute_with_rf: na_indexes zawiera indeksy nieobecne w df_incomplete.")

    y = pd.to_numeric(df_incomplete[incomplete_column], errors="coerce").to_numpy(dtype=float)
    X = df_incomplete[reference_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    train_mask = np.ones(len(df_incomplete), dtype=bool)
    train_mask[pos] = False
    train_mask &= ~np.isnan(y)
    if train_mask.sum() == 0:
        raise ValueError("impute_with_rf: brak wierszy treningowych (wszystko NaN po maskowaniu).")

    rf = RandomForestRegressor(
        n_estimators=int(n_estimators),
        random_state=int(random_state),
        n_jobs=int(n_jobs),
    )
    rf.fit(X[train_mask], y[train_mask])

    preds = rf.predict(X[pos])

    if precision is not None and precision > 0:
        decimals = int(max(0, np.ceil(-np.log10(precision))))
        preds = np.round(np.round(preds / precision) * precision, decimals=decimals)

    return {idx: float(val) for idx, val in zip(na_indexes, preds)}
