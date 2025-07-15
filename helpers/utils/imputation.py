from __future__ import annotations

import pandas as pd
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


def impute_mice(
        df: pd.DataFrame,
        incomplete_column: str,
        na_indexes,
        *,
        max_iter: int = 20,
        m: int = 5,
        estimator=None,
        random_state: int | None = None
) -> dict:
    """
    Imputuje jedną kolumnę metodą MICE i zwraca słownik {idx: wartość},
    gdzie wartość to średnia z m niezależnych imputacji.
    """
    if estimator is None:
        estimator = BayesianRidge()

    cat_cols = (
        df.select_dtypes(include=["object", "string", "category"])
        .columns.drop(incomplete_column, errors="ignore")
        .tolist()
    )
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    imputations = []

    for k in range(m):
        imp = IterativeImputer(
            estimator=estimator,
            max_iter=max_iter,
            sample_posterior=True,           # → Multiple Imputation
            random_state=None if random_state is None else random_state + k
        )
        imputed_array = imp.fit_transform(df)

        imputed_df = pd.DataFrame(
            imputed_array, columns=df.columns, index=df.index)

        imputations.append(imputed_df.loc[na_indexes, incomplete_column])

    # pooling (średnia wartości z m imputacji)
    mean_imputed = pd.concat(imputations, axis=1).mean(axis=1)
    return mean_imputed.to_dict()

def impute_with_rf(
    df: pd.DataFrame,
    column: str,
    na_idx,
    *,
    n_estimators: int = 200,
    max_depth: int | None = None,
    random_state: int | None = None,
    **rf_kwargs
):
    """
    Imputuje brakujące wartości w `column` za pomocą pojedynczego
    modelu Random Forest (regresja lub klasyfikacja).

    Zwraca słownik {index: przewidziana_wartość}.
    """
    train_idx = df.index.difference(na_idx)
    X_train = df.loc[train_idx].drop(columns=[column])
    y_train = df.loc[train_idx, column]
    X_pred  = df.loc[na_idx].drop(columns=[column])

    X_train_enc = pd.get_dummies(X_train, drop_first=True)
    X_pred_enc  = pd.get_dummies(X_pred, drop_first=True)
    X_pred_enc  = X_pred_enc.reindex(columns=X_train_enc.columns, fill_value=0)

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_enc)
    X_pred_imp  = imputer.transform(X_pred_enc)

    if pd.api.types.is_numeric_dtype(y_train):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **rf_kwargs
        )
    else:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            **rf_kwargs
        )

    model.fit(X_train_imp, y_train)
    y_imputed = model.predict(X_pred_imp)

    return dict(zip(na_idx, y_imputed))
