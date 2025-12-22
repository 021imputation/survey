import time
import pandas as pd
import gspread
import streamlit as st
from google.oauth2 import service_account
import gspread
from gspread_dataframe import set_with_dataframe

from helpers.utils.dataset import build_results_df


# from gsheetsdb import connect

class GSAuthentication:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GSAuthentication, cls).__new__(cls)
            credentials = service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"],
                scopes=[
                    "https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive.readonly",
                ],
            )
            print("created")
            # TODO: refactor to use SQL not custom code
            # conn = connect(credentials=credentials)
            cls.gc = gspread.Client(credentials)
            cls.url = st.secrets['private_gsheets_url']

        return cls._instance


def download_datasets(g, names):
    for name in names:
        gsheet = g.gc.open(name, st.secrets['private_data_folder_id'])
        data = gsheet.sheet1.get_all_values()
        df = pd.DataFrame(data[1:], columns=data[0])
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                pass
        st.session_state[f'ds_{name}'] = df


def save_results(g, values, annotator_id, seed, dataset_name, na_fraction, projection_key, MAE, RMSE,
                 mean_preds, cluster_mean_preds, knn_preds, cluster_knn_preds,
                 mice_preds,
                 rf_preds,
                 true_values
                 ):

    df=build_results_df(values=values, true_values=true_values,
        mean_preds=mean_preds, cluster_mean_preds=cluster_mean_preds,
        knn_preds=knn_preds, cluster_knn_preds=cluster_knn_preds,mice_preds=mice_preds,random_forest_preds=rf_preds,
        annotator_id=annotator_id, seed=seed, dataset_name=dataset_name,
        na_fraction=na_fraction, projection=projection_key, MAE=MAE, RMSE=RMSE)
    gsheet = g.gc.open_by_url(g.url)
    datasets = {"iris": 0, "wola": 1, "stamp_type": 2}
    wsheet = gsheet.get_worksheet(datasets[dataset_name])

    # 2. Sprawdź, czy arkusz jest pusty
    existing_data = wsheet.get_all_values()
    is_empty = (len(existing_data) == 0)

    # 3. Wstaw DataFrame (z nagłówkami tylko jeśli arkusz był pusty)
    start_row = 1 if is_empty else len(existing_data) + 1

    set_with_dataframe(
        worksheet=wsheet,
        dataframe=df,
        row=start_row,
        include_column_header=is_empty,  # nagłówki tylko przy pierwszym zapisie
        include_index=False,
        resize=False  # nie zmieniaj wymiarów arkusza
    )

