import time
import pandas as pd
import gspread
import streamlit as st
from google.oauth2 import service_account
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


def save_results(g, values, annotator_id, seed, dataset_name, na_fraction, projection_key, MAE, RMSE):
    gsheet = g.gc.open_by_url(g.url)
    datasets = {"iris": 0, "wola": 1, "stamp_type": 2}
    worksheet_number = datasets[dataset_name]
    wsheet = gsheet.get_worksheet(worksheet_number)
    data = pd.DataFrame(wsheet.get_all_values())
    now = time.localtime(time.time())
    row_data = [
        time.strftime("%m/%d/%Y, %H:%M:%S", now),
        annotator_id,
        seed,
        na_fraction,
        projection_key,
        str(values),
        str(MAE),
        str(RMSE)
    ]
    wsheet.insert_row(row_data, len(data) + 1)
