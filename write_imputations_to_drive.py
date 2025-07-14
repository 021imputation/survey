#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
write_imputations_row_append_cols.py
------------------------------------
• Każdy seed z arkusza 'iris' (lub innego wskazanego w DATASETS)
  → odtwarza braki, liczy MICE i Random-Forest
  → dopisuje wyniki w dwóch kolejnych pustych kolumnach (JSON)
"""

from __future__ import annotations
import json, time, random, warnings
import numpy as np, pandas as pd, streamlit as st
from gspread.utils import rowcol_to_a1
from helpers.drive.api        import GSAuthentication
from helpers.streamlit.utils  import read_settings_from_file
from helpers.utils.imputation import impute_mice, impute_with_rf
import gspread

DATASETS = ["stamp_type"]                       # ← podmień na ["wola"] lub ["stamp_type"]

warnings.filterwarnings("ignore", category=UserWarning)
cfg = {d["name"]: d for d in read_settings_from_file()}
gc  = GSAuthentication().gc

# ── helpers ──────────────────────────────────────────────────────────
def make_incomplete(df: pd.DataFrame, col: str,
                    na_frac: float, seed: int):
    na_idx = df.sample(frac=na_frac, random_state=seed).index.tolist()
    out = df.copy(deep=True)
    out.loc[na_idx, col] = np.nan
    return out, na_idx


def with_retry(callable_, *args, **kwargs):
    """ Wykonaj funkcję gspread z 3-krotnym powtórzeniem przy rozłączeniu. """
    for attempt in range(3):
        try:
            return callable_(*args, **kwargs)
        except (gspread.exceptions.APIError,
                gspread.exceptions.HTTPError):
            if attempt == 2:
                raise             # po 3 próbie poddajemy się
            time.sleep(random.uniform(1.0, 2.0))
# ────────────────────────────────────────────────────────────────────

# 1) raw datasets -----------------------------------------------------
raw = {}
for name in DATASETS:
    vals = gc.open(name, st.secrets["private_data_folder_id"]).sheet1.get_all_values()
    raw[name] = (pd.DataFrame(vals[1:], columns=vals[0])
                   .apply(pd.to_numeric, errors="ignore"))

# 2) workbook & arkusz z seedami -------------------------------------
wb = gc.open_by_url(st.secrets["private_gsheets_url"])
ws = wb.worksheet("stamp_type")                 # ← nazwa zakładki z seedami

# 3) pętla po seedach -------------------------------------------------
row_counter = 1           # u Ciebie w arkuszu brak nagłówka

for ds in DATASETS:
    data = with_retry(ws.get_all_values)  # cały arkusz w pamięci
    inc_col = cfg[ds]["incomplete_column"]

    for rec in data:
        try:
            seed = int(rec[2])            # kol-3
            na_f = float(rec[3])          # kol-4
        except (ValueError, IndexError):
            row_counter += 1
            continue
        if not 0 < na_f <= 1:
            row_counter += 1
            continue

        df_inc, na_idx = make_incomplete(raw[ds], inc_col, na_f, seed)
        if not na_idx:
            row_counter += 1
            continue

        num_cols = df_inc.select_dtypes(include=[np.number]).columns.tolist()
        if inc_col not in num_cols:
            num_cols.append(inc_col)

        mice = impute_mice(df_inc[num_cols], inc_col, na_idx, random_state=seed)
        rf   = impute_with_rf(df_inc,        inc_col, na_idx, random_state=seed)

        # —— pierwsza wolna kolumna w tym wierszu --------------------
        col_start = len(with_retry(ws.row_values, row_counter)) + 1
        rng = f"{rowcol_to_a1(row_counter, col_start)}:{rowcol_to_a1(row_counter, col_start+1)}"
        payload = [[json.dumps(mice, ensure_ascii=False),
                    json.dumps(rf,   ensure_ascii=False)]]

        with_retry(ws.update, rng, payload)   # ↔ stary podpis (range, values)

        print(f"✔ seed {seed:<10} → wiersz {row_counter}, kol {col_start}-{col_start+1}")
        row_counter += 1

print(f"\n🏁  Zakończono – zapisano {row_counter-1} par MICE | RF.")
