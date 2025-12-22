import pandas as pd
import numpy as np
import streamlit as st
import time
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from helpers.streamlit.callbacks import update_point_color
from helpers.streamlit.utils import init_session_state, read_settings_from_file, validate, init_new_annotation_task
from helpers.utils.dataset import add_projection_dimensions_to_df_incomplete, save_results, generate_incomplete_dataset
from helpers.utils.imputation import impute_global_mean, impute_cluster_mean, impute_knn_mean, impute_cluster_knn_mean, impute_mice, impute_with_rf
from helpers.utils.projections import projections_dict, plot_scatter

st.set_page_config(layout="wide")



def get_y_pred_uncertain_str():
    return [str(y_pred) if y_pred is not None else None for y_pred in st.session_state['y_pred_uncertain']]


st.sidebar.markdown('## Authorization')
user_id = st.sidebar.text_input('Your ID')
user_password = st.sidebar.text_input('Password', type="password")

#Logowanie do aplikacji
if user_id and user_password:
    is_authorized = validate(user_id) and user_password == st.secrets["password"]
    if not is_authorized:
        st.error("⚠️ Invalid ID or password. ⚠️")
else:
    is_authorized = False

if not is_authorized:
    st.stop()


datasets_settings = read_settings_from_file()
init_session_state(datasets_settings)

with st.expander("INSTRUCTION"):
    st.video("https://www.youtube.com/watch?v=YcGcMBnQ0I4")

with st.sidebar:
    st.markdown('## Settings')

    # Inicjalizacja zmiennych sesji
    for key, default in {
        'seed_input': 0,
        'force_seed_reload': False,
        'pending_start': False
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Jeśli wcześniej kliknięto START z seedem = 0 → ustaw nowy seed i rerun
    if st.session_state.force_seed_reload:
        st.session_state.seed_input = int(time.time())
        st.session_state.force_seed_reload = False
        st.rerun()

    # Widget do ustawienia seedu
    seed_input = st.number_input('Insert seed', step=1, key='seed_input')

    # Wybór datasetu i frakcji braków
    dataset_name_selectbox = st.selectbox('Select dataset', [ds['name'] for ds in datasets_settings])
    na_fraction_selectbox = st.selectbox('Select na_fraction', [0.1, 0.2, 0.3], key="na_fraction_selectbox")

    # START button
    if st.button('START'):
        if st.session_state.seed_input == 0:
            st.session_state.force_seed_reload = True
            st.session_state.pending_start = True
            st.rerun()
        else:
            st.session_state['dataset_generation_seed'] = int(st.session_state.seed_input)
            st.session_state.df_incomplete = init_new_annotation_task(
                next(ds for ds in datasets_settings if ds['name'] == dataset_name_selectbox),
                st.session_state['dataset_generation_seed']
            )
            st.session_state.started = True

# Po rerunie – kontynuacja rozpoczęcia sesji anotacji
if st.session_state.get("pending_start", False) and st.session_state.seed_input != 0:
    st.session_state.pending_start = False
    st.session_state['dataset_generation_seed'] = int(st.session_state.seed_input)
    st.session_state.df_incomplete = init_new_annotation_task(
        next(ds for ds in datasets_settings if ds['name'] == dataset_name_selectbox),
        st.session_state['dataset_generation_seed']
    )
    st.session_state.started = True


dataset_settings = next(ds for ds in datasets_settings if ds['name'] == dataset_name_selectbox)
incomplete_column = dataset_settings['incomplete_column']
reference_columns = dataset_settings['reference_columns']



if st.session_state['started']:
    with st.sidebar:
        projection_key = st.selectbox('Set projection', projections_dict.keys())
    try:
        df_incomplete = add_projection_dimensions_to_df_incomplete(
            str(projection_key),
            st.session_state.X,
            st.session_state.df_incomplete
        )

        n_of_points_to_annotate = len(st.session_state['border_points'])
        if n_of_points_to_annotate > 0:

            current_null_index = st.session_state['border_points'][0]

            points_to_show_mask = df_incomplete[incomplete_column].notnull()
            points_to_show_mask[current_null_index] = True
            df_to_show = df_incomplete[points_to_show_mask]

            st.session_state['current_null_index'] = current_null_index
            slider_values = np.around(np.arange(
                st.session_state['min_value'] - 3 * dataset_settings['precision'],
                st.session_state['max_value'] + 3 * dataset_settings['precision'],
                step=dataset_settings['precision']),
                3).tolist()

            f"Records to be labeled: {len(st.session_state['border_points'])}"
            st.progress(
                st.session_state['annotated_points'] / (
                        st.session_state['annotated_points'] + n_of_points_to_annotate))

            'You may experiment with different values on the chart slider below.'
            plot_scatter(df_to_show, incomplete_column, reference_columns, current_null_index, slider_values)

            col0, col1, col2, col3 = st.columns((1, 1, 5, 4))

            with col1:
                "Finally, write the selected value and click SUBMIT."
            with col2:
                value = st.number_input(
                    "",
                    st.session_state['min_value'] - 3 * dataset_settings['precision'],
                    st.session_state['max_value'] + 3 * dataset_settings['precision'],
                    step=dataset_settings['precision'],
                    key='numerical_input',
                    format='%.3f'
                )
            with col3:
                if st.button('Submit'):
                    st.session_state['imputed_values'][current_null_index] = value
                    st.session_state['border_points'].pop(0)
                    st.session_state['annotated_points'] += 1
                    st.rerun()
                if st.button('Cancel'):
                    st.session_state['started'] = False
                    st.rerun()

        else:
            # Iteration finished
            na_indexes = list(st.session_state['imputed_values'].keys())
            global_means = impute_global_mean(df_incomplete, incomplete_column, na_indexes)
            cluster_means = impute_cluster_mean(df_incomplete, incomplete_column, na_indexes)
            knn = impute_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
            cluster_knn = impute_cluster_knn_mean(df_incomplete, incomplete_column, na_indexes, reference_columns)
            mice = impute_mice(
                df_incomplete,
                incomplete_column,
                na_indexes,
                random_state=st.session_state['dataset_generation_seed'],
                reference_columns=reference_columns,
                max_iter=10,
                precision=dataset_settings['precision'],
            )

            random_forest = impute_with_rf(
                df_incomplete,
                incomplete_column,
                na_indexes,
                random_state=st.session_state['dataset_generation_seed'],
                reference_columns=reference_columns,
                n_estimators=300,
                n_jobs=1,
                precision=dataset_settings['precision'],
            )

            mean_preds = pd.Series(global_means).reindex(na_indexes).to_numpy()
            cluster_mean_preds = pd.Series(cluster_means).reindex(na_indexes).to_numpy()
            knn_preds = pd.Series(knn).reindex(na_indexes).to_numpy()
            cluster_knn_preds = pd.Series(cluster_knn).reindex(na_indexes).to_numpy()
            mice_preds = pd.Series(mice).reindex(na_indexes).to_numpy()
            rf_preds = pd.Series(random_forest).reindex(na_indexes).to_numpy()
            real_values = st.session_state[f'ds_{dataset_settings["name"]}'][incomplete_column][na_indexes].tolist()

            rows = ["Annotator", "Mean", "Cluster mean", "knn", "cluster knn", "MICE","RF"]
            imputations = [st.session_state['imputed_values'], global_means, cluster_means, knn, cluster_knn, mice,random_forest]
            MAE = [mean_absolute_error(list(imputation.values()), real_values) for imputation in imputations]
            RMSE = [root_mean_squared_error(list(imputation.values()), real_values) for imputation in imputations]

            results = pd.DataFrame(MAE, columns=["MAE"], index=rows)
            results["RMSE"] = RMSE
            st.table(results)

            if st.button("OK, SAVE RESULTS"):
                st.session_state['started'] = False
                save_results(st.session_state['imputed_values'],
                             st.session_state['dataset_generation_seed'],
                             dataset_name_selectbox,
                             na_fraction_selectbox,
                             projection_key,
                             MAE,  # Mean absolute error of imputations in order
                             RMSE,  # Root Mean Squared Error
                             mean_preds,cluster_mean_preds,
                             knn_preds,
                             cluster_knn_preds,
                             mice_preds,
                             rf_preds,
                             real_values
                             )

                st.success('Your answers have been saved')
                st.balloons()
                time.sleep(1)
                st.rerun()

    except (ValueError, KeyError):
        st.error("Click START after changing the dataset to restart the annotation process")
