from umap.umap_ import UMAP
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import streamlit as st
import plotly.express as px
import pandas as pd

@st.cache_data
def perform_pca_projection(X):
    tsne = PCA(n_components=2, random_state=0)
    X_projected = tsne.fit_transform(X)
    return X_projected


@st.cache_data
def perform_umap_projection(X):
    umap_2d = UMAP(n_components=2, init='random', random_state=0)
    X_projected = umap_2d.fit_transform(X)
    return X_projected


@st.cache_data
def perform_tsne_projection(X):
    tsne = TSNE(n_components=2, random_state=0)
    X_projected = tsne.fit_transform(X)
    return X_projected


projections_dict = {
    "tsne": perform_tsne_projection,
    "pca": perform_pca_projection,
    "umap": perform_umap_projection,
}


def plot_scatter(df, incomplete_column, reference_columns, current_null_index, slider_values):
    uncertain_x, uncertain_y = df[['x', 'y']].loc[current_null_index]
    hover_data = {'x': False, 'y': False, incomplete_column: ':.3f'}
    hover_data.update({col: True for col in df[reference_columns].columns})
    x_std = df['x'].std()
    y_std = df['y'].std()
    min_value = min(slider_values)
    max_value = max(slider_values)
    trace_points = px.scatter(df[~df[incomplete_column].isna()], x="x", y="y",
                     color=incomplete_column,
                     color_continuous_scale=px.colors.sequential.Inferno,
                     range_color=(min_value, max_value),
                     symbol_map={"uncertain": "x"},
                     hover_data=hover_data,
                     range_x=[uncertain_x - x_std, uncertain_x + x_std],
                     range_y=[uncertain_y - y_std, uncertain_y + y_std],
                     # hover_name=target_column,
                     height=600,
                     width=800
                     )

    single_point = df.loc[[current_null_index]].copy(deep=True)
    single_point = pd.concat([single_point] * len(slider_values), ignore_index=True)
    single_point[incomplete_column] = slider_values
    trace_annotation = px.scatter(single_point, x="x", y="y",
                     color=incomplete_column,
                     animation_frame=incomplete_column,
                     color_continuous_scale=px.colors.sequential.Inferno,
                     range_color=(min_value, max_value),
                     symbol_map={"uncertain": "x"},
                     range_x=[uncertain_x - x_std, uncertain_x + x_std],
                     range_y=[uncertain_y - y_std, uncertain_y + y_std],
                     hover_data=hover_data,
                     height=600,
                     width=800
                    )
    fig = trace_annotation
    fig.add_traces(trace_points.data)

    fig.update_traces(textposition='top center', textfont_size=8,
                      textfont_color="#636363", marker_size=12
                      )
    fig.layout.pop("updatemenus")


    fig.layout.showlegend = False
    # fig.update_layout(hovermode="incomplete_column")
    add_annotation(fig, uncertain_x, uncertain_y)

    scatter = st.plotly_chart(fig, key="scatter")


def add_annotation(fig, x, y):
    fig.add_annotation(
        x=x,
        y=y,
        yshift=10,
        xref="x",
        yref="y",
        text="x",
        showarrow=True,
        font=dict(
            family="Courier New, monospace",
            size=16,
            color="#ffffff"
        ),
        align="center",
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#636363",
        ax=0,
        ay=-50,
        bordercolor="#c7c7c7",
        borderwidth=1,
        borderpad=3,
        bgcolor="#636363",
        opacity=0.5
    )
