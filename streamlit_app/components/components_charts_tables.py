"""
streamlit_app/components/charts.py y tables.py
===============================================

Gráficos y tablas reutilizables
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# CHARTS.PY - GRÁFICOS REUTILIZABLES
# ============================================================================


def plot_distribution(data: pd.Series, title: str = "", nbins: int = 30):
    """Crea un histograma de distribución"""
    fig = px.histogram(
        x=data.dropna(),
        nbins=nbins,
        title=title,
        template="plotly_white",
        color_discrete_sequence=["#FF5A5F"],
    )
    fig.update_layout(xaxis_title=data.name, yaxis_title="Frecuencia", showlegend=False)
    return fig


def plot_boxplot(df: pd.DataFrame, x: str, y: str, title: str = ""):
    """Crea un boxplot"""
    fig = px.box(
        df,
        x=x,
        y=y,
        title=title,
        template="plotly_white",
        color=x,
        color_discrete_sequence=["#FF5A5F", "#00A699", "#FC642D"],
    )
    return fig


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: str = None,
    size: str = None,
    title: str = "",
):
    """Crea un scatter plot"""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        size=size,
        title=title,
        template="plotly_white",
        opacity=0.7,
        hover_data=["name"] if "name" in df.columns else None,
    )
    fig.update_traces(marker=dict(line=dict(width=0.5, color="DarkSlateGrey")))
    return fig


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str = "", color: str = None):
    """Crea un gráfico de barras"""
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        template="plotly_white",
        color=color if color else x,
        text_auto=".2s",
        color_discrete_sequence=["#FF5A5F", "#00A699", "#FC642D"],
    )
    fig.update_traces(textposition="outside")
    return fig


def plot_pie(df: pd.DataFrame, values: str, names: str, title: str = ""):
    """Crea un gráfico de pastel"""
    fig = px.pie(
        df,
        values=values,
        names=names,
        title=title,
        template="plotly_white",
        color_discrete_sequence=["#FF5A5F", "#00A699", "#FC642D", "#FFB400", "#007A87"],
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "Correlaciones"):
    """Crea un heatmap de correlaciones"""
    corr_matrix = df.corr(numeric_only=True)

    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text:.2f}",
            textfont={"size": 10},
        )
    )

    fig.update_layout(title=title, width=700, height=700)

    return fig


def plot_comparison_bars(data: list, labels: list, title: str = ""):
    """Crea barras comparativas"""
    fig = go.Figure(
        data=[go.Bar(name=labels[i], x=labels, y=data[i]) for i in range(len(data))]
    )

    fig.update_layout(title=title, barmode="group", template="plotly_white")

    return fig


def plot_metric_card(
    title: str, value: str, delta: str = "", color: str = "#FF5A5F", icon: str = ""
):
    """Renderiza una tarjeta de métrica con HTML personalizado"""
    html = f"""
    <div style='
        background: linear-gradient(135deg, {color}33 0%, {color}11 100%);
        border-left: 4px solid {color};
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 10px;
    '>
        <div style='font-size: 12px; color: gray; margin-bottom: 5px;'>{icon} {title}</div>
        <div style='font-size: 28px; font-weight: bold; color: {color};'>{value}</div>
        <div style='font-size: 11px; color: gray; margin-top: 5px;'>{delta}</div>
    </div>
    """
    return html


# ============================================================================
# TABLES.PY - TABLAS REUTILIZABLES
# ============================================================================


def styled_dataframe(
    df: pd.DataFrame, highlight_cols: list = None, format_dict: dict = None
):
    """
    Renderiza DataFrame con estilos personalizados

    Args:
        df: DataFrame a mostrar
        highlight_cols: Columnas a resaltar
        format_dict: Diccionario de formatos {col: 'format_spec'}
    """

    styled_df = df.style

    # Aplicar formato a columnas
    if format_dict:
        for col, fmt in format_dict.items():
            if col in df.columns:
                styled_df = styled_df.format({col: fmt})

    # Resaltar máximos y mínimos
    if highlight_cols:
        for col in highlight_cols:
            if col in df.columns:
                if df[col].dtype in ["float64", "int64"]:
                    styled_df = styled_df.highlight_max(
                        subset=[col], color="lightgreen"
                    ).highlight_min(subset=[col], color="lightcoral")

    return styled_df


def comparison_table(df: pd.DataFrame, highlight_best: str = None):
    """
    Crea tabla de comparación con resaltado del mejor

    Args:
        df: DataFrame con resultados
        highlight_best: Columna para resaltar el mejor valor
    """

    styled_df = df.style.set_properties(
        **{"border": "1px solid #ddd", "padding": "8px"}
    )

    if highlight_best and highlight_best in df.columns:
        max_idx = df[highlight_best].idxmax()

        def highlight_row(row):
            if row.name == max_idx:
                return ["background-color: lightgreen"] * len(row)
            else:
                return [""] * len(row)

        styled_df = styled_df.apply(highlight_row, axis=1)

    return styled_df


def model_comparison_table(models_results: pd.DataFrame):
    """Tabla especializada para comparación de modelos"""

    # Formatear columnas numéricas
    format_dict = {
        "R² Score": "{:.4f}",
        "RMSE": "${:.2f}",
        "MAE": "${:.2f}",
        "MAPE": "{:.2f}%",
        "Accuracy": "{:.4f}",
        "F1 Score": "{:.4f}",
    }

    # Aplicar formato
    for col, fmt in format_dict.items():
        if col in models_results.columns:
            models_results[col] = models_results[col].apply(lambda x: fmt.format(x))

    return comparison_table(models_results, highlight_best="R² Score")


def feature_importance_table(importance_df: pd.DataFrame, top_n: int = 20):
    """Tabla de importancia de características"""

    top_features = importance_df.head(top_n).copy()

    # Crear barra visual
    max_importance = importance_df["importance"].max()

    styled = top_features.style.bar(
        subset=["importance"], vmin=0, vmax=max_importance, color="#FF5A5F", width=100
    )

    return styled


def cluster_profile_table(profiles_df: pd.DataFrame):
    """Tabla especializada para perfiles de clusters"""

    format_dict = {
        "Precio Promedio": "${:.2f}",
        "Reseñas Promedio": "{:.1f}",
        "Disponibilidad": "{:.0f}",
        "Tamaño": "{:,}",
    }

    for col, fmt in format_dict.items():
        if col in profiles_df.columns:
            try:
                profiles_df[col] = profiles_df[col].apply(lambda x: fmt.format(x))
            except Exception:
                pass

    return styled_dataframe(profiles_df)


def summary_stats_table(df: pd.DataFrame, numeric_only: bool = True):
    """Tabla de estadísticas resumidas"""

    if numeric_only:
        df = df.select_dtypes(include=["number"])

    summary = pd.DataFrame(
        {
            "Count": df.count(),
            "Mean": df.mean(),
            "Std": df.std(),
            "Min": df.min(),
            "25%": df.quantile(0.25),
            "50%": df.quantile(0.50),
            "75%": df.quantile(0.75),
            "Max": df.max(),
        }
    )

    return summary.style.format("{:.2f}")
