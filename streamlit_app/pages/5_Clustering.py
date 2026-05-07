"""
Clustering y Segmentación de Mercado - Airbnb NYC
=================================================
Aprendizaje no supervisado mediante K-Means y validación de consenso.
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Clustering", layout="wide")

st.title("Clustering - Aprendizaje No Supervisado")

root_dir = Path(__file__).resolve().parent.parent.parent
FIGS_DIR = root_dir / "results" / "figures" / "clustering"
TABLES_DIR = root_dir / "results" / "tables" / "clustering"

st.markdown("""
### Estrategia de Segmentación
Se ha aplicado una estrategia de clustering particional para identificar perfiles de propiedades en el espacio PCA.
Se utilizaron los siguientes métodos de validación para determinar el número óptimo de grupos (K):
""")

# 1. Validación de K (Imágenes)
st.header("1. Validación del Número de Clusters (K)")
col1, col2, col3 = st.columns(3)

with col1:
    elbow = FIGS_DIR / "elbow_method.png"
    if elbow.exists():
        st.image(str(elbow), caption="Método del Codo")
with col2:
    sil = FIGS_DIR / "silhouette_analysis.png"
    if sil.exists():
        st.image(str(sil), caption="Análisis de Silueta")
with col3:
    gap = FIGS_DIR / "gap_statistic.png"
    if gap.exists():
        st.image(str(gap), caption="Gap Statistic")

# Carga de Datos y Control de Errores
CLUSTERED_DATA = root_dir / "results" / "tables" / "final_processed_data.csv"

if not CLUSTERED_DATA.exists():
    st.warning(
        "El archivo final_processed_data.csv no existe. Ejecute el pipeline maestro primero."
    )
    st.stop()

df_clustered = pd.read_csv(CLUSTERED_DATA)

if "cluster" not in df_clustered.columns:
    st.warning(
        "Los datos están procesados pero no contienen la columna 'cluster'. Ejecute la fase de clustering para generarlos."
    )
    st.stop()

# 2. Métricas y Estadísticas por Cluster
st.header("2. Validación del K-Óptimo")

optimal_k = df_clustered["cluster"].nunique()
st.markdown(f"**K-Óptimo detectado e implementado:** {optimal_k} clusters")

metrics_path = TABLES_DIR / "clustering_metrics_k.csv"
if metrics_path.exists():
    df_metrics = pd.read_csv(metrics_path)
    if optimal_k in df_metrics["k"].values:
        row = df_metrics[df_metrics["k"] == optimal_k].iloc[0]
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Silhouette Score", f"{row['Silhouette']:.4f}")
        col_m2.metric("Gap Statistic", f"{row['Gap']:.4f}")
        col_m3.metric("Inertia", f"{row['Inertia']:,.0f}")

    with st.expander("Ver tabla completa de métricas de validación"):
        st.dataframe(df_metrics, use_container_width=True)

st.header("3. Perfiles de Cluster")
st.markdown(
    "Promedios de variables clave y análisis modal para interpretar la identidad de cada segmento de mercado."
)

profile_data = []
for cluster_id, group in df_clustered.groupby("cluster"):
    mode_rt = (
        group["room_type"].mode().iloc[0] if "room_type" in group.columns else "N/A"
    )
    rt_pct = (
        (group["room_type"] == mode_rt).mean() * 100
        if "room_type" in group.columns
        else 0
    )

    profile_data.append(
        {
            "Cluster": cluster_id,
            "Most Frequent Room Type": mode_rt,
            "Room Type Percentage": f"{rt_pct:.1f}%",
            "Average Price": round(group["price"].mean(), 2)
            if "price" in group.columns
            else None,
            "Average Distance to Center": round(group["dist_times_square"].mean(), 2)
            if "dist_times_square" in group.columns
            else None,
            "Count of Listings": len(group),
        }
    )

profile_df = pd.DataFrame(profile_data).set_index("Cluster")
st.dataframe(profile_df, use_container_width=True)

# 4. Mapa de Clusters (Visualización Espacial)
st.header("4. Distribución Geográfica por Cluster")
st.markdown(
    "Mapa interactivo de la ciudad donde los colores representan la pertenencia a un cluster específico."
)

df_clustered["cluster_cat"] = df_clustered["cluster"].astype(str)
fig = px.scatter_mapbox(
    df_clustered,
    lat="latitude",
    lon="longitude",
    color="cluster_cat",
    hover_name="neighbourhood",
    hover_data=["price", "room_type"],
    zoom=10,
    height=600,
    title="Segmentación Geográfica de Airbnb NYC",
    color_discrete_sequence=px.colors.qualitative.Set1,
)
fig.update_layout(
    mapbox_style="carto-positron", margin={"r": 0, "t": 40, "l": 0, "b": 0}
)
st.plotly_chart(fig, use_container_width=True)
