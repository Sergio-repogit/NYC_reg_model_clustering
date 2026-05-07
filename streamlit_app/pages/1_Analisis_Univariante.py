"""
Análisis Univariante - Airbnb NYC 2019
=======================================
Visualización de distribuciones univariantes generadas durante el EDA.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Análisis Univariante", layout="wide")

st.title("Análisis Univariante")
st.markdown("""
En esta sección se presentan las distribuciones individuales de las variables clave del dataset.
Se incluyen gráficos de sectores para variables categóricas e histogramas con diagramas de caja para las numéricas.
""")

EDA_DIR = Path("results/figures/eda")

if not EDA_DIR.exists():
    st.error("No se encontraron los resultados del EDA en 'results/figures/eda/'. Ejecute el pipeline principal primero.")
else:
    # 1. Variables Categóricas (Pie Charts)
    st.header("1. Distribución de Variables Categóricas")
    cat_files = list(EDA_DIR.glob("eda_cat_*.png"))
    if cat_files:
        for i in range(0, len(cat_files), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(cat_files):
                    file = cat_files[i+j]
                    with cols[j]:
                        st.image(str(file), use_container_width=True)
    else:
        st.info("No se encontraron gráficos de sectores.")

    # 2. Variables Numéricas (Histogramas + Boxplots)
    st.header("2. Distribución de Variables Numéricas")
    num_files = list(EDA_DIR.glob("eda_num_*.png"))
    if num_files:
        for i in range(0, len(num_files), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(num_files):
                    file = num_files[i+j]
                    with cols[j]:
                        st.image(str(file), use_container_width=True)
    else:
        st.info("No se encontraron gráficos numéricos.")
