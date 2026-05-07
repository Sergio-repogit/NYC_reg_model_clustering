"""
Preprocesamiento y Feature Engineering - Airbnb NYC
===================================================
Documentación técnica de transformaciones y PCA.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Preprocesamiento", layout="wide")

st.title(" Preprocesamiento y Feature Engineering")

st.markdown("""
### Pipeline de Transformación Aplicado
Para el modelado de alta complejidad, se han implementado las siguientes técnicas:

1. **Modificación de Variables**:
   - Cálculo de la **distancia a lugares turísticos** (Times Square, Central Park y Grand Central) para capturar la influencia de la localización estratégica.
   - Cálculo de days_since_review (días desde la última reseña) a partir de last_review.
   - Eliminación de variables no relevantes para el modelo (id, host_id, name, host_name,calculated_host_listings_count, last_review).
   - Imputación de valores faltantes: last_review imputado con la media y reviews_per_month imputado por 0 al considerarse nuevo, se consideran MAR (Missing at Random - Faltantes al Azar).
2. **Codificación de Categóricas**:
   - **Target Encoding** para la variable `neighbourhood` (vecindario), permitiendo capturar el valor medio por zona sin aumentar drásticamente el número de columnas.
   - **One-Hot Encoding** para `neighbourhood_group` y `room_type`.
3. **Escalado**:
   - Transformaciones logarítmicas en variables con alto sesgo.
   - Aplicación de `StandardScaler` para normalizar las magnitudes de las características.
4. **Reducción de Dimensionalidad (PCA)**:
   - Se ha aplicado PCA para reducir el espacio de características manteniendo el **95% de la varianza explicada**.
""")


# ============================================================================
# VISUALIZACIONES DE PCA
# ============================================================================
st.header("1. Análisis Visual de PCA")

col_biplot, col_scree = st.columns(2)

with col_biplot:
    st.subheader("PCA Biplot")
    BIPLOT_PATH = Path("results/figures/preprocessing/pca_biplot.png")
    if BIPLOT_PATH.exists():
        st.image(
            str(BIPLOT_PATH),
            caption="Relación entre variables originales y componentes principales.",
        )
    else:
        st.warning("Biplot no encontrado.")

with col_scree:
    st.subheader("PCA Scree Plot")
    SCREE_PATH = Path("results/figures/preprocessing/pca_variance_scree.png")
    if SCREE_PATH.exists():
        st.image(
            str(SCREE_PATH), caption="Varianza individual y acumulada por componente."
        )
    else:
        st.warning("Scree Plot no encontrado.")

# ============================================================================
# PCA LOADINGS TABLE
# ============================================================================
st.header("2. Pesos de las Variables (PCA Loadings)")
LOADINGS_PATH = Path("results/tables/pca/pca_loadings.csv")
if LOADINGS_PATH.exists():
    df_loadings = pd.read_csv(LOADINGS_PATH, index_col=0)
    st.markdown(
        "Esta tabla muestra la contribución de cada variable original a los componentes principales."
    )
    st.dataframe(
        df_loadings.style.background_gradient(cmap="coolwarm"), use_container_width=True
    )
else:
    st.warning("Tabla de pesos PCA no encontrada.")
