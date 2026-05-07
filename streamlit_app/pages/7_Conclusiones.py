import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Conclusiones", layout="wide")
st.title("Conclusiones y Síntesis del Proyecto")

# Dynamic data loading
root_dir = Path(__file__).resolve().parent.parent.parent

winner_name = "No Determinado"
r2_score = 0.0
mae_score = 0.0
gap = 0.0

csv_path = root_dir / 'results' / 'tables' / 'models' / 'model_comparison_master.csv'
if csv_path.exists():
    df_res = pd.read_csv(csv_path)
    stable_models = df_res[df_res['Overfitting Gap'] < 0.05]
    if stable_models.empty:
        winner = df_res.sort_values('Hold-out R2', ascending=False).iloc[0]
    else:
        winner = stable_models.sort_values('Hold-out R2', ascending=False).iloc[0]
        
    winner_name = winner['Modelo']
    r2_score = winner['Hold-out R2']
    mae_score = winner['MAE ($)']
    gap = winner['Overfitting Gap']

clustered_data_path = root_dir / 'results' / 'tables' / 'final_processed_data.csv'
n_clusters = 0
if clustered_data_path.exists():
    n_clusters = pd.read_csv(clustered_data_path)['cluster'].nunique()

st.markdown(f"""
### 1. Resumen de resultados

1. **Predicción de Precios en NYC:** Se evaluó una suite expandida de 12 modelos de regresión. El objetivo predictivo se alcanzó con éxito, coronando a **{winner_name}** como el modelo ganador con un $R^2$ en el conjunto de prueba de **{r2_score:.4f}** y un Error Absoluto Medio (MAE) de **${mae_score:.2f} USD**.
2. **Inteligencia Espacial:** Se integraron exitosamente métricas geoespaciales avanzadas (distancias a Times Square, Central Park y Grand Central) en el pipeline de ingeniería de características.
3. **Segmentación de Mercado:** Mediante aprendizaje no supervisado (K-Means), se segmentó el mercado en **{n_clusters}** clústeres distintos.

### 2. Hallazgos Clave 

**Ingeniería de Variables (Feature Engineering):** 
- La incorporación de variables espaciales (`dist_times_square`, `dist_central_park`, `dist_grand_central`) demostró tener un impacto significativo en la capacidad predictiva del modelo.
- Patrón claro de Manhattan > Brooklyn > Queens > Bronx > Staten Island.
- El tipo de habitación es el factor diferenciador principal del precio.
- Mayor disponibilidad → Precios menores.

**Reducción de Dimensionalidad (PCA):** 
- La aplicación de Análisis de Componentes Principales permitió reducir la complejidad del espacio de características manteniendo la varianza más significativa. Esto optimizó el tiempo de entrenamiento y mitigó la multicolinealidad inherente entre las variables espaciales y de disponibilidad.

**Clustering no Jerárquico (k-means):** 
El análisis no supervisado reveló **{n_clusters}** perfiles de mercado distintos.

### 3. Trabajo Futuro y Recomendaciones

**1. Mejoras de Datos**:
- Incluir datos históricos 2016-2024
- Incorporar variables externas (metros cuadrados,piscina, garaje, etc.)

2. **Actualización de Precios en Tiempo Real:** Implementar una API que alimente el modelo con datos dinámicos del mercado para ajustar las predicciones a fluctuaciones inmediatas de la oferta y la demanda.
---

### Referencias y Metadatos del Proyecto

- **Kaggle Dataset:** NYC Airbnb Open Data
- **Stack Tecnológico:** Scikit-Learn, XGBoost, Pandas, Streamlit
- **Institución:** UNIE (Universidad) | Grado: Matemáticas | Asignatura: Minería de Datos
""")