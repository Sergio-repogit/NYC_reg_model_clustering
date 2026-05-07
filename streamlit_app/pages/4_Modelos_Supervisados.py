"""
Modelos Supervisados - Airbnb NYC 2019
=======================================
Comparativa dinámica y diagnósticos de rendimiento.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Modelos Supervisados", layout="wide")

st.title("Modelos de Aprendizaje Supervisado")

COMPARISON_PATH = Path("results/tables/models/model_comparison_master.csv")
MODELS_FIG_DIR = Path("results/figures/models")
EVAL_FIG_DIR = Path("results/figures/evaluation")

if not COMPARISON_PATH.exists():
    st.error("No se encontró la tabla de comparación. Ejecute el entrenamiento primero.")
else:
    # 1. Tabla de Rendimiento Dinámica
    st.header("1. Comparativa de Rendimiento")
    df_results = pd.read_csv(COMPARISON_PATH)
    
    # Estilizar la tabla: Resaltar mejores R2 (max) y menores errores (min)
    styled_df = df_results.style.highlight_max(axis=0, subset=['CV Mean R2', 'Hold-out R2'], color='#00A699') \
                               .highlight_min(axis=0, subset=['MAE ($)', 'RMSE ($)'], color='#FF5A5F')
    
    st.dataframe(styled_df, use_container_width=True)

    # 2. Diagnósticos Visuales por Modelo
    st.header("2. Diagnósticos e Importancia de Variables")
    model_name = st.selectbox("Seleccione un modelo para ver detalles:", df_results['Modelo'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Diagnósticos
        diag_path = MODELS_FIG_DIR / f"{model_name}_diagnostics.png"
        if diag_path.exists():
            st.image(str(diag_path), caption=f"Diagnósticos: {model_name}")
        
        # Curvas de Aprendizaje
        lc_path = EVAL_FIG_DIR / f"{model_name}_learning_curve.png"
        if lc_path.exists():
            st.image(str(lc_path), caption=f"Curva de Aprendizaje: {model_name}")

    with col2:
        # Importancia de Variables
        imp_path = MODELS_FIG_DIR / f"{model_name}_feature_importance.png"
        if imp_path.exists():
            st.image(str(imp_path), caption=f"Importancia de Variables: {model_name}")
        else:
            st.info("Este modelo no soporta visualización de importancia de variables directa o lineal.")

    # 3. Métricas en Test Set (Modelo Ganador)
    st.header("3. Evaluación del Modelo Ganador")
    winner = df_results.sort_values('Hold-out R2', ascending=False).iloc[0]
    
    st.success(f"El modelo seleccionado como ganador es: **{winner['Modelo']}**")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    col_r1.metric("R² Score (Test)", f"{winner['Hold-out R2']:.4f}")
    col_r2.metric("MAE ($) (Test)", f"${winner['MAE ($)']:.2f}")
    col_r3.metric("RMSE ($) (Test)", f"${winner['RMSE ($)']:.2f}")

    st.info("Nota: Las métricas de error ($) representan la desviación media del modelo en dólares reales por noche.")