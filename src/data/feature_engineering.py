"""
Feature Engineering - Airbnb NYC Data Mining
=============================================

Módulo refinado para la creación de indicadores, ingeniería de dominio
y reducción de dimensionalidad mediante PCA con exportación de pesos.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils.config import (
    RANDOM_STATE, TARGET_COL, FIGURES_CLUSTERING_DIR, 
    TABLES_PCA_DIR, FIGURES_PREPROCESSING_DIR
)
from utils.helpers import setup_logging, timer, print_custom_report
from evaluation.visualization import plot_pca_biplot, plot_pca_variance

logger = setup_logging(__name__)

# ============================================================================
# INGENIERÍA DE VARIABLES
# ============================================================================

@timer
def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea indicadores de nuevos listings y características de proximidad."""
    df = df.copy()
    df['is_new_listing'] = (df['number_of_reviews'].isna() | df['last_review'].isna()).astype(int)
    
    points = {
        'times_square': (40.7580, -73.9855),
        'central_park': (40.7812, -73.9665),
        'grand_central': (40.7527, -73.9772)
    }
    
    def haversine_np(lat1, lon1, lat2, lon2):
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km
    
    if 'latitude' in df.columns and 'longitude' in df.columns:
        for name, coords in points.items():
            df[f'dist_{name}'] = haversine_np(df['latitude'], df['longitude'], coords[0], coords[1])
            
    return df

# ============================================================================
# PCA CON ANÁLISIS DE PESOS (LOADINGS)
# ============================================================================

@timer
def apply_pca_with_report(X: pd.DataFrame, n_components: float = 0.95):
    """Aplica PCA, genera Biplot y exporta tabla de pesos (Loadings)."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    
    # Exportar Tabla de Pesos (Loadings)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=X.columns
    )
    
    # Identificar variables con mayor peso por PC
    loadings.to_csv(TABLES_PCA_DIR / 'pca_loadings.csv')
    
    # Reporte por consola
    pca_metrics = {
        "Dimensiones originales": X.shape[1],
        "Componentes principales": X_pca.shape[1],
        "Varianza total": f"{np.sum(pca.explained_variance_ratio_) * 100:.2f}%"
    }
    print_custom_report("PCA", pca_metrics)
    
    # Guardar métricas para Streamlit
    import json
    with open(TABLES_PCA_DIR / 'pca_metrics.json', 'w') as f:
        json.dump({
            "orig_dim": X.shape[1],
            "reduced_dim": X_pca.shape[1],
            "total_variance": float(np.sum(pca.explained_variance_ratio_)),
            "orig_features": X.columns.tolist(),
            "reduced_features": [f'PC{i+1}' for i in range(X_pca.shape[1])]
        }, f)
    
    # Biplot y Scree Plot
    plot_pca_biplot(X, pca, FIGURES_PREPROCESSING_DIR)
    plot_pca_variance(pca, FIGURES_PREPROCESSING_DIR)
    
    df_pca = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=X.index)
    return df_pca, pca

# ============================================================================
# PIPELINE INTEGRADO
# ============================================================================

@timer
def complete_feature_engineering(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Orquestador completo con exportación de metadatos de PCA."""
    from data.preprocessing import apply_log_transform, encode_categorical_hybrid, scale_features, handle_missing_values
    
    df_fe = df.copy()
    df_fe = create_domain_features(df_fe)
    
    cols_to_drop = ['id', 'name', 'host_id', 'host_name', 'calculated_host_listings_count']
    df_fe = df_fe.drop(columns=[c for c in cols_to_drop if c in df_fe.columns])
    
    df_fe['reviews_per_month'] = df_fe['reviews_per_month'].fillna(0)
    
    if 'last_review' in df_fe.columns:
        df_fe['last_review'] = pd.to_datetime(df_fe['last_review'], errors='coerce')
        ref_date = df_fe['last_review'].max()
        df_fe['days_since_review'] = (ref_date - df_fe['last_review']).dt.days
        df_fe['days_since_review'] = df_fe['days_since_review'].fillna(df_fe['days_since_review'].median())
        df_fe = df_fe.drop(columns=['last_review'])
        
    df_fe = handle_missing_values(df_fe, strategy='mean')
    
    log_cols = ['price', 'minimum_nights', 'number_of_reviews', 'days_since_review', 'reviews_per_month']
    df_fe = apply_log_transform(df_fe, columns=log_cols)
    
    target_y = df_fe['price_log']
    df_fe = encode_categorical_hybrid(df_fe, y=target_y)
    
    X = df_fe.drop(columns=[TARGET_COL, 'price_log'], errors='ignore')
    y = df_fe[TARGET_COL]
    X_scaled, scaler = scale_features(X)
    
    df_pca, pca = apply_pca_with_report(X_scaled)
    
    # Guardar artefactos
    import joblib
    from pathlib import Path
    models_dir = Path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(pca, models_dir / "pca_model.pkl")
    
    df_final = pd.concat([df_pca, y], axis=1)
    if 'price_log' in df_fe.columns:
        df_final['price_log'] = df_fe['price_log']
        
    return df_final, {'pca': pca, 'df_fe': df_fe}
