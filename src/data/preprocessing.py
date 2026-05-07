"""
Preprocesamiento de Datos - Airbnb NYC Data Mining
===================================================

Módulo optimizado para limpieza, imputación y codificación selectiva.
Implementa Target Encoding para variables de alta cardinalidad.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, TargetEncoder

from utils.config import RANDOM_STATE
from utils.helpers import setup_logging, timer

logger = setup_logging(__name__)

# ============================================================================
# TRATAMIENTO DE VALORES FALTANTES
# ============================================================================

@timer
def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = 'mean',
    numeric_cols: Optional[list] = None,
    categorical_cols: Optional[list] = None
) -> pd.DataFrame:
    """Imputa valores faltantes en el DataFrame."""
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if strategy == 'mean':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    elif strategy == 'median':
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    return df

# ============================================================================
# TRANSFORMACIONES
# ============================================================================

@timer
def apply_log_transform(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Aplica np.log1p a las columnas indicadas."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
    return df

# ============================================================================
# CODIFICACIÓN SELECTIVA (OHE + TARGET)
# ============================================================================

@timer
def encode_categorical_hybrid(
    df: pd.DataFrame,
    y: pd.Series,
    target_cols: list[str] = ['neighbourhood'],
    ohe_cols: list[str] = ['neighbourhood_group', 'room_type']
) -> pd.DataFrame:
    """
    Implementa una codificación híbrida:
    - Target Encoding para variables de alta cardinalidad (ej. neighbourhood).
    - One-Hot Encoding para variables de baja cardinalidad (ej. neighbourhood_group).
    """
    df = df.copy()

    # 1. Target Encoding
    if target_cols:
        target_cols = [c for c in target_cols if c in df.columns]
        te = TargetEncoder(random_state=RANDOM_STATE)
        # El TargetEncoder de sklearn espera X (n_samples, n_features)
        df[target_cols] = te.fit_transform(df[target_cols], y)
        logger.info(f" Target Encoding aplicado a: {target_cols}")

    # 2. One-Hot Encoding
    if ohe_cols:
        ohe_cols = [c for c in ohe_cols if c in df.columns]
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True, dtype=int)
        logger.info(f" One-Hot Encoding aplicado a: {ohe_cols}")

    return df

# ============================================================================
# ESCALADO
# ============================================================================

@timer
def scale_features(df: pd.DataFrame, numeric_cols: Optional[list] = None) -> tuple[pd.DataFrame, StandardScaler]:
    """Escala variables numéricas usando StandardScaler."""
    df = df.copy()
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Evitar escalar targets
        numeric_cols = [c for c in numeric_cols if 'price' not in c]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df, scaler
