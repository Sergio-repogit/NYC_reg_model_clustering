"""
Carga de Datos - Airbnb NYC Data Mining
========================================

Módulo para cargar, validar y preparar datos para el proyecto.

Author: [Tu Nombre]
Date: 2026
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.config import (
    CATEGORICAL_COLS,
    DATA_PROCESSED_DIR,
    DATA_RAW_PATH,
    ID_COLS,
    NUMERIC_COLS,
    TARGET_COL,
)
from utils.helpers import check_data_quality, print_dataframe_info, setup_logging, timer

logger = setup_logging(__name__)

# ============================================================================
# CARGA DE DATOS
# ============================================================================

@timer
def load_raw_data(filepath: Path = DATA_RAW_PATH) -> pd.DataFrame:
    """
    Carga el dataset bruto de Airbnb NYC 2019.

    Args:
        filepath: Ruta del archivo CSV

    Returns:
        DataFrame con datos brutos

    Example:
        >>> df = load_raw_data()
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"Archivo no encontrado: {filepath}")
        raise FileNotFoundError(f"Dataset no encontrado en: {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f" Dataset cargado: {df.shape[0]:,} filas, {df.shape[1]} columnas")
        return df

    except Exception as e:
        logger.error(f" Error al cargar datos: {str(e)}")
        raise


@timer
def load_processed_data(
    split: str = 'train',
    directory: Path = DATA_PROCESSED_DIR
) -> pd.DataFrame:
    """
    Carga datos procesados (train/validation/test).

    Args:
        split: 'train', 'validation' o 'test'
        directory: Directorio donde están los datos procesados

    Returns:
        DataFrame procesado

    Example:
        >>> X_train = load_processed_data('train')
    """
    filepath = directory / f'{split}.csv'

    if not filepath.exists():
        logger.error(f"Archivo no encontrado: {filepath}")
        raise FileNotFoundError(f"Datos procesados no encontrados: {filepath}")

    try:
        df = pd.read_csv(filepath)
        logger.info(f" Datos {split} cargados: {df.shape[0]:,} filas")
        return df

    except Exception as e:
        logger.error(f" Error al cargar datos {split}: {str(e)}")
        raise


# ============================================================================
# VALIDACIÓN DE DATOS
# ============================================================================

def validate_columns(df: pd.DataFrame, required_cols: list) -> bool:
    """
    Valida que el DataFrame contenga las columnas requeridas.

    Args:
        df: DataFrame a validar
        required_cols: Lista de columnas requeridas

    Returns:
        True si todas las columnas están presentes

    Example:
        >>> validate_columns(df, ['price', 'room_type'])
    """
    missing_cols = set(required_cols) - set(df.columns)

    if missing_cols:
        logger.error(f"Columnas faltantes: {missing_cols}")
        return False

    logger.info(" Validación de columnas exitosa")
    return True


def validate_data_types(df: pd.DataFrame, expected_dtypes: dict) -> bool:
    """
    Valida que las columnas tengan los tipos de datos esperados.

    Args:
        df: DataFrame a validar
        expected_dtypes: Diccionario {columna: tipo}

    Returns:
        True si todos los tipos coinciden

    Example:
        >>> validate_data_types(df, {'price': 'float64', 'name': 'object'})
    """
    mismatches = []

    for col, expected_dtype in expected_dtypes.items():
        if col in df.columns:
            if str(df[col].dtype) != expected_dtype:
                mismatches.append((col, expected_dtype, str(df[col].dtype)))

    if mismatches:
        logger.warning("Desajustes de tipos de datos:")
        for col, expected, actual in mismatches:
            logger.warning(f"  {col}: esperado {expected}, actual {actual}")
        return False

    logger.info(" Validación de tipos de datos exitosa")
    return True


# ============================================================================
# INSPECCIÓN DE DATOS
# ============================================================================

def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Obtiene un resumen estadístico del dataset.

    Args:
        df: DataFrame a analizar

    Returns:
        Diccionario con información del dataset

    Example:
        >>> summary = get_data_summary(df)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    summary = {
        'shape': df.shape,
        'rows': df.shape[0],
        'cols': df.shape[1],
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'missing_values': int(df.isna().sum().sum()),
        'duplicates': int(df.duplicated().sum()),
        'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        'dtypes': df.dtypes.to_dict()
    }

    return summary


def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Obtiene descripción estadística de variables numéricas.

    Args:
        df: DataFrame a analizar

    Returns:
        DataFrame con estadísticas

    Example:
        >>> stats = describe_numeric(df)
    """
    return df.describe().T


def describe_categorical(df: pd.DataFrame) -> dict:
    """
    Obtiene descripción de variables categóricas.

    Args:
        df: DataFrame a analizar

    Returns:
        Diccionario con información categórica

    Example:
        >>> cat_info = describe_categorical(df)
    """
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    cat_info = {}
    for col in categorical_cols:
        cat_info[col] = {
            'unique_values': df[col].nunique(),
            'most_common': df[col].value_counts().index[0],
            'missing_values': int(df[col].isna().sum())
        }

    return cat_info


# ============================================================================
# PARTICIÓN DE DATOS
# ============================================================================

@timer
def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 42,
    stratify_col: Optional[str] = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en train/validation/test.

    Args:
        df: DataFrame completo
        test_size: Proporción para test (ej: 0.2 = 20%)
        val_size: Proporción para validation del 80% restante (ej: 0.25 = 20%)
        random_state: Semilla aleatoria
        stratify_col: Columna para estratificar (para mantener proporción)

    Returns:
        Tupla (train_df, val_df, test_df)

    Example:
        >>> train, val, test = split_data(df, test_size=0.2, val_size=0.2)
    """
    from sklearn.model_selection import train_test_split

    logger.info(f"Dividiendo dataset: test={test_size*100:.0f}%, val={val_size*80:.0f}%, train={100-test_size*100-val_size*80:.0f}%")

    # Primera división: train+val vs test
    stratify = df[stratify_col] if stratify_col else None

    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    # Segunda división: train vs val
    stratify = train_val[stratify_col] if stratify_col else None

    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify
    )

    logger.info(" Dataset dividido:")
    logger.info(f"   Train: {train.shape[0]:,} filas ({train.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"   Val:   {val.shape[0]:,} filas ({val.shape[0]/len(df)*100:.1f}%)")
    logger.info(f"   Test:  {test.shape[0]:,} filas ({test.shape[0]/len(df)*100:.1f}%)")

    return train, val, test


# ============================================================================
# GUARDADO DE DATOS
# ============================================================================

@timer
def save_processed_data(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    directory: Path = DATA_PROCESSED_DIR
) -> None:
    """
    Guarda los conjuntos de train/val/test procesados.

    Args:
        train: DataFrame de entrenamiento
        val: DataFrame de validación
        test: DataFrame de test
        directory: Directorio donde guardar

    Example:
        >>> save_processed_data(train, val, test)
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    try:
        train.to_csv(directory / 'train.csv', index=False)
        val.to_csv(directory / 'validation.csv', index=False)
        test.to_csv(directory / 'test.csv', index=False)

        logger.info(f" Datos procesados guardados en: {directory}")

    except Exception as e:
        logger.error(f" Error al guardar datos: {str(e)}")
        raise


# ============================================================================
# PIPELINE DE CARGA
# ============================================================================

@timer
def load_and_validate() -> pd.DataFrame:
    """
    Carga, valida e imprime información del dataset bruto.

    Returns:
        DataFrame validado

    Example:
        >>> df = load_and_validate()
    """
    # Cargar datos
    df = load_raw_data()

    # Validar columnas
    all_cols = NUMERIC_COLS + CATEGORICAL_COLS + ID_COLS + [TARGET_COL]
    validate_columns(df, all_cols)

    # Información del dataset
    print_dataframe_info(df, "Dataset Bruto - Airbnb NYC 2019")

    # Chequeo de calidad
    check_data_quality(df)

    return df


if __name__ == '__main__':
    df = load_and_validate()
