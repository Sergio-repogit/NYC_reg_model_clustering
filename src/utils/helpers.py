"""
Funciones Auxiliares del Proyecto - Airbnb NYC Data Mining
===========================================================

Módulo optimizado para una salida por terminal limpia y profesional.
Gestiona el logging, timing y formateo de reportes.
"""

import logging
import sys
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from utils.config import LOG_FILE, LOG_FORMAT, LOG_LEVEL

# ============================================================================
# CONFIGURACIÓN DE LOGGING (SALIDA LIMPIA)
# ============================================================================

def setup_logging(name: str = __name__, level: str = LOG_LEVEL) -> logging.Logger:
    """Configura el logging con una salida por consola minimalista."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    if not logger.handlers:
        file_formatter = logging.Formatter(LOG_FORMAT)
        console_formatter = logging.Formatter('%(message)s')

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        if LOG_FILE:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_FILE)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)

    return logger

logger = setup_logging(__name__)

# ============================================================================
# DECORADORES Y REPORTES
# ============================================================================

def timer(func: Callable) -> Callable:
    """Decorador que mide el tiempo, logueando de forma discreta."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        inicio = time.time()
        try:
            resultado = func(*args, **kwargs)
            tiempo_total = time.time() - inicio
            # Solo mostramos en terminal si no es una operación interna muy pequeña
            if tiempo_total > 0.01:
                logger.info(f" -> {func.__name__} finalizado ({tiempo_total:.2f}s)")
            return resultado
        except Exception as e:
            logger.error(f" [ERROR] {func.__name__} falló: {str(e)}")
            raise

    return wrapper

def print_section(title: str) -> None:
    """Imprime un separador visual elegante para la terminal."""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def print_custom_report(title: str, metrics: dict) -> None:
    """Imprime un reporte formateado siguiendo los requerimientos del usuario."""
    print("-" * 60)
    print(f"[{title}] PROCESO COMPLETADO")
    for label, value in metrics.items():
        print(f"- {label}: {value}")
    print("-" * 60 + "\n")

# ============================================================================
# FUNCIONES DE COMPATIBILIDAD (RESTORED)
# ============================================================================

def print_dataframe_info(df: pd.DataFrame, name: str = "DF") -> None:
    """Versión minimalista de info de dataframe."""
    logger.info(f" Info [{name}]: {df.shape[0]:,} filas, {df.shape[1]} columnas.")

def check_data_quality(df: pd.DataFrame) -> dict:
    """Versión silenciosa de chequeo de calidad."""
    return {"rows": df.shape[0], "cols": df.shape[1]}

def save_model(model: Any, filepath: Path) -> None:
    """Guarda modelo usando joblib."""
    joblib.dump(model, filepath)
    logger.info(f" Modelo guardado: {filepath.name}")

def load_model(filepath: Path) -> Any:
    """Carga modelo usando joblib."""
    return joblib.load(filepath)

# ============================================================================
# GESTIÓN DE DIRECTORIOS
# ============================================================================

def create_directories() -> None:
    """Crea la estructura de directorios del proyecto."""
    from utils.config import (
        DATA_PROCESSED_DIR,
        DATA_RAW_DIR,
        FIGURES_CLUSTERING_DIR,
        FIGURES_EDA_DIR,
        FIGURES_EVALUATION_DIR,
        MODELS_DIR,
    )
    dirs = [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
            FIGURES_EDA_DIR, FIGURES_CLUSTERING_DIR, FIGURES_EVALUATION_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
