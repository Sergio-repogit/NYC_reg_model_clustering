"""
Métricas de Evaluación - Airbnb NYC Data Mining
================================================

Módulo especializado en el cálculo de métricas de regresión y análisis
de residuos para la validación de modelos de precios.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
from sklearn.model_selection import cross_validate

from utils.config import CV_FOLDS, CV_SCORING, RANDOM_STATE
from utils.helpers import setup_logging, timer

logger = setup_logging(__name__)

# ============================================================================
# MÉTRICAS DE REGRESIÓN
# ============================================================================

def calculate_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calcula métricas de regresión fundamentales: MSE, RMSE, MAE, MAPE y R2.
    """
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }
    
    # R² ajustado
    n = len(y_true)
    p = 1  # Por defecto, asumir p features
    if n > p + 1:
        metrics['r2_adjusted'] = 1 - (1 - metrics['r2']) * (n - 1) / (n - p - 1)
    else:
        metrics['r2_adjusted'] = metrics['r2']
    
    return metrics

# ============================================================================
# VALIDACIÓN CRUZADA
# ============================================================================

@timer
def calculate_cv_scores(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = CV_FOLDS,
    scoring: str = CV_SCORING,
    return_train_score: bool = True
) -> Dict[str, Tuple[np.ndarray, float, float]]:
    """
    Calcula métricas de validación cruzada para asegurar robustez.
    """
    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring={'score': scoring},
        return_train_score=return_train_score
    )
    
    scores = {
        'test_scores': cv_results['test_score'],
        'test_mean': cv_results['test_score'].mean(),
        'test_std': cv_results['test_score'].std(),
    }
    
    if return_train_score:
        scores['train_scores'] = cv_results['train_score']
        scores['train_mean'] = cv_results['train_score'].mean()
        scores['train_std'] = cv_results['train_score'].std()
        scores['gap'] = scores['train_mean'] - scores['test_mean']
    
    logger.info(f" Validación Cruzada completada ({cv} folds)")
    logger.info(f"   Test Mean ({scoring}): {scores['test_mean']:.4f}")
    
    return scores

# ============================================================================
# ANÁLISIS DE RESIDUOS
# ============================================================================

def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Realiza un análisis estadístico de los errores de predicción (residuos).
    """
    residuals = y_true - y_pred
    
    analysis = {
        'residuals_mean': np.mean(residuals),
        'residuals_std': np.std(residuals),
        'residuals_min': np.min(residuals),
        'residuals_max': np.max(residuals),
        'residuals_median': np.median(residuals),
        'residuals_skewness': pd.Series(residuals).skew(),
        'residuals_kurtosis': pd.Series(residuals).kurtosis(),
    }
    
    return analysis

# ============================================================================
# COMPARACIÓN DE MODELOS
# ============================================================================

@timer
def compare_models(
    models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """
    Compara el rendimiento de múltiples modelos de regresión.
    """
    results = []
    
    for model_name, model in models.items():
        logger.info(f"Evaluando: {model_name}")
        y_pred = model.predict(X_test)
        metrics = calculate_regression_metrics(y_test, y_pred)
        metrics['model'] = model_name
        results.append(metrics)
    
    comparison_df = pd.DataFrame(results).set_index('model')
    comparison_df = comparison_df.sort_values('r2', ascending=False)
    
    logger.info(f"\n Comparación de {len(models)} modelos de regresión completada\n")
    
    return comparison_df

# ============================================================================
# RESUMEN DE MODELO
# ============================================================================

@timer
def get_model_summary(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = 'Model'
) -> Dict:
    """
    Genera un informe consolidado del rendimiento de un modelo de regresión.
    """
    y_pred = model.predict(X_test)
    
    summary = {
        'model_name': model_name,
        'test_samples': len(y_test),
        'metrics': calculate_regression_metrics(y_test, y_pred),
        'residuals': analyze_residuals(y_test, y_pred)
    }
    
    return summary
