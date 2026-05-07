"""
Modelos Baseline - Airbnb NYC Data Mining
==========================================

Módulo con modelos lineales y no lineales simples adaptados para la
estrategia de regresión basada en componentes principales (PCA).
"""


import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

from utils.config import (
    CV_FOLDS,
    LASSO_PARAMS,
    LINEAR_PARAMS,
    RANDOM_STATE,
    RIDGE_PARAMS,
)
from utils.helpers import setup_logging, timer

logger = setup_logging(__name__)

# ============================================================================
# REGRESIÓN LINEAL
# ============================================================================

@timer
def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    **kwargs
):
    """
    Entrena modelo de Regresión Lineal sobre componentes de PCA.
    """
    model = LinearRegression(**LINEAR_PARAMS, **kwargs)
    model.fit(X_train, y_train)

    logger.info(" Regresión Lineal entrenada")
    logger.info(f"   R2 train: {model.score(X_train, y_train):.4f}")

    return model

# ============================================================================
# REGRESIÓN REGULARIZADA (RIDGE & LASSO)
# ============================================================================

@timer
def train_ridge(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    **kwargs
):
    """
    Entrena modelo Ridge Regression para manejar multicolinealidad residual.
    """
    params = RIDGE_PARAMS.copy()
    params['alpha'] = alpha
    params.update(kwargs)

    model = Ridge(**params)
    model.fit(X_train, y_train)

    logger.info(f" Ridge Regression entrenada (alpha={alpha})")
    logger.info(f"   R2 train: {model.score(X_train, y_train):.4f}")

    return model

@timer
def train_lasso(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    alpha: float = 1.0,
    **kwargs
):
    """
    Entrena modelo Lasso Regression para selección de componentes dominantes.
    """
    params = LASSO_PARAMS.copy()
    params['alpha'] = alpha
    params.update(kwargs)

    model = Lasso(**params)
    model.fit(X_train, y_train)

    n_features_selected = np.sum(model.coef_ != 0)
    logger.info(f" Lasso Regression entrenada (alpha={alpha})")
    logger.info(f"   Componentes activos: {n_features_selected}/{len(model.coef_)}")
    logger.info(f"   R2 train: {model.score(X_train, y_train):.4f}")

    return model

# ============================================================================
# ÁRBOLES DE DECISIÓN
# ============================================================================

@timer
def train_decision_tree_regressor(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_depth: int = 5,
    min_samples_split: int = 2,
    **kwargs
):
    """
    Entrena Árbol de Decisión para Regresión sobre el espacio latente.
    """
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=RANDOM_STATE,
        **kwargs
    )
    model.fit(X_train, y_train)

    logger.info(" Árbol de Decisión (Regresión) entrenado")
    logger.info(f"   R2 train: {model.score(X_train, y_train):.4f}")

    return model

# ============================================================================
# GLM CON STATSMODELS
# ============================================================================

@timer
def train_glm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    family='gaussian'
):
    """
    Entrena GLM (Generalized Linear Model) para análisis estadístico de componentes.
    """
    X_train_const = sm.add_constant(X_train)

    if family == 'gaussian':
        fam = sm.families.Gaussian()
    elif family == 'poisson':
        fam = sm.families.Poisson()
    else:
        raise ValueError(f"Familia no compatible con regresión de precios: {family}")

    model = sm.GLM(y_train, X_train_const, family=fam)
    result = model.fit()

    logger.info(f" GLM entrenado (familia={family})")
    logger.info(f"   Log-Likelihood: {result.llf:.2f}")

    return result

# ============================================================================
# OPTIMIZACIÓN DE HIPERPARÁMETROS
# ============================================================================

@timer
def optimize_baseline_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = 'ridge',
    cv: int = CV_FOLDS
):
    """
    Optimiza hiperparámetros para modelos baseline de regresión.
    """
    if model_type == 'ridge':
        model = Ridge(random_state=RANDOM_STATE)
        param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}
    elif model_type == 'lasso':
        model = Lasso(random_state=RANDOM_STATE)
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1]}
    elif model_type == 'decision_tree':
        model = DecisionTreeRegressor(random_state=RANDOM_STATE)
        param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
    else:
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    grid_search = GridSearchCV(
        model, param_grid, cv=cv,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    logger.info(f" Optimización {model_type} finalizada. Mejores parámetros: {grid_search.best_params_}")
    return grid_search.best_estimator_

# ============================================================================
# ORQUESTACIÓN DE BASELINES
# ============================================================================

@timer
def train_all_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> dict:
    """
    Entrena la suite completa de modelos baseline de regresión.
    """
    logger.info(" Entrenando suite de regresión baseline sobre componentes PCA...")

    baselines = {
        'Linear Regression': train_linear_regression(X_train, y_train),
        'Ridge': train_ridge(X_train, y_train),
        'Lasso': train_lasso(X_train, y_train),
        'Decision Tree': train_decision_tree_regressor(X_train, y_train)
    }

    return baselines

def get_baseline_feature_importance(
    model,
    X_train: pd.DataFrame
) -> pd.DataFrame:
    """
    Obtiene la importancia de los componentes PCA para el modelo baseline.
    """
    if hasattr(model, 'coef_'):
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': np.abs(model.coef_)
        })
    elif hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        })
    else:
        return pd.DataFrame()

    return importance.sort_values('importance', ascending=False)
