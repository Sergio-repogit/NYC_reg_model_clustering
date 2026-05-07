"""
Modelos Flexibles - Airbnb NYC Data Mining
===========================================

Módulo con modelos no paramétricos y de alta flexibilidad: Splines, GAM, SVM y KNN.
Optimizado exclusivamente para tareas de regresión de precios.
"""

import pandas as pd
from pygam import LinearGAM, s
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

from utils.helpers import setup_logging, timer

logger = setup_logging(__name__)

# ============================================================================
# GAM (GENERALIZED ADDITIVE MODELS) CON SPLINES
# ============================================================================


@timer
def train_gam(X_train: pd.DataFrame, y_train: pd.Series, n_splines: int = 25):
    """
    Entrena un GAM lineal usando splines para capturar relaciones no lineales.
    """
    # Construcción dinámica de términos s(i) para todas las columnas
    terms = s(0, n_splines=n_splines)
    for i in range(1, X_train.shape[1]):
        terms += s(i, n_splines=n_splines)

    model = LinearGAM(terms)
    model.fit(X_train, y_train)

    logger.info(
        f" GAM (Splines) entrenado. Pseudo-R2: {model.statistics_['pseudo_r2']['explained_deviance']:.4f}"
    )
    return model


# ============================================================================
# SUPPORT VECTOR REGRESSION (SVR)
# ============================================================================


@timer
def train_svr(X_train, y_train, kernel="rbf", C=1.0, epsilon=0.1):
    """
    Entrena SVR con escalado interno mandatorio.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = SVR(kernel=kernel, C=C, epsilon=epsilon)
    model.fit(X_scaled, y_train)
    model.scaler = scaler

    logger.info(f" SVR ({kernel}) entrenado. R2: {model.score(X_scaled, y_train):.4f}")
    return model


# ============================================================================
# K-NEAREST NEIGHBORS (KNN REGRESSOR)
# ============================================================================


@timer
def train_knn_regressor(X_train, y_train, n_neighbors=5):
    """
    Entrena KNN para regresión sobre el espacio de componentes.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)
    model.fit(X_scaled, y_train)
    model.scaler = scaler

    logger.info(f" KNN Regressor entrenado (k={n_neighbors})")
    return model


# ============================================================================
# REGRESIÓN POLINOMIAL (PIPELINE)
# ============================================================================


@timer
def train_polynomial_regression(X_train, y_train, degree=2):
    """
    Pipeline de regresión polinomial.
    """
    model = Pipeline(
        [("poly", PolynomialFeatures(degree=degree)), ("linear", LinearRegression())]
    )
    model.fit(X_train, y_train)

    logger.info(f" Regresión Polinomial (grado={degree}) entrenada")
    return model


# ============================================================================
# ORQUESTACIÓN
# ============================================================================


@timer
def train_all_flexible_models(X_train, y_train) -> dict:
    """Entrena la suite completa de modelos flexibles."""
    return {
        "GAM": train_gam(X_train, y_train),
        "SVR": train_svr(X_train, y_train),
        "KNN": train_knn_regressor(X_train, y_train),
        "Polynomial": train_polynomial_regression(X_train, y_train),
    }
