"""
Módulo de Entrenamiento de Modelos - NYC Airbnb
================================================
Wrapper para orquestar el entrenamiento y evaluación.
"""

import pandas as pd
from models.ensemble_models import train_xgboost
from evaluation.metrics import calculate_regression_metrics
from utils.config import TARGET_COL, ID_COLS
from utils.helpers import setup_logging, timer

logger = setup_logging(__name__)

@timer
def train_model(df):
    """
    Entrena el modelo final y lo evalúa.
    """
    logger.info(" Iniciando entrenamiento del modelo...")
    
    # Solo conservar columnas numéricas para el modelo
    X = df.select_dtypes(include=['number'])
    
    # Asegurar que el target esté separado
    if TARGET_COL in X.columns:
        y = X[TARGET_COL]
        X = X.drop(columns=[TARGET_COL])
    else:
        y = df[TARGET_COL]
        
    logger.info(f" Features finales para entrenamiento: {list(X.columns)}")
    
    # Entrenar XGBoost
    model = train_xgboost(X, y)
    
    # Evaluación
    y_pred = model.predict(X)
    metrics = calculate_regression_metrics(y, y_pred)
    
    logger.info(f" Resultados Entrenamiento: {metrics}")
    
    return model, metrics
