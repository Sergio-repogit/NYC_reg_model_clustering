"""
Ensembles de Regresión - Airbnb NYC Data Mining
================================================

Módulo para entrenamiento de modelos avanzados con validación cruzada,
ajuste de hiperparámetros y exportación de tablas comparativas.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from evaluation.visualization import (
    plot_feature_importance,
    plot_learning_curve_diagnostic,
    plot_model_performance_comparison,
    plot_regression_diagnostics,
)
from utils.config import RANDOM_STATE, TABLES_MODELS_DIR
from utils.helpers import setup_logging, timer

logger = setup_logging(__name__)

# ============================================================================
# AUDIT Y SELECCIÓN DE MODELO (BIAS-VARIANCE)
# ============================================================================

@timer
def train_and_select_best_model(X, y) -> tuple[Any, str]:
    """
    Entrena candidatos, ajusta hiperparámetros y selecciona basado en estabilidad.
    Prioriza modelos con Overfitting Gap < 0.05.
    """
    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Definición de Candidatos y Espacio de Búsqueda Optimizado
    from sklearn.ensemble import VotingRegressor

    candidates = {
        'LinearRegression': (LinearRegression(), {}),
        'Ridge': (Ridge(random_state=RANDOM_STATE), {'alpha': [0.1, 1.0, 10.0]}),
        'Lasso': (Lasso(random_state=RANDOM_STATE), {'alpha': [0.01, 0.1, 1.0]}),
        'DecisionTree': (DecisionTreeRegressor(random_state=RANDOM_STATE), {'max_depth': [5, 10, 15]}),
        'RandomForest': (RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1), {
            'n_estimators': [100], 'max_depth': [10]
        }),
        'AdaBoost': (AdaBoostRegressor(random_state=RANDOM_STATE), {'n_estimators': [50, 100], 'learning_rate': [0.1, 1.0]}),
        'XGBoost': (XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=500), {
            'learning_rate': [0.1],
            'max_depth': [3, 5]
        }),
        'LinearSVR': (LinearSVR(dual=False, loss='squared_epsilon_insensitive', random_state=RANDOM_STATE), {'C': [0.1, 1.0]}),
        'SVR_RBF': (SVR(kernel='rbf'), {'C': [1, 10], 'gamma': ['scale']}),
        'Bagging_LSVR': (BaggingRegressor(estimator=LinearSVR(dual=False, loss='squared_epsilon_insensitive', random_state=RANDOM_STATE),
                                               n_estimators=10, n_jobs=-1, random_state=RANDOM_STATE), {
            'n_estimators': [5, 10]
        }),
        'Voting_Assembly': (VotingRegressor(
            estimators=[
                ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
                ('xgb', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE)),
                ('ridge', Ridge(alpha=1.0, random_state=RANDOM_STATE))
            ]
        ), {
            'weights': [[0.4, 0.4, 0.2], [0.33, 0.33, 0.34], [0.5, 0.3, 0.2]]
        }),
        'Stacking': (StackingRegressor(
            estimators=[
                ('ridge', Ridge(random_state=RANDOM_STATE)),
                ('lasso', Lasso(alpha=0.1, random_state=RANDOM_STATE)),
                ('lsvr', LinearSVR(dual=False, loss='squared_epsilon_insensitive', random_state=RANDOM_STATE))
            ],
            final_estimator=Lasso(alpha=0.01, random_state=RANDOM_STATE),
            cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
            n_jobs=-1
        ), {})
    }

    results = []
    best_models = {}

    total_start = time.time()
    for name, (model, params) in candidates.items():
        start_m = time.time()

        cv_fold = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

        # Ajuste de Hiperparámetros (Parallelized)
        grid = GridSearchCV(model, params, cv=cv_fold, scoring='r2', n_jobs=-1)

        # Early Stopping Logic only for XGBoost during GridSearch
        if name == 'XGBoost':
            grid.fit(X_train, y_train, eval_set=[(X_holdout, y_holdout)], verbose=False)
        else:
            grid.fit(X_train, y_train)

        m = grid.best_estimator_

        # Ensure XGBoost does not attempt early stopping in cross_val_score
        if name == 'XGBoost':
            m.set_params(early_stopping_rounds=None)

        best_models[name] = m

        # Evaluación robusta (Sin Early Stopping para evitar ValueError en folds)
        cv_scores = cross_val_score(m, X_train, y_train, cv=cv_fold, scoring='r2', n_jobs=-1)
        cv_mean = np.mean(cv_scores)
        train_r2 = m.score(X_train, y_train)
        holdout_r2 = m.score(X_holdout, y_holdout)
        y_pred = m.predict(X_holdout)
        mean_absolute_error(y_holdout, y_pred)
        np.sqrt(mean_squared_error(y_holdout, y_pred))
        y_holdout_usd = np.expm1(y_holdout)
        y_pred_usd = np.expm1(y_pred)
        mae_usd = mean_absolute_error(y_holdout_usd, y_pred_usd)
        rmse_usd = np.sqrt(mean_squared_error(y_holdout_usd, y_pred_usd))

        duration = time.time() - start_m

        # Ahora los resultados incluyen:
        results.append({
            'Modelo': name,
            'CV Mean R2': cv_mean,
            'Hold-out R2': holdout_r2,
            'MAE ($)': mae_usd,
            'RMSE ($)': rmse_usd,
            'Overfitting Gap': train_r2 - holdout_r2,
            'Total Time (s)': duration,
            'Train Time (s)': grid.refit_time_,
            'Mejores Params': str(grid.best_params_)
        })

        # Diagnósticos Visuales
        y_pred = m.predict(X_holdout)
        plot_regression_diagnostics(y_holdout, y_pred, name)
        plot_feature_importance(m, X.columns, name)
        plot_learning_curve_diagnostic(m, X_train, y_train, name)

    df_results = pd.DataFrame(results)
    df_results.to_csv(TABLES_MODELS_DIR / 'model_comparison_master.csv', index=False)

    # Selección de Ganador (Mejor R2 en Hold-out con Gap < 0.05)
    stable_models = df_results[df_results['Overfitting Gap'] < 0.05]
    if stable_models.empty:
        winner_row = df_results.sort_values('Hold-out R2', ascending=False).iloc[0]
    else:
        winner_row = stable_models.sort_values('Hold-out R2', ascending=False).iloc[0]

    winner_name = winner_row['Modelo']
    winner_model = best_models[winner_name]

    # Reporte de Auditoría de Depuración
    (time.time() - total_start) / 60
    plot_model_performance_comparison(df_results)

    return winner_model, winner_name
