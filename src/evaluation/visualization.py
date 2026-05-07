"""
Suite de Visualización Avanzada - Airbnb NYC Data Mining
=========================================================

Generación de gráficos diagnósticos multimodelo, análisis de importancia
de variables y biplots de PCA.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import learning_curve

from utils.config import (
    FIGURES_CLUSTERING_DIR,
    FIGURES_EDA_DIR,
    FIGURES_MODELS_DIR,
)

# ============================================================================
# EDA Y ANÁLISIS DESCRIPTIVO
# ============================================================================


def plot_clustering_validation(df_metrics: pd.DataFrame):
    """Genera las gráficas de validación para Clustering (Codo, Silueta, Gap)."""
    FIGURES_CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(df_metrics.index, df_metrics["Inertia"], marker="o")
    plt.title("Método del Codo (Elbow Method)")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Inercia")
    plt.grid(True)
    plt.savefig(FIGURES_CLUSTERING_DIR / "elbow_method.png")
    plt.close()

    # 2. Silhouette
    plt.figure(figsize=(8, 5))
    plt.plot(df_metrics.index, df_metrics["Silhouette"], marker="o", color="green")
    plt.title("Análisis de Silueta")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.savefig(FIGURES_CLUSTERING_DIR / "silhouette_analysis.png")
    plt.close()

    # 3. Gap Statistic
    plt.figure(figsize=(8, 5))
    plt.plot(df_metrics.index, df_metrics["Gap"], marker="o", color="orange")
    plt.title("Gap Statistic")
    plt.xlabel("Número de Clusters (K)")
    plt.ylabel("Gap Value")
    plt.grid(True)
    plt.savefig(FIGURES_CLUSTERING_DIR / "gap_statistic.png")
    plt.close()


def plot_distributions(df: pd.DataFrame, columns: list, title: str, filename: str):
    """Genera histogramas con KDE para las columnas especificadas."""
    FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
    n_cols = len(columns)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    if n_cols == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue")
        axes[i].set_title(f"Distribución de {col}")
        axes[i].set_xlabel(col)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(FIGURES_EDA_DIR / filename)
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame):
    """Genera un mapa de calor de la matriz de correlación."""
    FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
    corr = df.select_dtypes(include=[np.number]).corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Matriz de Correlación de Variables Numéricas")
    plt.tight_layout()
    plt.savefig(FIGURES_EDA_DIR / "correlation_matrix.png")
    plt.close()


def plot_numerical_eda(df: pd.DataFrame, columns: list):
    """Genera visualización combinada (Histograma + Boxplot) para variables numéricas."""
    FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
    for col in columns:
        if col not in df.columns:
            continue

        fig, (ax_box, ax_hist) = plt.subplots(
            2, sharex=True, gridspec_kw={"height_ratios": (0.15, 0.85)}, figsize=(10, 8)
        )

        sns.boxplot(x=df[col], ax=ax_box, color="skyblue")
        sns.histplot(x=df[col], ax=ax_hist, kde=True, color="skyblue")

        ax_box.set(xlabel="")
        ax_box.set_title(f"Análisis de Distribución y Outliers: {col}")
        plt.savefig(FIGURES_EDA_DIR / f"eda_num_{col}.png")
        plt.close()


def plot_categorical_eda(df: pd.DataFrame, columns: list):
    """Genera gráficos de sectores (Pie Charts) para variables categóricas."""
    FIGURES_EDA_DIR.mkdir(parents=True, exist_ok=True)
    for col in columns:
        if col not in df.columns:
            continue

        plt.figure(figsize=(10, 8))
        df[col].value_counts().plot.pie(
            autopct="%1.1f%%", startangle=90, cmap="Pastel1"
        )
        plt.title(f"Distribución de Categorías: {col}")
        plt.ylabel("")
        plt.savefig(FIGURES_EDA_DIR / f"eda_cat_{col}.png")
        plt.close()


# ============================================================================
# PREPROCESAMIENTO Y PCA
# ============================================================================


def plot_pca_biplot(X_scaled, pca, directory: Path):
    """Genera un Biplot de PCA mostrando las cargas de las variables."""
    directory.mkdir(parents=True, exist_ok=True)
    coords = pca.components_.T[:, :2]
    features = X_scaled.columns

    plt.figure(figsize=(12, 10))
    for i, feature in enumerate(features):
        plt.arrow(
            0, 0, coords[i, 0], coords[i, 1], color="r", alpha=0.5, head_width=0.02
        )
        plt.text(
            coords[i, 0] * 1.15,
            coords[i, 1] * 1.15,
            feature,
            color="g",
            ha="center",
            va="center",
        )

    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("PCA Biplot: Influencia de Variables Originales")
    plt.grid()
    plt.savefig(directory / "pca_biplot.png")
    plt.close()


def plot_pca_variance(pca, directory: Path):
    """Genera Scree Plot y Varianza Acumulada."""
    directory.mkdir(parents=True, exist_ok=True)
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    plt.figure(figsize=(12, 6))

    # Bar plot for individual variance
    plt.bar(
        range(1, len(exp_var) + 1),
        exp_var,
        alpha=0.5,
        align="center",
        label="Varianza Individual",
    )
    # Step plot for cumulative variance
    plt.step(
        range(1, len(cum_var) + 1), cum_var, where="mid", label="Varianza Acumulada"
    )

    plt.ylabel("Ratio de Varianza Explicada")
    plt.xlabel("Componentes Principales")
    plt.title("PCA: Scree Plot y Varianza Acumulada")
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(directory / "pca_variance_scree.png")
    plt.close()


# ============================================================================
# COMPARACIÓN DE MODELOS
# ============================================================================


def plot_model_performance_comparison(df_comparison: pd.DataFrame):
    """Compara métricas (R2, RMSE, MAE) entre todos los modelos evaluados."""
    FIGURES_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = ["CV Mean R2", "Hold-out R2"]

    plt.figure(figsize=(12, 6))
    df_melted = df_comparison.melt(id_vars="Modelo", value_vars=metrics)
    sns.barplot(data=df_melted, x="value", y="Modelo", hue="variable")
    plt.title("Comparación de Rendimiento R²: CV vs Hold-out")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_MODELS_DIR / "model_comparison_r2.png")
    plt.close()


def plot_feature_importance(model, features, model_name: str):
    """Genera gráfico de importancia de variables para modelos basados en árboles o lineales."""
    importances = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances[0]

    if importances is not None:
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 8))
        plt.title(f"Importancia de Variables: {model_name}")
        plt.barh(range(len(indices[:15])), importances[indices[:15]], align="center")
        plt.yticks(range(len(indices[:15])), [features[i] for i in indices[:15]])
        plt.xlabel("Importancia (Absoluta para lineales)")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_MODELS_DIR / f"{model_name}_feature_importance.png")
        plt.close()


# ============================================================================
# DIAGNÓSTICOS DE REGRESIÓN
# ============================================================================


def plot_regression_diagnostics(y_true, y_pred, model_name: str):
    """Genera gráficos de Predicción vs Real y Residuos para un modelo."""
    FIGURES_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Real vs Predicho
    ax1.scatter(y_true, y_pred, alpha=0.3)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
    ax1.set_xlabel("Valores Reales")
    ax1.set_ylabel("Predicciones")
    ax1.set_title(f"{model_name}: Real vs Predicho")

    # Residuos
    residuals = y_true - y_pred
    sns.histplot(residuals, kde=True, ax=ax2)
    ax2.set_xlabel("Residuo")
    ax2.set_title(f"{model_name}: Distribución de Residuos")

    plt.tight_layout()
    plt.savefig(FIGURES_MODELS_DIR / f"{model_name}_diagnostics.png")
    plt.close()


def plot_learning_curve_diagnostic(model, X, y, model_name: str):
    """Genera Curvas de Aprendizaje para diagnóstico de Bias vs Variance."""
    from utils.config import FIGURES_EVALUATION_DIR

    FIGURES_EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=3, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring="r2"
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
    plt.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")

    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g"
    )

    plt.title(f"Curva de Aprendizaje: {model_name}")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("R² Score")
    plt.legend(loc="best")
    plt.grid()
    plt.savefig(FIGURES_EVALUATION_DIR / f"{model_name}_learning_curve.png")
    plt.close()
