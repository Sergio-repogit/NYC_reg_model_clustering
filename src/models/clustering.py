"""
Clustering y Segmentación - Airbnb NYC Data Mining
===================================================

Módulo para clustering mediante K-Means con exportación de tablas de métricas.
Valida la estructura óptima mediante Codo, Silueta y Gap Statistic.
"""


import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils.config import RANDOM_STATE, TABLES_CLUSTERING_DIR
from utils.helpers import setup_logging, timer

logger = setup_logging(__name__)

# ============================================================================
# VALIDACIÓN DE K-MEANS
# ============================================================================

def calculate_gap_statistic(X: np.ndarray, n_clusters: int, n_references: int = 5) -> float:
    """Calcula el Gap Statistic contra una distribución nula."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10).fit(X)
    dispersion = np.log(kmeans.inertia_)

    ref_dispersions = []
    for _ in range(n_references):
        X_random = np.random.uniform(X.min(axis=0), X.max(axis=0), X.shape)
        kmeans_ref = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=5).fit(X_random)
        ref_dispersions.append(np.log(kmeans_ref.inertia_))

    return np.mean(ref_dispersions) - dispersion

def find_elbow_k(k_values: list[int], inertias: list[float]) -> int:
    """Detecta el punto del codo geométricamente."""
    p1 = np.array([k_values[0], inertias[0]])
    p2 = np.array([k_values[-1], inertias[-1]])

    distances = []
    for i in range(len(k_values)):
        p0 = np.array([k_values[i], inertias[i]])
        d = np.abs(np.cross(p2-p1, p1-p0)) / np.linalg.norm(p2-p1)
        distances.append(d)

    return k_values[np.argmax(distances)]

@timer
def validate_kmeans_strategy(
    X: pd.DataFrame,
    k_range: list[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10]
) -> tuple[pd.DataFrame, int]:
    """
    Ejecuta validación cruzada y exporta tabla de métricas (results/tables/clustering/).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        results.append({
            'k': k,
            'Inertia': kmeans.inertia_,
            'Gap': calculate_gap_statistic(X_scaled, k),
            'Silhouette': silhouette_score(X_scaled, kmeans.labels_)
        })

    df_metrics = pd.DataFrame(results).set_index('k')

    # Exportar Tabla de Métricas
    df_metrics.to_csv(TABLES_CLUSTERING_DIR / 'clustering_metrics_k.csv')

    # Selección de K
    k_elbow = find_elbow_k(k_range, df_metrics['Inertia'].tolist())
    consensus_k = int(np.median([k_elbow, df_metrics['Silhouette'].idxmax(), df_metrics['Gap'].idxmax()]))

    # Import and call visualization to generate metric images
    from evaluation.visualization import plot_clustering_validation
    plot_clustering_validation(df_metrics)

    return df_metrics, consensus_k

@timer
def train_final_kmeans(X: pd.DataFrame, n_clusters: int) -> tuple[pd.DataFrame, KMeans]:
    """Entrena el modelo final y lo exporta."""
    # X_clustering is already PCA transformed, so scaling is redundant but we keep it identical or remove it.
    # Let's remove it so we don't need a second scaler in production.
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X)
    df_res = X.copy()
    df_res['cluster'] = labels

    from pathlib import Path

    import joblib
    models_dir = Path("results/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, models_dir / "cluster_model.pkl")

    return df_res, kmeans
