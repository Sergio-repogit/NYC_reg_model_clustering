"""
Script Maestro de Ejecución (main.py) - NYC Airbnb Data Mining
============================================================
Orquestador del pipeline completo con generación masiva de tablas e imágenes.
"""

import sys
import argparse
from pathlib import Path

# Inyección de ruta raíz
root = Path(__file__).resolve().parent
if str(root / "src") not in sys.path:
    sys.path.append(str(root / "src"))

from utils.config import create_directories, NUMERIC_COLS, CATEGORICAL_COLS
from utils.helpers import setup_logging, print_section, timer
from data.load_data import load_raw_data
from data.feature_engineering import complete_feature_engineering
from models.clustering import validate_kmeans_strategy, train_final_kmeans
from models.ensemble_models import train_and_select_best_model
from evaluation.visualization import (
    plot_distributions, plot_correlation_matrix, 
    plot_numerical_eda, plot_categorical_eda
)
import joblib
logger = setup_logging(__name__)

@timer
def main():
    parser = argparse.ArgumentParser(description="Pipeline Maestro NYC Airbnb")
    parser.add_argument("--all", action="store_true", help="Ejecutar pipeline completo")
    args = parser.parse_args()

    if not args.all:
        parser.print_help()
        return

    # Inicialización de arquitectura de resultados
    create_directories()
    
    # 1. Ingestión y EDA
    print_section("FASE 1: INGESTIÓN Y EDA")
    df_raw = load_raw_data()
    plot_distributions(df_raw, ['price', 'availability_365'], "EDA Inicial", "eda_initial.png")
    plot_correlation_matrix(df_raw)
    plot_numerical_eda(df_raw, NUMERIC_COLS)
    plot_categorical_eda(df_raw, CATEGORICAL_COLS)
    
    # 2. Ingeniería de Variables y PCA
    print_section("FASE 2: FEATURE ENGINEERING Y PCA")
    df_pca, fe_info = complete_feature_engineering(df_raw)
    
    # 3. Clustering (Segmentación con Tabulación)
    print_section("FASE 3: CLUSTERING DE MERCADO")
    X_clustering = df_pca.drop(columns=[c for c in df_pca.columns if 'price' in c])
    df_metrics, consensus_k = validate_kmeans_strategy(X_clustering)
    df_clustered, kmeans_model = train_final_kmeans(X_clustering, n_clusters=consensus_k)
    
    # Save the final processed data with the cluster column and engineered features
    df_final_data = df_raw.copy()
    for col in ['dist_times_square', 'dist_central_park', 'dist_grand_central', 'is_new_listing']:
        if col in fe_info['df_fe'].columns:
            df_final_data[col] = fe_info['df_fe'][col]
    
    df_final_data['cluster'] = df_clustered['cluster']
    
    final_csv_path = root / 'results' / 'tables' / 'final_processed_data.csv'
    df_final_data.to_csv(final_csv_path, index=False)

    # 4. Modelado Predictivo (Auditoría Bias-Variance)
    print_section("FASE 4: MODELADO PREDICTIVO Y AUDITORÍA")
    X_train = df_pca.drop(columns=['price', 'price_log'])
    y_train = df_pca['price_log']
    
    # Esta función ahora centraliza la generación de tablas y diagnósticos multimodelo
    best_model, winner_name = train_and_select_best_model(X_train, y_train)

    joblib.dump(best_model, root / 'results' / 'models' / 'best_model.pkl')

    print_section("PIPELINE FINALIZADO EXITOSAMENTE")

if __name__ == "__main__":
    main()
