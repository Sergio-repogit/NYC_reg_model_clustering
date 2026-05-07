"""
Configuración Global - Airbnb NYC Data Mining
==============================================

Definición de rutas, constantes y parámetros compartidos.
"""

from pathlib import Path

# ============================================================================
# RUTAS DEL PROYECTO
# ============================================================================

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_RAW_PATH = DATA_RAW_DIR / "AB_NYC_2019.csv"

# Resultados
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Subdirectorios de figuras
FIGURES_EDA_DIR = FIGURES_DIR / "eda"
FIGURES_PREPROCESSING_DIR = FIGURES_DIR / "preprocessing"
FIGURES_CLUSTERING_DIR = FIGURES_DIR / "clustering"
FIGURES_EVALUATION_DIR = FIGURES_DIR / "evaluation"
FIGURES_MODELS_DIR = FIGURES_DIR / "models"

# Subdirectorios de tablas
TABLES_CLUSTERING_DIR = TABLES_DIR / "clustering"
TABLES_MODELS_DIR = TABLES_DIR / "models"
TABLES_PCA_DIR = TABLES_DIR / "pca"

# ============================================================================
# CONSTANTES Y PARÁMETROS
# ============================================================================

RANDOM_STATE = 42
TARGET_COL = "price"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(message)s"
LOG_FILE = ROOT_DIR / "logs" / "pipeline.log"

# Columnas identificadas
ID_COLS = ['id', 'name', 'host_id', 'host_name']
CATEGORICAL_COLS = ['neighbourhood_group', 'neighbourhood', 'room_type']
NUMERIC_COLS = [
    'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
    'reviews_per_month', 'calculated_host_listings_count', 'availability_365'
]

def create_directories():
    """Crea la estructura completa de carpetas del proyecto."""
    dirs = [
        DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR,
        FIGURES_EDA_DIR, FIGURES_PREPROCESSING_DIR, FIGURES_CLUSTERING_DIR, 
        FIGURES_EVALUATION_DIR, FIGURES_MODELS_DIR,
        TABLES_CLUSTERING_DIR, TABLES_MODELS_DIR, TABLES_PCA_DIR,
        LOG_FILE.parent
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
