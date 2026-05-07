# Predicción de Precios y Clustering en Airbnb NYC

> Proyecto final — Minería de Datos · Grado en Matemáticas · UNIE Universidad

[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](#)
![Python](https://img.shields.io/badge/python-3.13-blue)
![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)
[![CI](https://github.com/Sergio-repogit/NYC_reg_model_clustering/actions/workflows/ci.yml/badge.svg)](https://github.com/Sergio-repogit/NYC_reg_model_clustering/actions/workflows/ci.yml)
[![Docs](https://github.com/Sergio-repogit/NYC_reg_model_clustering/actions/workflows/docs.yml/badge.svg)](https://sergio-repogit.github.io/NYC_reg_model_clustering/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
---

## Description

Este proyecto implementa un pipeline integral de Machine Learning para predecir precios de alojamientos y agrupar vecindarios de Airbnb en la ciudad de Nueva York. Utilizando regresión y clustering no jerárquico, el estudio aplica técnicas de preprocesamiento,incluyendo PCA y codificación híbrida, optimizando el rendimiento mediante una arquitectura en backend que se sincroniza con un dashboard interactivo.

## Documentation

Full documentation en **[docs/RESULTS.md](docs/RESULTS.md)** e **[informe_tecnico/main.pdf](docs/informe_tecnico/main.pdf)**.

## Installation

```bash
git clone https://github.com/Sergio-repogit/NYC_reg_model_clustering.git
cd NYC_reg_model_clustering
pip install uv
uv sync --group dev
```

## Execution

Para ejecutar el pipeline completo de entrenamiento (Preprocesamiento -> Regresión -> Clustering):

```bash
python -m src.models.train_model
```

## Development Commands

```bash
python -m ruff check .                    # linter
python -m ruff format .                   # formateador
```

## Project Structure

```
NYC_reg_model_clustering/
├── src/                   # Código modular
│   ├── data/              # Carga y preprocesamiento de datos
│   ├── evaluation/        # Métricas (RMSE, MAE) y visualización
│   ├── models/            # Modelos de regresión, ensemble y clustering
│   └── utils/             # Configuración y utilidades
├── tests/                 # Tests unitarios
├── docs/                  # Documentación y reportes
│   ├── RESULTS.md
│   └── informe_tecnico/
├── data/                  # Datos crudos y procesados
├── results/               
│   ├── tables/            # Tablas CSV generadas dinámicamente
│   ├── figures/           # Gráficos y matrices
│   └── models/            # Modelos serializados (Pickle)
├── app/                   # Aplicación web en Streamlit
├── pyproject.toml         # Dependencias
├── LICENSE                # MIT License
└── README.md              # Este archivo
```

## Model Performance

Los resultados son cargados dinámicamente desde `results/tables/model_comparison.csv` en el entorno de evaluación.

---

## Methodology

El análisis se centra en la aplicación de algoritmos paramétricos y basados en árboles, junto a técnicas de segmentación:
* **Hybrid Encoding:** Combinación de Target Encoding para variables categóricas de alta cardinalidad (ej. vecindarios) y One-Hot Encoding para el resto.
* **Dimensionality Reduction (PCA):** Aplicado sobre el espacio de características para eliminar colinealidad y optimizar la convergencia de modelos lineales.
* **Regression Modeling:** Entrenamiento comparativo y ajuste de hiperparámetros (Grid Search).
* **K-Means Clustering:** Segmentación particional basada en el consenso de índices de validación interna (Silhouette, GAP e Inercia) para extraer perfiles de alojamientos.

## Feature Engineering Process

Dada la complejidad del mercado de Airbnb, se aplica un pipeline propio:
1. **Outlier Treatment:** Aplicación de log-transform sobre variables para mitigar el sesgo a la derecha.
2. **Missing Value Imputation:** Imputación basada en distribuciones locales e iterativa, pudiendo elegir la métrica a utilizar (mediana o media).
3. **Scaling:** Normalización y estandarización para asegurar el correcto desempeño del algoritmo K-Means y los estimadores lineales.

## Data Source & Acquisition

El proyecto utiliza el dataset público de **Airbnb NYC (Inside Airbnb)**.
* **Resolución:** Registro individual por alojamiento (Listing).
* **Características:** Incluye precio, geolocalización, tipo de habitación, reseñas y disponibilidad.

## Dashboard Interactivo

Se ha desarrollado una aplicación web con **Streamlit** para la exploración fluida de los resultados.

* **Exploración Dinámica:** Gráficos del EDA, PCA, y métricas de clustering servidas directamente desde los assets del backend.
* **Inferencia en Tiempo Real:** Un motor de estimación de precio integrado que permite ingresar características manuales y predecir el coste usando el modelo exportado.
* **Auditoría Continua:** Sincronización automática con los directorios de salida del pipeline de datos.

**Para ejecutar el dashboard localmente:**

```bash
# Entorno virtual con uv
uv run streamlit run app/main.py
```

## Uso de herramientas de IA

Durante el desarrollo de este proyecto se han utilizado herramientas de inteligencia artificial como apoyo puntual, principalmente para consultas específicas de implementación, resolución de errores y mejora de la eficiencia del código.

En concreto, se han empleado modelos de lenguaje Claude Sonnet 4.5 y Gemini 3.1 como asistentes para:
- Aclarar dudas sobre librerías y funciones concretas.
- Sugerir posibles enfoques de implementación.
- Ayudar en la depuración de errores puntuales.

El diseño del proyecto, la selección de metodologías, el desarrollo principal del código y la interpretación de los resultados han sido realizados de forma autónoma. Las herramientas de IA se han utilizado únicamente como soporte, de manera similar a la consulta de documentación técnica o foros especializados.

Se garantiza la comprensión completa de todo el código presentado, así como de las decisiones metodológicas adoptadas.

## Author

**Sergio Mínguez Cruces** · [github.com/Sergio-repogit](https://github.com/Sergio-repogit)

---

*Minería de Datos · Grado en Matemáticas · UNIE Universidad · 2025–2026*
