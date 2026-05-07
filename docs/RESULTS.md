# Resultados del Proyecto: Predicción y Segmentación de Airbnb en NYC

Este documento sintetiza los resultados finales obtenidos tras ejecutar el pipeline completo de machine learning sobre el dataset de Airbnb NYC 2019 (48,895 propiedades).

## 1. Ingeniería de Características e Inteligencia Espacial

El enriquecimiento del dataset mediante inteligencia espacial resultó fundamental para el rendimiento del modelo. La adición de distancias geográficas exactas (`haversine`) a los principales hitos de la ciudad (Times Square, Central Park, Grand Central) proporcionó a los modelos una señal mucho más fuerte sobre la centralidad de la propiedad, reduciendo la dependencia exclusiva de los nombres de los barrios.

## 2. Reducción de Dimensionalidad (PCA)

La aplicación de **Análisis de Componentes Principales (PCA)** permitió sintetizar un espacio de variables altamente correlacionado (como la disponibilidad a 365 días y el ratio de reviews) sin sacrificar la interpretabilidad general. Esto ayudó a mitigar la multicolinealidad antes de entrenar algoritmos de regresión lineal y sirvió como base estable para el aprendizaje no supervisado.

## 3. Segmentación del Mercado (Clustering)

A través del algoritmo K-Means con selección de hiperparámetros por consenso (mediana entre Método del Codo, Silhouette y Gap Statistic), se logró una segmentación automática del mercado de alojamientos.

*   **Identidad de los Clústeres:** La variable `room_type` (tipo de alojamiento) y el vector de `price` probaron ser los ejes discriminantes de la oferta. 
*   **Resultados:** Se separó la oferta premium (alojamientos enteros céntricos y de alta disponibilidad) de las opciones más económicas (habitaciones compartidas periféricas), creando perfiles estadísticos limpios para cada clúster.

## 4. Rendimiento de Modelos Predictivos

Se entrenó una suite masiva de regresores, desde modelos lineales simples hasta arquitecturas avanzadas de ensemble (Boosting, Bagging, Voting y Stacking). A continuación, se presenta un resumen de rendimiento (validación en Hold-out test set):

| Modelo | Hold-out R² | MAE ($) | RMSE ($) | Overfitting Gap |
| :--- | :--- | :--- | :--- | :--- |
| **SVR_RBF** | **0.5898** | **54.29** | **194.85** | **-0.0026** |
| Voting_Assembly | 0.5866 | 55.15 | 193.95 | 0.0414 |
| XGBoost | 0.5863 | 55.40 | 193.10 | 0.0287 |
| RandomForest | 0.5794 | 55.49 | 194.48 | 0.0760 |
| LinearRegression | 0.5471 | 57.32 | 196.29 | -0.0199 |

*Nota: Los modelos con un "Overfitting Gap" mayor a 0.05 (como Random Forest) fueron penalizados en la selección para garantizar estabilidad.*

### El Modelo Ganador

El algoritmo de regresión **SVR_RBF** (Support Vector Regression con kernel Radial) fue seleccionado dinámicamente como el ganador absoluto.
*   Logró el **R² más alto (58.98%)** entre los modelos estables.
*   Alcanzó el **MAE más bajo ($54.29 USD)**.
*   Presentó un Overfitting Gap negativo (-0.0026), lo cual indica una excelente capacidad de generalización sobre datos nunca antes vistos.

## 5. Conclusión y Trabajo Futuro

El pipeline ha logrado extraer todo el valor predictivo posible de los metadatos tabulares estructurados (coordenadas, tipos de cuarto, número de reviews). Para traspasar la barrera actual del ~60% de varianza explicada, se recomienda enfáticamente:

1.  **NLP y Análisis de Texto:** Extraer sentimientos y características explícitas a partir de las descripciones escritas y las reseñas de los usuarios.
2.  **Modelos Especializados:** Dividir el dataset y entrenar ensamblajes independientes por `neighbourhood_group` (ej. un modelo exclusivo para Manhattan y otro para la periferia).
3.  **Actualización Dinámica:** Conectar el simulador de Streamlit a una API para obtener las fluctuaciones temporales reales de ocupación, capturando el efecto estacional que el dataset estático original no provee.
