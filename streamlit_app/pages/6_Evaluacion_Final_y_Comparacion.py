"""
Evaluación Final y Comparación (Simulador Interactivo)
=======================================================
Motor de inferencia dinámica basado en el modelo ganador.
Inspirado en la lógica de practica.py.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Price Estimator", layout="wide")

st.title("Evaluación Final y Simulador de Precios")

# ============================================================================
# CARGA DE ARTEFACTOS Y DATOS
# ============================================================================

# Inyección de ruta raíz robusta
root_dir = Path(__file__).resolve().parent.parent.parent

@st.cache_resource
def load_inference_artifacts():
    try:
        model = joblib.load(root_dir / "results" / "models" / "best_model.pkl")
        scaler = joblib.load(root_dir / "results" / "models" / "scaler.pkl")
        pca = joblib.load(root_dir / "results" / "models" / "pca_model.pkl")
        kmeans = joblib.load(root_dir / "results" / "models" / "cluster_model.pkl")
        return model, scaler, pca, kmeans
    except Exception as e:
        st.error(f"Error al cargar modelos de producción: {e}")
        return None, None, None, None

@st.cache_data
def load_ref_data():
    try:
        df_raw = pd.read_csv(root_dir / "data" / "raw" / "AB_NYC_2019.csv")
        df_final = pd.read_csv(root_dir / "results" / "tables" / "final_processed_data.csv")
        return df_raw, df_final
    except Exception:
        # Fallback si no está el procesado
        return pd.read_csv(root_dir / "data" / "raw" / "AB_NYC_2019.csv"), None

model, scaler, pca, kmeans = load_inference_artifacts()
df_ref, df_final = load_ref_data()

# ============================================================================
# INTERFAZ DE USUARIO
# ============================================================================
if model and df_ref is not None:
    st.markdown("### Ingrese los detalles del alojamiento para obtener una estimación:")
    
    col_in, col_out = st.columns([1, 1.2])
    
    with col_in:
        # 1. Selectores de zona
        distrito = st.selectbox("Distrito principal (Neighbourhood Group):", sorted(df_ref['neighbourhood_group'].unique()))
        barrios = sorted(df_ref[df_ref['neighbourhood_group'] == distrito]['neighbourhood'].unique())
        barrio = st.selectbox("Vecindario (Neighbourhood):", barrios)

        # 2. Características
        room_type = st.selectbox("Tipo de Alojamiento:", sorted(df_ref['room_type'].unique()))
        
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            min_nights = st.slider("Mínimo de noches:", 1, 30, 1)
        with col_s2:
            availability = st.slider("Disponibilidad anual (días):", 0, 365, 180)
            
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            reviews = st.number_input("Número de reseñas:", min_value=0, value=10)
        with col_r2:
            is_new = st.checkbox("¿Listing Nuevo?", value=False)
        with col_r3:
            days_review = st.number_input("Días desde última reseña:", min_value=0, value=30)
        
        subset = df_final[
            (df_final["neighbourhood_group"] == distrito) &
            (df_final["neighbourhood"] == barrio)
                    ]
        
        st.markdown("### Selecciona la ubicación en el mapa")

        #Límites y centro de la latitud y longitud del barrio
        lat_min, lat_max = subset["latitude"].min(), subset["latitude"].max()
        lon_min, lon_max = subset["longitude"].min(), subset["longitude"].max()
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2

        # Crear el mapa centrado en el barrio y creación del cuadrdado para indicar al usuario el barrio
        m = folium.Map(location=[lat_center, lon_center], zoom_start=14) 
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color="blue", fill=True, fill_opacity=0.1
        ).add_to(m)
        folium.LatLngPopup().add_to(m)

        # Mostrar el mapa y guardar el click
        map_output = st_folium(m, width=700, height=500)

        # Se inicieliza la latitud y longitud del click en None por si el usuario no pulsara el mapa
        lat_click, lon_click = None, None
        if map_output["last_clicked"] is not None:
            lat_click = map_output["last_clicked"]["lat"]
            lon_click = map_output["last_clicked"]["lng"]
            st.success(f"Ubicación seleccionada: lat={lat_click:.5f}, lon={lon_click:.5f}")

        if lat_click != None:
            if ((lat_min >= lat_click) or (lat_click >= lat_max) or (lon_min >= lon_click) or  (lon_click >= lon_max)):
                st.warning("La ubicación seleccionada está fuera del barrio. Se usará el centro del área.")
                lat_click, lon_click = lat_center, lon_center

        
        predict_btn = st.button("Estimar Precio y Cluster", use_container_width=True)

    with col_out:
        if predict_btn:
            # LÓGICA DE INFERENCIA
            try:
                # 1. Feature Engineering en tiempo real (Coincidencia con Pipeline)
                def haversine_np(lat1, lon1, lat2, lon2):
                    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
                    dlon = lon2 - lon1
                    dlat = lat2 - lat1
                    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
                    c = 2 * np.arcsin(np.sqrt(a))
                    km = 6367 * c
                    return km
                
                # Crear vector de entrada basado en los features del scaler
                input_dict = {col: 0 for col in scaler.feature_names_in_}
                
                # Rellenar valores conocidos
                if 'latitude' in input_dict: input_dict['latitude'] = lat_click
                if 'longitude' in input_dict: input_dict['longitude'] = lon_click
                
                # Variables espaciales
                if 'dist_times_square' in input_dict: input_dict['dist_times_square'] = haversine_np(lat_click, lon_click, 40.7580, -73.9855)
                if 'dist_central_park' in input_dict: input_dict['dist_central_park'] = haversine_np(lat_click, lon_click, 40.7812, -73.9665)
                if 'dist_grand_central' in input_dict: input_dict['dist_grand_central'] = haversine_np(lat_click, lon_click, 40.7527, -73.9772)
                
                if 'minimum_nights_log' in input_dict: input_dict['minimum_nights_log'] = np.log1p(min_nights)
                if 'number_of_reviews_log' in input_dict: input_dict['number_of_reviews_log'] = np.log1p(reviews)
                if 'availability_365' in input_dict: input_dict['availability_365'] = availability
                if 'is_new_listing' in input_dict: input_dict['is_new_listing'] = 1 if is_new else 0
                if 'days_since_review_log' in input_dict: input_dict['days_since_review_log'] = np.log1p(days_review)
                
                # OHE features (Aproximación para el Dashboard)
                rt_col = f"room_type_{room_type}"
                if rt_col in input_dict: input_dict[rt_col] = 1
                ng_col = f"neighbourhood_group_{distrito}"
                if ng_col in input_dict: input_dict[ng_col] = 1
                
                # Target encoding approx
                if 'neighbourhood' in input_dict:
                    precio_medio = subset['price'].mean() if not subset.empty else df_ref['price'].mean()
                    input_dict['neighbourhood'] = np.log1p(precio_medio)
                    
                X_input = pd.DataFrame([input_dict])
                
                # 3. Aplicar Scaler y PCA
                X_scaled = scaler.transform(X_input)
                X_pca = pca.transform(X_scaled)
                X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(X_pca.shape[1])])
                
                # 4. Predicción del Modelo
                pred_log = model.predict(X_pca_df)[0]
                precio_final = np.expm1(pred_log)
                
                # UI
                st.header("Resultado de la Estimación")
                st.metric("Precio Estimado por Noche", f"${precio_final:.2f} USD")
                
                # 2. Identificación de Cluster
                if kmeans is not None:
                    # Drop price columns if needed for clustering
                    # But KMeans was trained on df_pca excluding price
                    cluster_id = kmeans.predict(X_pca_df)[0]
                    st.info(f"Este alojamiento pertenece al **Cluster {cluster_id}** según el análisis de segmentación.")
                else:
                    st.info("Cluster no disponible.")
                
                # 3. Comparativa con Vecindario
                st.markdown("### Comparativa de Precios")
                avg_neigh = subset['price'].mean() if not subset.empty else df_ref['price'].mean()
                
                if df_final is not None and kmeans is not None:
                    # Comparativa en base a Cluster en lugar del barrio entero si es posible
                    cluster_subset = df_final[df_final['cluster'] == cluster_id]
                    avg_cluster = cluster_subset['price'].mean() if not cluster_subset.empty else avg_neigh
                    comp_data = pd.DataFrame({
                        "Nivel": [f"Media en {barrio}", f"Media en Cluster {cluster_id}", "Tu Estimación"],
                        "Precio": [avg_neigh, avg_cluster, precio_final]
                    })
                else:
                    avg_group = df_ref[df_ref['neighbourhood_group'] == distrito]['price'].mean()
                    comp_data = pd.DataFrame({
                        "Nivel": [f"Media en {barrio}", f"Media en {distrito}", "Tu Estimación"],
                        "Precio": [avg_neigh, avg_group, precio_final]
                    })
                    
                st.bar_chart(comp_data, x="Nivel", y="Precio")

            except Exception as e:
                st.error(f"Error en el motor de inferencia: {e}")
        else:
            st.info("Ajuste las características y pulse el botón para realizar la simulación.")

else:
    st.warning("Los modelos o el dataset no están disponibles para la inferencia.")