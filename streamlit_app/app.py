"""
Streamlit App Principal - Airbnb NYC 2019 Data Mining
=====================================================
Rediseño solicitado por el usuario con Mapa 3D Global en tema OSCURO.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from pathlib import Path

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================
st.set_page_config(
    page_title="Airbnb NYC 2019 - Data Mining Suite",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Estilos CSS Profesionales
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    h1, h2, h3 { color: #1e293b; font-family: 'Inter', sans-serif; }
    .data-dict { font-size: 14px; background-color: #f1f5f9; padding: 20px; border-radius: 8px; border-left: 5px solid #3b82f6; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CARGA DE DATOS
# ============================================================================
@st.cache_data
def load_global_data():
    path = Path("data/raw/AB_NYC_2019.csv")
    if not path.exists(): return None
    df = pd.read_csv(path)
    df.dropna(subset=['price', 'latitude', 'longitude'], inplace=True)
    df = df[df['price'] > 0]
    return df

# ============================================================================
# CUERPO DE LA APP
# ============================================================================
def main():
    st.title("Airbnb NYC 2019: Data Mining Pipeline")
    
    # 1. Objetivos del Análisis
    st.header("Objetivos del Análisis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - **Entender los factores** que más influyen en el precio de un alojamiento.
        - **Identificar patrones geográficos** y por tipo de habitación.
        """)
    with col2:
        st.markdown("""
        - **Desarrollar modelos predictivos** robustos para estimar el precio.
        - **Segmentar el mercado** en grupos coherentes mediante clustering (K-Means).
        """)

    df = load_global_data()
    if df is not None:
        # 2. Información del Dataset
        st.header("Información del Dataset")
        tab1, tab2 = st.tabs(["Muestra de Datos", "Estadísticas Descriptivas"])
        
        with tab1:
            st.dataframe(df.head(100), use_container_width=True)
        with tab2:
            st.dataframe(df.describe(), use_container_width=True)
            
        # 3. Diccionario de Variables
        st.markdown("### Diccionario de Variables")
        st.markdown("""
        <div class="data-dict">
        <ul>
            <li><b>id / name</b>: Identificador único y descripción del anuncio.</li>
            <li><b>host_id / host_name</b>: Identificador y nombre del anfitrión.</li>
            <li><b>neighbourhood_group</b>: Distrito principal de Nueva York.</li>
            <li><b>neighbourhood</b>: Vecindario específico.</li>
            <li><b>latitude / longitude</b>: Coordenadas geográficas.</li>
            <li><b>room_type</b>: Tipo de estancia.</li>
            <li><b>price</b>: Precio por noche en USD.</li>
            <li><b>minimum_nights</b>: Estancia mínima.</li>
            <li><b>number_of_reviews</b>: Número total de valoraciones.</li>
            <li><b>last_review</b>: Fecha de la última valoración.</li>
            <li><b>reviews_per_month</b>: Frecuencia mensual de reseñas.</li>
            <li><b>calculated_host_listings_count</b>: Cantidad de propiedades del anfitrión.</li>
            <li><b>availability_365</b>: Disponibilidad al año.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # 4. Sección de Visualización Global
        st.header("Visualización Espacial Global")
        
        # Colores consistentes
        color_map = {
            'Entire home/apt': [0, 128, 255, 180],
            'Private room': [0, 255, 128, 180],
            'Shared room': [255, 0, 128, 180],
        }
        
        # Aplicar color y asegurar que es una lista
        df['color'] = df['room_type'].map(color_map)
        df['color'] = df['color'].apply(lambda x: x if isinstance(x, list) else [128, 128, 128, 180])
        
        # Configuración de Capa 3D
        layer = pdk.Layer(
            "ColumnLayer",
            df,
            get_position=["longitude", "latitude"],
            get_elevation="price",
            elevation_scale=0.5,
            radius=100,
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        
        view_state = pdk.ViewState(
            latitude=df['latitude'].mean(),
            longitude=df['longitude'].mean(),
            zoom=10,
            pitch=55 # Aumento del ángulo para mejor efecto 3D
        )
        
        # Usar el estilo OSCURO integrado de PyDeck
        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            map_style="dark", # Cambiado de 'light' a 'dark' por si ell usuario quier cambiarlo
            tooltip={
                "html": "<b>Alojamiento:</b> {name}<br><b>Tipo:</b> {room_type}<br><b>Precio:</b> ${price}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
        )
        
        st.pydeck_chart(r)
        
        st.markdown("**Leyenda de Colores:** 🔵 Casa/Apto entero | 🟢 Habitación privada | 🔴 Habitación compartida")

if __name__ == "__main__":
    main()
