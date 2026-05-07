"""
streamlit_app/components/sidebar.py
===================================

Componentes del sidebar reutilizables en todas las páginas
"""

import streamlit as st
import pandas as pd


def render_sidebar_header():
    """Renderiza el encabezado del sidebar"""
    st.sidebar.markdown("##  Navegación")
    st.sidebar.markdown("---")


def render_sidebar_info():
    """Renderiza información del proyecto en el sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Acerca de")
    st.sidebar.info("""
    **Proyecto**: Airbnb NYC Data Mining  
    **Dataset**: 48,895 propiedades (2019)  
    **Objetivo**: Predicción de precios  
    **Defensa**: 14 de mayo 2026  
    """)


def render_sidebar_settings():
    """Renderiza opciones de configuración en el sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ️ Opciones")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        dark_mode = st.checkbox(" Modo oscuro", value=False)
    
    with col2:
        show_code = st.checkbox(" Ver código", value=False)
    
    return dark_mode, show_code


def render_sidebar_filters(df: pd.DataFrame, show_advanced: bool = False):
    """
    Renderiza filtros en el sidebar
    
    Args:
        df: DataFrame con datos
        show_advanced: Si mostrar filtros avanzados
        
    Returns:
        dict: Diccionario con filtros seleccionados
    """
    st.sidebar.markdown("###  Filtros Básicos")
    
    filters = {}
    
    # Distrito
    if 'neighbourhood_group' in df.columns:
        filters['neighbourhood_group'] = st.sidebar.selectbox(
            " Distrito:",
            sorted(df['neighbourhood_group'].unique()),
            help="Selecciona el distrito principal"
        )
    
    # Vecindario
    if 'neighbourhood' in df.columns and 'neighbourhood_group' in filters:
        df_filtered = df[df['neighbourhood_group'] == filters['neighbourhood_group']]
        filters['neighbourhood'] = st.sidebar.selectbox(
            "️ Vecindario:",
            sorted(df_filtered['neighbourhood'].unique()),
            help="Selecciona el vecindario específico"
        )
    
    # Tipo de habitación
    if 'room_type' in df.columns:
        filters['room_type'] = st.sidebar.multiselect(
            "️ Tipo de alojamiento:",
            sorted(df['room_type'].unique()),
            default=sorted(df['room_type'].unique()),
            help="Selecciona tipo(s) de habitación"
        )
    
    # Rango de precios
    if 'price' in df.columns:
        st.sidebar.markdown("###  Rango de Precios")
        col1, col2 = st.sidebar.columns(2)
        with col1:
            filters['min_price'] = st.number_input(
                "Mínimo ($):",
                min_value=0,
                value=0,
                step=10
            )
        with col2:
            filters['max_price'] = st.number_input(
                "Máximo ($):",
                min_value=0,
                value=int(df['price'].max()),
                step=10
            )
    
    # Filtros avanzados
    if show_advanced:
        st.sidebar.markdown("---")
        st.sidebar.markdown("###  Filtros Avanzados")
        
        # Mínimo de noches
        if 'minimum_nights' in df.columns:
            filters['min_nights'] = st.sidebar.slider(
                "Mínimo de noches:",
                min_value=0,
                max_value=int(df['minimum_nights'].max()),
                value=0,
                step=5
            )
        
        # Número de reseñas
        if 'number_of_reviews' in df.columns:
            filters['min_reviews'] = st.sidebar.slider(
                "Mínimo de reseñas:",
                min_value=0,
                max_value=int(df['number_of_reviews'].max()),
                value=0,
                step=5
            )
        
        # Disponibilidad
        if 'availability_365' in df.columns:
            filters['min_availability'] = st.sidebar.slider(
                "Disponibilidad mínima (días):",
                min_value=0,
                max_value=365,
                value=0,
                step=10
            )
    
    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Aplica filtros al DataFrame
    
    Args:
        df: DataFrame original
        filters: Diccionario con filtros
        
    Returns:
        DataFrame filtrado
    """
    df_filtered = df.copy()
    
    # Aplicar cada filtro si existe
    if 'neighbourhood_group' in filters and filters['neighbourhood_group']:
        df_filtered = df_filtered[df_filtered['neighbourhood_group'] == filters['neighbourhood_group']]
    
    if 'neighbourhood' in filters and filters['neighbourhood']:
        df_filtered = df_filtered[df_filtered['neighbourhood'] == filters['neighbourhood']]
    
    if 'room_type' in filters and filters['room_type']:
        df_filtered = df_filtered[df_filtered['room_type'].isin(filters['room_type'])]
    
    if 'min_price' in filters and 'max_price' in filters:
        df_filtered = df_filtered[
            (df_filtered['price'] >= filters['min_price']) &
            (df_filtered['price'] <= filters['max_price'])
        ]
    
    if 'min_nights' in filters:
        df_filtered = df_filtered[df_filtered['minimum_nights'] >= filters['min_nights']]
    
    if 'min_reviews' in filters:
        df_filtered = df_filtered[df_filtered['number_of_reviews'] >= filters['min_reviews']]
    
    if 'min_availability' in filters:
        df_filtered = df_filtered[df_filtered['availability_365'] >= filters['min_availability']]
    
    return df_filtered


def render_sidebar_download(df: pd.DataFrame, filename: str = "data"):
    """
    Renderiza opción de descarga en el sidebar
    
    Args:
        df: DataFrame a descargar
        filename: Nombre del archivo
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("###  Descargas")
    
    csv = df.to_csv(index=False)
    st.sidebar.download_button(
        label=" Descargar CSV",
        data=csv,
        file_name=f"{filename}.csv",
        mime="text/csv"
    )


def render_sidebar_footer():
    """Renderiza footer del sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='text-align: center; font-size: 11px; color: gray;'>
        <p> Airbnb NYC Data Mining</p>
        <p>UNIE 2026 | Matemáticas</p>
    </div>
    """, unsafe_allow_html=True)
