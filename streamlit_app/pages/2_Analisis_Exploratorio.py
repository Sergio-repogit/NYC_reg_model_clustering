"""
Página 2: Análisis Exploratorio
================================

EDA con visualizaciones, mapas 3D y filtros interactivos
Inspirado en practica.py con pydeck y filtros en sidebar
"""

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(
    page_title="Análisis Exploratorio",
    page_icon="",
    layout="wide"
)

st.title(" Análisis Exploratorio de Datos (EDA)")

# ============================================================================
# CARGAR DATOS
# ============================================================================

@st.cache_data
def load_data():
    """Carga el dataset de Airbnb NYC 2019"""
    try:
        df = pd.read_csv('data/raw/AB_NYC_2019.csv')
        df.dropna(subset=['price', 'latitude', 'longitude'], inplace=True)
        df = df[df['price'] > 0]
        return df
    except FileNotFoundError:
        st.error("Dataset no encontrado")
        return None

df = load_data()

if df is not None:

    # ============================================================================
    # SIDEBAR - FILTROS
    # ============================================================================

    st.sidebar.title(" Criterios de Búsqueda")
    st.sidebar.markdown("---")

    # Selector de distrito (neighbourhood_group)
    neighbourhood_groups = sorted(df['neighbourhood_group'].unique())
    selected_group = st.sidebar.selectbox(
        "Selecciona un distrito:",
        neighbourhood_groups,
        help="Selecciona el distrito principal (Manhattan, Brooklyn, etc.)"
    )

    # Filtrar vecindarios según distrito seleccionado
    df_group = df[df['neighbourhood_group'] == selected_group]
    neighbourhoods = sorted(df_group['neighbourhood'].unique())
    selected_neighbourhood = st.sidebar.selectbox(
        "Selecciona un vecindario:",
        neighbourhoods,
        help="Selecciona el vecindario específico"
    )

    # Tipos de habitación
    room_types = sorted(df['room_type'].unique())
    selected_room_types = st.sidebar.multiselect(
        "Tipos de alojamiento:",
        room_types,
        default=room_types,
        help="Selecciona uno o más tipos de habitación"
    )

    # Rango de precios
    col_price1, col_price2 = st.sidebar.columns(2)
    with col_price1:
        min_price = st.number_input(
            "Precio mín. ($):",
            min_value=0,
            value=0,
            step=10
        )
    with col_price2:
        max_price = st.number_input(
            "Precio máx. ($):",
            min_value=0,
            value=int(df['price'].max()),
            step=10
        )

    # Mínimo de noches
    col_nights1, col_nights2 = st.sidebar.columns(2)
    with col_nights1:
        min_nights = st.number_input(
            "Mín. noches:",
            min_value=0,
            value=0,
            step=1
        )
    with col_nights2:
        max_nights = st.number_input(
            "Máx. noches:",
            min_value=0,
            value=365,
            step=10
        )

    # Slider de número de reseñas
    min_reviews = st.sidebar.slider(
        "Mínimo de reseñas:",
        min_value=0,
        max_value=int(df['number_of_reviews'].max()),
        value=0,
        step=5
    )

    # ============================================================================
    # APLICAR FILTROS
    # ============================================================================

    df_filtered = df[
        (df['neighbourhood_group'] == selected_group) &
        (df['neighbourhood'] == selected_neighbourhood) &
        (df['room_type'].isin(selected_room_types)) &
        (df['price'] >= min_price) &
        (df['price'] <= max_price) &
        (df['minimum_nights'] >= min_nights) &
        (df['minimum_nights'] <= max_nights) &
        (df['number_of_reviews'] >= min_reviews)
    ]

    # Información de filtros
    st.markdown(f"""
    ###  Resultados del Filtro

    De los **{len(df):,}** alojamientos disponibles en total,
    **{len(df_filtered):,}** cumplen tus criterios de búsqueda ({len(df_filtered)/len(df)*100:.1f}%).
    """)

    if len(df_filtered) == 0:
        st.warning("️ No hay resultados con los filtros seleccionados. Intenta ajustar los parámetros.")

    else:
        # ============================================================================
        # TABS CON CONTENIDO
        # ============================================================================

        tab1, tab2, tab3, tab4 = st.tabs([
            " Análisis General",
            " Gráficos Comparativos",
            "️ Exploración Espacial",
            " Datos Filtrados"
        ])

        # ============================================================================
        # TAB 1: ANÁLISIS GENERAL
        # ============================================================================

        with tab1:
            st.markdown("###  Distribuciones Clave")
            st.markdown("---")

            col1, col2 = st.columns(2)

            # Precio promedio por tipo de habitación
            with col1:
                room_price = df_filtered.groupby('room_type')['price'].mean().reset_index().sort_values('price', ascending=False)
                fig1 = px.bar(
                    room_price,
                    x='room_type',
                    y='price',
                    title=' Precio Promedio por Tipo de Habitación',
                    color='room_type',
                    text_auto='.2s',
                    template='plotly_white'
                )
                fig1.update_traces(textposition='outside')
                st.plotly_chart(fig1, use_container_width=True)

            # Número de alojamientos por tipo
            with col2:
                room_count = df_filtered['room_type'].value_counts().reset_index()
                room_count.columns = ['room_type', 'count']
                fig2 = px.pie(
                    room_count,
                    values='count',
                    names='room_type',
                    title=' Distribución por Tipo de Habitación',
                    template='plotly_white'
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("---")

            col3, col4 = st.columns(2)

            # Distribución de precios
            with col3:
                fig3 = px.histogram(
                    df_filtered,
                    x='price',
                    nbins=50,
                    title=' Distribución de Precios',
                    template='plotly_white',
                    color_discrete_sequence=['#FF5A5F']
                )
                st.plotly_chart(fig3, use_container_width=True)

            # Distribución de reseñas
            with col4:
                fig4 = px.histogram(
                    df_filtered,
                    x='number_of_reviews',
                    nbins=40,
                    title=' Distribución de Reseñas',
                    template='plotly_white',
                    color_discrete_sequence=['#00A699']
                )
                st.plotly_chart(fig4, use_container_width=True)

        # ============================================================================
        # TAB 2: GRÁFICOS COMPARATIVOS
        # ============================================================================

        with tab2:
            st.markdown("###  Análisis de Relaciones")
            st.markdown("---")

            # Precio vs Noches Mínimas
            st.subheader(" Precio vs Mínimo de Noches")
            fig_scatter1 = px.scatter(
                df_filtered,
                x='minimum_nights',
                y='price',
                color='number_of_reviews',
                size='availability_365',
                hover_data=['name', 'room_type', 'neighbourhood'],
                title='Relación entre Noches Mínimas, Precio y Reseñas',
                template='plotly_white',
                color_continuous_scale='Viridis'
            )
            fig_scatter1.update_traces(
                marker=dict(size=8, opacity=0.7, line=dict(width=0.5))
            )
            st.plotly_chart(fig_scatter1, use_container_width=True)

            st.markdown("---")

            # Disponibilidad vs Precio
            st.subheader(" Disponibilidad Anual vs Precio")
            fig_scatter2 = px.scatter(
                df_filtered,
                x='availability_365',
                y='price',
                color='room_type',
                size='number_of_reviews',
                hover_data=['name', 'neighbourhood'],
                trendline='ols',
                title='Mayor Disponibilidad no implica Precios Menores',
                template='plotly_white'
            )
            fig_scatter2.update_traces(
                marker=dict(size=7, opacity=0.6)
            )
            st.plotly_chart(fig_scatter2, use_container_width=True)

            st.markdown("---")

            # Boxplot de precios por tipo
            st.subheader("Variabilidad de Precios por Tipo")
            fig_box = px.box(
                df_filtered,
                x='room_type',
                y='price',
                color='room_type',
                title='Distribución de Precios por Tipo de Habitación',
                template='plotly_white'
            )
            st.plotly_chart(fig_box, use_container_width=True)

        # ============================================================================
        # TAB 3: EXPLORACIÓN ESPACIAL CON MAPAS 3D
        # ============================================================================

        with tab3:
            st.markdown(f"### ️ Exploración Espacial de {selected_neighbourhood}")
            st.markdown("---")

            col_map, col_stats = st.columns([1, 1.2])

            with col_map:
                # Mapa 3D con PyDeck
                st.markdown(f"**Mapa 3D de {selected_neighbourhood}**")

                # Convertir tipos de datos
                df_map = df_filtered.copy()
                df_map['latitude'] = df_map['latitude'].astype(float)
                df_map['longitude'] = df_map['longitude'].astype(float)
                df_map['price'] = df_map['price'].astype(float)

                # Mapeo de colores por tipo de habitación
                color_map = {
                    'Entire home/apt': [0, 128, 255],    # Azul
                    'Private room': [0, 255, 128],       # Verde
                    'Shared room': [255, 0, 128],        # Rosa
                }

                df_map['color'] = df_map['room_type'].map(color_map)

                # Crear capa 3D
                layer = pdk.Layer(
                    'ColumnLayer',
                    data=df_map,
                    get_position='[longitude, latitude]',
                    get_elevation='price',
                    elevation_scale=0.5,
                    radius=40,
                    get_fill_color='color',
                    pickable=True,
                    auto_highlight=True,
                )

                # Configurar vista del mapa
                view_state = pdk.ViewState(
                    latitude=df_map['latitude'].mean(),
                    longitude=df_map['longitude'].mean(),
                    zoom=13,
                    pitch=50
                )

                # Crear mapa
                pydeck_map = pdk.Deck(
                    map_style='https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
                    layers=[layer],
                    initial_view_state=view_state,
                    tooltip={
                        'html': (
                            '<b>Nombre:</b> {name}<br>'
                            '<b>Tipo:</b> {room_type}<br>'
                            '<b>Precio:</b> ${price}<br>'
                            '<b>Reseñas:</b> {number_of_reviews}<br>'
                            '<b>Disponibilidad:</b> {availability_365} días'
                        ),
                        'style': {'backgroundColor': 'steelblue', 'color': 'white'}
                    }
                )

                st.pydeck_chart(pydeck_map)

                # Leyenda de colores
                st.markdown("**Leyenda de Colores:**")
                col_leg1, col_leg2, col_leg3 = st.columns(3)
                with col_leg1:
                    st.color_picker('Entire home/apt', '#0080FF')
                with col_leg2:
                    st.color_picker('Private room', '#00FF80')
                with col_leg3:
                    st.color_picker('Shared room', '#FF0080')

            with col_stats:
                # Estadísticas por barrio
                st.markdown(f"**Estadísticas de {selected_neighbourhood}**")

                col_s1, col_s2 = st.columns(2)
                with col_s1:
                    st.metric("Total alojamientos", len(df_filtered))
                    st.metric("Precio promedio", f"${df_filtered['price'].mean():.2f}")
                with col_s2:
                    st.metric("Reseñas promedio", f"{df_filtered['number_of_reviews'].mean():.1f}")
                    st.metric("Disponibilidad prom.", f"{df_filtered['availability_365'].mean():.0f} días")

                st.markdown("---")

                # Gráfico de distribución por tipo en el barrio
                st.markdown(f"**Distribución en {selected_neighbourhood}**")
                room_dist = df_filtered['room_type'].value_counts().reset_index()
                room_dist.columns = ['room_type', 'count']

                fig_barrio = px.bar(
                    room_dist,
                    x='room_type',
                    y='count',
                    color='room_type',
                    text='count',
                    template='plotly_white'
                )
                fig_barrio.update_traces(textposition='outside')
                st.plotly_chart(fig_barrio, use_container_width=True)

                st.markdown("---")

                # Gráfico de distribución por tipo en el distrito
                st.markdown(f"**Distribución en {selected_group}**")
                df_district = df[df['neighbourhood_group'] == selected_group]
                df_district = df_district[df_district['room_type'].isin(selected_room_types)]
                district_dist = df_district['room_type'].value_counts().reset_index()
                district_dist.columns = ['room_type', 'count']

                fig_district = px.bar(
                    district_dist,
                    x='room_type',
                    y='count',
                    color='room_type',
                    text='count',
                    template='plotly_white'
                )
                fig_district.update_traces(textposition='outside')
                st.plotly_chart(fig_district, use_container_width=True)

        # ============================================================================
        # TAB 4: DATOS FILTRADOS
        # ============================================================================

        with tab4:
            st.markdown("###  Tabla de Datos Filtrados")

            # Selector de columnas a mostrar
            cols_to_show = st.multiselect(
                "Selecciona columnas a mostrar:",
                df_filtered.columns.tolist(),
                default=['name', 'room_type', 'price', 'number_of_reviews', 'availability_365'],
                help="Elige qué columnas mostrar en la tabla"
            )

            if cols_to_show:
                st.dataframe(
                    df_filtered[cols_to_show].sort_values('price', ascending=False),
                    use_container_width=True,
                    height=500
                )

                # Descargar datos filtrados
                csv = df_filtered[cols_to_show].to_csv(index=False)
                st.download_button(
                    label=" Descargar datos filtrados (CSV)",
                    data=csv,
                    file_name=f"airbnb_{selected_neighbourhood}.csv",
                    mime="text/csv"
                )

            st.markdown("---")

            # Resumen estadístico
            st.markdown("###  Resumen Estadístico")
            st.dataframe(
                df_filtered.describe(),
                use_container_width=True
            )

else:
    st.error("No se pudo cargar el dataset")
