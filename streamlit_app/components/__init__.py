"""
streamlit_app/components/__init__.py
====================================

Módulo de componentes reutilizables para Streamlit
Centraliza todas las importaciones de componentes
"""

# Importar componentes del sidebar
# Importar componentes de gráficos
# Importar componentes de tablas
from .charts import (
    cluster_profile_table,
    comparison_table,
    feature_importance_table,
    model_comparison_table,
    plot_bar,
    plot_boxplot,
    plot_comparison_bars,
    plot_correlation_heatmap,
    plot_distribution,
    plot_metric_card,
    plot_pie,
    plot_scatter,
    styled_dataframe,
    summary_stats_table,
)
from .sidebar import (
    apply_filters,
    render_sidebar_download,
    render_sidebar_filters,
    render_sidebar_footer,
    render_sidebar_header,
    render_sidebar_info,
    render_sidebar_settings,
)

__all__ = [
    # Sidebar
    'render_sidebar_header',
    'render_sidebar_info',
    'render_sidebar_settings',
    'render_sidebar_filters',
    'apply_filters',
    'render_sidebar_download',
    'render_sidebar_footer',

    # Charts
    'plot_distribution',
    'plot_boxplot',
    'plot_scatter',
    'plot_bar',
    'plot_pie',
    'plot_correlation_heatmap',
    'plot_comparison_bars',
    'plot_metric_card',

    # Tables
    'styled_dataframe',
    'comparison_table',
    'model_comparison_table',
    'feature_importance_table',
    'cluster_profile_table',
    'summary_stats_table',
]
