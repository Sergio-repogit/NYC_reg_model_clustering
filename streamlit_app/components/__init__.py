"""
streamlit_app/components/__init__.py
====================================

Módulo de componentes reutilizables para Streamlit
Centraliza todas las importaciones de componentes
"""

# Importar componentes del sidebar
from .sidebar import (
    render_sidebar_header,
    render_sidebar_info,
    render_sidebar_settings,
    render_sidebar_filters,
    apply_filters,
    render_sidebar_download,
    render_sidebar_footer
)

# Importar componentes de gráficos
from .charts import (
    plot_distribution,
    plot_boxplot,
    plot_scatter,
    plot_bar,
    plot_pie,
    plot_correlation_heatmap,
    plot_comparison_bars,
    plot_metric_card
)

# Importar componentes de tablas
from .charts import (
    styled_dataframe,
    comparison_table,
    model_comparison_table,
    feature_importance_table,
    cluster_profile_table,
    summary_stats_table
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
