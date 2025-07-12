# src/category_admin.py - VERSIÓN OPTIMIZADA PARA RENDIMIENTO Y SIN RERUNS
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import json
import time
import math
from datetime import datetime, timedelta
from decimal import Decimal

import plotly.graph_objects as go
import plotly.express as px

# Opcional para mejor suavizado (si está disponible):
try:
    from scipy.ndimage import gaussian_filter1d
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Importar módulos locales
from hype_cycle_positioning import HypeCyclePositioner


class CategoryAdminInterface:
    """Interfaz OPTIMIZADA de administración de categorías y tecnologías del Hype Cycle"""
    
    def __init__(self, hype_storage, context_prefix: str = "default"):
        """
        Inicializa la interfaz de administración optimizada
        
        Args:
            hype_storage: Instancia de HypeCycleStorage
            context_prefix: Prefijo único para evitar conflictos de keys
        """
        self.storage = hype_storage
        
        # FIJO: Usar context_prefix estable sin timestamps
        if not context_prefix or context_prefix == "default":
            self.context_prefix = "category_admin"
        else:
            self.context_prefix = context_prefix
        
        # BASE KEY ESTABLE para todos los estados
        self._state_key_base = f"admin_state_{self.context_prefix}"
        
        # Importar positioner
        from hype_cycle_positioning import HypeCyclePositioner
        self.positioner = HypeCyclePositioner()
        
        # Cache local para mejorar rendimiento
        self._local_cache = {}
        self._cache_timestamp = {}
        self._cache_ttl = 300  # 5 minutos
        
        # Inicializar estados estables
        self._init_stable_states()
    
    def _init_stable_states(self):
        """Inicializa estados estables que persisten entre reruns"""
        # Estados principales con keys fijas
        state_keys = {
            f"{self._state_key_base}_selected_category_for_chart": None,
            f"{self._state_key_base}_chart_category_name": None,
            f"{self._state_key_base}_chart_show_labels": True,
            f"{self._state_key_base}_chart_show_confidence": False,
            f"{self._state_key_base}_refresh_trigger": 0,
            # Estados para gestión avanzada
            f"{self._state_key_base}_mgmt_selected_tech": "",
            f"{self._state_key_base}_mgmt_target_category": "",
            f"{self._state_key_base}_mgmt_confirm_delete": False,
            f"{self._state_key_base}_mgmt_delete_tech": "",
            # Estados para duplicados
            f"{self._state_key_base}_duplicates_to_delete": [],
            f"{self._state_key_base}_batch_operation_type": "none"
        }
        
        # Solo inicializar si no existen
        for key, default_value in state_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def _get_cached_data(self, key: str, fetch_function, *args, **kwargs):
        """Obtiene datos del cache local o los obtiene si no existen/expiraron"""
        current_time = time.time()
        
        # Verificar si tenemos cache válido
        if (key in self._local_cache and 
            key in self._cache_timestamp and 
            current_time - self._cache_timestamp[key] < self._cache_ttl):
            return self._local_cache[key]
        
        # Obtener datos frescos
        try:
            data = fetch_function(*args, **kwargs)
            self._local_cache[key] = data
            self._cache_timestamp[key] = current_time
            return data
        except Exception as e:
            st.error(f"Error obteniendo datos: {str(e)}")
            return self._local_cache.get(key, [])
    
    def _invalidate_cache(self):
        """Invalida el cache local"""
        self._local_cache.clear()
        self._cache_timestamp.clear()
    
    def _safe_float_format(self, value, format_str=".2f", default="0.00"):
        """
        Formatea un valor como float de forma segura
        """
        try:
            # Convertir Decimal, int, float a float
            if isinstance(value, Decimal):
                num_value = float(value)
            elif isinstance(value, (int, float)):
                num_value = float(value)
            elif isinstance(value, str):
                # Intentar convertir string a float
                num_value = float(value.replace(',', '').replace('%', ''))
            elif value is None:
                return default
            else:
                return str(value)
            
            # Verificar que no sea NaN o infinito
            if math.isnan(num_value) or math.isinf(num_value):
                return default
            
            # Aplicar formato
            return f"{num_value:{format_str}}"
            
        except (ValueError, TypeError, decimal.InvalidOperation):
            return default
    
    def _safe_int_format(self, value, default=0):
        """
        Convierte un valor a int de forma segura
        """
        try:
            if isinstance(value, Decimal):
                return int(value)
            elif isinstance(value, (int, float)):
                return int(value)
            elif isinstance(value, str):
                return int(float(value.replace(',', '')))
            elif value is None:
                return default
            else:
                return default
        except (ValueError, TypeError, decimal.InvalidOperation):
            return default
    
    def show_admin_interface(self):
        """Muestra la interfaz principal de administración OPTIMIZADA"""
        st.header("🏷️ Administración de Categorías - Hype Cycle")
        
        st.write("""
        Gestiona las tecnologías analizadas por categoría y visualiza su posición 
        en el Hype Cycle de Gartner. **Versión optimizada para mejor rendimiento.**
        """)
        
        # Pestañas principales
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Vista por Categorías",
            "🎯 Gráfica Hype Cycle", 
            "⚙️ Gestión Avanzada",
            "🧹 Limpieza de Datos"
        ])
        
        with tab1:
            self._show_category_overview_optimized()
        
        with tab2:
            self._show_hype_cycle_chart_optimized()
        
        with tab3:
            self._show_advanced_management_optimized()
        
        with tab4:
            self._show_data_cleanup()
    
    def _show_category_overview_optimized(self):
        """OPTIMIZADA: Vista general de categorías y tecnologías"""
        st.subheader("📋 Vista General por Categorías")
        
        # Obtener categorías con cache
        categories = self._get_cached_data(
            "categories",
            lambda: self.storage.storage.get_all_categories()
        )
        
        if not categories:
            st.info("No hay categorías disponibles. Crea una nueva categoría en la pestaña de análisis.")
            return
        
        # Obtener todas las consultas con cache
        all_queries = self._get_cached_data(
            "all_queries",
            lambda: self.storage.get_all_hype_cycle_queries()
        )
        
        # Mostrar estadísticas generales
        total_queries = len(all_queries)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Tecnologías", total_queries)
        with col2:
            if all_queries:
                phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in all_queries]
                most_common = max(set(phases), key=phases.count) if phases else "N/A"
                st.metric("Fase Más Común", most_common)
        with col3:
            current_month = datetime.now().strftime("%Y-%m")
            recent = len([q for q in all_queries if q.get("execution_date", "").startswith(current_month)])
            st.metric("Analizadas Este Mes", recent)
        
        # Procesar cada categoría (optimizado)
        category_data = []
        for category in categories:
            category_id = category.get("category_id") or category.get("id")
            category_name = category.get("name", "Sin nombre")
            
            # Filtrar consultas en memoria (más rápido que consulta BD)
            queries = [q for q in all_queries if q.get("category_id") == category_id]
            
            if queries:
                category_data.append({
                    'category_id': category_id,
                    'category_name': category_name,
                    'queries': queries,
                    'query_count': len(queries)
                })
        
        # Ordenar por número de consultas
        category_data.sort(key=lambda x: x['query_count'], reverse=True)
        
        # Mostrar categorías con paginación para mejor rendimiento
        categories_per_page = 5
        total_pages = math.ceil(len(category_data) / categories_per_page)
        
        if total_pages > 1:
            page = st.selectbox(
                f"Página (mostrando {categories_per_page} categorías por página)",
                options=list(range(1, total_pages + 1)),
                key=f"{self._state_key_base}_category_page"
            )
            start_idx = (page - 1) * categories_per_page
            end_idx = start_idx + categories_per_page
            page_categories = category_data[start_idx:end_idx]
        else:
            page_categories = category_data
        
        # Mostrar categorías de la página actual
        for cat_data in page_categories:
            with st.expander(
                f"📁 **{cat_data['category_name']}** ({cat_data['query_count']} tecnologías)", 
                expanded=False
            ):
                self._show_category_details_optimized(
                    cat_data['category_id'], 
                    cat_data['category_name'], 
                    cat_data['queries']
                )
    
    def _show_category_details_optimized(self, category_id: str, category_name: str, queries: List[Dict]):
        """OPTIMIZADA: Muestra detalles de una categoría específica"""
        
        # Procesar datos de tecnologías (optimizado)
        tech_data = []
        phase_distribution = {}
        
        for query in queries:
            hype_metrics = query.get("hype_metrics", {})
            
            # Datos para tabla
            phase = hype_metrics.get("phase", "Unknown")
            phase_distribution[phase] = phase_distribution.get(phase, 0) + 1
            
            # Extraer nombre de tecnología
            tech_name = (
                query.get("technology_name") or 
                query.get("name") or 
                query.get("search_query", "")[:30]
            )
            
            # Formatear fecha
            exec_date = query.get("execution_date", "")
            try:
                if exec_date:
                    formatted_date = datetime.fromisoformat(exec_date.replace('Z', '+00:00')).strftime("%Y-%m-%d")
                else:
                    formatted_date = "No disponible"
            except:
                formatted_date = exec_date[:10] if len(exec_date) >= 10 else "No disponible"
            
            # FORMATEO SEGURO DE MÉTRICAS
            confidence_raw = hype_metrics.get('confidence', 0)
            confidence_formatted = self._safe_float_format(confidence_raw, ".2f", "0.00")
            
            total_mentions_raw = hype_metrics.get('total_mentions', 0)
            total_mentions_formatted = self._safe_int_format(total_mentions_raw, 0)
            
            tech_data.append({
                "🔬 Tecnología": tech_name,
                "📍 Fase": phase,
                "🎯 Confianza": confidence_formatted,
                "⏱️ Tiempo al Plateau": hype_metrics.get("time_to_plateau", "N/A"),
                "📅 Última Actualización": formatted_date,
                "📊 Menciones": total_mentions_formatted,
                "🆔 ID": query.get("query_id", query.get("analysis_id", ""))[:8]
            })
        
        # Mostrar tabla de tecnologías (limitada para rendimiento)
        if tech_data:
            # Limitar a 20 tecnologías para mejor rendimiento
            display_data = tech_data[:20]
            df = pd.DataFrame(display_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            if len(tech_data) > 20:
                st.info(f"Mostrando las primeras 20 de {len(tech_data)} tecnologías")
            
            # Estadísticas de la categoría
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**📊 Distribución por Fases:**")
                for phase, count in list(phase_distribution.items())[:5]:  # Limitar a 5
                    percentage = (count / len(queries)) * 100
                    st.write(f"• {phase}: {count} ({percentage:.1f}%)")
            
            with col2:
                # Tecnología más mencionada
                if tech_data:
                    try:
                        max_mentions = 0
                        most_mentioned = tech_data[0]
                        
                        for tech in tech_data:
                            mentions = self._safe_int_format(tech["📊 Menciones"], 0)
                            if mentions > max_mentions:
                                max_mentions = mentions
                                most_mentioned = tech
                        
                        st.write("**🔥 Más Mencionada:**")
                        st.write(f"• {most_mentioned['🔬 Tecnología'][:20]}...")
                        st.write(f"• {max_mentions} menciones")
                    except:
                        st.write("**🔥 Más Mencionada:**")
                        st.write("• Error calculando")
            
            with col3:
                # Fecha más reciente
                try:
                    most_recent = max(tech_data, key=lambda x: x["📅 Última Actualización"])
                    st.write("**🕒 Más Reciente:**")
                    st.write(f"• {most_recent['🔬 Tecnología'][:20]}...")
                    st.write(f"• {most_recent['📅 Última Actualización']}")
                except:
                    st.write("**🕒 Más Reciente:**")
                    st.write("• Error calculando")
        
        # BOTONES CON KEYS ESTABLES (optimizados)
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            chart_button_key = f"{self._state_key_base}_chart_btn_{category_id}"
            
            if st.button(f"📊 Ver Gráfica", key=chart_button_key, type="primary"):
                # Establecer la categoría seleccionada en estados estables
                st.session_state[f"{self._state_key_base}_selected_category_for_chart"] = category_id
                st.session_state[f"{self._state_key_base}_chart_category_name"] = category_name
                
                # Incrementar trigger para forzar actualización
                current_trigger = st.session_state.get(f"{self._state_key_base}_refresh_trigger", 0)
                st.session_state[f"{self._state_key_base}_refresh_trigger"] = current_trigger + 1
                
                st.success(f"✅ Categoría '{category_name}' seleccionada para visualización.")
                st.info("👆 **Haz clic en la pestaña '🎯 Gráfica Hype Cycle' arriba para ver la gráfica.**")
        
        with col2:
            export_button_key = f"{self._state_key_base}_export_btn_{category_id}"
            if st.button(f"📤 Exportar CSV", key=export_button_key):
                self._export_category_data(category_name, tech_data)
        
        with col3:
            update_button_key = f"{self._state_key_base}_update_btn_{category_id}"
            if st.button(f"🔄 Actualizar Cache", key=update_button_key):
                self._invalidate_cache()
                st.success("✅ Cache actualizado")
                st.rerun()
        
        with col4:
            copy_button_key = f"{self._state_key_base}_copy_btn_{category_id}"
            if st.button(f"📋 Copiar IDs", key=copy_button_key):
                ids = [item["🆔 ID"] for item in tech_data[:10]]  # Solo primeros 10
                st.code(", ".join(ids))
    
    def _show_hype_cycle_chart_optimized(self):
        """OPTIMIZADA: Muestra la gráfica principal del Hype Cycle"""
        st.subheader("🎯 Gráfica del Hype Cycle por Categorías")
        
        st.write("""
        **Visualización profesional del Hype Cycle de Gartner optimizada para presentaciones.**  
        Versión optimizada para mejor rendimiento y menor tiempo de carga.
        """)
        
        # Obtener categorías con cache
        categories = self._get_cached_data(
            "categories_chart",
            lambda: self.storage.storage.get_all_categories()
        )
        
        if not categories:
            st.warning("No hay categorías disponibles para mostrar.")
            return
        
        # Preparar opciones de categorías (optimizado)
        category_options = {}
        for cat in categories:
            cat_id = cat.get("category_id") or cat.get("id")
            cat_name = cat.get("name", "Sin nombre")
            
            # Solo incluir categorías que tengan consultas (check rápido en cache)
            all_queries = self._get_cached_data(
                "all_queries_chart",
                lambda: self.storage.get_all_hype_cycle_queries()
            )
            
            queries = [q for q in all_queries if q.get("category_id") == cat_id]
            if queries:
                category_options[cat_name] = cat_id
        
        if not category_options:
            st.info("No hay categorías con tecnologías analizadas para mostrar en la gráfica.")
            return
        
        # SELECTBOX CON KEY ESTABLE
        chart_category_selector_key = f"{self._state_key_base}_chart_category_selector"
        
        # Determinar índice basado en categoría preseleccionada
        selected_category_id = st.session_state.get(f"{self._state_key_base}_selected_category_for_chart")
        selected_category_name_saved = st.session_state.get(f"{self._state_key_base}_chart_category_name")
        
        try:
            if selected_category_name_saved and selected_category_name_saved in category_options.keys():
                default_index = list(category_options.keys()).index(selected_category_name_saved)
            elif selected_category_id and selected_category_id in category_options.values():
                # Encontrar el nombre correspondiente al ID
                found_name = None
                for name, cat_id in category_options.items():
                    if cat_id == selected_category_id:
                        found_name = name
                        break
                
                if found_name:
                    default_index = list(category_options.keys()).index(found_name)
                else:
                    default_index = 0
            else:
                default_index = 0
        except:
            default_index = 0
        
        # Interfaz de control
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            # Selector de categoría con estado estable
            selected_category_name = st.selectbox(
                "🏷️ Selecciona una categoría para visualizar:",
                options=list(category_options.keys()),
                index=default_index,
                key=chart_category_selector_key
            )
        
        with col2:
            # Opciones de visualización con keys estables
            show_labels_key = f"{self._state_key_base}_show_labels"
            show_labels = st.checkbox(
                "📝 Etiquetas", 
                value=st.session_state.get(f"{self._state_key_base}_chart_show_labels", True),
                key=show_labels_key
            )
            st.session_state[f"{self._state_key_base}_chart_show_labels"] = show_labels
        
        with col3:
            show_confidence_key = f"{self._state_key_base}_show_confidence"
            show_confidence = st.checkbox(
                "🎯 Confianza", 
                value=st.session_state.get(f"{self._state_key_base}_chart_show_confidence", False),
                key=show_confidence_key
            )
            st.session_state[f"{self._state_key_base}_chart_show_confidence"] = show_confidence
        
        # Actualizar estados si hay cambio de categoría
        current_selected_id = category_options[selected_category_name]
        if current_selected_id != selected_category_id:
            st.session_state[f"{self._state_key_base}_selected_category_for_chart"] = current_selected_id
            st.session_state[f"{self._state_key_base}_chart_category_name"] = selected_category_name
        
        # Obtener tecnologías de la categoría seleccionada (desde cache)
        all_queries = self._get_cached_data(
            f"queries_cat_{current_selected_id}",
            lambda: [q for q in self._get_cached_data("all_queries_chart", lambda: self.storage.get_all_hype_cycle_queries()) 
                    if q.get("category_id") == current_selected_id]
        )
        
        active_queries = [q for q in all_queries if q.get("is_active", True)]
        
        if not active_queries:
            st.warning(f"No hay tecnologías activas en la categoría '{selected_category_name}'")
            return
        
        # Información previa a la gráfica con formateo seguro
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔬 Tecnologías", len(active_queries))
        
        with col2:
            # Fase más común
            phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in active_queries]
            if phases:
                most_common_phase = max(set(phases), key=phases.count)
                st.metric("📍 Fase Dominante", most_common_phase)
        
        with col3:
            # Confianza promedio con formateo seguro
            confidences = []
            for q in active_queries:
                conf_raw = q.get("hype_metrics", {}).get("confidence", 0)
                conf_float = self._safe_float_format(conf_raw, "", "0")
                try:
                    confidences.append(float(conf_float))
                except:
                    confidences.append(0.0)
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                st.metric("🎯 Confianza Promedio", self._safe_float_format(avg_confidence, ".2f"))
        
        # Generar y mostrar gráfica (optimizada)
        try:
            with st.spinner(f"🎨 Generando visualización para {len(active_queries)} tecnologías..."):
                # Limitar a 30 tecnologías para mejor rendimiento
                limited_queries = active_queries[:30] if len(active_queries) > 30 else active_queries
                
                fig = self._create_hype_cycle_chart_optimized(
                    limited_queries, 
                    selected_category_name,
                    show_labels=show_labels,
                    show_confidence=show_confidence
                )
            
            if fig and len(fig.data) > 0:
                # KEY ESTABLE para el gráfico
                chart_plot_key = f"{self._state_key_base}_chart_plot_{current_selected_id}"
                
                # Mostrar la gráfica con configuración optimizada
                st.plotly_chart(
                    fig, 
                    use_container_width=True,
                    key=chart_plot_key,
                    config={
                        'displayModeBar': True,
                        'displaylogo': False,
                        'modeBarButtonsToAdd': ['downloadSvg'],
                        'modeBarButtonsToRemove': ['select2d', 'lasso2d', 'autoScale2d'],
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'hype_cycle_{selected_category_name}',
                            'height': 750,
                            'width': 1200,
                            'scale': 2
                        }
                    }
                )
                
                if len(active_queries) > 30:
                    st.info(f"Mostrando las primeras 30 de {len(active_queries)} tecnologías para mejor rendimiento")
                
                # Mostrar leyenda de la gráfica (simplificada)
                self._show_chart_legend_optimized(limited_queries)
            
            else:
                st.error("❌ Error: La gráfica está vacía o no se pudo generar")
                
        except Exception as e:
            st.error(f"❌ Error generando la gráfica: {str(e)}")
    
    def _create_hype_cycle_chart_optimized(self, queries: List[Dict], category_name: str, 
                        show_labels: bool = True, show_confidence: bool = False) -> go.Figure:
        """
        OPTIMIZADA: Crea la gráfica del Hype Cycle con mejor rendimiento
        """
        # Crear figura con dimensiones amplias
        fig = go.Figure()
        
        # Curva optimizada (menos puntos para mejor rendimiento)
        x_curve = np.linspace(0, 100, 500)  # Reducido de 1000 a 500
        
        # Rediseñar curva con Peak más amplio y definido
        trigger = 15 * np.exp(-((x_curve - 12)/8)**2)
        peak = 70 * np.exp(-((x_curve - 26)/12)**2)
        trough = -25 * np.exp(-((x_curve - 55)/12)**2)
        slope_rise = 15 * (1 / (1 + np.exp(-(x_curve - 75)/5)))
        plateau = 25 * (1 / (1 + np.exp(-(x_curve - 90)/4)))
        
        baseline = 25
        y_curve = baseline + trigger + peak + trough + slope_rise + plateau
        
        # Suavizar la curva (optimizado)
        if SCIPY_AVAILABLE:
            y_curve = gaussian_filter1d(y_curve, sigma=2.0)
        else:
            window = 7  # Reducido para mejor rendimiento
            y_smooth = np.convolve(y_curve, np.ones(window)/window, mode='same')
            y_curve = y_smooth
        
        y_curve = np.clip(y_curve, 8, 90)
        
        # Función optimizada para obtener posición exacta sobre la curva
        def get_exact_position_on_curve(x_pos):
            if x_pos < 0 or x_pos > 100:
                return None
            idx = int(x_pos * (len(x_curve) - 1) / 100)
            idx = min(max(idx, 0), len(y_curve) - 1)
            return float(x_curve[idx]), float(y_curve[idx])
        
        # Añadir curva principal
        fig.add_trace(go.Scatter(
            x=x_curve, 
            y=y_curve,
            mode='lines',
            name='Hype Cycle',
            line=dict(
                color='#2E86AB',
                width=6,
                shape='spline',
                smoothing=1.3
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Definir zonas optimizadas
        phase_positions = {
            "Innovation Trigger": {
                "x_range": list(range(8, 18, 2)),
                "max_capacity": 5
            },
            "Peak of Inflated Expectations": {
                "x_range": list(range(20, 36, 2)),  # Reducido paso
                "max_capacity": 12  # Reducido
            },
            "Trough of Disillusionment": {
                "x_range": list(range(45, 66, 3)),  # Reducido
                "max_capacity": 8
            },
            "Slope of Enlightenment": {
                "x_range": list(range(68, 83, 3)),
                "max_capacity": 6
            },
            "Plateau of Productivity": {
                "x_range": list(range(85, 97, 3)),
                "max_capacity": 5
            },
            "Unknown": {
                "x_range": [50],
                "max_capacity": 1
            }
        }
        
        # Procesar y posicionar tecnologías (optimizado)
        technologies = []
        phase_counters = {phase: 0 for phase in phase_positions.keys()}
        
        # Procesar solo las tecnologías limitadas
        for i, query in enumerate(queries):
            try:
                if not isinstance(query, dict):
                    continue
                    
                hype_metrics = query.get("hype_metrics", {})
                
                if not isinstance(hype_metrics, dict):
                    hype_metrics = {}
                
                phase = hype_metrics.get("phase", "Unknown")
                
                # FORMATEO SEGURO Y OPTIMIZADO
                confidence = float(self._safe_float_format(hype_metrics.get("confidence", 0.5), "", "0.5"))
                total_mentions = self._safe_int_format(hype_metrics.get("total_mentions", 0), 0)
                
                # Obtener posición X según disponibilidad en la fase
                phase_info = phase_positions.get(phase, phase_positions["Unknown"])
                available_positions = phase_info["x_range"]
                counter = phase_counters[phase]
                
                if counter < len(available_positions):
                    x_pos = available_positions[counter]
                else:
                    base_idx = counter % len(available_positions)
                    x_pos = available_positions[base_idx] + (counter // len(available_positions)) * 0.5
                
                phase_counters[phase] += 1
                
                # Obtener posición exacta sobre la curva
                exact_x, exact_y = get_exact_position_on_curve(x_pos)
                
                # Extraer información de la tecnología
                tech_name = (
                    query.get("technology_name") or 
                    query.get("name") or 
                    query.get("search_query", f"Tecnología_{i}")[:15]  # Reducido para rendimiento
                )
                
                time_to_plateau = hype_metrics.get("time_to_plateau", "N/A")
                sentiment_avg = float(self._safe_float_format(hype_metrics.get("sentiment_avg", 0), "", "0.0"))
                
                technologies.append({
                    "name": tech_name,
                    "phase": phase,
                    "confidence": confidence,
                    "position_x": exact_x,
                    "position_y": exact_y,
                    "query_id": query.get("query_id", f"query_{i}"),
                    "time_to_plateau": time_to_plateau,
                    "total_mentions": total_mentions,
                    "sentiment_avg": sentiment_avg
                })
                
            except Exception as e:
                # En caso de error con una tecnología específica, continuar con la siguiente
                continue
        
        # Añadir tecnologías con posicionamiento optimizado
        for i, tech in enumerate(technologies):
            # Tamaño del punto optimizado
            base_size = 10  # Reducido
            confidence_factor = tech["confidence"] * 4  # Reducido
            mentions_factor = min(tech["total_mentions"] / 300, 3)  # Reducido
            size = base_size + confidence_factor + mentions_factor
            
            color = self._get_color_for_time_to_plateau(tech["time_to_plateau"])
            
            # Punto exactamente sobre la curva
            fig.add_trace(go.Scatter(
                x=[tech["position_x"]],
                y=[tech["position_y"]],
                mode='markers',
                name=tech["name"],
                marker=dict(
                    size=size,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=1),  # Reducido
                    opacity=0.9
                ),
                hovertemplate=f"""
                    <b>{tech['name']}</b><br>
                    Fase: {tech['phase']}<br>
                    Confianza: {tech['confidence']:.1%}<br>
                    Tiempo al Plateau: {tech['time_to_plateau']}<br>
                    Menciones: {tech['total_mentions']:,}<br>
                    <extra></extra>
                """,
                showlegend=False
            ))
            
            # Etiquetas optimizadas (solo si se solicitan y hay pocas tecnologías)
            if show_labels and len(technologies) <= 20:
                label_x, label_y = self._calculate_simple_label_position(
                    tech["position_x"], tech["position_y"], i
                )
                
                # Línea conectora simple
                fig.add_shape(
                    type="line",
                    x0=tech["position_x"], 
                    y0=tech["position_y"],
                    x1=label_x, 
                    y1=label_y,
                    line=dict(
                        color=color,
                        width=1
                    ),
                    layer="below"
                )
                
                # Etiqueta simplificada
                fig.add_annotation(
                    x=label_x,
                    y=label_y,
                    text=f'<b>{tech["name"][:10]}...</b>',  # Limitado para rendimiento
                    showarrow=False,
                    font=dict(
                        size=9, 
                        color='#2C3E50',
                        family="Arial"
                    ),
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=3,
                    xanchor='center',
                    yanchor='middle',
                    opacity=0.9
                )
        
        # Etiquetas de fases (simplificadas)
        phase_labels = [
            {"name": "Innovation<br>Trigger", "x": 12, "y": -20},
            {"name": "Peak<br>Expectations", "x": 28, "y": -20},
            {"name": "Trough<br>Disillusionment", "x": 55, "y": -20},
            {"name": "Slope<br>Enlightenment", "x": 75, "y": -20},
            {"name": "Plateau<br>Productivity", "x": 90, "y": -20}
        ]
        
        for label in phase_labels:
            fig.add_annotation(
                x=label["x"], 
                y=label["y"],
                text=f"<b>{label['name']}</b>",
                showarrow=False,
                font=dict(size=9, color='#7f8c8d', family="Arial"),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                borderpad=4,
                xanchor='center',
                yanchor='top',
                opacity=0.9
            )
        
        # Leyenda de tiempo al plateau (simplificada)
        legend_items = [
            {"label": "Ya alcanzado", "color": "#27AE60"},
            {"label": "< 2 años", "color": "#3498DB"},
            {"label": "2-5 años", "color": "#F39C12"},
            {"label": "5-10 años", "color": "#E67E22"},
            {"label": "> 10 años", "color": "#E74C3C"}
        ]
        
        for item in legend_items:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=item["color"]),
                name=item["label"],
                showlegend=True
            ))
        
        # Layout optimizado
        fig.update_layout(
            title=dict(
                text=f"<b>Hype Cycle - {category_name}</b><br><sub>({len(technologies)} tecnologías)</sub>",
                x=0.5,
                font=dict(size=18, color='#2C3E50')
            ),
            xaxis=dict(
                title=dict(
                    text="<b>TIME</b>",
                    font=dict(size=12, color='#7f8c8d')
                ),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[0, 100],
                showline=True,
                linecolor='#bdc3c7',
                linewidth=2
            ),
            yaxis=dict(
                title=dict(
                    text="<b>EXPECTATIONS</b>",
                    font=dict(size=12, color='#7f8c8d')
                ),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-30, 110],
                showline=True,
                linecolor='#bdc3c7',
                linewidth=2
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=700,  # Reducido
            width=1200,
            showlegend=True,
            font=dict(family="Arial"),
            margin=dict(t=100, l=70, r=180, b=90),
            hovermode='closest',
            legend=dict(
                title=dict(
                    text="<b>Tiempo al Plateau</b>",
                    font=dict(size=11, color="#2C3E50")
                ),
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                font=dict(size=9, color="#2C3E50"),
                itemsizing="constant"
            )
        )
        
        return fig

    def _calculate_simple_label_position(self, point_x: float, point_y: float, index: int) -> tuple:
        """Calcula posición simple de etiqueta para mejor rendimiento"""
        # Estrategia simplificada basada en índice
        if index % 4 == 0:  # Arriba
            label_x = point_x + (index % 3 - 1) * 8
            label_y = point_y + 20 + (index % 2) * 10
        elif index % 4 == 1:  # Derecha
            label_x = point_x + 15 + (index % 2) * 8
            label_y = point_y + (index % 3 - 1) * 8
        elif index % 4 == 2:  # Abajo
            label_x = point_x + (index % 3 - 1) * 8
            label_y = point_y - 15 - (index % 2) * 8
        else:  # Izquierda
            label_x = point_x - 15 - (index % 2) * 8
            label_y = point_y + (index % 3 - 1) * 8
        
        # Mantener dentro de límites
        label_x = max(5, min(95, label_x))
        label_y = max(-25, min(105, label_y))
        
        return (label_x, label_y)

    def _get_color_for_time_to_plateau(self, time_estimate: str) -> str:
        """Colores optimizados para tiempo al plateau"""
        time_colors = {
            "already": "#27AE60",
            "<2": "#3498DB", 
            "2-5": "#F39C12",
            "5-10": "#E67E22",
            ">10": "#E74C3C",
            "unknown": "#95A5A6"
        }
        
        time_lower = str(time_estimate).lower()
        
        if any(x in time_lower for x in ["ya alcanzado", "already", "reached"]):
            return time_colors["already"]
        elif any(x in time_lower for x in ["<2", "menos de 2", "1-2"]):
            return time_colors["<2"]
        elif any(x in time_lower for x in ["2-5", "3-5", "2-4"]):
            return time_colors["2-5"]
        elif any(x in time_lower for x in ["5-10", "6-10", "5-8"]):
            return time_colors["5-10"]
        elif any(x in time_lower for x in [">10", "más de 10", "10+"]):
            return time_colors[">10"]
        else:
            return time_colors["unknown"]
    
    def _show_chart_legend_optimized(self, queries: List[Dict]):
        """OPTIMIZADA: Muestra tabla explicativa simplificada"""
        st.subheader("📋 Tecnologías en la Gráfica")
        
        legend_data = []
        for query in queries[:15]:  # Limitar para mejor rendimiento
            hype_metrics = query.get("hype_metrics", {})
            
            tech_name = (
                query.get("technology_name") or 
                query.get("name") or 
                query.get("search_query", "")[:20]  # Limitado
            )
            
            confidence_formatted = self._safe_float_format(hype_metrics.get("confidence", 0), ".2f", "0.00")
            total_mentions_formatted = self._safe_int_format(hype_metrics.get("total_mentions", 0), 0)
            
            legend_data.append({
                "🔬 Tecnología": tech_name,
                "📍 Fase": hype_metrics.get("phase", "Unknown"),
                "🎯 Confianza": confidence_formatted,
                "⏱️ Tiempo al Plateau": hype_metrics.get("time_to_plateau", "N/A"),
                "📊 Menciones": total_mentions_formatted
            })
        
        df_legend = pd.DataFrame(legend_data)
        st.dataframe(df_legend, use_container_width=True, hide_index=True)
        
        if len(queries) > 15:
            st.info(f"Mostrando las primeras 15 de {len(queries)} tecnologías")
    
    def _show_advanced_management_optimized(self):
        """OPTIMIZADA: Gestión avanzada con mejor rendimiento"""
        st.subheader("⚙️ Gestión Avanzada")
        
        st.write("""
        Herramientas optimizadas para gestionar tecnologías: cambiar categorías, 
        eliminar registros y realizar operaciones masivas.
        """)
        
        # Sub-pestañas para organizar mejor
        subtab1, subtab2, subtab3 = st.tabs([
            "🔄 Mover Tecnologías", 
            "🗑️ Eliminar Registros",
            "📊 Operaciones Masivas"
        ])
        
        with subtab1:
            self._show_move_technologies_form()
        
        with subtab2:
            self._show_delete_technologies_form()
        
        with subtab3:
            self._show_massive_operations_optimized()
    
    def _show_move_technologies_form(self):
        """CORREGIDA: Formulario para mover tecnologías sin bloqueos"""
        st.write("### 🔄 Mover Tecnologías Entre Categorías")
        
        # Obtener datos con cache
        all_queries = self._get_cached_data(
            "all_queries_move",
            lambda: self.storage.get_all_hype_cycle_queries()
        )
        
        categories = self._get_cached_data(
            "categories_move",
            lambda: self.storage.storage.get_all_categories()
        )
        
        if not all_queries:
            st.info("No hay tecnologías para mover.")
            return
        
        if len(categories) < 2:
            st.warning("Se necesitan al menos 2 categorías para mover tecnologías.")
            return
        
        # USAR FORMULARIO SIN DISABLED PARA EVITAR BLOQUEOS
        with st.form(key=f"{self._state_key_base}_move_form_opt", clear_on_submit=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### 🔬 Seleccionar Tecnología")
                
                # Crear opciones simplificadas (limitadas para rendimiento)
                tech_options = []
                tech_data = {}
                
                for query in all_queries[:50]:  # Limitar para mejor rendimiento
                    query_id = query.get("query_id", query.get("analysis_id"))
                    tech_name = (
                        query.get("technology_name") or 
                        query.get("search_query", "")[:25] or 
                        "Sin nombre"
                    )
                    
                    # Obtener categoría actual
                    current_cat_id = query.get("category_id", "unknown")
                    current_cat_name = "Sin categoría"
                    for cat in categories:
                        if cat.get("category_id") == current_cat_id:
                            current_cat_name = cat.get("name", "Sin nombre")
                            break
                    
                    phase = query.get("hype_metrics", {}).get("phase", "Unknown")
                    display_name = f"{tech_name} | {current_cat_name} | {phase}"
                    
                    tech_options.append(display_name)
                    tech_data[display_name] = {
                        "query_id": query_id,
                        "tech_name": tech_name,
                        "current_cat_id": current_cat_id,
                        "current_cat_name": current_cat_name
                    }
                
                selected_tech_display = st.selectbox(
                    f"Tecnología a mover (mostrando {len(tech_options)} de {len(all_queries)}):",
                    options=tech_options
                )
                
                selected_tech_info = tech_data[selected_tech_display]
            
            with col2:
                st.write("#### 🎯 Nueva Categoría")
                
                # Filtrar categorías disponibles (excluir la actual)
                current_cat_id = selected_tech_info["current_cat_id"]
                available_categories = []
                category_data = {}
                
                for cat in categories:
                    if cat.get("category_id") != current_cat_id:
                        cat_name = cat.get("name", "Sin nombre")
                        available_categories.append(cat_name)
                        category_data[cat_name] = cat.get("category_id")
                
                if not available_categories:
                    st.warning("No hay otras categorías disponibles.")
                    return
                
                target_category_name = st.selectbox(
                    "Nueva categoría:",
                    options=available_categories
                )
                
                target_category_id = category_data[target_category_name]
                
                # Mostrar resumen del movimiento
                st.info(f"""
                **Movimiento a realizar:**
                
                🔬 **Tecnología:** {selected_tech_info['tech_name']}
                📁 **De:** {selected_tech_info['current_cat_name']}
                📁 **A:** {target_category_name}
                """)
            
            # Controles del formulario
            st.write("---")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                confirm_move = st.checkbox("✅ Confirmar movimiento")
                if not confirm_move:
                    st.warning("⚠️ Marca la casilla para habilitar el movimiento")
            
            with col2:
                # BOTÓN SIN DISABLED - Validación después del submit
                submitted = st.form_submit_button(
                    "🔄 EJECUTAR MOVIMIENTO", 
                    type="primary"
                )
            
            # Procesar cuando se envía el formulario
            if submitted:
                if not confirm_move:
                    st.error("❌ Debes confirmar el movimiento marcando la casilla")
                else:
                    with st.spinner("Moviendo tecnología..."):
                        # MÉTODO MEJORADO: Primero verificar que existe
                        query_to_move = selected_tech_info['query_id']
                        
                        # Debug: Mostrar información del item a mover
                        debug_info = self.storage.debug_query_ids(query_to_move)
                        
                        if debug_info.get('matching_items'):
                            # El item existe, proceder con movimiento
                            matching_item = debug_info['matching_items'][0]
                            st.info(f"🔍 Item encontrado: {matching_item['tech_name']}")
                            
                            success = self.storage.move_technology_to_category(
                                query_to_move, 
                                target_category_id
                            )
                            
                            if success:
                                st.success(f"✅ '{selected_tech_info['tech_name']}' movida exitosamente a '{target_category_name}'!")
                                
                                # Invalidar cache
                                self._invalidate_cache()
                                self.storage._invalidate_cache()
                                
                                # Limpiar cache de Streamlit
                                if hasattr(st, 'cache_data'):
                                    st.cache_data.clear()
                                
                                st.balloons()
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Error moviendo la tecnología. Revisa los logs.")
                                
                                # Mostrar información de debug
                                with st.expander("🔍 Información de Debug del Error"):
                                    st.write("**Item que se intentó mover:**")
                                    st.json(matching_item)
                                    
                                    st.write("**Información del movimiento:**")
                                    st.write(f"- ID usado: {query_to_move}")
                                    st.write(f"- Categoría origen: {selected_tech_info['current_cat_id']}")
                                    st.write(f"- Categoría destino: {target_category_id}")
                                    st.write(f"- Categoría destino nombre: {target_category_name}")
                                    
                                    st.write("**Posibles causas:**")
                                    st.write("- Permisos insuficientes en DynamoDB")
                                    st.write("- Error en la actualización de claves")
                                    st.write("- Categoría destino no válida")
                                    
                                    # Botón para intentar con ID alternativo
                                    if len(debug_info['matching_items']) > 1:
                                        st.write("**IDs alternativos encontrados:**")
                                        for i, alt_item in enumerate(debug_info['matching_items'][1:], 1):
                                            if st.button(f"🔄 Intentar con {alt_item['analysis_id'][:20]}...", key=f"{self._state_key_base}_alt_move_{i}"):
                                                alt_success = self.storage.move_technology_to_category(
                                                    alt_item['analysis_id'], 
                                                    target_category_id
                                                )
                                                if alt_success:
                                                    st.success("✅ Movimiento exitoso con ID alternativo!")
                                                    st.rerun()
                                                else:
                                                    st.error("❌ También falló con ID alternativo")
                        else:
                            # El item no existe
                            st.error(f"❌ No se encontró el item con ID: {query_to_move}")
                            
                            with st.expander("🔍 Información de Debug - Item No Encontrado"):
                                st.write(f"**ID buscado:** {query_to_move}")
                                st.write(f"**Total items en base:** {debug_info.get('total_items', 0)}")
                                
                                if debug_info.get('all_ids_sample'):
                                    st.write("**Muestra de IDs existentes:**")
                                    for sample in debug_info['all_ids_sample'][:5]:  # Solo primeros 5
                                        st.write(f"- query_id: {sample['query_id']}")
                                        st.write(f"  analysis_id: {sample['analysis_id']}")
                                        st.write(f"  tecnología: {sample['tech_name']}")
                                        st.write("---")
                                
                                st.write("**Posibles soluciones:**")
                                st.write("1. Refrescar la página y volver a cargar")
                                st.write("2. Limpiar cache y recargar datos")
                                st.write("3. El item pudo haber sido eliminado por otro proceso")
                                st.write("4. Usar la herramienta de Debug de IDs en la pestaña Debug")
                                
                                # Botón para limpiar cache y recargar
                                if st.button("🔄 Limpiar Cache y Recargar", key=f"{self._state_key_base}_debug_reload_move"):
                                    self._invalidate_cache()
                                    self.storage._invalidate_cache()
                                    if hasattr(st, 'cache_data'):
                                        st.cache_data.clear()
                                    st.rerun()
    
    def _show_delete_technologies_form(self):
        """CORREGIDA: Formulario para eliminar tecnologías sin bloqueos"""
        st.write("### 🗑️ Eliminar Registros de Tecnologías")
        st.write("⚠️ **CUIDADO:** Esta acción no se puede deshacer.")
        
        # Obtener datos con cache
        all_queries = self._get_cached_data(
            "all_queries_delete",
            lambda: self.storage.get_all_hype_cycle_queries()
        )
        
        if not all_queries:
            st.info("No hay tecnologías para eliminar.")
            return
        
        # USAR FORMULARIO SIN DISABLED PARA EVITAR BLOQUEOS
        with st.form(key=f"{self._state_key_base}_delete_form_opt", clear_on_submit=False):
            st.write("#### 🎯 Eliminar Tecnología")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Crear opciones para selectbox (limitadas)
                tech_delete_options = []
                tech_delete_data = {}
                
                for query in all_queries[:30]:  # Limitar para mejor rendimiento
                    query_id = query.get("query_id", query.get("analysis_id"))
                    tech_name = (
                        query.get("technology_name") or 
                        query.get("search_query", "")[:25] or 
                        "Sin nombre"
                    )
                    
                    # Obtener categoría
                    cat_id = query.get("category_id", "unknown") 
                    cat_name = "Sin categoría"
                    categories = self._get_cached_data(
                        "categories_delete",
                        lambda: self.storage.storage.get_all_categories()
                    )
                    for cat in categories:
                        if cat.get("category_id") == cat_id:
                            cat_name = cat.get("name", "Sin nombre")
                            break
                    
                    phase = query.get("hype_metrics", {}).get("phase", "Unknown")
                    display_name = f"{tech_name} | {cat_name} | {phase}"
                    
                    tech_delete_options.append(display_name)
                    tech_delete_data[display_name] = {
                        "query_id": query_id,
                        "tech_name": tech_name,
                        "cat_name": cat_name,
                        "phase": phase,
                        "timestamp": query.get("timestamp", ""),
                        "analysis_id": query.get("analysis_id", query_id)
                    }
                
                selected_delete_display = st.selectbox(
                    f"Tecnología a eliminar (mostrando {len(tech_delete_options)} de {len(all_queries)}):",
                    options=tech_delete_options
                )
                
                selected_delete_info = tech_delete_data[selected_delete_display]
                
                # Mostrar información de la tecnología seleccionada
                st.warning(f"""
                **Tecnología seleccionada:**
                
                🔬 **Nombre:** {selected_delete_info['tech_name']}
                📁 **Categoría:** {selected_delete_info['cat_name']}
                📍 **Fase:** {selected_delete_info['phase']}
                🆔 **ID:** {selected_delete_info['query_id'][:12]}...
                """)
            
            with col2:
                st.write("#### ⚠️ Confirmación")
                
                # Todas las confirmaciones en el formulario
                confirm1 = st.checkbox("☑️ Entiendo que esta acción no se puede deshacer")
                
                confirm2 = st.checkbox(
                    "☑️ Quiero eliminar permanentemente esta tecnología"
                )
                
                confirmation_text = ""
                if confirm1 and confirm2:
                    confirmation_text = st.text_input(
                        "Escribe 'ELIMINAR' para confirmar:",
                        placeholder="ELIMINAR"
                    )
                    
                    if confirmation_text and confirmation_text.upper().strip() != "ELIMINAR":
                        st.error("❌ Debes escribir exactamente 'ELIMINAR'")
                elif confirm1 and not confirm2:
                    st.warning("⚠️ Marca la segunda confirmación")
                elif not confirm1:
                    st.warning("⚠️ Marca la primera confirmación")
                
                # Mostrar estado de validación
                text_confirmed = confirmation_text.upper().strip() == "ELIMINAR"
                all_confirmed = confirm1 and confirm2 and text_confirmed
                
                if all_confirmed:
                    st.success("✅ Todas las confirmaciones completadas")
                
                # BOTÓN SIN DISABLED - Validación después del submit
                submitted = st.form_submit_button(
                    "🗑️ ELIMINAR PERMANENTEMENTE", 
                    type="secondary"
                )
            
            # Procesar cuando se envía el formulario
            if submitted:
                if not confirm1:
                    st.error("❌ Debes confirmar que entiendes que la acción no se puede deshacer")
                elif not confirm2:
                    st.error("❌ Debes confirmar que quieres eliminar permanentemente la tecnología")
                elif not text_confirmed:
                    st.error("❌ Debes escribir exactamente 'ELIMINAR' en el campo de confirmación")
                else:
                    with st.spinner("Eliminando tecnología..."):
                        # MÉTODO MEJORADO: Primero verificar que existe
                        query_to_delete = selected_delete_info['query_id']
                        
                        # Debug: Mostrar información del item a eliminar
                        debug_info = self.storage.debug_query_ids(query_to_delete)
                        
                        if debug_info.get('matching_items'):
                            # El item existe, proceder con eliminación
                            matching_item = debug_info['matching_items'][0]
                            st.info(f"🔍 Item encontrado: {matching_item['tech_name']}")
                            
                            success = self.storage.delete_query(query_to_delete)
                            
                            if success:
                                st.success(f"✅ Tecnología '{selected_delete_info['tech_name']}' eliminada exitosamente!")
                                
                                # Invalidar cache
                                self._invalidate_cache()
                                self.storage._invalidate_cache()
                                
                                # Limpiar cache de Streamlit
                                if hasattr(st, 'cache_data'):
                                    st.cache_data.clear()
                                
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Error eliminando la tecnología. Revisa los logs.")
                                
                                # Mostrar información de debug
                                with st.expander("🔍 Información de Debug del Error"):
                                    st.write("**Item que se intentó eliminar:**")
                                    st.json(matching_item)
                                    
                                    st.write("**Posibles causas:**")
                                    st.write("- Permisos insuficientes en DynamoDB")
                                    st.write("- Inconsistencia en las claves primary/sort")
                                    st.write("- Item eliminado por otro proceso")
                        else:
                            # El item no existe
                            st.error(f"❌ No se encontró el item con ID: {query_to_delete}")
                            
                            with st.expander("🔍 Información de Debug - Item No Encontrado"):
                                st.write(f"**ID buscado:** {query_to_delete}")
                                st.write(f"**Total items en base:** {debug_info.get('total_items', 0)}")
                                
                                if debug_info.get('all_ids_sample'):
                                    st.write("**Muestra de IDs existentes:**")
                                    for sample in debug_info['all_ids_sample']:
                                        st.write(f"- query_id: {sample['query_id']}")
                                        st.write(f"  analysis_id: {sample['analysis_id']}")
                                        st.write(f"  tecnología: {sample['tech_name']}")
                                        st.write("---")
                                
                                st.write("**Posibles soluciones:**")
                                st.write("1. Refrescar la página y volver a cargar")
                                st.write("2. Limpiar cache y recargar datos")
                                st.write("3. El item pudo haber sido eliminado por otro proceso")
                                
                                # Botón para limpiar cache y recargar
                                if st.button("🔄 Limpiar Cache y Recargar", key=f"{self._state_key_base}_debug_reload"):
                                    self._invalidate_cache()
                                    self.storage._invalidate_cache()
                                    if hasattr(st, 'cache_data'):
                                        st.cache_data.clear()
                                    st.rerun()
    
    def _show_massive_operations_optimized(self):
        """OPTIMIZADA: Operaciones masivas con mejor rendimiento"""
        st.write("### 📊 Operaciones Masivas")
        
        # Obtener estadísticas con cache
        all_queries = self._get_cached_data(
            "all_queries_massive",
            lambda: self.storage.get_all_hype_cycle_queries()
        )
        
        categories = self._get_cached_data(
            "categories_massive",
            lambda: self.storage.storage.get_all_categories()
        )
        
        # Estadísticas en tiempo real
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Tecnologías", len(all_queries))
        
        with col2:
            st.metric("Total Categorías", len(categories))
        
        with col3:
            # Tecnologías este mes
            current_month = datetime.now().strftime("%Y-%m")
            recent_queries = [q for q in all_queries if q.get("execution_date", "").startswith(current_month)]
            st.metric("Este Mes", len(recent_queries))
        
        with col4:
            # Fase más común
            if all_queries:
                phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in all_queries]
                most_common_phase = max(set(phases), key=phases.count)
                st.metric("Fase Dominante", most_common_phase[:10] + "...")
        
        st.write("---")
        
        # Operaciones disponibles
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 🧹 Operaciones de Limpieza")
            
            if st.button("🔍 Buscar Duplicados", type="primary", key=f"{self._state_key_base}_find_dupes_btn"):
                with st.spinner("Buscando duplicados..."):
                    duplicates = self.storage.find_duplicates()
                    
                    if duplicates:
                        st.success(f"✅ Encontrados {len(duplicates)} grupos de duplicados")
                        
                        total_to_delete = sum(len(dup['delete_queries']) for dup in duplicates)
                        st.metric("Consultas duplicadas para eliminar", total_to_delete)
                        
                        # Mostrar muestra
                        with st.expander("Ver muestra de duplicados", expanded=True):
                            for i, dup_group in enumerate(duplicates[:3]):  # Solo 3 para rendimiento
                                st.write(f"**Grupo {i+1}:** {dup_group['search_query'][:50]}...")
                                st.write(f"- Total duplicados: {dup_group['total_count']}")
                                st.write(f"- A eliminar: {len(dup_group['delete_queries'])}")
                                st.write("---")
                        
                        if st.button("🗑️ Eliminar Todos los Duplicados", type="secondary", key=f"{self._state_key_base}_delete_all_dupes"):
                            all_to_delete = []
                            for dup_group in duplicates:
                                all_to_delete.extend([q.get('query_id') for q in dup_group['delete_queries']])
                            
                            with st.spinner(f"Eliminando {len(all_to_delete)} duplicados..."):
                                results = self.storage.batch_delete_queries(all_to_delete)
                                
                                successful = sum(1 for success in results.values() if success)
                                failed = len(results) - successful
                                
                                if successful > 0:
                                    st.success(f"✅ {successful} duplicados eliminados")
                                    
                                    # Invalidar cache
                                    self._invalidate_cache()
                                    self.storage._invalidate_cache()
                                    if hasattr(st, 'cache_data'):
                                        st.cache_data.clear()
                                
                                if failed > 0:
                                    st.error(f"❌ {failed} duplicados no pudieron eliminarse")
                    else:
                        st.info("✅ No se encontraron duplicados")
            
            if st.button("🧹 Limpiar Cache", key=f"{self._state_key_base}_clear_cache_btn"):
                self._invalidate_cache()
                self.storage._invalidate_cache()
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
                st.success("✅ Cache limpiado")
        
        with col2:
            st.write("#### 📤 Operaciones de Exportación")
            
            if st.button("📥 Exportar Todas las Categorías", type="primary", key=f"{self._state_key_base}_export_all_btn"):
                self._export_all_categories_optimized(all_queries)
            
            if st.button("📊 Exportar Resumen Estadístico", key=f"{self._state_key_base}_export_stats_btn"):
                self._export_summary_statistics_optimized(all_queries, categories)
    
    def _show_data_cleanup(self):
        """NUEVA: Pestaña dedicada a limpieza de datos"""
        st.subheader("🧹 Limpieza Avanzada de Datos")
        
        st.write("""
        Herramientas especializadas para mantener la calidad de los datos 
        y eliminar información duplicada o incorrecta.
        """)
        
        # Obtener datos con cache
        all_queries = self._get_cached_data(
            "all_queries_cleanup",
            lambda: self.storage.get_all_hype_cycle_queries()
        )
        
        if not all_queries:
            st.info("No hay datos para limpiar.")
            return
        
        # Pestañas de limpieza
        cleanup_tab1, cleanup_tab2, cleanup_tab3 = st.tabs([
            "🔍 Detectar Problemas",
            "🗑️ Eliminar Duplicados",
            "📊 Estadísticas de Calidad"
        ])
        
        with cleanup_tab1:
            self._show_detect_problems(all_queries)
        
        with cleanup_tab2:
            self._show_remove_duplicates_interface(all_queries)
        
        with cleanup_tab3:
            self._show_data_quality_stats(all_queries)
    
    def _show_detect_problems(self, all_queries: List[Dict]):
        """Detecta problemas en los datos"""
        st.write("### 🔍 Detectar Problemas en los Datos")
        
        if st.button("🔍 Analizar Calidad de Datos", type="primary", key=f"{self._state_key_base}_analyze_quality"):
            with st.spinner("Analizando calidad de datos..."):
                # Análisis de problemas
                problems = {
                    'missing_tech_names': [],
                    'low_confidence': [],
                    'no_mentions': [],
                    'invalid_dates': [],
                    'duplicate_queries': [],
                    'missing_phase': []
                }
                
                seen_queries = {}
                
                for query in all_queries:
                    query_id = query.get('query_id', 'Unknown')
                    
                    # Nombre de tecnología faltante
                    tech_name = query.get('technology_name')
                    if not tech_name or tech_name.strip() == "":
                        problems['missing_tech_names'].append(query_id)
                    
                    # Confianza baja
                    hype_metrics = query.get('hype_metrics', {})
                    confidence = self._safe_float_format(hype_metrics.get('confidence', 0), "", "0")
                    if float(confidence) < 0.3:
                        problems['low_confidence'].append(query_id)
                    
                    # Sin menciones
                    mentions = self._safe_int_format(hype_metrics.get('total_mentions', 0), 0)
                    if mentions == 0:
                        problems['no_mentions'].append(query_id)
                    
                    # Fechas inválidas
                    exec_date = query.get('execution_date', '')
                    try:
                        if exec_date:
                            datetime.fromisoformat(exec_date.replace('Z', '+00:00'))
                    except:
                        problems['invalid_dates'].append(query_id)
                    
                    # Fase faltante
                    phase = hype_metrics.get('phase', '')
                    if not phase or phase == 'Unknown':
                        problems['missing_phase'].append(query_id)
                    
                    # Consultas duplicadas
                    search_query = query.get('search_query', '').strip().lower()
                    if search_query:
                        if search_query in seen_queries:
                            problems['duplicate_queries'].append(query_id)
                        else:
                            seen_queries[search_query] = query_id
                
                # Mostrar resultados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Nombres faltantes", len(problems['missing_tech_names']))
                    st.metric("Confianza baja (<30%)", len(problems['low_confidence']))
                
                with col2:
                    st.metric("Sin menciones", len(problems['no_mentions']))
                    st.metric("Fechas inválidas", len(problems['invalid_dates']))
                
                with col3:
                    st.metric("Fases faltantes", len(problems['missing_phase']))
                    st.metric("Consultas duplicadas", len(problems['duplicate_queries']))
                
                # Mostrar detalles de problemas
                if any(len(problem_list) > 0 for problem_list in problems.values()):
                    with st.expander("Ver detalles de problemas", expanded=True):
                        for problem_type, problem_list in problems.items():
                            if problem_list:
                                st.write(f"**{problem_type.replace('_', ' ').title()}:** {len(problem_list)} registros")
                                st.write(f"IDs: {', '.join(problem_list[:5])}{'...' if len(problem_list) > 5 else ''}")
                                st.write("---")
                else:
                    st.success("✅ No se detectaron problemas significativos en los datos")
    
    def _show_remove_duplicates_interface(self, all_queries: List[Dict]):
        """Interfaz para eliminar duplicados"""
        st.write("### 🗑️ Eliminar Consultas Duplicadas")
        
        if st.button("🔍 Buscar Duplicados", type="primary", key=f"{self._state_key_base}_find_duplicates_cleanup"):
            with st.spinner("Buscando duplicados..."):
                duplicates = self.storage.find_duplicates()
                st.session_state[f"{self._state_key_base}_duplicates_found"] = duplicates
        
        # Mostrar duplicados encontrados
        duplicates = st.session_state.get(f"{self._state_key_base}_duplicates_found", [])
        
        if duplicates:
            st.write(f"### 📊 Duplicados Encontrados: {len(duplicates)} grupos")
            
            total_duplicates = sum(len(dup['delete_queries']) for dup in duplicates)
            st.metric("Total de consultas duplicadas para eliminar", total_duplicates)
            
            # Interfaz de selección masiva
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("✅ Seleccionar Todos", key=f"{self._state_key_base}_select_all_dupes"):
                    selected_ids = []
                    for dup_group in duplicates:
                        selected_ids.extend([q.get('query_id') for q in dup_group['delete_queries']])
                    st.session_state[f"{self._state_key_base}_selected_duplicates"] = selected_ids
                    st.success(f"Seleccionados {len(selected_ids)} duplicados")
            
            with col2:
                if st.button("❌ Deseleccionar Todos", key=f"{self._state_key_base}_deselect_all_dupes"):
                    st.session_state[f"{self._state_key_base}_selected_duplicates"] = []
                    st.info("Duplicados deseleccionados")
            
            # Mostrar grupos de duplicados (limitado para rendimiento)
            for i, duplicate_group in enumerate(duplicates[:10]):  # Solo primeros 10
                with st.expander(f"Grupo {i+1}: {duplicate_group['search_query'][:50]}... ({duplicate_group['total_count']} duplicados)", expanded=False):
                    
                    st.write(f"**Consulta:** {duplicate_group['search_query']}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Mantener (más reciente):**")
                        keep_query = duplicate_group['keep_query']
                        st.write(f"- ID: {keep_query.get('query_id', 'N/A')[:12]}...")
                        st.write(f"- Fecha: {keep_query.get('execution_date', 'N/A')[:19]}")
                    
                    with col2:
                        st.write("**Eliminar:**")
                        for del_query in duplicate_group['delete_queries']:
                            query_id = del_query.get('query_id', 'N/A')
                            st.write(f"- {query_id[:12]}... | {del_query.get('execution_date', 'N/A')[:19]}")
            
            if len(duplicates) > 10:
                st.info(f"Mostrando los primeros 10 de {len(duplicates)} grupos para mejor rendimiento")
            
            # Eliminar duplicados seleccionados
            selected_duplicates = st.session_state.get(f"{self._state_key_base}_selected_duplicates", [])
            
            if selected_duplicates or total_duplicates > 0:
                st.write("---")
                st.write("### 🗑️ Eliminar Duplicados")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    auto_select = st.checkbox("Eliminar todos automáticamente", key=f"{self._state_key_base}_auto_select_dupes")
                
                with col2:
                    confirm_deletion = st.checkbox("Confirmar eliminación", key=f"{self._state_key_base}_confirm_mass_delete")
                
                with col3:
                    # BOTÓN SIN DISABLED - Validación después del submit  
                    if st.button(f"🗑️ ELIMINAR DUPLICADOS", type="secondary", key=f"{self._state_key_base}_execute_delete_dupes"):
                        # Validar confirmaciones después del click
                        if not confirm_deletion:
                            st.error("❌ Debes confirmar la eliminación marcando la casilla")
                        else:
                            if auto_select:
                                # Eliminar todos los duplicados
                                all_to_delete = []
                                for dup_group in duplicates:
                                    all_to_delete.extend([q.get('query_id') for q in dup_group['delete_queries']])
                            else:
                                all_to_delete = selected_duplicates
                            
                            if all_to_delete:
                                with st.spinner(f"Eliminando {len(all_to_delete)} duplicados..."):
                                    results = self.storage.batch_delete_queries(all_to_delete)
                                    
                                    successful = sum(1 for success in results.values() if success)
                                    failed = len(results) - successful
                                    
                                    if successful > 0:
                                        st.success(f"✅ {successful} duplicados eliminados")
                                        
                                        # Invalidar cache
                                        self._invalidate_cache()
                                        self.storage._invalidate_cache()
                                        if hasattr(st, 'cache_data'):
                                            st.cache_data.clear()
                                    
                                    if failed > 0:
                                        st.error(f"❌ {failed} duplicados no pudieron eliminarse")
                            else:
                                st.warning("No hay duplicados seleccionados para eliminar")
        else:
            st.info("Haz clic en 'Buscar Duplicados' para encontrar consultas duplicadas.")
    
    def _show_data_quality_stats(self, all_queries: List[Dict]):
        """Muestra estadísticas de calidad de datos"""
        st.write("### 📊 Estadísticas de Calidad de Datos")
        
        if not all_queries:
            st.info("No hay datos para analizar")
            return
        
        # Calcular estadísticas
        total_queries = len(all_queries)
        
        # Métricas de calidad
        with_tech_names = sum(1 for q in all_queries if q.get('technology_name', '').strip())
        with_high_confidence = sum(1 for q in all_queries 
                                 if float(self._safe_float_format(q.get('hype_metrics', {}).get('confidence', 0), "", "0")) >= 0.7)
        with_mentions = sum(1 for q in all_queries 
                          if self._safe_int_format(q.get('hype_metrics', {}).get('total_mentions', 0), 0) > 0)
        with_valid_dates = 0
        for q in all_queries:
            try:
                exec_date = q.get('execution_date', '')
                if exec_date:
                    datetime.fromisoformat(exec_date.replace('Z', '+00:00'))
                    with_valid_dates += 1
            except:
                pass
        
        # Mostrar métricas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            percentage = (with_tech_names / total_queries) * 100 if total_queries > 0 else 0
            st.metric("Con Nombres de Tecnología", f"{with_tech_names}/{total_queries}", f"{percentage:.1f}%")
        
        with col2:
            percentage = (with_high_confidence / total_queries) * 100 if total_queries > 0 else 0
            st.metric("Alta Confianza (≥70%)", f"{with_high_confidence}/{total_queries}", f"{percentage:.1f}%")
        
        with col3:
            percentage = (with_mentions / total_queries) * 100 if total_queries > 0 else 0
            st.metric("Con Menciones", f"{with_mentions}/{total_queries}", f"{percentage:.1f}%")
        
        with col4:
            percentage = (with_valid_dates / total_queries) * 100 if total_queries > 0 else 0
            st.metric("Fechas Válidas", f"{with_valid_dates}/{total_queries}", f"{percentage:.1f}%")
        
        # Gráfico de distribución de calidad
        st.write("### 📈 Distribución de Calidad")
        
        quality_data = {
            'Métrica': ['Nombres de Tecnología', 'Alta Confianza', 'Con Menciones', 'Fechas Válidas'],
            'Porcentaje': [
                (with_tech_names / total_queries) * 100,
                (with_high_confidence / total_queries) * 100,
                (with_mentions / total_queries) * 100,
                (with_valid_dates / total_queries) * 100
            ]
        }
        
        fig = px.bar(
            x=quality_data['Métrica'],
            y=quality_data['Porcentaje'],
            title="Porcentaje de Consultas con Datos de Calidad",
            color=quality_data['Porcentaje'],
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribución por fases
        st.write("### 🔄 Distribución por Fases del Hype Cycle")
        
        phase_counts = {}
        for query in all_queries:
            phase = query.get("hype_metrics", {}).get("phase", "Unknown")
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        if phase_counts:
            fig_phases = px.pie(
                values=list(phase_counts.values()),
                names=list(phase_counts.keys()),
                title="Distribución de Tecnologías por Fase"
            )
            st.plotly_chart(fig_phases, use_container_width=True)
    
    # ===== MÉTODOS AUXILIARES OPTIMIZADOS =====
    
    def _export_category_data(self, category_name: str, tech_data: List[Dict]):
        """OPTIMIZADA: Exporta datos de una categoría específica"""
        try:
            df = pd.DataFrame(tech_data)
            csv = df.to_csv(index=False)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"hype_cycle_{category_name}_{timestamp}.csv"
            
            st.download_button(
                label=f"📥 Descargar {filename}",
                data=csv,
                file_name=filename,
                mime="text/csv",
                key=f"{self._state_key_base}_download_{category_name}_{timestamp}"
            )
            
            st.success(f"✅ Archivo CSV preparado para descarga")
            
        except Exception as e:
            st.error(f"Error exportando datos: {str(e)}")
    
    def _export_all_categories_optimized(self, all_queries: List[Dict]):
        """OPTIMIZADA: Exporta datos de todas las categorías"""
        try:
            if not all_queries:
                st.warning("No hay datos para exportar.")
                return
            
            export_data = []
            for query in all_queries:
                hype_metrics = query.get("hype_metrics", {})
                
                # Formatear fecha
                exec_date = query.get("execution_date", "")
                try:
                    if exec_date:
                        formatted_date = datetime.fromisoformat(exec_date.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M")
                    else:
                        formatted_date = "No disponible"
                except:
                    formatted_date = exec_date[:16] if len(exec_date) >= 16 else "No disponible"
                
                export_data.append({
                    "ID": query.get("query_id", query.get("analysis_id", "")),
                    "Tecnologia": query.get("technology_name") or query.get("search_query", "")[:50],
                    "Categoria": query.get("category_name", "Sin categoría"),
                    "Fase": hype_metrics.get("phase", "Unknown"),
                    "Confianza": self._safe_float_format(hype_metrics.get("confidence", 0), ".3f"),
                    "Menciones_Total": self._safe_int_format(hype_metrics.get("total_mentions", 0)),
                    "Tiempo_al_Plateau": hype_metrics.get("time_to_plateau", "N/A"),
                    "Sentimiento_Promedio": self._safe_float_format(hype_metrics.get("sentiment_avg", 0), ".3f"),
                    "Posicion_X": self._safe_float_format(hype_metrics.get("hype_cycle_position_x", 0), ".2f"),
                    "Posicion_Y": self._safe_float_format(hype_metrics.get("hype_cycle_position_y", 0), ".2f"),
                    "Fecha_Analisis": formatted_date,
                    "Consulta_Original": query.get("search_query", "")
                })
            
            # Crear DataFrame y CSV
            df_export = pd.DataFrame(export_data)
            csv = df_export.to_csv(index=False)
            
            # Botón de descarga
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"hype_cycle_completo_{timestamp}.csv"
            
            st.download_button(
                label=f"📥 Descargar {filename}",
                data=csv,
                file_name=filename,
                mime="text/csv",
                key=f"{self._state_key_base}_download_all_{timestamp}"
            )
            
            st.success(f"✅ Preparado archivo con {len(export_data)} registros para descarga")
            
        except Exception as e:
            st.error(f"Error exportando datos: {str(e)}")
    
    def _export_summary_statistics_optimized(self, all_queries: List[Dict], categories: List[Dict]):
        """OPTIMIZADA: Exporta estadísticas resumidas"""
        try:
            summary_data = []
            
            for category in categories:
                cat_id = category.get("category_id")
                cat_name = category.get("name", "Sin nombre")
                
                # Filtrar consultas de esta categoría en memoria
                cat_queries = [q for q in all_queries if q.get("category_id") == cat_id]
                
                if not cat_queries:
                    continue
                
                # Calcular estadísticas
                phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in cat_queries]
                phase_counts = {phase: phases.count(phase) for phase in set(phases)}
                most_common_phase = max(phase_counts.items(), key=lambda x: x[1])[0] if phase_counts else "N/A"
                
                # Confianza promedio
                confidences = []
                for q in cat_queries:
                    conf_raw = q.get("hype_metrics", {}).get("confidence", 0)
                    conf_float = float(self._safe_float_format(conf_raw, "", "0"))
                    confidences.append(conf_float)
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                # Total menciones
                total_mentions = 0
                for q in cat_queries:
                    mentions_raw = q.get("hype_metrics", {}).get("total_mentions", 0)
                    mentions_int = self._safe_int_format(mentions_raw, 0)
                    total_mentions += mentions_int
                
                summary_data.append({
                    "Categoria": cat_name,
                    "Total_Tecnologias": len(cat_queries),
                    "Fase_Dominante": most_common_phase,
                    "Confianza_Promedio": round(avg_confidence, 3),
                    "Total_Menciones": total_mentions,
                    "Distribuciones_Fases": json.dumps(phase_counts)
                })
            
            # Crear DataFrame y CSV
            df_summary = pd.DataFrame(summary_data)
            csv_summary = df_summary.to_csv(index=False)
            
            # Botón de descarga
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            filename = f"hype_cycle_resumen_{timestamp}.csv"
            
            st.download_button(
                label=f"📊 Descargar {filename}",
                data=csv_summary,
                file_name=filename,
                mime="text/csv",
                key=f"{self._state_key_base}_download_summary_{timestamp}"
            )
            
            st.success(f"✅ Resumen estadístico preparado con {len(summary_data)} categorías")
            
        except Exception as e:
            st.error(f"Error exportando resumen: {str(e)}")
    
    def _recalculate_all_positions(self):
        """OPTIMIZADA: Recalcula las posiciones de todas las tecnologías"""
        try:
            updated_count = 0
            all_queries = self._get_cached_data(
                "all_queries_recalc",
                lambda: self.storage.get_all_hype_cycle_queries()
            )
            
            # Procesar en lotes para mejor rendimiento
            batch_size = 10
            total_batches = math.ceil(len(all_queries) / batch_size)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(all_queries))
                batch_queries = all_queries[start_idx:end_idx]
                
                status_text.text(f"Procesando lote {batch_idx + 1} de {total_batches}")
                
                for query in batch_queries:
                    hype_metrics = query.get("hype_metrics", {})
                    phase = hype_metrics.get("phase", "Unknown")
                    
                    # Formateo seguro de confidence
                    confidence_raw = hype_metrics.get("confidence", 0.5)
                    confidence = float(self._safe_float_format(confidence_raw, "", "0.5"))
                    
                    # Formateo seguro de total_mentions
                    total_mentions_raw = hype_metrics.get("total_mentions", 0)
                    total_mentions = self._safe_int_format(total_mentions_raw, 0)
                    
                    # Recalcular posición
                    pos_x, pos_y = self.positioner.calculate_position(phase, confidence, total_mentions)
                    
                    # Actualizar métricas (conceptualmente)
                    hype_metrics["hype_cycle_position_x"] = pos_x
                    hype_metrics["hype_cycle_position_y"] = pos_y
                    
                    updated_count += 1
                
                # Actualizar barra de progreso
                progress = (batch_idx + 1) / total_batches
                progress_bar.progress(progress)
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Recalculadas {updated_count} posiciones de tecnologías")
            
        except Exception as e:
            st.error(f"Error recalculando posiciones: {str(e)}")
    
    def _detect_duplicates_optimized(self):
        """OPTIMIZADA: Detecta posibles consultas duplicadas"""
        try:
            all_queries = self._get_cached_data(
                "all_queries_duplicates",
                lambda: self.storage.get_all_hype_cycle_queries()
            )
            
            seen_queries = {}
            duplicates = []
            
            # Procesar con progress bar para grandes volúmenes
            progress_bar = st.progress(0)
            
            for i, query in enumerate(all_queries):
                search_query = query.get("search_query", "").lower().strip()
                if search_query:
                    if search_query in seen_queries:
                        duplicates.append({
                            "original": seen_queries[search_query],
                            "duplicate": query
                        })
                    else:
                        seen_queries[search_query] = query
                
                # Actualizar progreso cada 10 elementos
                if i % 10 == 0:
                    progress = (i + 1) / len(all_queries)
                    progress_bar.progress(progress)
            
            progress_bar.empty()
            
            if duplicates:
                st.warning(f"⚠️ Encontrados {len(duplicates)} posibles duplicados")
                
                with st.expander("Ver duplicados encontrados", expanded=True):
                    # Mostrar solo los primeros 10 para rendimiento
                    display_duplicates = duplicates[:10]
                    
                    for i, dup in enumerate(display_duplicates):
                        st.write(f"**Duplicado {i+1}:**")
                        st.write(f"- Query: `{dup['duplicate'].get('search_query', '')[:60]}...`")
                        st.write(f"- Original ID: `{dup['original'].get('query_id', 'N/A')[:12]}...`")
                        st.write(f"- Duplicado ID: `{dup['duplicate'].get('query_id', 'N/A')[:12]}...`")
                        st.write("---")
                    
                    if len(duplicates) > 10:
                        st.write(f"... y {len(duplicates) - 10} duplicados más.")
                        st.info("💡 Usa la pestaña 'Limpieza de Datos' para gestionar todos los duplicados")
            else:
                st.success("✅ No se encontraron duplicados obvios")
                
        except Exception as e:
            st.error(f"Error detectando duplicados: {str(e)}")
    
    def get_cache_info(self) -> Dict:
        """Obtiene información del estado del cache"""
        return {
            'local_cache_keys': list(self._local_cache.keys()),
            'local_cache_size': len(self._local_cache),
            'cache_timestamps': {k: time.time() - v for k, v in self._cache_timestamp.items()},
            'cache_ttl': self._cache_ttl
        }
    
    def clear_all_caches(self):
        """Limpia todos los caches (local y Streamlit)"""
        # Cache local
        self._invalidate_cache()
        
        # Cache del storage
        if hasattr(self.storage, '_invalidate_cache'):
            self.storage._invalidate_cache()
        
        # Cache de Streamlit
        if hasattr(st, 'cache_data'):
            st.cache_data.clear()
        
        return True
    
    def get_performance_stats(self) -> Dict:
        """Obtiene estadísticas de rendimiento"""
        all_queries = self._get_cached_data(
            "all_queries_stats",
            lambda: self.storage.get_all_hype_cycle_queries()
        )
        
        categories = self._get_cached_data(
            "categories_stats",
            lambda: self.storage.storage.get_all_categories()
        )
        
        return {
            'total_queries': len(all_queries),
            'total_categories': len(categories),
            'cache_hits': len([k for k in self._local_cache.keys() if k.startswith('all_queries')]),
            'last_update': max(self._cache_timestamp.values()) if self._cache_timestamp else 0,
            'avg_queries_per_category': len(all_queries) / len(categories) if categories else 0
        }