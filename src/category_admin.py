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
    
    def _safe_float_format(self, value, format_type="float", format_str=".2f", default="0.00"):
        """
        Formatea un valor como float de forma segura - VERSIÓN UNIFICADA
        
        Args:
            value: Valor a formatear
            format_type: Tipo de formato ("float", "percent", "int")
            format_str: String de formato (ej: ".2f")
            default: Valor por defecto si hay error
        """
        try:
            if value is None:
                return default
            
            if isinstance(value, Decimal):
                numeric_value = float(value)
            elif isinstance(value, str):
                clean_value = value.replace(',', '').replace('%', '').strip()
                numeric_value = float(clean_value) if clean_value else 0.0
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                return str(value)
            
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                return default
            
            # Aplicar formato según tipo
            if format_type == "percent":
                return f"{numeric_value * 100:.1f}%"
            elif format_type == "int":
                return str(int(numeric_value))
            else:  # format_type == "float" o cualquier otro
                return f"{numeric_value:{format_str}}"
                
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
        
        # Pestañas principales - AMPLIADAS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Vista por Categorías",
            "🎯 Gráfica Hype Cycle", 
            "🏷️ Gestión de Categorías",  # NUEVA PESTAÑA
            "⚙️ Gestión Avanzada",
            "🧹 Limpieza de Datos"
        ])
        
        with tab1:
            self._show_category_overview_optimized()
        
        with tab2:
            self._show_hype_cycle_chart_optimized()
        
        with tab3:
            self._show_category_management()  # NUEVA FUNCIÓN
        
        with tab4:
            self._show_advanced_management_optimized()
        
        with tab5:
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
        VERSIÓN PROFESIONAL: Crea la gráfica del Hype Cycle estilo Gartner con nombres completos
        Optimizada para presentaciones internacionales sin cruces de líneas
        """
        # Crear figura con dimensiones profesionales
        fig = go.Figure()
        
        # Curva AMPLIADA para usar TODO EL ESPACIO VISUAL disponible
        x_curve = np.linspace(10, 90, 500)  # Curva más centrada pero amplia
        
        # Curva MÁS GRANDE y PRONUNCIADA para aprovechar todo el espacio
        # Innovation Trigger - más alto y visible
        trigger = 40 * np.exp(-((x_curve - 20)/8)**2)
        
        # Peak - MUY ALTO para usar todo el espacio vertical
        peak = 120 * np.exp(-((x_curve - 35)/12)**2)
        
        # Trough - MÁS PROFUNDO para contraste dramático  
        trough = -60 * np.exp(-((x_curve - 55)/15)**2)
        
        # Slope - gradual y amplio
        slope_rise = 35 * (1 / (1 + np.exp(-(x_curve - 70)/8)))
        
        # Plateau - alto y extendido
        plateau = 45 * (1 / (1 + np.exp(-(x_curve - 80)/6)))
        
        baseline = 40  # Línea base más alta para mejor visibilidad
        y_curve = baseline + trigger + peak + trough + slope_rise + plateau
        
        # Suavizar la curva
        if SCIPY_AVAILABLE:
            y_curve = gaussian_filter1d(y_curve, sigma=1.5)
        else:
            window = 5
            y_smooth = np.convolve(y_curve, np.ones(window)/window, mode='same')
            y_curve = y_smooth
        
        # Rango AMPLIO para usar todo el espacio vertical disponible
        y_curve = np.clip(y_curve, 10, 140)
        
        # Función optimizada para posición exacta en la curva AMPLIADA
        def get_exact_position_on_curve(x_pos):
            if x_pos < 10 or x_pos > 90:
                return None
            # Mapear x_pos al índice correcto en la curva ampliada
            idx = int((x_pos - 10) * (len(x_curve) - 1) / 80)
            idx = min(max(idx, 0), len(y_curve) - 1)
            return float(x_curve[idx]), float(y_curve[idx])
        
        # Añadir curva principal que usa TODO EL ESPACIO VISUAL
        fig.add_trace(go.Scatter(
            x=x_curve, 
            y=y_curve,
            mode='lines',
            name='Hype Cycle',
            line=dict(
                color='#2E86AB',
                width=8,  # Línea MÁS gruesa para mejor visibilidad
                shape='spline',
                smoothing=1.3
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Zonas REAJUSTADAS para la curva más amplia y centrada
        phase_positions = {
            "Innovation Trigger": {
                "x_range": list(range(15, 27, 2)),  # Ajustado a la nueva curva
                "label_zones": {
                    "top": list(range(160, 190, 8)),
                    "bottom": list(range(-40, -70, -8))
                }
            },
            "Peak of Inflated Expectations": {
                "x_range": list(range(28, 45, 1)),  # Rango amplio para el pico alto
                "label_zones": {
                    "top": list(range(170, 220, 6)),
                    "upper": list(range(155, 195, 6)),
                    "side_right": list(range(150, 180, 6)),
                    "side_left": list(range(150, 180, 6))
                }
            },
            "Trough of Disillusionment": {
                "x_range": list(range(47, 65, 3)),  # Ajustado al valle profundo
                "label_zones": {
                    "bottom": list(range(-50, -100, -10)),
                    "side": list(range(15, 45, 8))
                }
            },
            "Slope of Enlightenment": {
                "x_range": list(range(66, 78, 2)),  # En la pendiente
                "label_zones": {
                    "top": list(range(160, 190, 8)),
                    "side": list(range(120, 150, 6))
                }
            },
            "Plateau of Productivity": {
                "x_range": list(range(79, 88, 2)),  # En la meseta alta
                "label_zones": {
                    "top": list(range(165, 195, 8)),
                    "side": list(range(130, 160, 5))
                }
            }
        }
        
        # Procesar y organizar tecnologías estilo Gartner profesional
        technologies = []
        tech_by_phase = {}
        
        # PASO 1: Crear lista básica y agrupar por fases
        for i, query in enumerate(queries):
            try:
                if not isinstance(query, dict):
                    continue
                    
                hype_metrics = query.get("hype_metrics", {})
                if not isinstance(hype_metrics, dict):
                    hype_metrics = {}
                
                phase = hype_metrics.get("phase", "Unknown")
                
                # Formateo seguro
                confidence = float(self._safe_float_format(hype_metrics.get("confidence", 0.5), "", "0.5"))
                total_mentions = self._safe_int_format(hype_metrics.get("total_mentions", 0), 0)
                
                # Extraer nombre COMPLETO de la tecnología
                tech_name = (
                    query.get("technology_name") or 
                    query.get("name") or 
                    query.get("search_query", f"Tecnología_{i}")
                )
                
                # NO truncar el nombre - mostrar completo
                if len(tech_name) > 50:
                    # Solo si es excesivamente largo, hacer smart truncate
                    words = tech_name.split()
                    if len(words) > 6:
                        tech_name = " ".join(words[:6]) + "..."
                
                time_to_plateau = hype_metrics.get("time_to_plateau", "N/A")
                sentiment_avg = float(self._safe_float_format(hype_metrics.get("sentiment_avg", 0), "", "0.0"))
                
                tech_info = {
                    "name": tech_name,
                    "phase": phase,
                    "confidence": confidence,
                    "query_id": query.get("query_id", f"query_{i}"),
                    "time_to_plateau": time_to_plateau,
                    "total_mentions": total_mentions,
                    "sentiment_avg": sentiment_avg,
                    "original_index": i
                }
                
                technologies.append(tech_info)
                
                # Agrupar por fase
                if phase not in tech_by_phase:
                    tech_by_phase[phase] = []
                tech_by_phase[phase].append(tech_info)
                
            except Exception as e:
                continue
        
        # PASO 2: Asignar posiciones inteligentes en la curva y calcular etiquetas
        positioned_technologies = []
        
        for phase, techs in tech_by_phase.items():
            if phase not in phase_positions or not techs:
                continue
                
            phase_config = phase_positions[phase]
            x_positions = phase_config["x_range"]
            
            # Distribuir tecnologías a lo largo de la fase
            for i, tech in enumerate(techs):
                if i < len(x_positions):
                    x_pos = x_positions[i]
                else:
                    # Para tecnologías extra, interpolar posiciones
                    base_idx = i % len(x_positions)
                    offset = (i // len(x_positions)) * 0.5
                    x_pos = x_positions[base_idx] + offset
                
                # Obtener posición exacta sobre la curva
                exact_x, exact_y = get_exact_position_on_curve(x_pos)
                
                tech["position_x"] = exact_x
                tech["position_y"] = exact_y
                
                positioned_technologies.append(tech)
        
        # PASO 3: Añadir puntos de tecnologías
        for tech in positioned_technologies:
            # Tamaño del punto basado en importancia
            base_size = 12
            confidence_factor = tech["confidence"] * 5
            mentions_factor = min(tech["total_mentions"] / 200, 4)
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
                    line=dict(color='white', width=2),
                    opacity=0.95
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
        
        # PASO 4: ALGORITMO PROFESIONAL DE ETIQUETADO SIN CRUCES
        if show_labels and positioned_technologies:
            self._add_professional_labels_no_crossing(fig, positioned_technologies, tech_by_phase, phase_positions)
        
        # RESTAURAR Etiquetas de fases en la parte inferior 
        phase_labels = [
            {"name": "Innovation<br>Trigger", "x": 21, "y": -25},  # Ajustado a nueva curva
            {"name": "Peak of Inflated<br>Expectations", "x": 36, "y": -25},   # Centrado en el pico
            {"name": "Trough of<br>Disillusionment", "x": 56, "y": -25},  # Centrado en el valle
            {"name": "Slope of<br>Enlightenment", "x": 72, "y": -25},     # En la pendiente
            {"name": "Plateau of<br>Productivity", "x": 83, "y": -25}     # En la meseta
        ]
        
        for label in phase_labels:
            fig.add_annotation(
                x=label["x"], 
                y=label["y"],
                text=f"<b>{label['name']}</b>",
                showarrow=False,
                font=dict(size=12, color='#34495e', family="Arial Black"),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                borderpad=6,
                xanchor='center',
                yanchor='top',
                opacity=0.95
            )
        
        # RESTAURAR Leyenda de tiempo al plateau (que se había perdido)
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
                marker=dict(size=14, color=item["color"]),
                name=item["label"],
                showlegend=True
            ))
        
        # Layout CORREGIDO para curva más grande y visible
        fig.update_layout(
            title=dict(
                text=f"<b>Hype Cycle - {category_name}</b><br><sub>({len(positioned_technologies)} tecnologías analizadas)</sub>",
                x=0.5,
                font=dict(size=22, color='#2C3E50', family="Arial Black")
            ),
            xaxis=dict(
                title=dict(
                    text="<b>TIEMPO</b>",
                    font=dict(size=16, color='#34495e', family="Arial Black")
                ),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[5, 95],  # Rango ajustado para curva más centrada y grande
                showline=True,
                linecolor='#7f8c8d',
                linewidth=3
            ),
            yaxis=dict(
                title=dict(
                    text="<b>EXPECTATIVAS</b>",
                    font=dict(size=16, color='#34495e', family="Arial Black")
                ),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-45, 235],  # Rango optimizado para curva más grande
                showline=True,
                linecolor='#7f8c8d',
                linewidth=3
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=950,  # Altura optimizada
            width=1700,  # Ancho optimizado
            showlegend=True,
            font=dict(family="Arial"),
            margin=dict(t=140, l=80, r=220, b=120),  # Márgenes optimizados
            hovermode='closest',
            legend=dict(
                title=dict(
                    text="<b>Tiempo al Plateau</b>",
                    font=dict(size=13, color="#2C3E50", family="Arial Black")
                ),
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.01,  # Posición ajustada
                bgcolor='rgba(255,255,255,0.98)',
                bordercolor='#bdc3c7',
                borderwidth=2,
                font=dict(size=11, color="#2C3E50"),
                itemsizing="constant"
            )
        )
        
        return fig

    def _add_professional_labels_no_crossing(self, fig, technologies, tech_by_phase, phase_positions):
        """
        ALGORITMO INTELIGENTE: Evita superposiciones y cruces de líneas
        Usa detección de colisiones y líneas optimizadas
        """
        
        # Procesar cada fase con estrategia anti-colisión
        for phase, techs in tech_by_phase.items():
            if not techs or phase not in phase_positions:
                continue
                
            if phase == "Peak of Inflated Expectations":
                # Estrategia especial para el pico - con detección de colisiones
                self._add_peak_smart_no_overlap(fig, techs)
                
            elif phase == "Innovation Trigger":
                # Etiquetas organizadas sin superposición
                self._add_organized_trigger_labels(fig, techs)
                
            elif phase == "Trough of Disillusionment":
                # Etiquetas del valle organizadas
                self._add_organized_trough_labels(fig, techs)
                
            elif phase == "Slope of Enlightenment":
                # Etiquetas de la pendiente organizadas
                self._add_organized_slope_labels(fig, techs)
                
            elif phase == "Plateau of Productivity":
                # Etiquetas del plateau organizadas
                self._add_organized_plateau_labels(fig, techs)

    def _add_peak_smart_no_overlap(self, fig, techs):
        """
        ALGORITMO ANTI-COLISIÓN para el pico congestionado
        Usa zones predefinidas y detección de superposiciones
        """
        # Ordenar por posición X
        sorted_techs = sorted(techs, key=lambda t: t["position_x"])
        
        # Definir ZONES específicas para evitar superposiciones
        zones = {
            "top_left": {"x_range": (15, 30), "y_range": (180, 220), "capacity": 4},
            "top_center": {"x_range": (30, 45), "y_range": (200, 235), "capacity": 5},
            "top_right": {"x_range": (45, 60), "y_range": (180, 220), "capacity": 4},
            "middle_left": {"x_range": (10, 25), "y_range": (160, 180), "capacity": 3},
            "middle_right": {"x_range": (50, 65), "y_range": (160, 180), "capacity": 3},
            "side_left": {"x_range": (5, 20), "y_range": (140, 160), "capacity": 3},
            "side_right": {"x_range": (55, 70), "y_range": (140, 160), "capacity": 3}
        }
        
        # Asignar tecnologías a zones sin superposición
        zone_assignments = {}
        zone_usage = {zone: 0 for zone in zones.keys()}
        
        for i, tech in enumerate(sorted_techs):
            # Encontrar la zone más apropiada y disponible
            best_zone = None
            min_distance = float('inf')
            
            for zone_name, zone_info in zones.items():
                if zone_usage[zone_name] < zone_info["capacity"]:
                    # Calcular distancia al centro de la zone
                    zone_center_x = (zone_info["x_range"][0] + zone_info["x_range"][1]) / 2
                    zone_center_y = (zone_info["y_range"][0] + zone_info["y_range"][1]) / 2
                    
                    distance = ((tech["position_x"] - zone_center_x)**2 + 
                            (tech["position_y"] - zone_center_y)**2)**0.5
                    
                    if distance < min_distance:
                        min_distance = distance
                        best_zone = zone_name
            
            if best_zone:
                zone_assignments[tech["name"]] = best_zone
                zone_usage[best_zone] += 1
            else:
                # Fallback: usar zone con menos uso
                best_zone = min(zone_usage.keys(), key=lambda k: zone_usage[k])
                zone_assignments[tech["name"]] = best_zone
                zone_usage[best_zone] += 1
        
        # Posicionar etiquetas en sus zones asignadas
        zone_counters = {zone: 0 for zone in zones.keys()}
        
        for tech in sorted_techs:
            assigned_zone = zone_assignments.get(tech["name"])
            if not assigned_zone:
                continue
                
            zone_info = zones[assigned_zone]
            zone_counter = zone_counters[assigned_zone]
            
            # Calcular posición específica dentro de la zone
            zone_width = zone_info["x_range"][1] - zone_info["x_range"][0]
            zone_height = zone_info["y_range"][1] - zone_info["y_range"][0]
            
            # Distribución en grid dentro de la zone
            cols = 2 if zone_info["capacity"] > 3 else 1
            row = zone_counter // cols
            col = zone_counter % cols
            
            label_x = zone_info["x_range"][0] + (col + 0.5) * (zone_width / cols)
            label_y = zone_info["y_range"][0] + (row + 0.5) * (zone_height / max(1, zone_info["capacity"] // cols))
            
            zone_counters[assigned_zone] += 1
            
            color = self._get_color_for_time_to_plateau(tech["time_to_plateau"])
            
            # Línea DIRECTA y CORTA - sin curvas innecesarias
            fig.add_shape(
                type="line",
                x0=tech["position_x"], 
                y0=tech["position_y"],
                x1=label_x, 
                y1=label_y,
                line=dict(
                    color=color,
                    width=1.5,
                    dash="solid"
                ),
                layer="below"
            )
            
            # Etiqueta optimizada
            fig.add_annotation(
                x=label_x,
                y=label_y,
                text=f'<b>{tech["name"]}</b>',
                showarrow=False,
                font=dict(size=10, color='#2C3E50', family="Arial"),
                bgcolor='rgba(255, 255, 255, 0.95)',
                bordercolor=color,
                borderwidth=1.5,
                borderpad=4,
                xanchor='center',
                yanchor='middle',
                opacity=0.98
            )

    def _add_organized_trough_labels(self, fig, techs):
        """
        Etiquetas ESTRATÉGICAS para Trough - prioriza distancia MÁS CORTA
        """
        # Definir posiciones disponibles ordenadas por DISTANCIA desde el valle
        available_positions = []
        
        for tech in techs:
            # Generar posiciones candidatas ordenadas por distancia
            candidates = []
            
            # Posiciones CERCANAS abajo (prioridad 1)
            for offset_x in [-8, 0, 8]:
                for offset_y in [-25, -35, -45]:
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y, "bottom"))
            
            # Posiciones CERCANAS laterales (prioridad 2)
            for offset_x in [15, 25]:
                for offset_y in [-10, 0, 10]:
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y, "side"))
            
            # Solo si no hay opciones cercanas, usar posiciones lejanas
            for offset_x in [-15, -25]:
                for offset_y in [-10, 0, 10]:
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y, "far_side"))
            
            # Ordenar por distancia (más cercano primero)
            candidates.sort(key=lambda x: x[0])
            available_positions.append((tech, candidates))
        
        # Asignar posiciones evitando superposiciones pero priorizando cercanía
        used_positions = set()
        final_assignments = {}
        
        for tech, candidates in available_positions:
            best_position = None
            
            for distance, pos_x, pos_y, position_type in candidates:
                # Verificar si está muy cerca de una posición ya usada
                too_close = False
                for used_x, used_y in used_positions:
                    if ((pos_x - used_x)**2 + (pos_y - used_y)**2)**0.5 < 12:  # Distancia mínima entre etiquetas
                        too_close = True
                        break
                
                if not too_close:
                    best_position = (pos_x, pos_y)
                    used_positions.add((pos_x, pos_y))
                    break
            
            if best_position:
                final_assignments[tech["name"]] = best_position
        
        # Renderizar etiquetas con posiciones optimizadas
        for tech in techs:
            if tech["name"] in final_assignments:
                label_x, label_y = final_assignments[tech["name"]]
                
                color = self._get_color_for_time_to_plateau(tech["time_to_plateau"])
                
                # Línea CORTA y directa - solo la distancia necesaria
                fig.add_shape(
                    type="line", x0=tech["position_x"], y0=tech["position_y"],
                    x1=label_x, y1=label_y,
                    line=dict(color=color, width=1.5), layer="below"
                )
                
                fig.add_annotation(
                    x=label_x, y=label_y, text=f'<b>{tech["name"]}</b>',
                    showarrow=False, font=dict(size=10, color='#2C3E50', family="Arial"),
                    bgcolor='rgba(255, 255, 255, 0.95)', bordercolor=color,
                    borderwidth=1.5, borderpad=4, xanchor='center', yanchor='middle', opacity=0.98
                )

    def _add_organized_slope_labels(self, fig, techs):
        """
        Etiquetas ESTRATÉGICAS para Slope - prioriza distancia MÁS CORTA
        """
        available_positions = []
        
        for tech in techs:
            candidates = []
            
            # Posiciones CERCANAS arriba (prioridad 1)
            for offset_x in [-8, 0, 8]:
                for offset_y in [25, 35, 45]:
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y, "top"))
            
            # Posiciones CERCANAS laterales (prioridad 2)
            for offset_x in [-15, -25]:
                for offset_y in [-8, 0, 8]:
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y, "left"))
            
            # Solo si necesario, posiciones más lejanas
            for offset_x in [15, 25]:
                for offset_y in [-8, 0, 8]:
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y, "right"))
            
            candidates.sort(key=lambda x: x[0])
            available_positions.append((tech, candidates))
        
        # Asignar posiciones optimizadas
        used_positions = set()
        final_assignments = {}
        
        for tech, candidates in available_positions:
            best_position = None
            
            for distance, pos_x, pos_y, position_type in candidates:
                too_close = False
                for used_x, used_y in used_positions:
                    if ((pos_x - used_x)**2 + (pos_y - used_y)**2)**0.5 < 12:
                        too_close = True
                        break
                
                if not too_close:
                    best_position = (pos_x, pos_y)
                    used_positions.add((pos_x, pos_y))
                    break
            
            if best_position:
                final_assignments[tech["name"]] = best_position
        
        # Renderizar
        for tech in techs:
            if tech["name"] in final_assignments:
                label_x, label_y = final_assignments[tech["name"]]
                
                color = self._get_color_for_time_to_plateau(tech["time_to_plateau"])
                
                fig.add_shape(
                    type="line", x0=tech["position_x"], y0=tech["position_y"],
                    x1=label_x, y1=label_y,
                    line=dict(color=color, width=1.5), layer="below"
                )
                
                fig.add_annotation(
                    x=label_x, y=label_y, text=f'<b>{tech["name"]}</b>',
                    showarrow=False, font=dict(size=10, color='#2C3E50', family="Arial"),
                    bgcolor='rgba(255, 255, 255, 0.95)', bordercolor=color,
                    borderwidth=1.5, borderpad=4, xanchor='center', yanchor='middle', opacity=0.98
                )

    def _add_organized_trigger_labels(self, fig, techs):
        """
        Etiquetas ESTRATÉGICAS para Innovation Trigger - prioriza distancia MÁS CORTA
        """
        available_positions = []
        
        for tech in techs:
            candidates = []
            
            # Posiciones CERCANAS (prioridad por distancia)
            for offset_x in [-8, 0, 8]:
                for offset_y in [25, -25, 35, -35]:  # Alternar arriba/abajo cercano
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y))
            
            # Solo si necesario, posiciones más lejanas
            for offset_x in [-15, 15]:
                for offset_y in [45, -45]:
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y))
            
            candidates.sort(key=lambda x: x[0])
            available_positions.append((tech, candidates))
        
        # Asignar posiciones optimizadas
        used_positions = set()
        final_assignments = {}
        
        for tech, candidates in available_positions:
            best_position = None
            
            for distance, pos_x, pos_y in candidates:
                too_close = False
                for used_x, used_y in used_positions:
                    if ((pos_x - used_x)**2 + (pos_y - used_y)**2)**0.5 < 12:
                        too_close = True
                        break
                
                if not too_close:
                    best_position = (pos_x, pos_y)
                    used_positions.add((pos_x, pos_y))
                    break
            
            if best_position:
                final_assignments[tech["name"]] = best_position
        
        # Renderizar
        for tech in techs:
            if tech["name"] in final_assignments:
                label_x, label_y = final_assignments[tech["name"]]
                
                color = self._get_color_for_time_to_plateau(tech["time_to_plateau"])
                
                fig.add_shape(
                    type="line", x0=tech["position_x"], y0=tech["position_y"],
                    x1=label_x, y1=label_y,
                    line=dict(color=color, width=1.5), layer="below"
                )
                
                fig.add_annotation(
                    x=label_x, y=label_y, text=f'<b>{tech["name"]}</b>',
                    showarrow=False, font=dict(size=10, color='#2C3E50', family="Arial"),
                    bgcolor='rgba(255, 255, 255, 0.95)', bordercolor=color,
                    borderwidth=1.5, borderpad=4, xanchor='center', yanchor='middle', opacity=0.98
                )

    def _add_organized_plateau_labels(self, fig, techs):
        """
        Etiquetas ESTRATÉGICAS para Plateau - prioriza distancia MÁS CORTA
        """
        available_positions = []
        
        for tech in techs:
            candidates = []
            
            # Solo posiciones arriba (característica del plateau) pero optimizadas por distancia
            for offset_x in [-8, 0, 8]:
                for offset_y in [25, 35, 45, 55]:
                    pos_x = tech["position_x"] + offset_x
                    pos_y = tech["position_y"] + offset_y
                    if 8 <= pos_x <= 92 and -35 <= pos_y <= 210:
                        distance = (offset_x**2 + offset_y**2)**0.5
                        candidates.append((distance, pos_x, pos_y))
            
            candidates.sort(key=lambda x: x[0])
            available_positions.append((tech, candidates))
        
        # Asignar posiciones optimizadas
        used_positions = set()
        final_assignments = {}
        
        for tech, candidates in available_positions:
            best_position = None
            
            for distance, pos_x, pos_y in candidates:
                too_close = False
                for used_x, used_y in used_positions:
                    if ((pos_x - used_x)**2 + (pos_y - used_y)**2)**0.5 < 12:
                        too_close = True
                        break
                
                if not too_close:
                    best_position = (pos_x, pos_y)
                    used_positions.add((pos_x, pos_y))
                    break
            
            if best_position:
                final_assignments[tech["name"]] = best_position
        
        # Renderizar
        for tech in techs:
            if tech["name"] in final_assignments:
                label_x, label_y = final_assignments[tech["name"]]
                
                color = self._get_color_for_time_to_plateau(tech["time_to_plateau"])
                
                fig.add_shape(
                    type="line", x0=tech["position_x"], y0=tech["position_y"],
                    x1=label_x, y1=label_y,
                    line=dict(color=color, width=1.5), layer="below"
                )
                
                fig.add_annotation(
                    x=label_x, y=label_y, text=f'<b>{tech["name"]}</b>',
                    showarrow=False, font=dict(size=10, color='#2C3E50', family="Arial"),
                    bgcolor='rgba(255, 255, 255, 0.95)', bordercolor=color,
                    borderwidth=1.5, borderpad=4, xanchor='center', yanchor='middle', opacity=0.98
                )



        # Etiquetas de fases REPOSICIONADAS para la curva extendida
        phase_labels = [
            {"name": "Innovation<br>Trigger", "x": 21, "y": -25},  # Ajustado a nueva curva
            {"name": "Peak of Inflated<br>Expectations", "x": 36, "y": -25},   # Centrado en el pico
            {"name": "Trough of<br>Disillusionment", "x": 56, "y": -25},  # Centrado en el valle
            {"name": "Slope of<br>Enlightenment", "x": 72, "y": -25},     # En la pendiente
            {"name": "Plateau of<br>Productivity", "x": 83, "y": -25}     # En la meseta
        ]
        
        for label in phase_labels:
            fig.add_annotation(
                x=label["x"], 
                y=label["y"],
                text=f"<b>{label['name']}</b>",
                showarrow=False,
                font=dict(size=11, color='#34495e', family="Arial Black"),
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                borderpad=6,
                xanchor='center',
                yanchor='top',
                opacity=0.95
            )
        
        # Leyenda de tiempo al plateau
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
                marker=dict(size=12, color=item["color"]),
                name=item["label"],
                showlegend=True
            ))

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
    
    def _show_category_management(self):
        """NUEVA: Gestión completa de categorías - crear, editar, eliminar"""
        st.subheader("🏷️ Gestión Completa de Categorías")
        
        st.write("""
        Administra todas las categorías: crear nuevas, editar existentes y eliminar categorías completas.
        **¡CUIDADO!** Eliminar una categoría afectará todas las tecnologías asociadas.
        """)
        
        # Obtener categorías existentes
        try:
            categories = self.storage.storage.get_all_categories()
        except Exception as e:
            st.error(f"Error cargando categorías: {str(e)}")
            return
        
        # Pestañas para organizar funcionalidades
        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "➕ Crear Nueva", 
            "✏️ Editar Existente", 
            "🗑️ Eliminar Categoría"
        ])
        
        with sub_tab1:
            self._show_create_category_form()
        
        with sub_tab2:
            self._show_edit_category_form(categories)
        
        with sub_tab3:
            self._show_delete_category_form(categories)

    def _show_create_category_form(self):
        """Formulario para crear nuevas categorías"""
        st.write("### ➕ Crear Nueva Categoría")
        
        with st.form(key=f"{self._state_key_base}_create_category_form", clear_on_submit=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                category_name = st.text_input(
                    "Nombre de la categoría *",
                    placeholder="ej: Inteligencia Artificial, Blockchain, IoT...",
                    help="Nombre único para identificar la categoría"
                )
                
                category_description = st.text_area(
                    "Descripción (opcional)",
                    placeholder="Descripción detallada de qué tecnologías incluye esta categoría...",
                    height=100,
                    help="Descripción que ayude a identificar qué tecnologías pertenecen a esta categoría"
                )
            
            with col2:
                st.write("**Validaciones:**")
                
                # Validaciones en tiempo real
                if category_name:
                    if len(category_name.strip()) < 2:
                        st.error("Mínimo 2 caracteres")
                    elif len(category_name.strip()) > 50:
                        st.error("Máximo 50 caracteres")
                    else:
                        # Verificar duplicados
                        try:
                            existing_categories = self.storage.storage.get_all_categories()
                            existing_names = [cat.get('name', '').lower() for cat in existing_categories]
                            
                            if category_name.strip().lower() in existing_names:
                                st.error("❌ Nombre ya existe")
                            else:
                                st.success("✅ Nombre válido")
                        except:
                            st.warning("⚠️ Error verificando duplicados")
                else:
                    st.info("Ingresa un nombre")
            
            st.write("---")
            
            # Botón de creación
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                submitted = st.form_submit_button(
                    "✨ CREAR CATEGORÍA",
                    type="primary",
                    use_container_width=True
                )
            
            # Procesar creación
            if submitted:
                if not category_name or not category_name.strip():
                    st.error("❌ El nombre de la categoría es obligatorio")
                else:
                    with st.spinner("Creando categoría..."):
                        from hype_cycle_storage import HypeCycleStorage
                        
                        # Usar el método de validación
                        if hasattr(self.storage, 'create_category_with_validation'):
                            result = self.storage.create_category_with_validation(
                                category_name.strip(), 
                                category_description.strip()
                            )
                        else:
                            # Fallback a método básico
                            try:
                                category_id = self.storage.storage.add_category(
                                    category_name.strip(), 
                                    category_description.strip()
                                )
                                if category_id:
                                    result = {
                                        'success': True,
                                        'message': f"Categoría '{category_name}' creada exitosamente",
                                        'category_id': category_id
                                    }
                                else:
                                    result = {
                                        'success': False,
                                        'message': "Error creando la categoría",
                                        'category_id': None
                                    }
                            except Exception as e:
                                result = {
                                    'success': False,
                                    'message': f"Error: {str(e)}",
                                    'category_id': None
                                }
                        
                        if result['success']:
                            st.success(f"✅ {result['message']}")
                            st.balloons()
                            
                            # Limpiar cache
                            self._invalidate_cache()
                            if hasattr(st, 'cache_data'):
                                st.cache_data.clear()
                            
                            time.sleep(2)
                            st.rerun()
                        else:
                            st.error(f"❌ {result['message']}")

    def _show_edit_category_form(self, categories):
        """Formulario para editar categorías existentes"""
        st.write("### ✏️ Editar Categoría Existente")
        
        if not categories:
            st.info("No hay categorías para editar.")
            return
        
        # Filtrar categorías editables (excluir default)
        editable_categories = [cat for cat in categories if cat.get('category_id') != 'default']
        
        if not editable_categories:
            st.info("Solo existe la categoría por defecto, que no se puede editar.")
            return
        
        with st.form(key=f"{self._state_key_base}_edit_category_form", clear_on_submit=False):
            # Selector de categoría
            category_options = {}
            for cat in editable_categories:
                cat_name = cat.get('name', 'Sin nombre')
                cat_id = cat.get('category_id')
                
                # Mostrar uso de la categoría
                try:
                    usage_count = self.storage.storage.check_category_usage(cat_id)
                    display_name = f"{cat_name} ({usage_count} tecnologías)"
                except:
                    display_name = cat_name
                
                category_options[display_name] = cat
            
            selected_display_name = st.selectbox(
                "Selecciona categoría a editar:",
                options=list(category_options.keys()),
                help="Muestra el número de tecnologías en cada categoría"
            )
            
            selected_category = category_options[selected_display_name]
            
            # Mostrar información actual
            st.info(f"""
            **Categoría actual:**
            - **Nombre:** {selected_category.get('name', 'Sin nombre')}
            - **Descripción:** {selected_category.get('description', 'Sin descripción')}
            - **ID:** {selected_category.get('category_id', 'N/A')}
            """)
            
            # Campos de edición
            col1, col2 = st.columns(2)
            
            with col1:
                new_name = st.text_input(
                    "Nuevo nombre:",
                    value=selected_category.get('name', ''),
                    help="Deja vacío para no cambiar"
                )
            
            with col2:
                new_description = st.text_area(
                    "Nueva descripción:",
                    value=selected_category.get('description', ''),
                    height=100,
                    help="Deja vacío para no cambiar"
                )
            
            # Botón de actualización
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                submitted = st.form_submit_button(
                    "💾 ACTUALIZAR CATEGORÍA",
                    type="primary",
                    use_container_width=True
                )
            
            # Procesar actualización
            if submitted:
                category_id = selected_category.get('category_id')
                
                # Verificar si hay cambios
                current_name = selected_category.get('name', '')
                current_desc = selected_category.get('description', '')
                
                if new_name.strip() == current_name and new_description.strip() == current_desc:
                    st.warning("⚠️ No se detectaron cambios para actualizar")
                else:
                    with st.spinner("Actualizando categoría..."):
                        try:
                            success, message = self.storage.storage.update_category(
                                category_id, 
                                new_name.strip() if new_name.strip() != current_name else None,
                                new_description.strip() if new_description.strip() != current_desc else None
                            )
                            
                            if success:
                                st.success(f"✅ {message}")
                                
                                # Limpiar cache
                                self._invalidate_cache()
                                if hasattr(st, 'cache_data'):
                                    st.cache_data.clear()
                                
                                time.sleep(2)
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                                
                        except Exception as e:
                            st.error(f"❌ Error actualizando categoría: {str(e)}")

    def _show_delete_category_form(self, categories):
        """Formulario para eliminar categorías completas"""
        st.write("### 🗑️ Eliminar Categoría Completa")
        
        st.error("""
        ⚠️ **ADVERTENCIA IMPORTANTE**
        
        Eliminar una categoría es una acción **IRREVERSIBLE** que afectará:
        - La categoría será eliminada permanentemente
        - Todas las tecnologías asociadas serán afectadas
        
        **Opciones disponibles:**
        - **Mover tecnologías:** Las tecnologías se moverán a "Sin categoría"
        - **Eliminar todo:** Las tecnologías también serán eliminadas
        """)
        
        # Filtrar categorías eliminables (excluir default)
        deletable_categories = [cat for cat in categories if cat.get('category_id') != 'default']
        
        if not deletable_categories:
            st.info("Solo existe la categoría por defecto, que no se puede eliminar.")
            return
        
        with st.form(key=f"{self._state_key_base}_delete_category_form", clear_on_submit=False):
            # Selector de categoría
            category_options = {}
            for cat in deletable_categories:
                cat_name = cat.get('name', 'Sin nombre')
                cat_id = cat.get('category_id')
                
                # Mostrar uso de la categoría
                try:
                    usage_count = self.storage.storage.check_category_usage(cat_id)
                    display_name = f"{cat_name} ({usage_count} tecnologías)"
                    category_options[display_name] = {
                        'category': cat,
                        'usage_count': usage_count
                    }
                except:
                    display_name = cat_name
                    category_options[display_name] = {
                        'category': cat,
                        'usage_count': 0
                    }
            
            selected_display_name = st.selectbox(
                "Categoría a eliminar:",
                options=list(category_options.keys()),
                help="El número entre paréntesis indica cuántas tecnologías se verán afectadas"
            )
            
            selected_info = category_options[selected_display_name]
            selected_category = selected_info['category']
            usage_count = selected_info['usage_count']
            
            # Mostrar impacto de la eliminación
            if usage_count > 0:
                st.warning(f"""
                **Impacto de la eliminación:**
                
                📁 **Categoría:** {selected_category.get('name', 'Sin nombre')}
                🔬 **Tecnologías afectadas:** {usage_count}
                """)
            else:
                st.info(f"""
                **Categoría a eliminar:**
                
                📁 **Nombre:** {selected_category.get('name', 'Sin nombre')}
                ✅ **Sin tecnologías asociadas** - Eliminación segura
                """)
            
            # Opciones de eliminación
            if usage_count > 0:
                st.write("### 🎯 Opciones para las tecnologías")
                
                action_option = st.radio(
                    "¿Qué hacer con las tecnologías de esta categoría?",
                    options=[
                        "Mover a 'Sin categoría'",
                        "Eliminar todas las tecnologías"
                    ],
                    index=0,
                    help="Elige qué sucederá con las tecnologías cuando se elimine la categoría"
                )
                
                move_to_default = action_option == "Mover a 'Sin categoría'"
                
                if not move_to_default:
                    st.error(f"⚠️ Se eliminarán {usage_count} tecnologías PERMANENTEMENTE")
            else:
                move_to_default = True
            
            # Confirmaciones de seguridad
            st.write("### 🔒 Confirmaciones de Seguridad")
            
            col1, col2 = st.columns(2)
            
            with col1:
                confirm1 = st.checkbox(
                    f"☑️ Entiendo que se eliminará la categoría '{selected_category.get('name', '')}'"
                )
                
                if usage_count > 0:
                    if move_to_default:
                        confirm2 = st.checkbox(
                            f"☑️ Entiendo que {usage_count} tecnologías se moverán a 'Sin categoría'"
                        )
                    else:
                        confirm2 = st.checkbox(
                            f"☑️ Entiendo que {usage_count} tecnologías serán ELIMINADAS"
                        )
                else:
                    confirm2 = True
            
            with col2:
                if confirm1 and confirm2:
                    safety_text = st.text_input(
                        "Escribe 'ELIMINAR CATEGORÍA' para confirmar:",
                        placeholder="ELIMINAR CATEGORÍA"
                    )
                    
                    text_confirmed = safety_text.upper().strip() == "ELIMINAR CATEGORÍA"
                    
                    if not text_confirmed and safety_text:
                        st.error("❌ Debes escribir exactamente 'ELIMINAR CATEGORÍA'")
                else:
                    text_confirmed = False
                    st.info("Complete las confirmaciones arriba")
            
            # Botón de eliminación
            st.write("---")
            
            all_confirmed = confirm1 and confirm2 and text_confirmed
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col2:
                submitted = st.form_submit_button(
                    "🗑️ ELIMINAR CATEGORÍA",
                    type="secondary",
                    use_container_width=True
                )
            
            # Procesar eliminación
            if submitted:
                if not all_confirmed:
                    st.error("❌ Debes completar todas las confirmaciones de seguridad")
                else:
                    category_id = selected_category.get('category_id')
                    category_name = selected_category.get('name', 'Sin nombre')
                    
                    with st.spinner(f"Eliminando categoría '{category_name}'..."):
                        try:
                            if hasattr(self.storage, 'delete_category_complete'):
                                result = self.storage.delete_category_complete(
                                    category_id, 
                                    move_to_default
                                )
                            else:
                                # Fallback manual
                                # Primero obtener tecnologías
                                technologies = self.storage.get_queries_by_category(category_id)
                                
                                # Procesar tecnologías
                                for tech in technologies:
                                    tech_id = tech.get('query_id', tech.get('analysis_id'))
                                    if move_to_default:
                                        self.storage.move_technology_to_category(tech_id, "default")
                                    else:
                                        self.storage.delete_query(tech_id)
                                
                                # Eliminar categoría
                                success, message = self.storage.storage.delete_category(category_id)
                                
                                result = {
                                    'success': success,
                                    'message': message,
                                    'technologies_affected': len(technologies)
                                }
                            
                            if result['success']:
                                st.success(f"✅ {result['message']}")
                                
                                if result['technologies_affected'] > 0:
                                    if move_to_default:
                                        st.info(f"📁 {result['technologies_affected']} tecnologías movidas a 'Sin categoría'")
                                    else:
                                        st.info(f"🗑️ {result['technologies_affected']} tecnologías eliminadas")
                                
                                # Limpiar cache
                                self._invalidate_cache()
                                if hasattr(st, 'cache_data'):
                                    st.cache_data.clear()
                                
                                st.balloons()
                                time.sleep(3)
                                st.rerun()
                            else:
                                st.error(f"❌ {result['message']}")
                                
                        except Exception as e:
                            st.error(f"❌ Error eliminando categoría: {str(e)}")
        
    def show_admin_interface(self):
        """Muestra la interfaz principal de administración AMPLIADA CON IA"""
        st.header("🏷️ Administración de Categorías - Hype Cycle")
        
        st.write("""
        Gestiona las tecnologías analizadas por categoría y visualiza su posición 
        en el Hype Cycle de Gartner. **Versión optimizada con análisis IA.**
        """)
        
        # Pestañas principales - AMPLIADAS CON IA
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Vista por Categorías",
            "🎯 Gráfica Hype Cycle", 
            "🏷️ Gestión de Categorías",
            "⚙️ Gestión Avanzada",
            "🧹 Limpieza de Datos",
            "🤖 Análisis IA"  # NUEVA PESTAÑA
        ])
        
        with tab1:
            self._show_category_overview_optimized()
        
        with tab2:
            self._show_hype_cycle_chart_optimized()
        
        with tab3:
            self._show_category_management()
        
        with tab4:
            self._show_advanced_management_optimized()
        
        with tab5:
            self._show_data_cleanup()
        
        with tab6:
            self._show_ai_analysis_interface()  # NUEVA FUNCIÓN

    def _show_ai_analysis_interface(self):
        """NUEVA: Interfaz completa para análisis con IA"""
        st.subheader("🤖 Análisis Inteligente del Hype Cycle")
        
        st.write("""
        Genera insights automáticos sobre el estado del Hype Cycle usando inteligencia artificial.
        Analiza patrones, tendencias y genera recomendaciones estratégicas basadas en tus datos.
        """)
        
        # Verificar dependencias y configuración
        try:
            from hype_ai_analyzer import (
                HypeAIAnalyzer, validate_openai_key, estimate_analysis_cost, 
                check_env_setup, get_openai_key_from_env
            )
            ai_available = True
        except ImportError:
            st.error("❌ Módulo de IA no disponible. Instala: `pip install openai python-dotenv`")
            return
        
        # Verificar setup del entorno
        env_status = check_env_setup()
        
        # Mostrar estado del entorno
        self._show_environment_status(env_status)
        
        # Si no está listo automáticamente, permitir configuración manual
        api_key = None
        if env_status["ready"]:
            api_key = get_openai_key_from_env()
            st.success("✅ Configuración automática desde .env")
        else:
            api_key = self._show_manual_api_configuration(env_status)
        
        if not api_key:
            st.stop()  # No continuar sin API key válida
        
        # === CONFIGURACIÓN PRINCIPAL ===
        st.write("### ⚙️ Configuración del Análisis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Selector de categoría
            categories = self._get_cached_data(
                "categories_ai", 
                lambda: self.storage.storage.get_all_categories()
            )
            
            if not categories:
                st.error("No hay categorías disponibles para analizar")
                return
            
            category_options = {}
            category_stats = {}
            
            # Preparar opciones con estadísticas
            for cat in categories:
                cat_id = cat.get("category_id")
                cat_name = cat.get("name", "Sin nombre")
                
                # Obtener consultas de la categoría
                try:
                    queries = self.storage.get_queries_by_category(cat_id)
                    if queries:
                        category_options[f"{cat_name} ({len(queries)} tecnologías)"] = cat_id
                        category_stats[cat_id] = {
                            "name": cat_name,
                            "queries": queries,
                            "count": len(queries)
                        }
                except Exception as e:
                    continue
            
            if not category_options:
                st.warning("No hay categorías con tecnologías para analizar")
                return
            
            selected_category_display = st.selectbox(
                "📁 Selecciona categoría para analizar:",
                options=list(category_options.keys()),
                key=f"{self._state_key_base}_ai_category_selector",
                help="Selecciona la categoría que quieres que analice la IA"
            )
            
            selected_category_id = category_options[selected_category_display]
            selected_category_info = category_stats[selected_category_id]
            queries = selected_category_info["queries"]
            
            # Mostrar preview de datos
            self._show_category_preview_for_ai(queries, selected_category_info["name"])
        
        with col2:
            st.write("#### 🎛️ Configuración IA")
            
            # Mostrar fuente de API key
            if env_status["ready"]:
                st.success("🔑 API Key: desde .env")
            else:
                st.info("🔑 API Key: manual")
            
            # Configuraciones de análisis
            analysis_depth = st.selectbox(
                "📊 Profundidad del análisis:",
                options=["Ejecutivo", "Detallado", "Técnico"],
                index=1,  # "Detallado" por defecto
                key=f"{self._state_key_base}_analysis_depth",
                help="Ejecutivo: Resumen para C-level | Detallado: Análisis completo | Técnico: Análisis profundo"
            )
            
            # Modelo de IA
            ai_model = st.selectbox(
                "🧠 Modelo IA:",
                options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                index=0,  # GPT-4 por defecto
                key=f"{self._state_key_base}_ai_model",
                help="GPT-4: Mejor calidad | GPT-4-turbo: Más rápido | GPT-3.5: Más económico"
            )
            
            # Mostrar estimación de costo
            if queries:
                cost_estimate = estimate_analysis_cost(len(queries), analysis_depth)
                
                st.write("💰 **Estimación:**")
                st.caption(f"~{cost_estimate['estimated_tokens']} tokens")
                st.caption(f"~${cost_estimate['estimated_cost']:.4f} USD")
        
        # === OPCIONES AVANZADAS ===
        with st.expander("🔬 Opciones Avanzadas", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                custom_focus = st.text_area(
                    "🎯 Enfoque personalizado (opcional):",
                    placeholder="Ej: Enfócate en oportunidades de inversión\nAnaliza riesgos regulatorios\nCompara con competidores",
                    height=100,
                    key=f"{self._state_key_base}_custom_focus",
                    help="Instrucciones específicas para personalizar el análisis"
                )
            
            with col2:
                # Configuraciones adicionales
                include_comparisons = st.checkbox(
                    "📈 Incluir comparaciones históricas",
                    value=True,
                    key=f"{self._state_key_base}_include_comparisons"
                )
                
                focus_on_actionable = st.checkbox(
                    "🎯 Priorizar insights accionables",
                    value=True,
                    key=f"{self._state_key_base}_focus_actionable"
                )
                
                include_risks = st.checkbox(
                    "⚠️ Incluir análisis de riesgos",
                    value=False,
                    key=f"{self._state_key_base}_include_risks"
                )
        
        # === BOTÓN DE ANÁLISIS ===
        st.write("---")
        
        # Determinar si puede analizar
        can_analyze = bool(api_key and queries and len(queries) > 0)
        
        # Mostrar estado
        if not api_key:
            st.warning("⚠️ Configura tu OpenAI API Key para continuar")
        elif not queries:
            st.warning("⚠️ No hay tecnologías para analizar")
        elif len(queries) == 0:
            st.warning("⚠️ La categoría seleccionada está vacía")
        else:
            st.success("✅ Todo listo para el análisis IA")
        
        # Botón principal
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button(
                f"🚀 GENERAR ANÁLISIS IA",
                type="primary",
                disabled=not can_analyze,
                key=f"{self._state_key_base}_generate_analysis",
                use_container_width=True
            ):
                # EJECUTAR ANÁLISIS
                self._execute_ai_analysis(
                    api_key=api_key,
                    model=ai_model,
                    queries=queries,
                    category_name=selected_category_info["name"],
                    analysis_depth=analysis_depth,
                    custom_focus=custom_focus if custom_focus.strip() else None,
                    advanced_options={
                        "include_comparisons": include_comparisons,
                        "focus_actionable": focus_on_actionable,
                        "include_risks": include_risks
                    }
                )
        
        # === ANÁLISIS PREVIOS ===
        self._show_previous_ai_analyses()

    def _show_environment_status(self, env_status: Dict):
        """Muestra el estado del entorno y configuración"""
        st.write("### 🔧 Estado del Entorno")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if env_status["dotenv_available"]:
                st.success("✅ dotenv")
            else:
                st.error("❌ dotenv")
        
        with col2:
            if env_status["openai_available"]:
                st.success("✅ OpenAI")
            else:
                st.error("❌ OpenAI")
        
        with col3:
            if env_status["api_key_in_env"]:
                st.success("✅ API Key")
            else:
                st.warning("⚠️ Sin API Key")
        
        with col4:
            if env_status["ready"]:
                st.success("✅ Listo")
            else:
                st.warning("⚠️ Config necesaria")
        
        # Mostrar detalles si hay problemas
        if not env_status["ready"]:
            with st.expander("🔍 Detalles de Configuración", expanded=False):
                if not env_status["dotenv_available"]:
                    st.write("❌ **python-dotenv no instalado**")
                    st.code("pip install python-dotenv")
                
                if not env_status["openai_available"]:
                    st.write("❌ **openai no instalado**")
                    st.code("pip install openai")
                
                if not env_status["api_key_in_env"]:
                    st.write("❌ **OPENAI_API_KEY no encontrada en .env**")
                    st.write("Crea un archivo `.env` en la raíz del proyecto:")
                    st.code("OPENAI_API_KEY=sk-tu-api-key-aqui")
                
                elif not env_status["api_key_valid"]:
                    st.write("❌ **API Key en .env no es válida**")
                    st.write("Verifica que la key sea correcta y tengas créditos disponibles")

    def _show_manual_api_configuration(self, env_status: Dict) -> Optional[str]:
        """Permite configuración manual de API key si la automática no funciona"""
        
        if env_status["ready"]:
            return get_openai_key_from_env()
        
        st.write("### 🔑 Configuración Manual de API Key")
        
        # Explicar por qué necesita configuración manual
        if not env_status["api_key_in_env"]:
            st.info("💡 **Configuración recomendada:** Agrega `OPENAI_API_KEY=tu-key` a tu archivo `.env`")
        elif not env_status["api_key_valid"]:
            st.warning("⚠️ La API key en .env no es válida. Configura una key alternativa:")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            manual_api_key = st.text_input(
                "🔑 OpenAI API Key:",
                type="password",
                help="Tu API key de OpenAI como alternativa a .env",
                key=f"{self._state_key_base}_manual_openai_key",
                placeholder="sk-..."
            )
        
        with col2:
            if manual_api_key:
                if st.button("🔍 Validar", key=f"{self._state_key_base}_validate_manual_api"):
                    with st.spinner("Validando..."):
                        validation = validate_openai_key(manual_api_key)
                        
                        if validation["valid"]:
                            st.success(f"✅ {validation['message']}")
                            return manual_api_key
                        else:
                            st.error(f"❌ {validation['message']}")
                            return None
        
        return manual_api_key if manual_api_key else None

    def _show_category_preview_for_ai(self, queries: List[Dict], category_name: str):
        """Muestra preview de los datos que se analizarán"""
        st.write("#### 📊 Preview de Datos")
        
        if not queries:
            st.warning("No hay datos para preview")
            return
        
        # Estadísticas rápidas
        phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in queries]
        phase_counts = {}
        for phase in phases:
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        confidences = []
        mentions = []
        
        for q in queries:
            hype_metrics = q.get("hype_metrics", {})
            conf = self._safe_float_format(hype_metrics.get("confidence", 0), "float", "", "0")
            ment = self._safe_int_format(hype_metrics.get("total_mentions", 0), 0)
            
            try:
                confidences.append(float(conf))
                mentions.append(int(ment))
            except:
                continue
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        total_mentions = sum(mentions) if mentions else 0
        
        # Mostrar métricas
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🔬 Tecnologías", len(queries))
            st.metric("🎯 Confianza Promedio", f"{avg_confidence:.2f}")
        
        with col2:
            st.metric("📍 Fases Representadas", len(phase_counts))
            st.metric("📊 Total Menciones", f"{total_mentions:,}")
        
        # Distribución de fases
        if phase_counts:
            st.write("**Distribución por Fases:**")
            for phase, count in list(phase_counts.items())[:5]:  # Limitar para UI
                percentage = (count / len(queries)) * 100
                st.write(f"• {phase}: {count} ({percentage:.1f}%)")
        
        # Lista de tecnologías (primeras 5)
        with st.expander("Ver tecnologías incluidas", expanded=False):
            for i, query in enumerate(queries[:10]):  # Mostrar máximo 10
                tech_name = (
                    query.get("technology_name") or 
                    query.get("search_query", "")[:40] or 
                    f"Tecnología {i+1}"
                )
                phase = query.get("hype_metrics", {}).get("phase", "Unknown")
                st.write(f"{i+1}. **{tech_name}** - {phase}")
            
            if len(queries) > 10:
                st.write(f"... y {len(queries) - 10} tecnologías más")

    def _execute_ai_analysis(self, api_key: str, model: str, queries: List[Dict], 
                        category_name: str, analysis_depth: str, custom_focus: str = None,
                        advanced_options: Dict = None):
        """Ejecuta el análisis de IA y muestra resultados"""
        
        # Contenedor para el progreso
        progress_container = st.empty()
        results_container = st.container()
        
        try:
            with progress_container:
                with st.spinner(f"🤖 Generando análisis IA de '{category_name}'..."):
                    
                    # Inicializar analizador (ahora puede usar .env automáticamente)
                    from hype_ai_analyzer import HypeAIAnalyzer
                    
                    # Si api_key es None, el analizador lo cargará desde .env
                    analyzer = HypeAIAnalyzer(api_key=api_key, model=model)
                    
                    # Generar análisis
                    result = analyzer.analyze_category_hype(
                        queries=queries,
                        category_name=category_name,
                        analysis_depth=analysis_depth,
                        custom_focus=custom_focus
                    )
            
            # Limpiar spinner
            progress_container.empty()
            
            # Mostrar resultados
            with results_container:
                if result["success"]:
                    # ✅ ANÁLISIS EXITOSO
                    st.success("🎉 ¡Análisis IA completado exitosamente!")
                    
                    # Mostrar fuente de API key
                    api_source = getattr(analyzer, 'api_key_source', 'unknown')
                    if api_source == 'environment':
                        st.info("🔑 Usando API key desde archivo .env")
                    
                    # Métricas del análisis
                    st.write("### 📊 Métricas del Análisis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("🔬 Tecnologías", result["metadata"]["technologies_analyzed"])
                    
                    with col2:
                        st.metric("🧠 Modelo", result["metadata"]["model_used"])
                    
                    with col3:
                        st.metric("⏱️ Tiempo", f"{result['metadata']['processing_time']}s")
                    
                    with col4:
                        st.metric("💰 Costo", f"${result['cost']['total']:.4f}")
                    
                    # Detalles adicionales en expander
                    with st.expander("Ver detalles técnicos", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Uso de Tokens:**")
                            st.write(f"• Input: {result['usage']['prompt_tokens']:,}")
                            st.write(f"• Output: {result['usage']['completion_tokens']:,}")
                            st.write(f"• Total: {result['usage']['total_tokens']:,}")
                        
                        with col2:
                            st.write("**Desglose de Costos:**")
                            st.write(f"• Input: ${result['cost']['input_cost']:.4f}")
                            st.write(f"• Output: ${result['cost']['output_cost']:.4f}")
                            st.write(f"• Total: ${result['cost']['total']:.4f}")
                            st.write(f"• Fuente API: {api_source}")
                    
                    # === EL ANÁLISIS PRINCIPAL ===
                    st.write("---")
                    st.write(f"### 🧠 Análisis IA: {category_name}")
                    
                    # Mostrar el análisis con formato mejorado
                    analysis_text = result["analysis"]
                    
                    # Contenedor estilizado para el análisis
                    st.markdown(
                        f"""
                        <div style="
                            background-color: #f8f9fa;
                            color: #000000;
                            padding: 20px;
                            border-radius: 10px;
                            border-left: 5px solid #28a745;
                            margin: 10px 0;
                        ">
                        {analysis_text.replace(chr(10), '<br>')}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # === ACCIONES POST-ANÁLISIS ===
                    st.write("---")
                    st.write("### 🎬 Acciones")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button(
                            "📄 Exportar PDF", 
                            key=f"{self._state_key_base}_export_pdf",
                            help="Exportar análisis como PDF"
                        ):
                            self._export_ai_analysis_pdf(result, category_name)
                    
                    with col2:
                        if st.button(
                            "📋 Copiar Texto", 
                            key=f"{self._state_key_base}_copy_analysis",
                            help="Mostrar texto para copiar"
                        ):
                            st.code(analysis_text, language="markdown")
                    
                    with col3:
                        if st.button(
                            "💾 Guardar Análisis", 
                            key=f"{self._state_key_base}_save_analysis",
                            help="Guardar análisis en historial"
                        ):
                            self._save_ai_analysis_to_history(result, category_name)
                    
                    with col4:
                        if st.button(
                            "🔄 Nuevo Análisis", 
                            key=f"{self._state_key_base}_new_analysis",
                            help="Limpiar y hacer nuevo análisis"
                        ):
                            # Limpiar estados y rerun
                            for key in st.session_state.keys():
                                if "ai_analysis_result" in key:
                                    del st.session_state[key]
                            st.rerun()
                    
                    # Guardar resultado en session_state para acciones posteriores
                    st.session_state[f"{self._state_key_base}_last_ai_result"] = result
                    
                else:
                    # ❌ ERROR EN EL ANÁLISIS
                    st.error("❌ Error durante el análisis IA")
                    
                    error_msg = result.get("error", "Error desconocido")
                    st.error(f"**Error:** {error_msg}")
                    
                    # Sugerencias de solución específicas para configuración .env
                    st.write("### 🔧 Posibles soluciones:")
                    
                    if "api" in error_msg.lower() or "key" in error_msg.lower():
                        st.write("• Verifica tu archivo `.env` tiene: `OPENAI_API_KEY=sk-tu-key`")
                        st.write("• Asegúrate de tener créditos disponibles en OpenAI")
                        st.write("• Reinicia la aplicación después de modificar .env")
                    elif "token" in error_msg.lower():
                        st.write("• Reduce el número de tecnologías analizadas")
                        st.write("• Usa un análisis más breve ('Ejecutivo')")
                    elif "rate" in error_msg.lower():
                        st.write("• Espera unos minutos antes de volver a intentar")
                        st.write("• Considera usar un modelo más económico (GPT-3.5)")
                    else:
                        st.write("• Verifica tu conexión a internet")
                        st.write("• Verifica que el archivo .env esté en la raíz del proyecto")
                        st.write("• Intenta nuevamente en unos momentos")
                    
                    # Botón para reintentar
                    if st.button("🔄 Reintentar Análisis", key=f"{self._state_key_base}_retry_analysis"):
                        st.rerun()
        
        except Exception as e:
            progress_container.empty()
            st.error(f"❌ Error inesperado: {str(e)}")
            
            # Ayuda específica para problemas de configuración
            if "No se encontró API key" in str(e):
                st.write("### 🔧 Configurar API Key en .env")
                st.write("1. Crea un archivo `.env` en la raíz de tu proyecto")
                st.write("2. Agrega la línea: `OPENAI_API_KEY=sk-tu-api-key-aqui`")
                st.write("3. Reinicia la aplicación Streamlit")
            
            with st.expander("Ver detalles del error"):
                st.code(str(e))

    def _show_previous_ai_analyses(self):
        """Muestra análisis de IA previos guardados"""
        st.write("---")
        st.write("### 📚 Análisis Previos")
        
        # Verificar si hay análisis previos en session_state
        previous_analyses = []
        
        for key in st.session_state.keys():
            if key.startswith(f"{self._state_key_base}_saved_analysis_"):
                analysis = st.session_state[key]
                previous_analyses.append(analysis)
        
        if previous_analyses:
            # Ordenar por fecha (más recientes primero)
            previous_analyses.sort(
                key=lambda x: x.get("timestamp", ""), 
                reverse=True
            )
            
            for i, analysis in enumerate(previous_analyses[:5]):  # Mostrar últimos 5
                with st.expander(
                    f"📄 {analysis.get('category_name', 'Sin categoría')} - "
                    f"{analysis.get('timestamp', 'Sin fecha')[:16]}", 
                    expanded=False
                ):
                    # Información básica
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.write(f"**Categoría:** {analysis.get('category_name', 'N/A')}")
                        st.write(f"**Profundidad:** {analysis.get('analysis_depth', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Tecnologías:** {analysis.get('technologies_analyzed', 'N/A')}")
                        st.write(f"**Modelo:** {analysis.get('model_used', 'N/A')}")
                    
                    with col3:
                        st.write(f"**Costo:** ${analysis.get('cost', 0):.4f}")
                        st.write(f"**Tokens:** {analysis.get('total_tokens', 0):,}")
                    
                    # Botón para ver análisis completo
                    if st.button(f"👁️ Ver Análisis", key=f"{self._state_key_base}_view_prev_{i}"):
                        st.markdown("**Análisis completo:**")
                        st.markdown(analysis.get('analysis_text', 'No disponible'))
        else:
            st.info("No hay análisis previos guardados. Genera tu primer análisis IA arriba.")

    def _export_ai_analysis_pdf(self, result: Dict, category_name: str):
        """Exporta el análisis IA como PDF"""
        try:
            # Por ahora, mostrar opción de descarga como texto
            analysis_content = f"""
    ANÁLISIS IA DEL HYPE CYCLE
    Categoría: {category_name}
    Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    Modelo: {result['metadata']['model_used']}
    Tecnologías analizadas: {result['metadata']['technologies_analyzed']}

    {'-'*50}

    {result['analysis']}

    {'-'*50}

    Métricas del análisis:
    - Tokens utilizados: {result['usage']['total_tokens']:,}
    - Costo del análisis: ${result['cost']['total']:.4f}
    - Tiempo de procesamiento: {result['metadata']['processing_time']}s
    """
            
            st.download_button(
                label="📥 Descargar como .txt",
                data=analysis_content,
                file_name=f"hype_analysis_{category_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                key=f"{self._state_key_base}_download_txt"
            )
            
            st.success("✅ Archivo preparado para descarga")
            
        except Exception as e:
            st.error(f"Error preparando descarga: {str(e)}")

    def _save_ai_analysis_to_history(self, result: Dict, category_name: str):
        """Guarda el análisis IA en el historial local"""
        try:
            # Crear entrada de historial
            history_entry = {
                "category_name": category_name,
                "analysis_text": result["analysis"],
                "timestamp": datetime.now().isoformat(),
                "analysis_depth": result["metadata"]["analysis_depth"],
                "model_used": result["metadata"]["model_used"],
                "technologies_analyzed": result["metadata"]["technologies_analyzed"],
                "cost": result["cost"]["total"],
                "total_tokens": result["usage"]["total_tokens"]
            }
            
            # Guardar en session_state
            timestamp_key = datetime.now().strftime("%Y%m%d_%H%M%S")
            history_key = f"{self._state_key_base}_saved_analysis_{timestamp_key}"
            
            st.session_state[history_key] = history_entry
            
            st.success("💾 Análisis guardado en historial local")
            
        except Exception as e:
            st.error(f"Error guardando en historial: {str(e)}")