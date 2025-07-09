# src/category_admin.py - CORREGIDO PARA ESTADOS ESTABLES
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
    """Interfaz de administración de categorías y tecnologías del Hype Cycle con estados estables"""
    
    def __init__(self, hype_storage, context_prefix: str = "default"):
        """
        Inicializa la interfaz de administración
        
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
            f"{self._state_key_base}_refresh_trigger": 0
        }
        
        # Solo inicializar si no existen
        for key, default_value in state_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
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
        """Muestra la interfaz principal de administración"""
        st.header("🏷️ Administración de Categorías - Hype Cycle")
        
        st.write("""
        Gestiona las tecnologías analizadas por categoría y visualiza su posición 
        en el Hype Cycle de Gartner. Esta herramienta te permite comparar múltiples 
        tecnologías dentro de una misma categoría.
        """)
        
        # Pestañas principales
        tab1, tab2, tab3 = st.tabs([
            "📊 Vista por Categorías",
            "🎯 Gráfica Hype Cycle", 
            "⚙️ Gestión Avanzada"
        ])
        
        with tab1:
            self._show_category_overview()
        
        with tab2:
            self._show_hype_cycle_chart()
        
        with tab3:
            self._show_advanced_management()
    
    def _show_category_overview(self):
        """CORREGIDO: Vista general de categorías y tecnologías"""
        st.subheader("📋 Vista General por Categorías")
        
        # Obtener todas las categorías
        try:
            categories = self.storage.storage.get_all_categories()
        except Exception as e:
            st.error(f"Error obteniendo categorías: {str(e)}")
            return
        
        if not categories:
            st.info("No hay categorías disponibles. Crea una nueva categoría en la pestaña de análisis.")
            return
        
        # Mostrar estadísticas generales
        total_queries = len(self.storage.get_all_hype_cycle_queries())
        st.metric("Total de Tecnologías Analizadas", total_queries)
        
        # Procesar cada categoría
        for category in categories:
            category_id = category.get("category_id") or category.get("id")
            category_name = category.get("name", "Sin nombre")
            
            # Obtener consultas de esta categoría
            queries = self.storage.get_queries_by_category(category_id)
            
            if not queries:
                continue
            
            with st.expander(f"📁 **{category_name}** ({len(queries)} tecnologías)", expanded=False):
                self._show_category_details(category_id, category_name, queries)
    
    def _show_category_details(self, category_id: str, category_name: str, queries: List[Dict]):
        """CORREGIDO: Muestra detalles de una categoría específica con keys estables"""
        
        # Procesar datos de tecnologías
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
        
        # Mostrar tabla de tecnologías
        if tech_data:
            df = pd.DataFrame(tech_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Estadísticas de la categoría
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**📊 Distribución por Fases:**")
                for phase, count in phase_distribution.items():
                    percentage = (count / len(queries)) * 100
                    st.write(f"• {phase}: {count} ({percentage:.1f}%)")
            
            with col2:
                # Tecnología más mencionada
                if tech_data:
                    try:
                        # Convertir menciones a int para comparar
                        max_mentions = 0
                        most_mentioned = tech_data[0]
                        
                        for tech in tech_data:
                            mentions = self._safe_int_format(tech["📊 Menciones"], 0)
                            if mentions > max_mentions:
                                max_mentions = mentions
                                most_mentioned = tech
                        
                        st.write("**🔥 Más Mencionada:**")
                        st.write(f"• {most_mentioned['🔬 Tecnología']}")
                        st.write(f"• {max_mentions} menciones")
                    except:
                        st.write("**🔥 Más Mencionada:**")
                        st.write("• Error calculando")
            
            with col3:
                # Fecha más reciente
                try:
                    most_recent = max(tech_data, key=lambda x: x["📅 Última Actualización"])
                    st.write("**🕒 Más Reciente:**")
                    st.write(f"• {most_recent['🔬 Tecnología']}")
                    st.write(f"• {most_recent['📅 Última Actualización']}")
                except:
                    st.write("**🕒 Más Reciente:**")
                    st.write("• Error calculando")
        
        # BOTONES CON KEYS ESTABLES
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # BOTÓN "VER GRÁFICA" CON KEY ESTABLE
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
                st.balloons()
        
        with col2:
            export_button_key = f"{self._state_key_base}_export_btn_{category_id}"
            if st.button(f"📤 Exportar CSV", key=export_button_key):
                self._export_category_data(category_name, tech_data)
        
        with col3:
            update_button_key = f"{self._state_key_base}_update_btn_{category_id}"
            if st.button(f"🔄 Actualizar", key=update_button_key):
                st.info(f"Funcionalidad de actualización para {category_name} - En desarrollo")
        
        with col4:
            copy_button_key = f"{self._state_key_base}_copy_btn_{category_id}"
            if st.button(f"📋 Copiar IDs", key=copy_button_key):
                ids = [item["🆔 ID"] for item in tech_data]
                st.code(", ".join(ids))
    
    def _show_hype_cycle_chart(self):
        """CORREGIDO: Muestra la gráfica principal del Hype Cycle con estados estables"""
        st.subheader("🎯 Gráfica del Hype Cycle por Categorías")
        
        st.write("""
        **Visualización profesional del Hype Cycle de Gartner optimizada para presentaciones.**  
        Cada punto representa una tecnología, con flechas que conectan a etiquetas explicativas 
        y colores que indican el tiempo estimado hasta llegar al plateau de productividad.
        """)
        
        # Obtener categorías disponibles
        try:
            categories = self.storage.storage.get_all_categories()
        except Exception as e:
            st.error(f"Error obteniendo categorías: {str(e)}")
            return
        
        if not categories:
            st.warning("No hay categorías disponibles para mostrar.")
            return
        
        # Preparar opciones de categorías
        category_options = {}
        for cat in categories:
            cat_id = cat.get("category_id") or cat.get("id")
            cat_name = cat.get("name", "Sin nombre")
            
            # Solo incluir categorías que tengan consultas
            queries = self.storage.get_queries_by_category(cat_id)
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
                "📝 Etiquetas con flechas", 
                value=st.session_state.get(f"{self._state_key_base}_chart_show_labels", True),
                key=show_labels_key
            )
            st.session_state[f"{self._state_key_base}_chart_show_labels"] = show_labels
        
        with col3:
            show_confidence_key = f"{self._state_key_base}_show_confidence"
            show_confidence = st.checkbox(
                "🎯 Mostrar confianza", 
                value=st.session_state.get(f"{self._state_key_base}_chart_show_confidence", False),
                key=show_confidence_key
            )
            st.session_state[f"{self._state_key_base}_chart_show_confidence"] = show_confidence
        
        # Actualizar estados si hay cambio de categoría
        current_selected_id = category_options[selected_category_name]
        if current_selected_id != selected_category_id:
            st.session_state[f"{self._state_key_base}_selected_category_for_chart"] = current_selected_id
            st.session_state[f"{self._state_key_base}_chart_category_name"] = selected_category_name
        
        # Obtener tecnologías de la categoría seleccionada
        queries = self.storage.get_queries_by_category(current_selected_id)
        active_queries = [q for q in queries if q.get("is_active", True)]
        
        if not active_queries:
            st.warning(f"No hay tecnologías activas en la categoría '{selected_category_name}'")
            
            # Información de debug mejorada
            with st.expander("🔍 Información de Debug"):
                st.write(f"- **Categoría seleccionada:** {selected_category_name} (ID: {current_selected_id})")
                st.write(f"- **Total queries encontradas:** {len(queries)}")
                st.write(f"- **Queries activas:** {len(active_queries)}")
                
                if queries:
                    st.write("- **Estados de las tecnologías:**")
                    for i, q in enumerate(queries):
                        is_active = q.get("is_active", True)
                        tech_name = q.get("technology_name") or q.get("name") or q.get("search_query", "")[:30]
                        status = "✅ Activa" if is_active else "❌ Inactiva"
                        st.write(f"  • {tech_name}: {status}")
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
        
        # Generar y mostrar gráfica mejorada
        try:
            with st.spinner(f"🎨 Generando visualización profesional para {len(active_queries)} tecnologías..."):
                fig = self._create_hype_cycle_chart(
                    active_queries, 
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
                
                # Información adicional para presentaciones
                with st.expander("📋 Información para Presentaciones", expanded=False):
                    st.write("### 🎯 Consejos para Presentar")
                    st.write("""
                    - **📱 Descargar:** Usa el botón de descarga para obtener la imagen en alta resolución
                    - **🖥️ Proyección:** La gráfica está optimizada para pantallas de presentación
                    - **🎨 Colores:** Los colores representan tiempo estimado hasta el plateau
                    - **➡️ Flechas:** Conectan cada tecnología con su etiqueta explicativa
                    - **📊 Dimensiones:** 1200x750px optimizado para slides y documentos
                    """)
                    
                    # Estadísticas de la gráfica para contexto en presentaciones
                    st.write("### 📊 Datos de Contexto")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        phases_dist = {}
                        for q in active_queries:
                            phase = q.get("hype_metrics", {}).get("phase", "Unknown")
                            phases_dist[phase] = phases_dist.get(phase, 0) + 1
                        
                        st.write("**Distribución por Fases:**")
                        for phase, count in phases_dist.items():
                            percentage = (count / len(active_queries)) * 100
                            st.write(f"• {phase}: {count} ({percentage:.1f}%)")
                    
                    with col2:
                        time_dist = {}
                        for q in active_queries:
                            time_to_plateau = q.get("hype_metrics", {}).get("time_to_plateau", "N/A")
                            time_dist[time_to_plateau] = time_dist.get(time_to_plateau, 0) + 1
                        
                        st.write("**Tiempo al Plateau:**")
                        for time_est, count in time_dist.items():
                            st.write(f"• {time_est}: {count}")
                    
                    with col3:
                        # Menciones totales con formateo seguro
                        total_mentions = 0
                        for q in active_queries:
                            mentions_raw = q.get("hype_metrics", {}).get("total_mentions", 0)
                            mentions_int = self._safe_int_format(mentions_raw, 0)
                            total_mentions += mentions_int
                        
                        st.metric("Total Menciones", f"{total_mentions:,}")
                        
                        # Fecha más reciente
                        dates = [q.get("execution_date", "") for q in active_queries if q.get("execution_date")]
                        if dates:
                            latest_date = max(dates)
                            try:
                                formatted_date = datetime.fromisoformat(latest_date.replace('Z', '+00:00')).strftime("%Y-%m-%d")
                                st.write(f"**Análisis más reciente:** {formatted_date}")
                            except:
                                st.write(f"**Análisis más reciente:** {latest_date[:10]}")
                
                # Mostrar leyenda de la gráfica
                self._show_chart_legend(active_queries)
            
            else:
                st.error("❌ Error: La gráfica está vacía o no se pudo generar")
                st.write("**Información de debug:**")
                st.write(f"- Figura creada: {fig is not None}")
                st.write(f"- Número de trazas: {len(fig.data) if fig else 0}")
                
        except Exception as e:
            st.error(f"❌ Error generando la gráfica: {str(e)}")
            with st.expander("📋 Ver detalles del error"):
                import traceback
                st.code(traceback.format_exc())
    
    def _create_hype_cycle_chart(self, queries: List[Dict], category_name: str, 
                        show_labels: bool = True, show_confidence: bool = False) -> go.Figure:
        """
        Crea la gráfica del Hype Cycle estilo Gartner con formateo seguro
        """
        # Crear figura con dimensiones amplias
        fig = go.Figure()
        
        # Curva ampliada con zona peak más extensa
        x_curve = np.linspace(0, 100, 1000)
        
        # Rediseñar curva con Peak más amplio y definido
        trigger = 15 * np.exp(-((x_curve - 12)/8)**2)
        peak = 70 * np.exp(-((x_curve - 26)/12)**2)
        trough = -25 * np.exp(-((x_curve - 55)/12)**2)
        slope_rise = 15 * (1 / (1 + np.exp(-(x_curve - 75)/5)))
        plateau = 25 * (1 / (1 + np.exp(-(x_curve - 90)/4)))
        
        baseline = 25
        y_curve = baseline + trigger + peak + trough + slope_rise + plateau
        
        # Suavizar la curva
        try:
            from scipy.ndimage import gaussian_filter1d
            y_curve = gaussian_filter1d(y_curve, sigma=2.5)
        except:
            window = 9
            y_smooth = np.convolve(y_curve, np.ones(window)/window, mode='same')
            y_curve = y_smooth
        
        y_curve = np.clip(y_curve, 8, 90)
        
        # Función para obtener posición exacta sobre la curva
        def get_exact_position_on_curve(x_pos):
            if x_pos < 0 or x_pos > 100:
                return None
            idx = int(x_pos * (len(x_curve) - 1) / 100)
            idx = min(max(idx, 0), len(y_curve) - 1)
            return float(x_curve[idx]), float(y_curve[idx])
        
        # Función para calcular pendiente de la curva
        def get_curve_slope(x_pos):
            if x_pos <= 1 or x_pos >= 99:
                return 0
            
            x1 = max(0, x_pos - 1)
            x2 = min(100, x_pos + 1)
            
            _, y1 = get_exact_position_on_curve(x1)
            _, y2 = get_exact_position_on_curve(x2)
            
            slope = (y2 - y1) / (x2 - x1)
            return slope
        
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
        
        # Definir zonas con posiciones específicas sobre la curva
        phase_positions = {
            "Innovation Trigger": {
                "x_range": list(range(8, 18, 2)),
                "max_capacity": 5
            },
            "Peak of Inflated Expectations": {
                "x_range": list(range(20, 36, 1)),
                "max_capacity": 16
            },
            "Trough of Disillusionment": {
                "x_range": list(range(45, 66, 2)),
                "max_capacity": 11
            },
            "Slope of Enlightenment": {
                "x_range": list(range(68, 83, 2)),
                "max_capacity": 8
            },
            "Plateau of Productivity": {
                "x_range": list(range(85, 97, 2)),
                "max_capacity": 6
            },
            "Unknown": {
                "x_range": [50],
                "max_capacity": 1
            }
        }
        
        # Procesar y posicionar tecnologías
        technologies = []
        phase_counters = {phase: 0 for phase in phase_positions.keys()}
        
        # Limitar a 45 tecnologías para optimal display
        limited_queries = queries[:45] if len(queries) > 45 else queries
        
        for i, query in enumerate(limited_queries):
            # Extracción segura de datos
            try:
                if not isinstance(query, dict):
                    continue
                    
                hype_metrics = query.get("hype_metrics", {})
                
                if not isinstance(hype_metrics, dict):
                    hype_metrics = {}
                
                phase = hype_metrics.get("phase", "Unknown")
                
                # FORMATEO SEGURO DE VALORES NUMÉRICOS
                confidence_raw = hype_metrics.get("confidence", 0.5)
                confidence = float(self._safe_float_format(confidence_raw, "", "0.5"))
                
                total_mentions_raw = hype_metrics.get("total_mentions", 0)
                total_mentions = self._safe_int_format(total_mentions_raw, 0)
                
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
                    query.get("search_query", f"Tecnología_{i}")[:20]
                )
                
                time_to_plateau = hype_metrics.get("time_to_plateau", "N/A")
                
                # FORMATEO SEGURO DE SENTIMENT
                sentiment_avg_raw = hype_metrics.get("sentiment_avg", 0)
                sentiment_avg = float(self._safe_float_format(sentiment_avg_raw, "", "0.0"))
                
                technologies.append({
                    "name": tech_name,
                    "phase": phase,
                    "confidence": confidence,
                    "position_x": exact_x,
                    "position_y": exact_y,
                    "query_id": query.get("query_id", f"query_{i}"),
                    "time_to_plateau": time_to_plateau,
                    "total_mentions": total_mentions,
                    "sentiment_avg": sentiment_avg,
                    "slope": get_curve_slope(x_pos)
                })
                
            except Exception as e:
                # En caso de error con una tecnología específica, continuar con la siguiente
                print(f"Error procesando tecnología {i}: {str(e)}")
                continue
        
        # Añadir tecnologías con posicionamiento inteligente de etiquetas estilo Gartner
        for i, tech in enumerate(technologies):
            # Tamaño del punto basado en métricas
            base_size = 12
            confidence_factor = tech["confidence"] * 6
            mentions_factor = min(tech["total_mentions"] / 200, 4)
            size = base_size + confidence_factor + mentions_factor
            
            color = self._get_classic_color_for_time_to_plateau(tech["time_to_plateau"])
            
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
                    <b style="font-size:14px">{tech['name']}</b><br>
                    <b>Fase:</b> {tech['phase']}<br>
                    <b>Confianza:</b> {tech['confidence']:.1%}<br>
                    <b>Tiempo al Plateau:</b> {tech['time_to_plateau']}<br>
                    <b>Menciones:</b> {tech['total_mentions']:,}<br>
                    <extra></extra>
                """,
                showlegend=False
            ))
            
            # Etiquetas estilo Gartner con líneas conectoras
            if show_labels:
                label_x, label_y, _ = self._calculate_intelligent_label_position(
                    tech["position_x"], tech["position_y"], tech["slope"], tech["name"], i
                )
                
                # Añadir línea conectora simple (estilo Gartner)
                fig.add_shape(
                    type="line",
                    x0=tech["position_x"], 
                    y0=tech["position_y"],
                    x1=label_x, 
                    y1=label_y,
                    line=dict(
                        color=color,
                        width=1.5
                    ),
                    layer="below"
                )
                
                # Añadir etiqueta sin flecha (estilo Gartner)
                fig.add_annotation(
                    x=label_x,
                    y=label_y,
                    text=f'<b style="font-size:10px">{tech["name"]}</b>',
                    showarrow=False,
                    font=dict(
                        size=10, 
                        color='#2C3E50',
                        family="Arial"
                    ),
                    bgcolor='rgba(255, 255, 255, 0.95)',
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=4,
                    xanchor='center',
                    yanchor='middle',
                    opacity=0.95
                )
        
        # Etiquetas de fases
        phase_labels = [
            {"name": "Innovation<br>Trigger", "x": 12, "y": -25},
            {"name": "Peak of Inflated<br>Expectations", "x": 28, "y": -25},
            {"name": "Trough of<br>Disillusionment", "x": 55, "y": -25},
            {"name": "Slope of<br>Enlightenment", "x": 75, "y": -25},
            {"name": "Plateau of<br>Productivity", "x": 90, "y": -25}
        ]
        
        for label in phase_labels:
            fig.add_annotation(
                x=label["x"], 
                y=label["y"],
                text=f"<b>{label['name']}</b>",
                showarrow=False,
                font=dict(size=10, color='#7f8c8d', family="Arial"),
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                borderpad=6,
                xanchor='center',
                yanchor='top',
                opacity=0.9
            )
        
        # Líneas divisorias suaves
        division_lines = [18, 36, 65, 83]
        for x_pos in division_lines:
            fig.add_vline(
                x=x_pos, 
                line=dict(color="rgba(123, 139, 158, 0.2)", width=1, dash="dot"),
                layer="below"
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
        
        # Layout optimizado estilo Gartner
        fig.update_layout(
            title=dict(
                text=f"<b>Hype Cycle - {category_name}</b><br><sub>({len(limited_queries)} tecnologías analizadas)</sub>",
                x=0.5,
                font=dict(size=20, color='#2C3E50')
            ),
            xaxis=dict(
                title=dict(
                    text="<b>TIME</b>",
                    font=dict(size=14, color='#7f8c8d')
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
                    font=dict(size=14, color='#7f8c8d')
                ),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-35, 120],
                showline=True,
                linecolor='#bdc3c7',
                linewidth=2
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=850,
            width=1300,
            showlegend=True,
            font=dict(family="Arial"),
            margin=dict(t=120, l=80, r=200, b=100),
            hovermode='closest',
            legend=dict(
                title=dict(
                    text="<b>Tiempo al Plateau</b>",
                    font=dict(size=12, color="#2C3E50")
                ),
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                font=dict(size=10, color="#2C3E50"),
                itemsizing="constant"
            )
        )
        
        return fig

    def _calculate_intelligent_label_position(self, point_x: float, point_y: float, 
                                        slope: float, text: str, index: int) -> tuple:
        """Calcula posición inteligente de etiqueta estilo Gartner"""
        # Estrategia mejorada para zona del Peak
        if point_y > 70:  # Zona alta (Peak) - distribución mejorada
            level = index // 4
            position_in_level = index % 4
            
            if level % 2 == 0:  # Niveles pares: arriba
                label_x = point_x + (position_in_level - 1.5) * 10
                label_y = point_y + 25 + (level * 15)
            else:  # Niveles impares: abajo
                label_x = point_x + (position_in_level - 1.5) * 10
                label_y = point_y - 20 - (level * 10)
                
        elif point_x < 25:  # Innovation Trigger
            label_x = point_x - 3 + (index % 3) * 4
            label_y = point_y + 18 + (index % 2) * 8
            
        elif point_x > 75:  # Plateau
            label_x = point_x + (index % 4 - 2) * 4
            label_y = point_y + 16 + (index % 3) * 6
            
        elif point_y < 30:  # Trough
            label_x = point_x + (index % 5 - 2) * 5
            label_y = point_y - 16 - (index % 3) * 5
            
        else:  # Slope - distribución mixta
            if index % 2 == 0:
                label_x = point_x + (index % 4 - 1.5) * 4
                label_y = point_y + 15 + (index % 2) * 7
            else:
                label_x = point_x + (index % 4 - 1.5) * 4
                label_y = point_y - 12 - (index % 2) * 6
        
        # Mantener dentro de límites
        label_x = max(5, min(95, label_x))
        label_y = max(-30, min(115, label_y))
        
        return label_x, label_y, {"ax": 0, "ay": 0}

    def _get_classic_color_for_time_to_plateau(self, time_estimate: str) -> str:
        """Colores clásicos para tiempo al plateau"""
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
    
    def _show_chart_legend(self, queries: List[Dict]):
        """Muestra tabla explicativa de la gráfica con formateo seguro"""
        st.subheader("📋 Tecnologías en la Gráfica")
        
        legend_data = []
        for query in queries:
            hype_metrics = query.get("hype_metrics", {})
            
            # Extraer nombre de tecnología
            tech_name = (
                query.get("technology_name") or 
                query.get("name") or 
                query.get("search_query", "")[:30]
            )
            
            # FORMATEO SEGURO DE MÉTRICAS
            confidence_raw = hype_metrics.get("confidence", 0)
            confidence_formatted = self._safe_float_format(confidence_raw, ".2f", "0.00")
            
            total_mentions_raw = hype_metrics.get("total_mentions", 0)
            total_mentions_formatted = self._safe_int_format(total_mentions_raw, 0)
            
            sentiment_avg_raw = hype_metrics.get("sentiment_avg", 0)
            sentiment_formatted = self._safe_float_format(sentiment_avg_raw, ".2f", "0.00")
            
            legend_data.append({
                "🔬 Tecnología": tech_name,
                "📍 Fase Actual": hype_metrics.get("phase", "Unknown"),
                "🎯 Confianza": confidence_formatted,
                "⏱️ Tiempo al Plateau": hype_metrics.get("time_to_plateau", "N/A"),
                "📊 Menciones": total_mentions_formatted,
                "💭 Sentimiento": sentiment_formatted
            })
        
        # Ordenar por fase para mejor presentación
        phase_order = {
            "Innovation Trigger": 1,
            "Peak of Inflated Expectations": 2,
            "Trough of Disillusionment": 3,
            "Slope of Enlightenment": 4,
            "Plateau of Productivity": 5
        }
        
        legend_data.sort(key=lambda x: phase_order.get(x["📍 Fase Actual"], 6))
        
        df_legend = pd.DataFrame(legend_data)
        st.dataframe(df_legend, use_container_width=True, hide_index=True)
        
        # Estadísticas de la gráfica con formateo seguro
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tech = len(legend_data)
            st.metric("Total Tecnologías", total_tech)
        
        with col2:
            # Calcular confianza promedio de forma segura
            confidences = []
            for item in legend_data:
                try:
                    conf_val = float(item["🎯 Confianza"])
                    confidences.append(conf_val)
                except:
                    confidences.append(0.0)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            st.metric("Confianza Promedio", self._safe_float_format(avg_confidence, ".2f"))
        
        with col3:
            # Calcular total menciones de forma segura
            total_mentions = 0
            for item in legend_data:
                try:
                    mentions_val = int(item["📊 Menciones"])
                    total_mentions += mentions_val
                except:
                    pass
            
            st.metric("Total Menciones", total_mentions)
        
        with col4:
            # Calcular sentimiento promedio de forma segura
            sentiments = []
            for item in legend_data:
                try:
                    sent_val = float(item["💭 Sentimiento"])
                    sentiments.append(sent_val)
                except:
                    sentiments.append(0.0)
            
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
            st.metric("Sentimiento Promedio", self._safe_float_format(avg_sentiment, ".2f"))
    
    def _show_advanced_management(self):
        """Gestión avanzada de categorías y tecnologías"""
        st.subheader("⚙️ Gestión Avanzada")
        
        st.write("""
        Herramientas adicionales para la gestión y mantenimiento de las categorías 
        y tecnologías del Hype Cycle en DynamoDB.
        """)
        
        # Sección de operaciones masivas
        with st.expander("🔄 Operaciones Masivas", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Actualización de Datos")
                
                # KEYS ESTABLES para botones de operaciones
                recalc_positions_key = f"{self._state_key_base}_recalc_positions"
                regen_stats_key = f"{self._state_key_base}_regen_stats"
                
                if st.button("🔄 Recalcular Todas las Posiciones", type="secondary", key=recalc_positions_key):
                    with st.spinner("Recalculando posiciones..."):
                        self._recalculate_all_positions()
                
                if st.button("📊 Regenerar Estadísticas", type="secondary", key=regen_stats_key):
                    with st.spinner("Regenerando estadísticas..."):
                        st.info("Funcionalidad en desarrollo - Regenerar estadísticas")
            
            with col2:
                st.write("#### Limpieza de Datos")
                
                # KEYS ESTABLES para botones de limpieza
                cleanup_key = f"{self._state_key_base}_cleanup"
                detect_dupes_key = f"{self._state_key_base}_detect_dupes"
                
                if st.button("🗑️ Limpiar Consultas Inactivas", type="secondary", key=cleanup_key):
                    self._cleanup_inactive_queries()
                
                if st.button("🔍 Detectar Duplicados", type="secondary", key=detect_dupes_key):
                    self._detect_duplicates()
        
        # Sección de estadísticas globales
        with st.expander("📊 Estadísticas Globales", expanded=True):
            self._show_global_statistics()
        
        # Sección de exportación
        with st.expander("📤 Exportación y Backup", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                # KEYS ESTABLES para botones de exportación
                export_all_key = f"{self._state_key_base}_export_all"
                if st.button("📥 Exportar Todas las Categorías", key=export_all_key):
                    self._export_all_categories()
            
            with col2:
                backup_key = f"{self._state_key_base}_backup"
                if st.button("💾 Crear Backup Completo", key=backup_key):
                    self._create_full_backup()
    
    def _show_global_statistics(self):
        """Muestra estadísticas globales del sistema con formateo seguro"""
        try:
            all_queries = self.storage.get_all_hype_cycle_queries()
            
            if not all_queries:
                st.info("No hay datos para mostrar estadísticas globales.")
                return
            
            # Métricas generales
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Tecnologías", len(all_queries))
            
            with col2:
                categories = set(q.get("category_id", "unknown") for q in all_queries)
                st.metric("Categorías Activas", len(categories))
            
            with col3:
                # Confianza promedio con formateo seguro
                confidences = []
                for q in all_queries:
                    conf_raw = q.get("hype_metrics", {}).get("confidence", 0)
                    conf_float = float(self._safe_float_format(conf_raw, "", "0"))
                    confidences.append(conf_float)
                
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                st.metric("Confianza Promedio", self._safe_float_format(avg_confidence, ".2f"))
            
            with col4:
                # Tecnologías analizadas este mes
                current_month = datetime.now().strftime("%Y-%m")
                recent_queries = [q for q in all_queries if q.get("execution_date", "").startswith(current_month)]
                st.metric("Este Mes", len(recent_queries))
            
            # Distribución por fases
            st.write("#### 📊 Distribución Global por Fases")
            
            phase_counts = {}
            for query in all_queries:
                phase = query.get("hype_metrics", {}).get("phase", "Unknown")
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            
            if phase_counts:
                # Crear gráfico de barras
                phases = list(phase_counts.keys())
                counts = list(phase_counts.values())
                
                fig_phases = go.Figure([go.Bar(x=phases, y=counts)])
                fig_phases.update_layout(
                    title="Distribución de Tecnologías por Fase del Hype Cycle",
                    xaxis_title="Fase",
                    yaxis_title="Número de Tecnologías",
                    height=400
                )
                st.plotly_chart(fig_phases, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error mostrando estadísticas globales: {str(e)}")
    
    def _export_category_data(self, category_name: str, tech_data: List[Dict]):
        """Exporta datos de una categoría específica"""
        try:
            df = pd.DataFrame(tech_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label=f"📥 Descargar {category_name}.csv",
                data=csv,
                file_name=f"hype_cycle_{category_name}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error exportando datos: {str(e)}")
    
    def _recalculate_all_positions(self):
        """Recalcula las posiciones de todas las tecnologías"""
        try:
            updated_count = 0
            all_queries = self.storage.get_all_hype_cycle_queries()
            
            for query in all_queries:
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
                
                # Actualizar en el objeto (esto es conceptual)
                hype_metrics["hype_cycle_position_x"] = pos_x
                hype_metrics["hype_cycle_position_y"] = pos_y
                
                updated_count += 1
            
            st.success(f"✅ Recalculadas {updated_count} posiciones de tecnologías")
            
        except Exception as e:
            st.error(f"Error recalculando posiciones: {str(e)}")
    
    def _cleanup_inactive_queries(self):
        """Limpia consultas marcadas como inactivas"""
        st.info("🔄 Funcionalidad de limpieza - En desarrollo")
    
    def _detect_duplicates(self):
        """Detecta posibles consultas duplicadas"""
        try:
            all_queries = self.storage.get_all_hype_cycle_queries()
            seen_queries = {}
            duplicates = []
            
            for query in all_queries:
                search_query = query.get("search_query", "").lower().strip()
                if search_query in seen_queries:
                    duplicates.append({
                        "original": seen_queries[search_query],
                        "duplicate": query
                    })
                else:
                    seen_queries[search_query] = query
            
            if duplicates:
                st.warning(f"⚠️ Encontrados {len(duplicates)} posibles duplicados")
                for dup in duplicates[:5]:
                    st.write(f"• Query: {dup['duplicate'].get('search_query', '')[:50]}...")
            else:
                st.success("✅ No se encontraron duplicados")
                
        except Exception as e:
            st.error(f"Error detectando duplicados: {str(e)}")
    
    def _export_all_categories(self):
        """Exporta datos de todas las categorías"""
        st.info("📤 Funcionalidad de exportación completa - En desarrollo")
    
    def _create_full_backup(self):
        """Crea un backup completo del sistema"""
        st.info("💾 Funcionalidad de backup - En desarrollo")