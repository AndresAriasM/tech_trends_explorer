# src/category_admin.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import json
import time
from datetime import datetime, timedelta

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
    """Interfaz de administración de categorías y tecnologías del Hype Cycle"""
    
    def __init__(self, hype_storage, context_prefix: str = "default"):
        """
        Inicializa la interfaz de administración
        
        Args:
            hype_storage: Instancia de HypeCycleStorage
            context_prefix: Prefijo único para evitar conflictos de keys
        """
        self.storage = hype_storage
        self.context_prefix = context_prefix
        
        # AÑADIR ESTA LÍNEA QUE FALTA:
        from hype_cycle_positioning import HypeCyclePositioner
        self.positioner = HypeCyclePositioner()
        
        # USAR CONTEXT_PREFIX ESTABLE EN LUGAR DE TIMESTAMP ALEATORIO
        # El context_prefix debe ser pasado desde fuera y ser estable
        if not context_prefix or context_prefix == "default":
            # Solo usar timestamp como fallback si no se proporciona un contexto
            import time
            self.unique_id = f"admin_{int(time.time())}"
        else:
            # Usar el context_prefix como base para IDs estables
            self.unique_id = f"admin_{context_prefix}"
        
        # INICIALIZAR ESTADOS DE MANERA MÁS CONTROLADA
        # Estados específicos para esta instancia de admin
        admin_state_key = f"admin_state_{self.unique_id}"
        if admin_state_key not in st.session_state:
            st.session_state[admin_state_key] = {
                'selected_category_for_chart': None,
                'admin_refresh_trigger': 0,
                'last_chart_category': None
            }
        
        # Referencias directas para facilitar acceso
        self.admin_state = st.session_state[admin_state_key]
    
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
        """Vista general de categorías y tecnologías - VERSIÓN CORREGIDA"""
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
        
        # Procesar cada categoría - CORRECCIÓN ASEGURADA
        for category in categories:  # CORRECTO: usar 'category'
            category_id = category.get("id") or category.get("category_id")  # CORRECTO
            category_name = category.get("name", "Sin nombre")  # CORRECTO
            
            # Obtener consultas de esta categoría
            queries = self.storage.get_queries_by_category(category_id)
            
            if not queries:
                continue
            
            with st.expander(f"📁 **{category_name}** ({len(queries)} tecnologías)", expanded=False):
                self._show_category_details(category_id, category_name, queries)
    
    def _show_category_details(self, category_id: str, category_name: str, queries: List[Dict]):
        """Muestra detalles de una categoría específica"""
        
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
            
            tech_data.append({
                "🔬 Tecnología": tech_name,
                "📍 Fase": phase,
                "🎯 Confianza": f"{hype_metrics.get('confidence', 0):.2f}",
                "⏱️ Tiempo al Plateau": hype_metrics.get("time_to_plateau", "N/A"),
                "📅 Última Actualización": formatted_date,
                "📊 Menciones": hype_metrics.get("total_mentions", 0),
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
                most_mentioned = max(tech_data, key=lambda x: x["📊 Menciones"])
                st.write("**🔥 Más Mencionada:**")
                st.write(f"• {most_mentioned['🔬 Tecnología']}")
                st.write(f"• {most_mentioned['📊 Menciones']} menciones")
            
            with col3:
                # Fecha más reciente
                most_recent = max(tech_data, key=lambda x: x["📅 Última Actualización"])
                st.write("**🕒 Más Reciente:**")
                st.write(f"• {most_recent['🔬 Tecnología']}")
                st.write(f"• {most_recent['📅 Última Actualización']}")
        
        # Botones de acción
        st.write("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            # BOTÓN "VER GRÁFICA" MEJORADO
            chart_button_key = f"chart_btn_{category_id}_{self.context_prefix}"
            
            if st.button(f"📊 Ver Gráfica", key=chart_button_key, type="primary"):
                # MÉTODO MEJORADO PARA PRESELECCIONAR CATEGORÍA
                
                # 1. Establecer la categoría en el estado del selectbox directamente
                chart_selector_key = "hype_chart_category_selector_static"
                st.session_state[chart_selector_key] = category_name
                
                # 2. También mantener compatibilidad con estados antiguos (por si acaso)
                st.session_state['selected_category_for_chart'] = category_id
                st.session_state['chart_category_id'] = category_id
                st.session_state['chart_category_name'] = category_name
                
                # 3. Forzar actualización del estado de categoría previa
                st.session_state['hype_chart_previous_category'] = None  # Para forzar detección de cambio
                
                # 4. Limpiar cache de gráficas previas
                for key in list(st.session_state.keys()):
                    if key.startswith('chart_cache_') or key.startswith('plot_data_'):
                        del st.session_state[key]
                
                # 5. Mostrar confirmación
                st.success(f"✅ Categoría '{category_name}' seleccionada para visualización.")
                st.info("👆 **Haz clic en la pestaña '🎯 Gráfica Hype Cycle' arriba para ver la gráfica.**")
                
                # 6. Opcional: Auto-scroll o indicación visual
                st.balloons()  # Efecto visual para confirmar la acción
        
        with col2:
            export_button_key = f"export_btn_{category_id}_{self.context_prefix}"
            if st.button(f"📤 Exportar CSV", key=export_button_key):
                self._export_category_data(category_name, tech_data)
        
        with col3:
            update_button_key = f"update_btn_{category_id}_{self.context_prefix}"
            if st.button(f"🔄 Actualizar", key=update_button_key):
                st.info(f"Funcionalidad de actualización para {category_name} - En desarrollo")
        
        with col4:
            copy_button_key = f"copy_btn_{category_id}_{self.context_prefix}"
            if st.button(f"📋 Copiar IDs", key=copy_button_key):
                ids = [item["🆔 ID"] for item in tech_data]
                st.code(", ".join(ids))

        # ADICIONAL: Botón de debug para limpiar todo el estado (temporal, para testing)
        if st.checkbox("🔧 Modo Debug", key=f"debug_mode_{category_id}_{self.unique_id}"):
            st.write("**Estado actual de la sesión (categorías):**")
            category_states = {
                k: v for k, v in st.session_state.items() 
                if any(term in k.lower() for term in ['category', 'chart', 'hype'])
            }
            
            if category_states:
                for key, value in category_states.items():
                    st.write(f"- {key}: {value}")
                    
                if st.button("🧹 Limpiar Estado Completo", key=f"clear_all_state_{category_id}_{self.unique_id}"):
                    for key in list(category_states.keys()):
                        if key in st.session_state:
                            del st.session_state[key]
                    st.success("Estado limpiado completamente")
                    st.rerun()
            else:
                st.write("No hay estados de categoría activos")
    
    def _show_hype_cycle_chart(self):
        """Muestra la gráfica principal del Hype Cycle - VERSIÓN CORREGIDA"""
        st.subheader("🎯 Gráfica del Hype Cycle por Categorías")
        
        st.write("""
        Visualiza todas las tecnologías de una categoría posicionadas en el Hype Cycle de Gartner.
        Cada punto representa una tecnología, con colores que indican el tiempo estimado hasta llegar al plateau.
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
            cat_id = cat.get("id") or cat.get("category_id")
            cat_name = cat.get("name", "Sin nombre")
            
            # Solo incluir categorías que tengan consultas
            queries = self.storage.get_queries_by_category(cat_id)
            if queries:
                category_options[cat_name] = cat_id
        
        if not category_options:
            st.info("No hay categorías con tecnologías analizadas para mostrar en la gráfica.")
            return
        
        # CLAVE ESTÁTICA PARA EL SELECTBOX (SIN TIMESTAMP)
        CHART_CATEGORY_KEY = "hype_chart_category_selector_static"
        
        # Inicializar el estado si no existe
        if CHART_CATEGORY_KEY not in st.session_state:
            # Verificar si hay categoría preseleccionada
            preselected_id = (
                st.session_state.get('selected_category_for_chart') or 
                st.session_state.get('chart_category_id')
            )
            
            if preselected_id and preselected_id in category_options.values():
                # Encontrar el nombre de la categoría preseleccionada
                preselected_name = None
                for name, cat_id in category_options.items():
                    if cat_id == preselected_id:
                        preselected_name = name
                        break
                st.session_state[CHART_CATEGORY_KEY] = preselected_name or list(category_options.keys())[0]
            else:
                # Usar la primera categoría disponible
                st.session_state[CHART_CATEGORY_KEY] = list(category_options.keys())[0]
        
        # SELECTOR DE CATEGORÍA CON KEY ESTÁTICA
        selected_category_name = st.selectbox(
            "🏷️ Selecciona una categoría para visualizar:",
            options=list(category_options.keys()),
            index=list(category_options.keys()).index(st.session_state[CHART_CATEGORY_KEY]),
            key=CHART_CATEGORY_KEY  # KEY ESTÁTICA
        )
        
        # Obtener ID de categoría seleccionada
        selected_category_id = category_options[selected_category_name]
        
        # DETECTAR CAMBIO DE CATEGORÍA Y LIMPIAR ESTADOS PREVIOS
        previous_category_key = "hype_chart_previous_category"
        if previous_category_key not in st.session_state:
            st.session_state[previous_category_key] = selected_category_id
        
        category_changed = st.session_state[previous_category_key] != selected_category_id
        if category_changed:
            st.session_state[previous_category_key] = selected_category_id
            # Limpiar cualquier estado relacionado con gráficas previas
            for key in list(st.session_state.keys()):
                if key.startswith('chart_cache_') or key.startswith('plot_data_'):
                    del st.session_state[key]
            st.info(f"📊 Categoría cambiada a: **{selected_category_name}**")
        
        # Opciones de visualización
        col1, col2 = st.columns(2)
        with col1:
            show_labels = st.checkbox("📝 Mostrar etiquetas de tecnologías", value=True, key="show_labels_hype_chart")
        with col2:
            show_confidence = st.checkbox("🎯 Mostrar niveles de confianza", value=False, key="show_confidence_hype_chart")
        
        # Obtener tecnologías de la categoría seleccionada
        queries = self.storage.get_queries_by_category(selected_category_id)
        active_queries = [q for q in queries if q.get("is_active", True)]
        
        if not active_queries:
            st.warning(f"No hay tecnologías activas en la categoría '{selected_category_name}'")
            
            # Información de debug mejorada
            with st.expander("🔍 Información de Debug"):
                st.write(f"- **Categoría seleccionada:** {selected_category_name} (ID: {selected_category_id})")
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
        
        # GENERAR Y MOSTRAR GRÁFICA
        try:
            # Usar cache simple basado en la categoría y número de queries
            cache_key = f"chart_cache_{selected_category_id}_{len(active_queries)}"
            
            # Mostrar progreso
            with st.spinner(f"📊 Generando gráfica para {len(active_queries)} tecnologías de '{selected_category_name}'..."):
                fig = self._create_hype_cycle_chart(
                    active_queries, 
                    selected_category_name,
                    show_labels=show_labels,
                    show_confidence=show_confidence
                )
            
            if fig and len(fig.data) > 0:
                # Mostrar la gráfica
                st.plotly_chart(fig, use_container_width=True, key=f"hype_chart_plot_{selected_category_id}")
                
                # Mostrar leyenda de la gráfica
                self._show_chart_legend(active_queries)
                
                # Limpiar estados de preselección después de mostrar exitosamente
                if 'selected_category_for_chart' in st.session_state:
                    del st.session_state['selected_category_for_chart']
                if 'chart_category_id' in st.session_state:
                    del st.session_state['chart_category_id']
                
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
                        show_labels: bool = True, show_confidence: bool = False,
                        chart_key: str = None) -> go.Figure:
        """
        Crea la gráfica del Hype Cycle estilo Gartner clásico - VERSIÓN CORREGIDA
        
        Args:
            queries: Lista de consultas/tecnologías
            category_name: Nombre de la categoría
            show_labels: Si mostrar etiquetas
            show_confidence: Si mostrar niveles de confianza
            chart_key: Clave única para evitar problemas de caché
        """
        # VALIDACIÓN: Asegurar que tenemos datos válidos
        if not queries:
            st.warning("No hay consultas para procesar en la gráfica")
            return go.Figure()
        
        # DEBUG: Mostrar información de las consultas que se están procesando
        st.write(f"**DEBUG:** Procesando {len(queries)} consultas para '{category_name}'")
        for i, q in enumerate(queries[:3]):  # Mostrar solo las primeras 3 para debug
            tech_name = q.get("technology_name", q.get("name", "Sin nombre"))
            phase = q.get("hype_metrics", {}).get("phase", "Unknown")
            st.write(f"  • {tech_name}: {phase}")
        
        # Crear figura
        fig = go.Figure()
        
        # CURVA MEJORADA Y SUAVE ESTILO GARTNER
        x_curve = np.linspace(0, 100, 500)
        
        # Crear curva suave usando funciones gaussianas superpuestas
        peak1 = 70 * np.exp(-((x_curve - 20)/12)**2)
        trough = -25 * np.exp(-((x_curve - 50)/15)**2)
        plateau = 25 * (1 / (1 + np.exp(-(x_curve - 80)/8)))
        baseline = 20
        
        y_curve = baseline + peak1 + trough + plateau
        
        # Suavizar la curva
        try:
            from scipy.ndimage import gaussian_filter1d
            y_curve = gaussian_filter1d(y_curve, sigma=1.5)
        except:
            window = 5
            y_smooth = np.convolve(y_curve, np.ones(window)/window, mode='same')
            y_curve = y_smooth
        
        y_curve = np.clip(y_curve, 5, 85)
        
        # AÑADIR CURVA PRINCIPAL
        fig.add_trace(go.Scatter(
            x=x_curve, 
            y=y_curve,
            mode='lines',
            name='Hype Cycle',
            line=dict(
                color='#1f77b4',
                width=5,
                shape='spline',
                smoothing=1.3
            ),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Función mejorada para obtener Y en la curva
        def get_y_on_curve(x_pos):
            if x_pos < 0:
                return y_curve[0]
            elif x_pos > 100:
                return y_curve[-1]
            else:
                idx = int(x_pos * (len(x_curve) - 1) / 100)
                idx = min(max(idx, 0), len(y_curve) - 1)
                return float(y_curve[idx])
        
        # Procesar tecnologías
        technologies = []
        colors_palette = [
            '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6',
            '#1ABC9C', '#E67E22', '#34495E', '#E91E63', '#00BCD4'
        ]
        
        # POSICIONES X MEJORADAS PARA EVITAR SUPERPOSICIÓN
        phase_x_positions = {
            "Innovation Trigger": [8, 12],
            "Peak of Inflated Expectations": [18, 24],
            "Trough of Disillusionment": [42, 48, 54],
            "Slope of Enlightenment": [65, 72, 78],
            "Plateau of Productivity": [85, 91],
            "Unknown": [50]
        }
        
        phase_counters = {phase: 0 for phase in phase_x_positions.keys()}
        
        for i, query in enumerate(queries):
            hype_metrics = query.get("hype_metrics", {})
            phase = hype_metrics.get("phase", "Unknown")
            confidence = float(hype_metrics.get("confidence", 0.5))
            total_mentions = int(hype_metrics.get("total_mentions", 0))
            
            # Obtener posición X
            available_positions = phase_x_positions.get(phase, [50])
            counter = phase_counters[phase]
            
            if counter < len(available_positions):
                pos_x = available_positions[counter]
            else:
                base_x = available_positions[counter % len(available_positions)]
                offset = (counter // len(available_positions)) * 6
                pos_x = base_x + offset
            
            phase_counters[phase] += 1
            
            # Obtener Y de la curva
            pos_y = get_y_on_curve(pos_x)
            pos_y += np.random.uniform(-1, 3)  # Variación mínima
            pos_x += np.random.uniform(-0.5, 0.5)
            
            # Extraer nombre de tecnología
            tech_name = (
                query.get("technology_name") or 
                query.get("name") or 
                query.get("search_query", "")[:20]
            )
            
            time_to_plateau = hype_metrics.get("time_to_plateau", "N/A")
            
            technologies.append({
                "name": tech_name,
                "phase": phase,
                "confidence": confidence,
                "position_x": pos_x,
                "position_y": pos_y,
                "query_id": query.get("query_id", ""),
                "time_to_plateau": time_to_plateau,
                "total_mentions": total_mentions,
                "sentiment_avg": float(hype_metrics.get("sentiment_avg", 0)),
                "color": colors_palette[i % len(colors_palette)]
            })
        
        # AÑADIR TECNOLOGÍAS CON ETIQUETAS MEJORADAS
        for i, tech in enumerate(technologies):
            base_size = 12
            confidence_factor = tech["confidence"] * 6
            mentions_factor = min(tech["total_mentions"] / 200, 4)
            size = base_size + confidence_factor + mentions_factor
            
            color = self._get_classic_color_for_time_to_plateau(tech["time_to_plateau"])
            
            hover_text = f"""
                <b style="font-size:14px">{tech['name']}</b><br>
                <b>Fase:</b> {tech['phase']}<br>
                <b>Confianza:</b> {tech['confidence']:.1%}<br>
                <b>Tiempo al Plateau:</b> {tech['time_to_plateau']}<br>
                <b>Menciones:</b> {tech['total_mentions']:,}<br>
                <b>Sentimiento:</b> {tech['sentiment_avg']:+.2f}
            """
            
            # Punto principal
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
                    opacity=0.9
                ),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False
            ))
            
            # ETIQUETAS CON MEJOR CONTRASTE Y POSICIONAMIENTO
            if show_labels:
                # Determinar posición de etiqueta para evitar superposición
                label_y = tech["position_y"] + 8
                
                # Ajustar posición Y si está muy arriba
                if label_y > 80:
                    label_y = tech["position_y"] - 8
                    arrow_direction = 15
                else:
                    arrow_direction = -15
                
                # FONDO CON MEJOR CONTRASTE
                fig.add_annotation(
                    x=tech["position_x"],
                    y=label_y,
                    text=f'<b>{tech["name"]}</b>',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    font=dict(
                        size=10, 
                        color='white',  # TEXTO BLANCO
                        family="Arial Black"
                    ),
                    bgcolor=color,  # FONDO DEL MISMO COLOR QUE EL PUNTO
                    bordercolor='white',
                    borderwidth=2,
                    borderpad=5,
                    ax=0,
                    ay=arrow_direction,
                    opacity=0.9
                )
        
        # ETIQUETAS DE FASES SIN SUPERPOSICIÓN
        phase_labels = [
            {"name": "Innovation<br>Trigger", "x": 10, "y": -8},
            {"name": "Peak of Inflated<br>Expectations", "x": 21, "y": -8},
            {"name": "Trough of<br>Disillusionment", "x": 48, "y": -8},
            {"name": "Slope of<br>Enlightenment", "x": 72, "y": -8},
            {"name": "Plateau of<br>Productivity", "x": 88, "y": -8}
        ]
        
        for label in phase_labels:
            fig.add_annotation(
                x=label["x"], 
                y=label["y"],
                text=f"<b>{label['name']}</b>",
                showarrow=False,
                font=dict(
                    size=10, 
                    color='white',  # TEXTO BLANCO
                    family="Arial"
                ),
                bgcolor='#34495E',  # FONDO GRIS OSCURO
                bordercolor='white',
                borderwidth=2,
                borderpad=6,
                xanchor='center',
                yanchor='top',
                opacity=0.9
            )
        
        # Líneas divisorias más sutiles
        division_lines = [16, 30, 60, 82]
        for x_pos in division_lines:
            fig.add_vline(
                x=x_pos, 
                line=dict(color="rgba(189, 195, 199, 0.4)", width=1, dash="dot"),
                layer="below"
            )
        
        # LEYENDA MEJORADA CON MEJOR CONTRASTE
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
                marker=dict(
                    size=12, 
                    color=item["color"],
                    symbol='circle',
                    line=dict(color='white', width=2)
                ),
                name=item["label"],
                showlegend=True
            ))
        
        # LAYOUT OPTIMIZADO
        fig.update_layout(
            title=dict(
                text=f"<b>Hype Cycle - {category_name}</b>",
                x=0.5,
                font=dict(size=20, color='#2C3E50', family="Arial")
            ),
            xaxis=dict(
                title=dict(
                    text="<b>TIME</b>",
                    font=dict(size=14, color='#2C3E50')
                ),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[0, 100],
                showline=True,
                linecolor='#BDC3C7',
                linewidth=2
            ),
            yaxis=dict(
                title=dict(
                    text="<b>EXPECTATIONS</b>",
                    font=dict(size=14, color='#2C3E50')
                ),
                showgrid=False,
                showticklabels=False,
                zeroline=False,
                range=[-15, 90],  # RANGO AMPLIADO PARA LAS ETIQUETAS
                showline=True,
                linecolor='#BDC3C7',
                linewidth=2
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=650,
            showlegend=True,
            font=dict(family="Arial, sans-serif"),
            margin=dict(t=80, l=80, r=160, b=100),
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
                bgcolor='rgba(248,249,250,0.95)',  # FONDO MÁS CONTRASTANTE
                bordercolor='#34495E',
                borderwidth=2,
                font=dict(size=11, color="#2C3E50"),
                itemsizing="constant"
            )
        )
        if chart_key:
            fig.update_layout(
                annotations=[
                    dict(
                        text=f"Chart Key: {chart_key}",
                        x=0.99, y=0.01,
                        xref="paper", yref="paper",
                        showarrow=False,
                        font=dict(size=8, color="lightgray"),
                        visible=False  # Oculto pero presente para forzar regeneración
                    )
                ]
            )

        return fig

    def _get_classic_color_for_time_to_plateau(self, time_estimate: str) -> str:
        """Colores clásicos para tiempo al plateau"""
        time_colors = {
            "already": "#27AE60",      # Verde
            "<2": "#3498DB",           # Azul
            "2-5": "#F39C12",          # Naranja
            "5-10": "#E67E22",         # Naranja oscuro
            ">10": "#E74C3C",          # Rojo
            "unknown": "#95A5A6"       # Gris
        }
        
        time_lower = time_estimate.lower()
        
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

    def _add_classic_time_legend(self, fig: go.Figure):
        """Leyenda clásica y limpia"""
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
        
        fig.update_layout(
            legend=dict(
                title="Tiempo al Plateau",
                orientation="v",
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=1.02,
                bgcolor='white',
                bordercolor='#BDC3C7',
                borderwidth=1
            )
        )

    def _get_enhanced_color_for_time_to_plateau(self, time_estimate: str) -> str:
        """
        Retorna color mejorado basado en tiempo estimado al plateau
        """
        # Paleta de colores más moderna y profesional
        time_colors = {
            "already": "#27AE60",      # Verde éxito
            "<2": "#3498DB",           # Azul claro
            "2-5": "#9B59B6",          # Púrpura
            "5-10": "#E67E22",         # Naranja
            ">10": "#E74C3C",          # Rojo
            "unknown": "#95A5A6"       # Gris
        }
        
        time_lower = time_estimate.lower()
        
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

    def _add_enhanced_time_legend(self, fig: go.Figure):
        """Añade leyenda de colores mejorada para tiempo al plateau"""
        legend_items = [
            {"label": "Ya alcanzado", "color": "#27AE60", "icon": "●"},
            {"label": "< 2 años", "color": "#3498DB", "icon": "●"},
            {"label": "2-5 años", "color": "#9B59B6", "icon": "●"},
            {"label": "5-10 años", "color": "#E67E22", "icon": "●"},
            {"label": "> 10 años", "color": "#E74C3C", "icon": "●"}
        ]
        
        # Añadir puntos invisibles para la leyenda con mejor diseño
        for i, item in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=12, 
                    color=item["color"],
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                name=f"{item['icon']} {item['label']}",
                showlegend=True
            ))
        
        # Configurar leyenda mejorada
        fig.update_layout(
            legend=dict(
                title=dict(
                    text="<b>Tiempo al Plateau</b>",
                    font=dict(size=12, color="#2E3440")
                ),
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='rgba(176, 196, 222, 0.8)',
                borderwidth=1,
                font=dict(size=10, color="#2E3440"),
                itemsizing="constant"
            )
        )

    def _add_enhanced_time_legend(self, fig: go.Figure):
        """Añade leyenda de colores mejorada para tiempo al plateau"""
        legend_items = [
            {"label": "Ya alcanzado", "color": "#27AE60", "icon": "●"},
            {"label": "< 2 años", "color": "#3498DB", "icon": "●"},
            {"label": "2-5 años", "color": "#9B59B6", "icon": "●"},
            {"label": "5-10 años", "color": "#E67E22", "icon": "●"},
            {"label": "> 10 años", "color": "#E74C3C", "icon": "●"}
        ]
        
        # Añadir puntos invisibles para la leyenda con mejor diseño
        for i, item in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(
                    size=12, 
                    color=item["color"],
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                name=f"{item['icon']} {item['label']}",
                showlegend=True
            ))
        
        # Configurar leyenda mejorada
        fig.update_layout(
            legend=dict(
                title=dict(
                    text="<b>Tiempo al Plateau</b>",
                    font=dict(size=12, color="#2E3440")
                ),
                orientation="v",
                yanchor="top",
                y=0.98,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.95)',
                bordercolor='rgba(176, 196, 222, 0.8)',
                borderwidth=1,
                font=dict(size=10, color="#2E3440"),
                itemsizing="constant"
            )
        )
    
    def _add_time_legend(self, fig: go.Figure):
        """Añade leyenda de colores para tiempo al plateau"""
        legend_items = [
            {"label": "< 2 años", "color": "#E3F2FD"},
            {"label": "2-5 años", "color": "#2196F3"},
            {"label": "5-10 años", "color": "#1976D2"},
            {"label": "> 10 años", "color": "#0D47A1"},
            {"label": "Ya alcanzado", "color": "#4CAF50"}
        ]
        
        # Añadir puntos invisibles para la leyenda
        for i, item in enumerate(legend_items):
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=item["color"]),
                name=item["label"],
                showlegend=True
            ))
        
        # Configurar leyenda
        fig.update_layout(
            legend=dict(
                title="Tiempo al Plateau:",
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#bdc3c7',
                borderwidth=1
            )
        )
    
    def _show_chart_legend(self, queries: List[Dict]):
        """Muestra tabla explicativa de la gráfica"""
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
            
            legend_data.append({
                "🔬 Tecnología": tech_name,
                "📍 Fase Actual": hype_metrics.get("phase", "Unknown"),
                "🎯 Confianza": f"{hype_metrics.get('confidence', 0):.2f}",
                "⏱️ Tiempo al Plateau": hype_metrics.get("time_to_plateau", "N/A"),
                "📊 Menciones": hype_metrics.get("total_mentions", 0),
                "💭 Sentimiento": f"{hype_metrics.get('sentiment_avg', 0):.2f}"
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
        
        # Estadísticas de la gráfica
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_tech = len(legend_data)
            st.metric("Total Tecnologías", total_tech)
        
        with col2:
            avg_confidence = sum(float(item["🎯 Confianza"]) for item in legend_data) / len(legend_data)
            st.metric("Confianza Promedio", f"{avg_confidence:.2f}")
        
        with col3:
            total_mentions = sum(item["📊 Menciones"] for item in legend_data)
            st.metric("Total Menciones", total_mentions)
        
        with col4:
            avg_sentiment = sum(float(item["💭 Sentimiento"]) for item in legend_data) / len(legend_data)
            st.metric("Sentimiento Promedio", f"{avg_sentiment:.2f}")
    
    def _show_advanced_management(self):
        """Gestión avanzada de categorías y tecnologías"""
        st.subheader("⚙️ Gestión Avanzada")
        
        st.write("""
        Herramientas adicionales para la gestión y mantenimiento de las categorías 
        y tecnologías del Hype Cycle.
        """)
        
        # Sección de operaciones masivas
        with st.expander("🔄 Operaciones Masivas", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("#### Actualización de Datos")
                if st.button("🔄 Recalcular Todas las Posiciones", type="secondary", key=f"recalc_positions_{self.unique_id}"):
                    with st.spinner("Recalculando posiciones..."):
                        self._recalculate_all_positions()
                
                if st.button("📊 Regenerar Estadísticas", type="secondary", key=f"regen_stats_{self.unique_id}"):
                    with st.spinner("Regenerando estadísticas..."):
                        st.info("Funcionalidad en desarrollo - Regenerar estadísticas")
            
            with col2:
                st.write("#### Limpieza de Datos")
                if st.button("🗑️ Limpiar Consultas Inactivas", type="secondary", key=f"cleanup_{self.unique_id}"):
                    self._cleanup_inactive_queries()
                
                if st.button("🔍 Detectar Duplicados", type="secondary", key=f"detect_dupes_{self.unique_id}"):
                    self._detect_duplicates()
        
        # Sección de estadísticas globales
        with st.expander("📊 Estadísticas Globales", expanded=True):
            self._show_global_statistics()
        
        # Sección de exportación
        with st.expander("📤 Exportación y Backup", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Exportar Todas las Categorías"):
                    self._export_all_categories()
            
            with col2:
                if st.button("💾 Crear Backup Completo"):
                    self._create_full_backup()
    
    def _show_global_statistics(self):
        """Muestra estadísticas globales del sistema"""
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
                confidences = [q.get("hype_metrics", {}).get("confidence", 0) for q in all_queries]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                st.metric("Confianza Promedio", f"{avg_confidence:.2f}")
            
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
                confidence = hype_metrics.get("confidence", 0.5)
                total_mentions = hype_metrics.get("total_mentions", 0)
                
                # Recalcular posición
                pos_x, pos_y = self.positioner.calculate_position(phase, confidence, total_mentions)
                
                # Actualizar en el objeto (esto es conceptual, en producción necesitarías update a BD)
                hype_metrics["hype_cycle_position_x"] = pos_x
                hype_metrics["hype_cycle_position_y"] = pos_y
                
                updated_count += 1
            
            st.success(f"✅ Recalculadas {updated_count} posiciones de tecnologías")
            
        except Exception as e:
            st.error(f"Error recalculando posiciones: {str(e)}")
    
    def _cleanup_inactive_queries(self):
        """Limpia consultas marcadas como inactivas"""
        st.info("🔄 Funcionalidad de limpieza - En desarrollo")
        # Implementar lógica de limpieza según necesidades
    
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
                for dup in duplicates[:5]:  # Mostrar solo los primeros 5
                    st.write(f"• Query: {dup['duplicate'].get('search_query', '')[:50]}...")
            else:
                st.success("✅ No se encontraron duplicados")
                
        except Exception as e:
            st.error(f"Error detectando duplicados: {str(e)}")

    def _show_advanced_management(self):
        """Gestión avanzada de categorías y tecnologías con funcionalidades completas"""
        st.subheader("⚙️ Gestión Avanzada")
        
        st.write("""
        Herramientas avanzadas para la gestión completa de categorías y tecnologías.
        **¡Cuidado!** Algunas operaciones son irreversibles.
        """)
        
        # Pestañas de gestión avanzada
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗑️ Eliminar Datos", 
            "📝 Editar Categorías", 
            "🔄 Mover Tecnologías",
            "📊 Operaciones Masivas"
        ])
        
        with tab1:
            self._show_delete_management()
        
        with tab2:
            self._show_edit_categories()
        
        with tab3:
            self._show_move_technologies()
        
        with tab4:
            self._show_mass_operations()

    def _show_delete_management(self):
        """Interfaz para eliminar categorías y tecnologías - VERSIÓN CORREGIDA"""
        st.write("### 🗑️ Eliminación de Datos")
        
        st.warning("⚠️ **ADVERTENCIA**: Las eliminaciones son permanentes e irreversibles.")
        
        # Sección 1: Eliminar tecnologías individuales
        with st.expander("🔬 Eliminar Tecnologías Individuales", expanded=True):
            
            all_queries = self.storage.get_all_hype_cycle_queries()
            
            if not all_queries:
                st.info("No hay tecnologías para eliminar.")
            else:
                # Crear lista de tecnologías con información detallada
                tech_options = {}
                for query in all_queries:
                    query_id = query.get("query_id", query.get("analysis_id"))
                    tech_name = (
                        query.get("technology_name") or 
                        query.get("name") or 
                        query.get("search_query", "")[:30]
                    )
                    category_id = query.get("category_id", "unknown")
                    
                    # Obtener nombre de categoría
                    try:
                        category = self.storage.storage.get_category_by_id(category_id)
                        category_name = category.get("name") if category else "Sin categoría"
                    except:
                        category_name = "Sin categoría"
                    
                    # Fecha y fase para mostrar
                    exec_date = query.get("execution_date", "")
                    try:
                        if exec_date:
                            date_obj = datetime.fromisoformat(exec_date.replace('Z', '+00:00'))
                            formatted_date = date_obj.strftime("%Y-%m-%d")
                        else:
                            formatted_date = "Sin fecha"
                    except:
                        formatted_date = "Fecha inválida"
                    
                    phase = query.get("hype_metrics", {}).get("phase", "Unknown")
                    
                    display_name = f"{tech_name} | {category_name} | {phase} | {formatted_date}"
                    tech_options[display_name] = {
                        "query_id": query_id,
                        "tech_name": tech_name,
                        "category_name": category_name,
                        "query": query,
                        "formatted_date": formatted_date
                    }
                
                # Filtros para tecnologías
                col1, col2 = st.columns(2)
                
                with col1:
                    # Filtro por categoría
                    categories = set(info["category_name"] for info in tech_options.values())
                    filter_category = st.selectbox(
                        "Filtrar por categoría:",
                        options=["Todas"] + sorted(list(categories)),
                        key=f"delete_filter_category_{self.unique_id}"
                    )
                
                with col2:
                    # Filtro por fase
                    phases = set(
                        info["query"].get("hype_metrics", {}).get("phase", "Unknown") 
                        for info in tech_options.values()
                    )
                    filter_phase = st.selectbox(
                        "Filtrar por fase:",
                        options=["Todas"] + sorted(list(phases)),
                        key=f"delete_filter_phase_{self.unique_id}"
                    )
                
                # Aplicar filtros
                filtered_options = tech_options.copy()
                
                if filter_category != "Todas":
                    filtered_options = {
                        name: info for name, info in filtered_options.items()
                        if info["category_name"] == filter_category
                    }
                
                if filter_phase != "Todas":
                    filtered_options = {
                        name: info for name, info in filtered_options.items()
                        if info["query"].get("hype_metrics", {}).get("phase", "Unknown") == filter_phase
                    }
                
                if not filtered_options:
                    st.info("No hay tecnologías que coincidan con los filtros seleccionados.")
                    return
                
                # Selector de tecnología a eliminar
                selected_tech = st.selectbox(
                    f"Selecciona la tecnología a eliminar ({len(filtered_options)} disponibles):",
                    options=list(filtered_options.keys()),
                    key=f"delete_tech_selector_{self.unique_id}"
                )
                
                if selected_tech:
                    tech_info = filtered_options[selected_tech]
                    
                    # Mostrar información detallada de la tecnología
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📋 Información de la tecnología:**")
                        st.write(f"• **Nombre:** {tech_info['tech_name']}")
                        st.write(f"• **Categoría:** {tech_info['category_name']}")
                        st.write(f"• **ID:** {tech_info['query_id']}")
                        st.write(f"• **Fecha:** {tech_info['formatted_date']}")
                    
                    with col2:
                        st.write("**📊 Métricas del Hype Cycle:**")
                        hype_metrics = tech_info['query'].get('hype_metrics', {})
                        
                        st.write(f"• **Fase:** {hype_metrics.get('phase', 'Unknown')}")
                        st.write(f"• **Confianza:** {hype_metrics.get('confidence', 0):.2f}")
                        st.write(f"• **Menciones:** {hype_metrics.get('total_mentions', 0)}")
                        
                        # Tiempo al plateau
                        time_to_plateau = hype_metrics.get('time_to_plateau', 'N/A')
                        st.write(f"• **Tiempo al Plateau:** {time_to_plateau}")
                    
                    # ZONA DE PELIGRO - Confirmación de eliminación
                    st.write("---")
                    st.error("🚨 **ZONA DE PELIGRO**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Doble confirmación
                        confirm_delete = st.checkbox(
                            f"Confirmar eliminación de '{tech_info['tech_name']}'",
                            key=f"confirm_delete_tech_{self.unique_id}"
                        )
                        
                        if confirm_delete:
                            final_confirm = st.checkbox(
                                "Estoy seguro de que quiero eliminar esta tecnología permanentemente",
                                key=f"final_confirm_delete_tech_{self.unique_id}"
                            )
                        else:
                            final_confirm = False
                    
                    with col2:
                        if confirm_delete and final_confirm and st.button(
                            "🗑️ ELIMINAR TECNOLOGÍA", 
                            type="secondary",
                            key=f"delete_tech_btn_{self.unique_id}"
                        ):
                            with st.spinner(f"Eliminando '{tech_info['tech_name']}'..."):
                                success = self._delete_technology(tech_info['query_id'])
                                
                                if success:
                                    st.success(f"✅ Tecnología '{tech_info['tech_name']}' eliminada correctamente")
                                    
                                    # Limpiar caché
                                    for key in list(st.session_state.keys()):
                                        if key.startswith('chart_cache_'):
                                            del st.session_state[key]
                                    
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ Error al eliminar la tecnología")
        
        # Sección 2: Eliminar categorías completas - VERSIÓN CORREGIDA
        with st.expander("📁 Eliminar Categorías Completas", expanded=False):
            
            categories = self.storage.storage.get_all_categories()
            
            # Filtrar categorías que no sean "default"
            deletable_categories = [
                category for category in categories 
                if category.get("id") != "default" and category.get("category_id") != "default"
            ]
            
            if not deletable_categories:
                st.info("No hay categorías eliminables (la categoría 'default' no se puede eliminar).")
            else:
                st.warning("⚠️ **ATENCIÓN**: Eliminar una categoría también eliminará TODAS las tecnologías asociadas.")
                
                # Mostrar información detallada de categorías
                for category in deletable_categories:
                    # CORREGIDO: usar 'category' en lugar de 'cat'
                    cat_id = category.get("id") or category.get("category_id")
                    cat_name = category.get("name", "Sin nombre")
                    
                    # Obtener información completa de la categoría
                    cat_info = self._get_category_info(cat_id)
                    stats = cat_info.get("statistics", {})
                    tech_count = stats.get("total_technologies", 0)
                    
                    with st.container():
                        st.write(f"### 📁 {cat_name}")
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"• **Tecnologías:** {tech_count}")
                            st.write(f"• **ID:** {cat_id}")
                            st.write(f"• **Activas:** {stats.get('active_technologies', 0)}")
                            
                            if tech_count > 0:
                                st.error(f"⚠️ **PELIGRO**: Eliminar esta categoría también eliminará {tech_count} tecnologías")
                                
                                # Mostrar distribución de fases
                                phase_dist = stats.get("phase_distribution", {})
                                if phase_dist:
                                    st.write("**Tecnologías por fase:**")
                                    for phase, count in phase_dist.items():
                                        st.write(f"  • {phase}: {count}")
                        
                        with col2:
                            # Confirmaciones múltiples para categorías
                            confirm_cat = st.checkbox(
                                "Entiendo los riesgos",
                                key=f"confirm_delete_cat_risk_{cat_id}_{self.unique_id}"
                            )
                            
                            if confirm_cat and tech_count > 0:
                                confirm_tech_loss = st.checkbox(
                                    f"Acepto perder {tech_count} tecnologías",
                                    key=f"confirm_tech_loss_{cat_id}_{self.unique_id}"
                                )
                            else:
                                confirm_tech_loss = True  # Si no hay tecnologías, no necesita confirmación adicional
                        
                        with col3:
                            if confirm_cat and confirm_tech_loss and st.button(
                                "🗑️ ELIMINAR CATEGORÍA", 
                                type="secondary",
                                key=f"delete_cat_btn_{cat_id}_{self.unique_id}"
                            ):
                                with st.spinner(f"Eliminando categoría '{cat_name}' y {tech_count} tecnologías..."):
                                    success = self._delete_category(cat_id)
                                    
                                    if success:
                                        st.success(f"✅ Categoría '{cat_name}' eliminada correctamente")
                                        
                                        # Limpiar caché
                                        for key in list(st.session_state.keys()):
                                            if key.startswith('chart_cache_'):
                                                del st.session_state[key]
                                        
                                        time.sleep(1)
                                        st.rerun()
                                    else:
                                        st.error("❌ Error al eliminar la categoría")
                        
                        st.write("---")

    def _show_edit_categories(self):
        """Interfaz para editar categorías - VERSIÓN CORREGIDA"""
        st.write("### 📝 Editar Categorías")
        
        categories = self.storage.storage.get_all_categories()
        
        if not categories:
            st.info("No hay categorías para editar.")
            return
        
        # Filtrar categorías editables
        editable_categories = [
            category for category in categories  # CORRECTO: usar 'category'
            if category.get("id") != "default" and category.get("category_id") != "default"  # CORRECTO
        ]
        
        if not editable_categories:
            st.info("Solo existe la categoría 'default' que no se puede editar.")
            return
        
        # Selector de categoría a editar
        category_options = {}
        for category in editable_categories:  # CORRECTO: usar 'category'
            cat_id = category.get("id") or category.get("category_id")  # CORRECTO: usar 'category'
            cat_name = category.get("name", "Sin nombre")  # CORRECTO: usar 'category'
            
            # Obtener estadísticas de la categoría
            cat_info = self._get_category_info(cat_id)
            tech_count = cat_info.get("statistics", {}).get("total_technologies", 0)
            
            display_name = f"{cat_name} ({tech_count} tecnologías)"
            category_options[display_name] = {
                "category": category,  # CORRECTO
                "cat_id": cat_id,
                "info": cat_info
            }
        
        selected_cat_display = st.selectbox(
            "Selecciona una categoría para editar:",
            options=list(category_options.keys()),
            key=f"edit_cat_selector_{self.unique_id}"
        )
        
        if selected_cat_display:
            cat_data = category_options[selected_cat_display]
            category = cat_data["category"]
            cat_id = cat_data["cat_id"]
            cat_info = cat_data["info"]
            
            # Mostrar información actual de la categoría
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Formulario de edición
                with st.form(f"edit_category_form_{self.unique_id}"):
                    st.write(f"**Editando categoría:** {category.get('name', 'Sin nombre')}")
                    
                    # Campos editables
                    new_name = st.text_input(
                        "Nuevo nombre:",
                        value=category.get("name", ""),
                        key=f"edit_cat_name_{self.unique_id}",
                        help="Nombre que aparecerá en las listas y gráficas"
                    )
                    
                    new_description = st.text_area(
                        "Nueva descripción:",
                        value=category.get("description", ""),
                        height=100,
                        key=f"edit_cat_desc_{self.unique_id}",
                        help="Descripción opcional para documentar el propósito de la categoría"
                    )
                    
                    # Validaciones en tiempo real
                    if new_name:
                        # Verificar que no existe otra categoría con el mismo nombre
                        existing_names = [
                            c.get("name", "") for c in categories 
                            if (c.get("id") != cat_id and c.get("category_id") != cat_id)
                        ]
                        
                        if new_name in existing_names:
                            st.error("❌ Ya existe una categoría con ese nombre")
                            name_valid = False
                        elif len(new_name.strip()) < 2:
                            st.warning("⚠️ El nombre debe tener al menos 2 caracteres")
                            name_valid = False
                        else:
                            st.success(f"✅ Nombre válido: '{new_name}'")
                            name_valid = True
                    else:
                        st.error("❌ El nombre no puede estar vacío")
                        name_valid = False
                    
                    # Mostrar preview de cambios
                    if new_name != category.get("name", "") or new_description != category.get("description", ""):
                        st.info("📝 **Preview de cambios:**")
                        if new_name != category.get("name", ""):
                            st.write(f"• **Nombre:** '{category.get('name', '')}' → '{new_name}'")
                        if new_description != category.get("description", ""):
                            st.write(f"• **Descripción:** Actualizada")
                    
                    # Botón de guardar
                    submit_changes = st.form_submit_button(
                        "💾 Guardar Cambios", 
                        disabled=not name_valid,
                        type="primary"
                    )
                    
                    if submit_changes and name_valid:
                        with st.spinner(f"Actualizando categoría '{new_name}'..."):
                            success = self._update_category(cat_id, new_name, new_description)
                            
                            if success:
                                st.success(f"✅ Categoría actualizada correctamente")
                                
                                # Limpiar cualquier caché relacionado
                                for key in list(st.session_state.keys()):
                                    if key.startswith('chart_cache_') or key.startswith('category_'):
                                        del st.session_state[key]
                                
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Error al actualizar la categoría")
            
            with col2:
                # Información y estadísticas de la categoría
                st.write("#### 📊 Información de la Categoría")
                
                stats = cat_info.get("statistics", {})
                
                # Métricas principales
                st.metric("Total Tecnologías", stats.get("total_technologies", 0))
                st.metric("Tecnologías Activas", stats.get("active_technologies", 0))
                st.metric("Confianza Promedio", f"{stats.get('average_confidence', 0):.2f}")
                
                # Información adicional
                st.write("**📋 Detalles:**")
                st.write(f"• **ID:** {cat_id}")
                
                # Fecha de creación
                created_at = category.get("created_at")
                if created_at:
                    try:
                        date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        formatted_date = date_obj.strftime("%Y-%m-%d")
                        st.write(f"• **Creada:** {formatted_date}")
                    except:
                        st.write(f"• **Creada:** {created_at}")
                
                # Distribución de fases
                phase_dist = stats.get("phase_distribution", {})
                if phase_dist:
                    st.write("**🎯 Distribución por Fases:**")
                    total_tech = sum(phase_dist.values())
                    
                    for phase, count in sorted(phase_dist.items()):
                        percentage = (count / total_tech * 100) if total_tech > 0 else 0
                        st.write(f"• {phase}: {count} ({percentage:.1f}%)")
                
                # Acciones adicionales
                st.write("---")
                st.write("**⚙️ Acciones:**")
                
                if st.button(
                    "📊 Ver Tecnologías", 
                    key=f"view_tech_in_cat_{cat_id}_{self.unique_id}"
                ):
                    self._show_technologies_in_category(cat_info)
                
                if st.button(
                    "📈 Ver en Gráfica", 
                    key=f"view_chart_from_edit_{cat_id}_{self.unique_id}"
                ):
                    # Configurar para mostrar en la gráfica
                    st.session_state['selected_category_for_chart'] = cat_id
                    st.session_state['chart_category_id'] = cat_id
                    st.session_state['chart_category_name'] = category.get("name", "Sin nombre")
                    
                    st.success(f"✅ Categoría seleccionada para visualización. Ve a la pestaña 'Gráfica Hype Cycle'.")

    def _show_technologies_in_category(self, cat_info: dict):
        """Muestra detalles de las tecnologías en una categoría"""
        st.write("### 🔬 Tecnologías en la Categoría")
        
        technologies = cat_info.get("technologies", [])
        
        if not technologies:
            st.info("No hay tecnologías en esta categoría.")
            return
        
        # Crear tabla de tecnologías
        tech_data = []
        for tech in technologies:
            hype_metrics = tech.get("hype_metrics", {})
            
            # Nombre de tecnología
            tech_name = (
                tech.get("technology_name") or 
                tech.get("name") or 
                tech.get("search_query", "")[:30]
            )
            
            # Fecha
            exec_date = tech.get("execution_date", "")
            try:
                if exec_date:
                    date_obj = datetime.fromisoformat(exec_date.replace('Z', '+00:00'))
                    formatted_date = date_obj.strftime("%Y-%m-%d")
                else:
                    formatted_date = "Sin fecha"
            except:
                formatted_date = "Fecha inválida"
            
            tech_data.append({
                "🔬 Tecnología": tech_name,
                "📍 Fase": hype_metrics.get("phase", "Unknown"),
                "🎯 Confianza": f"{hype_metrics.get('confidence', 0):.2f}",
                "📊 Menciones": hype_metrics.get("total_mentions", 0),
                "⏱️ Tiempo al Plateau": hype_metrics.get("time_to_plateau", "N/A"),
                "📅 Fecha": formatted_date,
                "✅ Activa": "Sí" if tech.get("is_active", True) else "No",
                "🆔 ID": tech.get("query_id", tech.get("analysis_id", ""))[:8]
            })
        
        # Mostrar tabla
        if tech_data:
            df = pd.DataFrame(tech_data)
            
            # Opciones de filtrado
            col1, col2, col3 = st.columns(3)
            
            with col1:
                phase_filter = st.selectbox(
                    "Filtrar por fase:",
                    options=["Todas"] + sorted(df["📍 Fase"].unique().tolist()),
                    key=f"tech_phase_filter_{self.unique_id}"
                )
            
            with col2:
                status_filter = st.selectbox(
                    "Filtrar por estado:",
                    options=["Todas", "Activas", "Inactivas"],
                    key=f"tech_status_filter_{self.unique_id}"
                )
            
            with col3:
                sort_by = st.selectbox(
                    "Ordenar por:",
                    options=["Fecha", "Confianza", "Menciones", "Nombre"],
                    key=f"tech_sort_filter_{self.unique_id}"
                )
            
            # Aplicar filtros
            filtered_df = df.copy()
            
            if phase_filter != "Todas":
                filtered_df = filtered_df[filtered_df["📍 Fase"] == phase_filter]
            
            if status_filter == "Activas":
                filtered_df = filtered_df[filtered_df["✅ Activa"] == "Sí"]
            elif status_filter == "Inactivas":
                filtered_df = filtered_df[filtered_df["✅ Activa"] == "No"]
            
            # Aplicar ordenamiento
            if sort_by == "Fecha":
                filtered_df = filtered_df.sort_values("📅 Fecha", ascending=False)
            elif sort_by == "Confianza":
                filtered_df["_conf_sort"] = filtered_df["🎯 Confianza"].astype(float)
                filtered_df = filtered_df.sort_values("_conf_sort", ascending=False)
                filtered_df = filtered_df.drop("_conf_sort", axis=1)
            elif sort_by == "Menciones":
                filtered_df = filtered_df.sort_values("📊 Menciones", ascending=False)
            elif sort_by == "Nombre":
                filtered_df = filtered_df.sort_values("🔬 Tecnología")
            
            # Mostrar tabla filtrada
            st.write(f"**Mostrando {len(filtered_df)} de {len(df)} tecnologías**")
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)
            
            # Estadísticas de la vista filtrada
            if len(filtered_df) > 0:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_conf = filtered_df["🎯 Confianza"].astype(float).mean()
                    st.metric("Confianza Promedio", f"{avg_conf:.2f}")
                
                with col2:
                    total_mentions = filtered_df["📊 Menciones"].sum()
                    st.metric("Total Menciones", total_mentions)
                
                with col3:
                    active_count = (filtered_df["✅ Activa"] == "Sí").sum()
                    st.metric("Activas", active_count)
                
                with col4:
                    # Fase más común
                    most_common_phase = filtered_df["📍 Fase"].mode().iloc[0] if not filtered_df["📍 Fase"].mode().empty else "N/A"
                    st.metric("Fase Más Común", most_common_phase)

    def _show_category_statistics(self):
        """Muestra estadísticas globales de categorías"""
        st.write("### 📊 Estadísticas Globales de Categorías")
        
        # Obtener todas las categorías y sus estadísticas
        categories = self.storage.storage.get_all_categories()
        all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not categories or not all_queries:
            st.info("No hay suficientes datos para mostrar estadísticas.")
            return
        
        # Calcular estadísticas por categoría
        category_stats = []
        
        for cat in categories:
            cat_id = cat.get("id") or cat.get("category_id")
            cat_name = cat.get("name", "Sin nombre")
            
            cat_info = self._get_category_info(cat_id)
            stats = cat_info.get("statistics", {})
            
            category_stats.append({
                "Categoría": cat_name,
                "Total Tecnologías": stats.get("total_technologies", 0),
                "Tecnologías Activas": stats.get("active_technologies", 0),
                "Confianza Promedio": stats.get("average_confidence", 0),
                "ID": cat_id
            })
        
        # Crear DataFrame para análisis
        df_stats = pd.DataFrame(category_stats)
        
        # Métricas globales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_categories = len(df_stats)
            st.metric("Total Categorías", total_categories)
        
        with col2:
            total_technologies = df_stats["Total Tecnologías"].sum()
            st.metric("Total Tecnologías", total_technologies)
        
        with col3:
            active_technologies = df_stats["Tecnologías Activas"].sum()
            st.metric("Tecnologías Activas", active_technologies)
        
        with col4:
            overall_avg_conf = df_stats["Confianza Promedio"].mean()
            st.metric("Confianza Global", f"{overall_avg_conf:.2f}")
        
        # Tabla de estadísticas por categoría
        st.write("#### 📋 Estadísticas por Categoría")
        
        # Ordenar por número de tecnologías
        df_display = df_stats.sort_values("Total Tecnologías", ascending=False)
        
        # Configurar columnas para mostrar
        column_config = {
            "Categoría": st.column_config.TextColumn("Categoría", width="medium"),
            "Total Tecnologías": st.column_config.NumberColumn("Total", width="small"),
            "Tecnologías Activas": st.column_config.NumberColumn("Activas", width="small"),
            "Confianza Promedio": st.column_config.NumberColumn("Confianza", format="%.2f", width="small"),
            "ID": st.column_config.TextColumn("ID", width="small")
        }
        
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config=column_config
        )
        
        # Gráficos de distribución
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras de tecnologías por categoría
            if not df_stats.empty:
                import plotly.express as px
                
                fig_bar = px.bar(
                    df_stats.sort_values("Total Tecnologías", ascending=True),
                    x="Total Tecnologías",
                    y="Categoría",
                    orientation='h',
                    title="Tecnologías por Categoría",
                    color="Confianza Promedio",
                    color_continuous_scale="viridis"
                )
                
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Gráfico de pastel de distribución
            if not df_stats.empty and df_stats["Total Tecnologías"].sum() > 0:
                fig_pie = px.pie(
                    df_stats[df_stats["Total Tecnologías"] > 0],
                    values="Total Tecnologías",
                    names="Categoría",
                    title="Distribución de Tecnologías"
                )
                
                fig_pie.update_layout(height=400)
                st.plotly_chart(fig_pie, use_container_width=True)

    def _show_move_technologies(self):
        """Interfaz para mover tecnologías entre categorías - VERSIÓN IMPLEMENTADA"""
        st.write("### 🔄 Mover Tecnologías Entre Categorías")
        
        all_queries = self.storage.get_all_hype_cycle_queries()
        categories = self.storage.storage.get_all_categories()
        
        if not all_queries or not categories:
            st.info("No hay suficientes datos para mover tecnologías.")
            return
        
        # Preparar opciones de tecnologías
        tech_options = {}
        for query in all_queries:
            query_id = query.get("query_id", query.get("analysis_id"))
            tech_name = (
                query.get("technology_name") or 
                query.get("name") or 
                query.get("search_query", "")[:30]
            )
            current_cat_id = query.get("category_id", "unknown")
            
            # Obtener nombre de categoría actual
            current_cat_name = "Sin categoría"
            for cat in categories:
                cat_id = cat.get("id") or cat.get("category_id")
                if cat_id == current_cat_id:
                    current_cat_name = cat.get("name", "Sin nombre")
                    break
            
            display_name = f"{tech_name} (Actual: {current_cat_name})"
            tech_options[display_name] = {
                "query_id": query_id,
                "tech_name": tech_name,
                "current_category_id": current_cat_id,
                "current_category_name": current_cat_name,
                "query": query
            }
        
        # Preparar opciones de categorías
        category_options = {}
        for cat in categories:
            cat_id = cat.get("id") or cat.get("category_id")
            cat_name = cat.get("name", "Sin nombre")
            category_options[cat_name] = cat_id
        
        # INTERFAZ MEJORADA DE MOVIMIENTO
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("#### 📋 Seleccionar Tecnología")
            
            # Filtro por categoría actual
            filter_category = st.selectbox(
                "Filtrar por categoría actual:",
                options=["Todas"] + list(category_options.keys()),
                key=f"move_filter_category_{self.unique_id}"
            )
            
            # Filtrar tecnologías según categoría seleccionada
            if filter_category != "Todas":
                filter_cat_id = category_options[filter_category]
                filtered_tech_options = {
                    name: info for name, info in tech_options.items()
                    if info["current_category_id"] == filter_cat_id
                }
            else:
                filtered_tech_options = tech_options
            
            if not filtered_tech_options:
                st.info("No hay tecnologías en la categoría seleccionada.")
                return
            
            selected_tech = st.selectbox(
                "Tecnología a mover:",
                options=list(filtered_tech_options.keys()),
                key=f"move_tech_selector_{self.unique_id}"
            )
            
            tech_info = filtered_tech_options[selected_tech]
            
            # Mostrar información de la tecnología seleccionada
            with st.expander("ℹ️ Información de la Tecnología", expanded=True):
                query_details = tech_info["query"]
                
                st.write(f"**Nombre:** {tech_info['tech_name']}")
                st.write(f"**Categoría actual:** {tech_info['current_category_name']}")
                st.write(f"**ID:** {tech_info['query_id']}")
                
                # Métricas del Hype Cycle
                hype_metrics = query_details.get("hype_metrics", {})
                if hype_metrics:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Fase", hype_metrics.get("phase", "Unknown"))
                        st.metric("Confianza", f"{hype_metrics.get('confidence', 0):.2f}")
                    with col_b:
                        st.metric("Menciones", hype_metrics.get("total_mentions", 0))
                        
                        # Fecha de análisis
                        try:
                            exec_date = query_details.get("execution_date", "")
                            if exec_date:
                                date_obj = datetime.fromisoformat(exec_date.replace('Z', '+00:00'))
                                formatted_date = date_obj.strftime("%Y-%m-%d")
                                st.write(f"**Fecha:** {formatted_date}")
                        except:
                            st.write("**Fecha:** No disponible")
        
        with col2:
            st.write("#### 🎯 Categoría Destino")
            
            # Excluir la categoría actual de las opciones
            current_cat_id = tech_info["current_category_id"]
            available_categories = {
                name: cat_id for name, cat_id in category_options.items()
                if cat_id != current_cat_id
            }
            
            if not available_categories:
                st.warning("No hay otras categorías disponibles para mover la tecnología.")
                return
            
            target_category = st.selectbox(
                "Mover a categoría:",
                options=list(available_categories.keys()),
                key=f"move_target_cat_{self.unique_id}"
            )
            
            target_cat_id = available_categories[target_category]
            
            # Mostrar información de la categoría destino
            target_cat_info = self._get_category_info(target_cat_id)
            
            if target_cat_info:
                with st.expander("ℹ️ Información de Categoría Destino", expanded=True):
                    stats = target_cat_info.get("statistics", {})
                    
                    st.write(f"**Nombre:** {target_category}")
                    st.write(f"**Tecnologías actuales:** {stats.get('total_technologies', 0)}")
                    st.write(f"**Tecnologías activas:** {stats.get('active_technologies', 0)}")
                    
                    # Distribución de fases en categoría destino
                    phase_dist = stats.get("phase_distribution", {})
                    if phase_dist:
                        st.write("**Distribución por fases:**")
                        for phase, count in phase_dist.items():
                            st.write(f"• {phase}: {count}")
            
            # BOTONES DE ACCIÓN
            st.write("---")
            
            # Confirmación visual del movimiento
            st.info(f"**Movimiento:** '{tech_info['tech_name']}' de '{tech_info['current_category_name']}' → '{target_category}'")
            
            # Checkbox de confirmación
            confirm_move = st.checkbox(
                f"Confirmar movimiento de tecnología",
                key=f"confirm_move_{self.unique_id}"
            )
            
            # Botón de ejecutar movimiento
            col_a, col_b = st.columns(2)
            
            with col_a:
                if confirm_move and st.button(
                    "🔄 MOVER TECNOLOGÍA", 
                    type="primary",
                    key=f"execute_move_{self.unique_id}"
                ):
                    with st.spinner(f"Moviendo '{tech_info['tech_name']}'..."):
                        success = self._move_technology(tech_info["query_id"], target_cat_id)
                        
                        if success:
                            st.success(f"✅ '{tech_info['tech_name']}' movida exitosamente a '{target_category}'")
                            
                            # Limpiar caché de gráficas para que se actualicen
                            for key in list(st.session_state.keys()):
                                if key.startswith('chart_cache_'):
                                    del st.session_state[key]
                            
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Error al mover la tecnología")
            
            with col_b:
                if st.button(
                    "📊 Preview Destino", 
                    key=f"preview_move_{self.unique_id}"
                ):
                    # Mostrar preview de cómo quedaría la categoría destino
                    self._show_move_preview(tech_info, target_cat_info)

    def _show_move_preview(self, tech_info: dict, target_cat_info: dict):
        """Muestra preview de cómo quedaría la categoría después del movimiento"""
        st.write("### 👀 Preview del Movimiento")
        
        # Información actual
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 📊 Estado Actual")
            current_stats = target_cat_info.get("statistics", {})
            
            st.metric("Tecnologías", current_stats.get("total_technologies", 0))
            st.metric("Promedio Confianza", f"{current_stats.get('average_confidence', 0):.2f}")
        
        with col2:
            st.write("#### 📈 Después del Movimiento")
            
            # Calcular nuevas estadísticas
            new_total = current_stats.get("total_technologies", 0) + 1
            
            # Calcular nuevo promedio de confianza
            current_avg = current_stats.get("average_confidence", 0)
            current_total = current_stats.get("total_technologies", 0)
            tech_confidence = tech_info["query"].get("hype_metrics", {}).get("confidence", 0)
            
            if current_total > 0:
                new_avg = ((current_avg * current_total) + tech_confidence) / new_total
            else:
                new_avg = tech_confidence
            
            st.metric("Tecnologías", new_total, delta=1)
            st.metric("Promedio Confianza", f"{new_avg:.2f}", delta=f"{new_avg - current_avg:+.2f}")
        
        # Distribución de fases actualizada
        st.write("#### 📊 Nueva Distribución por Fases")
        
        phase_dist = current_stats.get("phase_distribution", {}).copy()
        tech_phase = tech_info["query"].get("hype_metrics", {}).get("phase", "Unknown")
        phase_dist[tech_phase] = phase_dist.get(tech_phase, 0) + 1
        
        # Crear gráfico de distribución
        if phase_dist:
            import plotly.express as px
            
            phases = list(phase_dist.keys())
            counts = list(phase_dist.values())
            
            fig = px.pie(
                values=counts,
                names=phases,
                title="Distribución de Fases Después del Movimiento"
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def _show_mass_operations(self):
        """Operaciones masivas"""
        st.write("### 📊 Operaciones Masivas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### 🔄 Actualización Masiva")
            
            if st.button("🔄 Recalcular Todas las Posiciones", type="secondary", key=f"recalc_positions_{self.unique_id}"):
                with st.spinner("Recalculando posiciones..."):
                    result = self._recalculate_all_positions()
                    if result:
                        st.success(f"✅ {result} posiciones recalculadas")
                    else:
                        st.error("❌ Error en el recálculo")
            
            if st.button("📊 Regenerar Estadísticas", type="secondary", key=f"regen_stats_{self.unique_id}"):
                st.info("🔄 Funcionalidad en desarrollo")
        
        with col2:
            st.write("#### 🗑️ Limpieza Masiva")
            
            if st.button("🗑️ Limpiar Consultas Inactivas", type="secondary", key=f"cleanup_{self.unique_id}"):
                result = self._cleanup_inactive_queries()
                if result > 0:
                    st.success(f"✅ {result} consultas inactivas eliminadas")
                else:
                    st.info("No hay consultas inactivas para limpiar")
            
            if st.button("🔍 Detectar y Eliminar Duplicados", type="secondary", key=f"detect_dupes_{self.unique_id}"):
                result = self._detect_and_remove_duplicates()
                if result > 0:
                    st.success(f"✅ {result} duplicados eliminados")
                else:
                    st.info("No se encontraron duplicados")

    # MÉTODOS AUXILIARES PARA LAS OPERACIONES

    def _delete_technology(self, query_id: str) -> bool:
        """Elimina una tecnología específica"""
        try:
            return self.storage.delete_query(query_id)
        except Exception as e:
            st.error(f"Error eliminando tecnología: {str(e)}")
            return False

    def _delete_category(self, category_id: str) -> bool:
        """Elimina una categoría y todas sus tecnologías"""
        try:
            # Primero eliminar todas las tecnologías de la categoría
            cat_queries = self.storage.get_queries_by_category(category_id)
            
            for query in cat_queries:
                query_id = query.get("query_id", query.get("analysis_id"))
                if query_id:
                    self.storage.delete_query(query_id)
            
            # Luego eliminar la categoría (esto depende de tu implementación)
            # Por ahora, simplemente marcamos como eliminada
            st.info("Categoría marcada para eliminación (funcionalidad completa en desarrollo)")
            return True
            
        except Exception as e:
            st.error(f"Error eliminando categoría: {str(e)}")
            return False

    def _update_category(self, category_id: str, new_name: str, new_description: str) -> bool:
        """Actualiza los datos de una categoría"""
        try:
            # Esta funcionalidad depende de tu implementación específica de storage
            st.info("Funcionalidad de edición en desarrollo")
            return True
        except Exception as e:
            st.error(f"Error actualizando categoría: {str(e)}")
            return False

    def _move_technology(self, query_id: str, target_category_id: str) -> bool:
        """Mueve una tecnología a otra categoría"""
        try:
            # Esta funcionalidad requiere actualizar el campo category_id en la BD
            st.info("Funcionalidad de movimiento en desarrollo")
            return True
        except Exception as e:
            st.error(f"Error moviendo tecnología: {str(e)}")
            return False

    def _cleanup_inactive_queries(self) -> int:
        """Elimina consultas marcadas como inactivas"""
        try:
            all_queries = self.storage.get_all_hype_cycle_queries()
            inactive_count = 0
            
            for query in all_queries:
                if not query.get("is_active", True):
                    query_id = query.get("query_id", query.get("analysis_id"))
                    if query_id and self.storage.delete_query(query_id):
                        inactive_count += 1
            
            return inactive_count
        except Exception as e:
            st.error(f"Error en limpieza: {str(e)}")
            return 0

    def _detect_and_remove_duplicates(self) -> int:
        """Detecta y elimina consultas duplicadas"""
        try:
            all_queries = self.storage.get_all_hype_cycle_queries()
            seen_queries = {}
            duplicates_removed = 0
            
            for query in all_queries:
                search_query = query.get("search_query", "").lower().strip()
                category_id = query.get("category_id", "")
                key = f"{search_query}_{category_id}"
                
                if key in seen_queries:
                    # Es un duplicado, eliminar el más antiguo
                    query_id = query.get("query_id", query.get("analysis_id"))
                    if query_id and self.storage.delete_query(query_id):
                        duplicates_removed += 1
                else:
                    seen_queries[key] = query
            
            return duplicates_removed
        except Exception as e:
            st.error(f"Error detectando duplicados: {str(e)}")
            return 0
    
    def _export_all_categories(self):
        """Exporta datos de todas las categorías"""
        st.info("📤 Funcionalidad de exportación completa - En desarrollo")
    
    def _create_full_backup(self):
        """Crea un backup completo del sistema"""
        st.info("💾 Funcionalidad de backup - En desarrollo")
    
    def _move_technology(self, query_id: str, target_category_id: str) -> bool:
        """
        Mueve una tecnología a otra categoría - IMPLEMENTACIÓN COMPLETA
        
        Args:
            query_id: ID de la consulta/tecnología a mover
            target_category_id: ID de la categoría destino
            
        Returns:
            bool: True si se movió exitosamente, False en caso contrario
        """
        try:
            # 1. Obtener la tecnología actual
            current_query = self.storage.get_query_by_id(query_id)
            
            if not current_query:
                st.error(f"❌ No se encontró la tecnología con ID: {query_id}")
                return False
            
            current_category_id = current_query.get("category_id", "default")
            
            # 2. Verificar que la categoría destino existe
            target_category = self.storage.storage.get_category_by_id(target_category_id)
            if not target_category:
                st.error(f"❌ La categoría destino no existe: {target_category_id}")
                return False
            
            # 3. Verificar que no es la misma categoría
            if current_category_id == target_category_id:
                st.warning("⚠️ La tecnología ya está en esa categoría.")
                return False
            
            # 4. Actualizar la tecnología según el tipo de storage
            if hasattr(self.storage.storage, 'analyses_table'):
                # DYNAMODB - Actualizar item
                return self._move_technology_dynamodb(current_query, target_category_id)
            else:
                # LOCAL STORAGE - Actualizar archivo
                return self._move_technology_local(current_query, target_category_id)
                
        except Exception as e:
            st.error(f"❌ Error moviendo tecnología: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False

    def _move_technology_dynamodb(self, query: dict, target_category_id: str) -> bool:
        """Mueve tecnología en DynamoDB"""
        try:
            # Obtener claves primarias
            analysis_id = query.get("analysis_id") or query.get("query_id")
            timestamp = query.get("timestamp") or query.get("execution_date")
            
            if not analysis_id or not timestamp:
                st.error("❌ No se pueden obtener las claves primarias para DynamoDB")
                return False
            
            # Actualizar el item en DynamoDB
            response = self.storage.storage.analyses_table.update_item(
                Key={
                    'analysis_id': analysis_id,
                    'timestamp': timestamp
                },
                UpdateExpression='SET category_id = :cat_id, last_updated = :updated',
                ExpressionAttributeValues={
                    ':cat_id': target_category_id,
                    ':updated': datetime.now().isoformat()
                },
                ReturnValues='UPDATED_NEW'
            )
            
            # Verificar que la actualización fue exitosa
            if response.get('Attributes'):
                st.success(f"✅ Tecnología movida exitosamente en DynamoDB")
                return True
            else:
                st.error("❌ No se pudo confirmar la actualización en DynamoDB")
                return False
                
        except Exception as e:
            st.error(f"❌ Error en DynamoDB: {str(e)}")
            return False

    def _move_technology_local(self, query: dict, target_category_id: str) -> bool:
        """Mueve tecnología en almacenamiento local"""
        try:
            query_id = query.get("query_id") or query.get("analysis_id")
            
            # Buscar y actualizar en hype_cycle_queries
            hype_queries = self.storage.storage.data.get("hype_cycle_queries", [])
            updated = False
            
            for i, stored_query in enumerate(hype_queries):
                stored_id = stored_query.get("query_id") or stored_query.get("analysis_id")
                if stored_id == query_id:
                    # Actualizar categoría y timestamp
                    hype_queries[i]["category_id"] = target_category_id
                    hype_queries[i]["last_updated"] = datetime.now().isoformat()
                    updated = True
                    break
            
            # También buscar en searches generales (por compatibilidad)
            searches = self.storage.storage.data.get("searches", [])
            for i, search in enumerate(searches):
                search_id = search.get("id") or search.get("analysis_id")
                if search_id == query_id:
                    searches[i]["category_id"] = target_category_id
                    searches[i]["last_updated"] = datetime.now().isoformat()
                    updated = True
                    break
            
            if updated:
                # Guardar cambios
                success = self.storage.storage.save_data()
                if success:
                    st.success(f"✅ Tecnología movida exitosamente en almacenamiento local")
                    return True
                else:
                    st.error("❌ Error guardando cambios en almacenamiento local")
                    return False
            else:
                st.error(f"❌ No se encontró la tecnología con ID: {query_id}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error en almacenamiento local: {str(e)}")
            return False

    def _delete_technology(self, query_id: str) -> bool:
        """
        Elimina una tecnología específica - IMPLEMENTACIÓN COMPLETA
        
        Args:
            query_id: ID de la consulta/tecnología a eliminar
            
        Returns:
            bool: True si se eliminó exitosamente, False en caso contrario
        """
        try:
            # Verificar que la tecnología existe
            query = self.storage.get_query_by_id(query_id)
            if not query:
                st.error(f"❌ No se encontró la tecnología con ID: {query_id}")
                return False
            
            # Eliminar según el tipo de storage
            if hasattr(self.storage.storage, 'analyses_table'):
                # DYNAMODB
                return self._delete_technology_dynamodb(query)
            else:
                # LOCAL STORAGE
                return self._delete_technology_local(query_id)
                
        except Exception as e:
            st.error(f"❌ Error eliminando tecnología: {str(e)}")
            return False

    def _delete_technology_dynamodb(self, query: dict) -> bool:
        """Elimina tecnología de DynamoDB"""
        try:
            analysis_id = query.get("analysis_id") or query.get("query_id")
            timestamp = query.get("timestamp") or query.get("execution_date")
            
            if not analysis_id or not timestamp:
                st.error("❌ No se pueden obtener las claves primarias para eliminar")
                return False
            
            # Eliminar item de DynamoDB
            self.storage.storage.analyses_table.delete_item(
                Key={
                    'analysis_id': analysis_id,
                    'timestamp': timestamp
                }
            )
            
            st.success(f"✅ Tecnología eliminada de DynamoDB")
            return True
            
        except Exception as e:
            st.error(f"❌ Error eliminando de DynamoDB: {str(e)}")
            return False

    def _delete_technology_local(self, query_id: str) -> bool:
        """Elimina tecnología del almacenamiento local"""
        try:
            # Eliminar de hype_cycle_queries
            hype_queries = self.storage.storage.data.get("hype_cycle_queries", [])
            original_count = len(hype_queries)
            
            hype_queries[:] = [
                q for q in hype_queries 
                if q.get("query_id") != query_id and q.get("analysis_id") != query_id
            ]
            
            # Eliminar de searches generales (por compatibilidad)
            searches = self.storage.storage.data.get("searches", [])
            searches[:] = [
                s for s in searches 
                if s.get("id") != query_id and s.get("analysis_id") != query_id
            ]
            
            deleted_count = original_count - len(hype_queries)
            
            if deleted_count > 0:
                # Guardar cambios
                success = self.storage.storage.save_data()
                if success:
                    st.success(f"✅ Tecnología eliminada del almacenamiento local")
                    return True
                else:
                    st.error("❌ Error guardando cambios")
                    return False
            else:
                st.warning(f"⚠️ No se encontró la tecnología para eliminar: {query_id}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error en almacenamiento local: {str(e)}")
            return False

    def _update_category(self, category_id: str, new_name: str, new_description: str) -> bool:
        """
        Actualiza los datos de una categoría - IMPLEMENTACIÓN COMPLETA
        
        Args:
            category_id: ID de la categoría a actualizar
            new_name: Nuevo nombre de la categoría
            new_description: Nueva descripción
            
        Returns:
            bool: True si se actualizó exitosamente, False en caso contrario
        """
        try:
            # Validar datos de entrada
            if not new_name.strip():
                st.error("❌ El nombre de la categoría no puede estar vacío")
                return False
            
            # Verificar que la categoría existe
            current_category = self.storage.storage.get_category_by_id(category_id)
            if not current_category:
                st.error(f"❌ No se encontró la categoría con ID: {category_id}")
                return False
            
            # Actualizar según el tipo de storage
            if hasattr(self.storage.storage, 'categories_table'):
                # DYNAMODB
                return self._update_category_dynamodb(category_id, new_name, new_description)
            else:
                # LOCAL STORAGE
                return self._update_category_local(category_id, new_name, new_description)
                
        except Exception as e:
            st.error(f"❌ Error actualizando categoría: {str(e)}")
            return False

    def _update_category_dynamodb(self, category_id: str, new_name: str, new_description: str) -> bool:
        """Actualiza categoría en DynamoDB"""
        try:
            # Actualizar item en DynamoDB
            response = self.storage.storage.categories_table.update_item(
                Key={'category_id': category_id},
                UpdateExpression='SET #name = :name, description = :desc, updated_at = :updated',
                ExpressionAttributeNames={'#name': 'name'},  # 'name' es palabra reservada
                ExpressionAttributeValues={
                    ':name': new_name,
                    ':desc': new_description,
                    ':updated': datetime.now().isoformat()
                },
                ReturnValues='UPDATED_NEW'
            )
            
            if response.get('Attributes'):
                st.success(f"✅ Categoría actualizada en DynamoDB")
                return True
            else:
                st.error("❌ No se pudo confirmar la actualización")
                return False
                
        except Exception as e:
            st.error(f"❌ Error actualizando en DynamoDB: {str(e)}")
            return False

    def _update_category_local(self, category_id: str, new_name: str, new_description: str) -> bool:
        """Actualiza categoría en almacenamiento local"""
        try:
            categories = self.storage.storage.categories.get("categories", [])
            updated = False
            
            for i, category in enumerate(categories):
                cat_id = category.get("id") or category.get("category_id")
                if cat_id == category_id:
                    # Actualizar datos
                    categories[i]["name"] = new_name
                    categories[i]["description"] = new_description
                    categories[i]["updated_at"] = datetime.now().isoformat()
                    updated = True
                    break
            
            if updated:
                # Guardar cambios
                success = self.storage.storage.save_categories()
                if success:
                    st.success(f"✅ Categoría actualizada en almacenamiento local")
                    return True
                else:
                    st.error("❌ Error guardando cambios de categoría")
                    return False
            else:
                st.error(f"❌ No se encontró la categoría con ID: {category_id}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error en almacenamiento local: {str(e)}")
            return False

    def _delete_category(self, category_id: str) -> bool:
        """
        Elimina una categoría y todas sus tecnologías - IMPLEMENTACIÓN COMPLETA
        
        Args:
            category_id: ID de la categoría a eliminar
            
        Returns:
            bool: True si se eliminó exitosamente, False en caso contrario
        """
        try:
            # Verificar que no sea la categoría default
            if category_id == "default":
                st.error("❌ No se puede eliminar la categoría 'default'")
                return False
            
            # Verificar que la categoría existe
            category = self.storage.storage.get_category_by_id(category_id)
            if not category:
                st.error(f"❌ No se encontró la categoría con ID: {category_id}")
                return False
            
            # Obtener tecnologías de esta categoría
            cat_queries = self.storage.get_queries_by_category(category_id)
            
            # Confirmar eliminación si hay tecnologías
            if cat_queries:
                st.warning(f"⚠️ Esta operación eliminará {len(cat_queries)} tecnologías asociadas")
                
                # Eliminar todas las tecnologías primero
                deleted_tech_count = 0
                for query in cat_queries:
                    query_id = query.get("query_id") or query.get("analysis_id")
                    if query_id and self._delete_technology(query_id):
                        deleted_tech_count += 1
                
                st.info(f"📊 {deleted_tech_count} tecnologías eliminadas")
            
            # Eliminar la categoría según el tipo de storage
            if hasattr(self.storage.storage, 'categories_table'):
                # DYNAMODB
                return self._delete_category_dynamodb(category_id)
            else:
                # LOCAL STORAGE
                return self._delete_category_local(category_id)
                
        except Exception as e:
            st.error(f"❌ Error eliminando categoría: {str(e)}")
            return False

    def _delete_category_dynamodb(self, category_id: str) -> bool:
        """Elimina categoría de DynamoDB"""
        try:
            # Eliminar item de DynamoDB
            self.storage.storage.categories_table.delete_item(
                Key={'category_id': category_id}
            )
            
            st.success(f"✅ Categoría eliminada de DynamoDB")
            return True
            
        except Exception as e:
            st.error(f"❌ Error eliminando categoría de DynamoDB: {str(e)}")
            return False

    def _delete_category_local(self, category_id: str) -> bool:
        """Elimina categoría del almacenamiento local"""
        try:
            categories = self.storage.storage.categories.get("categories", [])
            original_count = len(categories)
            
            # Filtrar categoría a eliminar
            categories[:] = [
                cat for cat in categories 
                if cat.get("id") != category_id and cat.get("category_id") != category_id
            ]
            
            deleted_count = original_count - len(categories)
            
            if deleted_count > 0:
                # Guardar cambios
                success = self.storage.storage.save_categories()
                if success:
                    st.success(f"✅ Categoría eliminada del almacenamiento local")
                    return True
                else:
                    st.error("❌ Error guardando cambios")
                    return False
            else:
                st.warning(f"⚠️ No se encontró la categoría para eliminar: {category_id}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error en almacenamiento local: {str(e)}")
            return False

    # MÉTODO AUXILIAR PARA OBTENER INFORMACIÓN DE CATEGORÍA
    def _get_category_info(self, category_id: str) -> dict:
        """
        Obtiene información completa de una categoría incluyendo tecnologías asociadas - VERSIÓN CORREGIDA
        """
        try:
            # Obtener datos básicos de la categoría
            category = self.storage.storage.get_category_by_id(category_id)
            if not category:
                return {}
            
            # Obtener tecnologías asociadas
            queries = self.storage.get_queries_by_category(category_id)
            
            # Calcular estadísticas
            total_technologies = len(queries)
            active_technologies = len([q for q in queries if q.get("is_active", True)])
            
            # Distribución por fases
            phase_distribution = {}
            for query in queries:
                phase = query.get("hype_metrics", {}).get("phase", "Unknown")
                phase_distribution[phase] = phase_distribution.get(phase, 0) + 1
            
            # Promedio de confianza
            confidences = [
                q.get("hype_metrics", {}).get("confidence", 0) 
                for q in queries if q.get("hype_metrics", {}).get("confidence")
            ]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "category": category,
                "statistics": {
                    "total_technologies": total_technologies,
                    "active_technologies": active_technologies,
                    "phase_distribution": phase_distribution,
                    "average_confidence": avg_confidence
                },
                "technologies": queries
            }
            
        except Exception as e:
            st.error(f"❌ Error obteniendo información de categoría: {str(e)}")
            return {}

    def _validate_category_data(self, category_data):
        """Valida que los datos de categoría estén completos"""
        if not category_data:
            return False, "Datos de categoría vacíos"
        
        # Verificar campos obligatorios
        required_fields = ["id", "name"]
        for field in required_fields:
            if field not in category_data and f"category_{field}" not in category_data:
                return False, f"Campo requerido faltante: {field}"
        
        return True, "Válido"

    # 5. Método auxiliar para debug de errores de variables
    def _debug_category_data(self, categories):
        """Método de debug para verificar estructura de datos"""
        st.write("**🔍 Debug de Categorías:**")
        
        if not categories:
            st.write("• No hay categorías")
            return
        
        for i, category in enumerate(categories):
            st.write(f"**Categoría {i+1}:**")
            st.write(f"• Tipo: {type(category)}")
            st.write(f"• Keys disponibles: {list(category.keys()) if isinstance(category, dict) else 'No es dict'}")
            
            # Verificar campos comunes
            cat_id = category.get("id") or category.get("category_id", "NO_ID")
            cat_name = category.get("name", "NO_NAME")
            st.write(f"• ID: {cat_id}")
            st.write(f"• Nombre: {cat_name}")
            st.write("---")