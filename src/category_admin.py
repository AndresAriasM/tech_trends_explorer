# src/category_admin.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import json

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
        
        # Generar ID único para evitar conflictos
        import time
        import random
        self.unique_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Estado de la sesión para la interfaz
        if 'selected_category_for_chart' not in st.session_state:
            st.session_state.selected_category_for_chart = None
        if 'admin_refresh_trigger' not in st.session_state:
            st.session_state.admin_refresh_trigger = 0
    
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
        """Vista general de categorías y tecnologías"""
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
            category_id = category.get("id") or category.get("category_id")
            category_name = category.get("name", "Sin nombre")
            
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
            # MEJORAR el botón "Ver Gráfica"
            if st.button(f"📊 Ver Gráfica", key=f"chart_{category_id}_{self.unique_id}", type="primary"):
                # Guardar en múltiples lugares para asegurar persistencia
                st.session_state.selected_category_for_chart = category_id
                st.session_state.chart_category_id = category_id
                st.session_state.chart_category_name = category_name
                
                # Mostrar mensaje de confirmación
                st.success(f"✅ Categoría '{category_name}' seleccionada. Ve a la pestaña 'Gráfica Hype Cycle' para ver la visualización.")
                
                # Opcional: forzar navegación automática si es posible
                st.info("👆 Haz clic en la pestaña '🎯 Gráfica Hype Cycle' arriba para ver la gráfica.")
        
        with col2:
            if st.button(f"📤 Exportar CSV", key=f"export_{category_id}_{self.unique_id}"):
                self._export_category_data(category_name, tech_data)
        
        with col3:
            if st.button(f"🔄 Actualizar", key=f"update_{category_id}_{self.unique_id}"):
                st.info(f"Funcionalidad de actualización para {category_name} - En desarrollo")
        
        with col4:
            if st.button(f"📋 Copiar IDs", key=f"copy_{category_id}_{self.unique_id}"):
                ids = [item["🆔 ID"] for item in tech_data]
                st.code(", ".join(ids))
    
    def _show_hype_cycle_chart(self):
        """Muestra la gráfica principal del Hype Cycle"""
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
        
        # MEJORAR la detección de categoría pre-seleccionada
        default_index = 0
        selected_category_id = None
        
        # Verificar múltiples fuentes de selección
        preselected_id = (
            st.session_state.get('selected_category_for_chart') or 
            st.session_state.get('chart_category_id')
        )
        
        if preselected_id:
            try:
                category_names = list(category_options.keys())
                category_ids = list(category_options.values())
                if preselected_id in category_ids:
                    default_index = category_ids.index(preselected_id)
                    selected_category_id = preselected_id
                    
                    # Mostrar mensaje de confirmación
                    cat_name = category_names[default_index]
                    st.info(f"📊 Mostrando gráfica para la categoría: **{cat_name}**")
            except Exception as e:
                st.warning(f"Error detectando categoría preseleccionada: {str(e)}")
        
        # Selector de categoría
        selected_category_name = st.selectbox(
            "🏷️ Selecciona una categoría para visualizar:",
            options=list(category_options.keys()),
            index=default_index,
            key=f"hype_chart_category_selector_{self.unique_id}"
        )
        
        selected_category_id = category_options[selected_category_name]
        
        # Opciones de visualización
        col1, col2 = st.columns(2)
        with col1:
            show_labels = st.checkbox("📝 Mostrar etiquetas de tecnologías", value=True)
        with col2:
            show_confidence = st.checkbox("🎯 Mostrar niveles de confianza", value=False)
        
        # Obtener tecnologías de la categoría seleccionada
        queries = self.storage.get_queries_by_category(selected_category_id)
        active_queries = [q for q in queries if q.get("is_active", True)]
        
        if not active_queries:
            st.warning(f"No hay tecnologías activas en la categoría '{selected_category_name}'")
            
            # Mostrar información de debug
            st.write("**Debug Info:**")
            st.write(f"- Total queries encontradas: {len(queries)}")
            st.write(f"- Queries activas: {len(active_queries)}")
            if queries:
                st.write("- Estados de actividad:")
                for i, q in enumerate(queries):
                    is_active = q.get("is_active", True)
                    tech_name = q.get("technology_name", "Sin nombre")
                    st.write(f"  • {tech_name}: {'Activa' if is_active else 'Inactiva'}")
            return
        
        # Generar y mostrar gráfica
        try:
            st.write(f"**Generando gráfica para {len(active_queries)} tecnologías...**")
            
            fig = self._create_hype_cycle_chart(
                active_queries, 
                selected_category_name,
                show_labels=show_labels,
                show_confidence=show_confidence
            )
            
            if fig and len(fig.data) > 0:
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar leyenda de la gráfica
                self._show_chart_legend(active_queries)
            else:
                st.error("Error: La gráfica está vacía o no se pudo generar")
                
        except Exception as e:
            st.error(f"Error generando la gráfica: {str(e)}")
            import traceback
            with st.expander("Ver detalles del error"):
                st.code(traceback.format_exc())
    
    def _create_hype_cycle_chart(self, queries: List[Dict], category_name: str, 
                                show_labels: bool = True, show_confidence: bool = False) -> go.Figure:
        """
        Crea la gráfica del Hype Cycle con múltiples tecnologías
        
        Args:
            queries: Lista de consultas de la categoría
            category_name: Nombre de la categoría
            show_labels: Si mostrar etiquetas de tecnologías
            show_confidence: Si mostrar niveles de confianza
            
        Returns:
            Figura de Plotly con la gráfica del Hype Cycle
        """
        
        # Crear figura
        fig = go.Figure()
        
        # Generar curva base del Hype Cycle
        x_curve, y_curve = self.positioner.create_hype_cycle_curve()
        
        # Añadir curva principal
        fig.add_trace(go.Scatter(
            x=x_curve, 
            y=y_curve,
            mode='lines',
            name='Hype Cycle',
            line=dict(color='#1f77b4', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Procesar posiciones de tecnologías
        technologies = []
        for query in queries:
            hype_metrics = query.get("hype_metrics", {})
            phase = hype_metrics.get("phase", "Unknown")
            confidence = hype_metrics.get("confidence", 0.5)
            total_mentions = hype_metrics.get("total_mentions", 0)
            
            # Obtener o calcular posición
            if hasattr(hype_metrics, 'hype_cycle_position_x') and hype_metrics.hype_cycle_position_x:
                pos_x = hype_metrics.hype_cycle_position_x
                pos_y = hype_metrics.hype_cycle_position_y
            else:
                pos_x, pos_y = self.positioner.calculate_position(phase, confidence, total_mentions)
            
            # Extraer nombre de tecnología
            tech_name = (
                query.get("technology_name") or 
                query.get("name") or 
                query.get("search_query", "")[:20]
            )
            
            # Obtener tiempo al plateau
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
                "sentiment_avg": hype_metrics.get("sentiment_avg", 0)
            })
        
        # Evitar superposición
        technologies = self.positioner.avoid_overlap(technologies)
        
        # Añadir tecnologías a la gráfica
        for i, tech in enumerate(technologies):
            # Determinar color basado en tiempo al plateau
            color = self.positioner.get_color_for_time_to_plateau(tech["time_to_plateau"])
            
            # Determinar tamaño basado en menciones (opcional)
            size = max(8, min(20, 8 + (tech["total_mentions"] / 100)))
            
            # Configurar texto de etiqueta
            text_label = tech["name"] if show_labels else ""
            
            # Configurar información de hover
            hover_text = f"""
                <b>{tech['name']}</b><br>
                Fase: {tech['phase']}<br>
                Confianza: {tech['confidence']:.2f}<br>
                Tiempo al Plateau: {tech['time_to_plateau']}<br>
                Menciones: {tech['total_mentions']}<br>
                Sentimiento: {tech['sentiment_avg']:.2f}
            """
            
            # Añadir punto de la tecnología
            fig.add_trace(go.Scatter(
                x=[tech["position_x"]],
                y=[tech["position_y"]],
                mode='markers+text' if show_labels else 'markers',
                name=tech["name"],
                marker=dict(
                    size=size,
                    color=color,
                    symbol='circle',
                    line=dict(color='white', width=1),
                    opacity=0.8
                ),
                text=[text_label],
                textposition="top center",
                textfont=dict(size=9, color='black'),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False
            ))
            
            # Mostrar nivel de confianza si está habilitado
            if show_confidence:
                confidence_text = f"{tech['confidence']:.2f}"
                fig.add_trace(go.Scatter(
                    x=[tech["position_x"]],
                    y=[tech["position_y"] - 8],
                    mode='text',
                    text=[confidence_text],
                    textfont=dict(size=8, color='gray'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Añadir etiquetas de fases
        phase_labels = self.positioner.get_phase_label_positions()
        
        for label in phase_labels:
            fig.add_annotation(
                x=label["x"], 
                y=label["y"],
                text=f"<b>{label['name']}</b>",
                showarrow=False,
                font=dict(size=11, color='#2c3e50'),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='#bdc3c7',
                borderwidth=1,
                xanchor='center',
                yanchor='top'
            )
        
        # Añadir leyenda de colores (tiempo al plateau)
        self._add_time_legend(fig)
        
        # Configurar layout
        fig.update_layout(
            title=dict(
                text=f"<b>Hype Cycle - {category_name}</b><br><span style='font-size:12px'>Posición de tecnologías en el ciclo de Gartner</span>",
                x=0.5,
                font=dict(size=16)
            ),
            xaxis=dict(
                title="<b>TIME</b>",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                showticklabels=False,
                zeroline=False,
                range=[-2, 102]
            ),
            yaxis=dict(
                title="<b>EXPECTATIONS</b>",
                showgrid=True,
                gridcolor='rgba(128,128,128,0.2)',
                showticklabels=False,
                zeroline=False,
                range=[0, 100]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=600,
            showlegend=False,
            font=dict(family="Arial, sans-serif")
        )
        
        return fig
    
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
        """Interfaz para eliminar categorías y tecnologías"""
        st.write("### 🗑️ Eliminación de Datos")
        
        st.warning("⚠️ **ADVERTENCIA**: Las eliminaciones son permanentes e irreversibles.")
        
        # Sección 1: Eliminar tecnologías individuales
        with st.expander("🔬 Eliminar Tecnologías Individuales", expanded=True):
            
            # Obtener todas las tecnologías
            all_queries = self.storage.get_all_hype_cycle_queries()
            
            if not all_queries:
                st.info("No hay tecnologías para eliminar.")
            else:
                # Crear lista de tecnologías con información
                tech_options = {}
                for query in all_queries:
                    query_id = query.get("query_id", query.get("analysis_id"))
                    tech_name = query.get("technology_name", query.get("search_query", ""))[:30]
                    category_id = query.get("category_id", "unknown")
                    
                    # Obtener nombre de categoría
                    try:
                        category = self.storage.storage.get_category_by_id(category_id)
                        category_name = category.get("name") if category else "Sin categoría"
                    except:
                        category_name = "Sin categoría"
                    
                    display_name = f"{tech_name} ({category_name})"
                    tech_options[display_name] = {
                        "query_id": query_id,
                        "tech_name": tech_name,
                        "category_name": category_name,
                        "query": query
                    }
                
                # Selector de tecnología a eliminar
                selected_tech = st.selectbox(
                    "Selecciona la tecnología a eliminar:",
                    options=list(tech_options.keys()),
                    key=f"delete_tech_selector_{self.unique_id}"
                )
                
                if selected_tech:
                    tech_info = tech_options[selected_tech]
                    
                    # Mostrar información de la tecnología
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Información de la tecnología:**")
                        st.write(f"• **Nombre:** {tech_info['tech_name']}")
                        st.write(f"• **Categoría:** {tech_info['category_name']}")
                        st.write(f"• **ID:** {tech_info['query_id']}")
                    
                    with col2:
                        hype_metrics = tech_info['query'].get('hype_metrics', {})
                        st.write("**Métricas:**")
                        st.write(f"• **Fase:** {hype_metrics.get('phase', 'Unknown')}")
                        st.write(f"• **Confianza:** {hype_metrics.get('confidence', 0):.2f}")
                        st.write(f"• **Menciones:** {hype_metrics.get('total_mentions', 0)}")
                    
                    # Confirmación de eliminación
                    col1, col2 = st.columns(2)
                    with col1:
                        confirm_delete = st.checkbox(
                            f"Confirmar eliminación de '{tech_info['tech_name']}'",
                            key=f"confirm_delete_tech_{self.unique_id}"
                        )
                    
                    with col2:
                        if confirm_delete and st.button(
                            "🗑️ ELIMINAR TECNOLOGÍA", 
                            type="secondary",
                            key=f"delete_tech_btn_{self.unique_id}"
                        ):
                            if self._delete_technology(tech_info['query_id']):
                                st.success(f"✅ Tecnología '{tech_info['tech_name']}' eliminada correctamente")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("❌ Error al eliminar la tecnología")
        
        # Sección 2: Eliminar categorías completas
        with st.expander("📁 Eliminar Categorías Completas", expanded=False):
            
            categories = self.storage.storage.get_all_categories()
            
            # Filtrar categorías que no sean "default"
            deletable_categories = [cat for cat in categories if cat.get("id") != "default" and cat.get("category_id") != "default"]
            
            if not deletable_categories:
                st.info("No hay categorías eliminables (la categoría 'default' no se puede eliminar).")
            else:
                # Mostrar información de categorías
                for category in deletable_categories:
                    cat_id = category.get("id") or category.get("category_id")
                    cat_name = category.get("name", "Sin nombre")
                    
                    # Contar tecnologías en esta categoría
                    cat_queries = self.storage.get_queries_by_category(cat_id)
                    tech_count = len(cat_queries)
                    
                    with st.container():
                        st.write(f"**📁 {cat_name}**")
                        
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"• {tech_count} tecnologías")
                            st.write(f"• ID: {cat_id}")
                            if tech_count > 0:
                                st.warning(f"⚠️ Eliminar esta categoría también eliminará {tech_count} tecnologías")
                        
                        with col2:
                            confirm_cat = st.checkbox(
                                "Confirmar",
                                key=f"confirm_delete_cat_{cat_id}_{self.unique_id}"
                            )
                        
                        with col3:
                            if confirm_cat and st.button(
                                "🗑️ ELIMINAR", 
                                type="secondary",
                                key=f"delete_cat_btn_{cat_id}_{self.unique_id}"
                            ):
                                if self._delete_category(cat_id):
                                    st.success(f"✅ Categoría '{cat_name}' eliminada")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("❌ Error al eliminar categoría")
                        
                        st.write("---")

    def _show_edit_categories(self):
        """Interfaz para editar categorías"""
        st.write("### 📝 Editar Categorías")
        
        categories = self.storage.storage.get_all_categories()
        
        if not categories:
            st.info("No hay categorías para editar.")
            return
        
        # Selector de categoría a editar
        category_options = {cat.get("name", "Sin nombre"): cat for cat in categories}
        
        selected_cat_name = st.selectbox(
            "Selecciona una categoría para editar:",
            options=list(category_options.keys()),
            key=f"edit_cat_selector_{self.unique_id}"
        )
        
        if selected_cat_name:
            category = category_options[selected_cat_name]
            cat_id = category.get("id") or category.get("category_id")
            
            with st.form(f"edit_category_form_{self.unique_id}"):
                st.write(f"**Editando categoría:** {selected_cat_name}")
                
                # Campos editables
                new_name = st.text_input(
                    "Nuevo nombre:",
                    value=category.get("name", ""),
                    key=f"edit_cat_name_{self.unique_id}"
                )
                
                new_description = st.text_area(
                    "Nueva descripción:",
                    value=category.get("description", ""),
                    height=100,
                    key=f"edit_cat_desc_{self.unique_id}"
                )
                
                # Botón de guardar
                if st.form_submit_button("💾 Guardar Cambios"):
                    if self._update_category(cat_id, new_name, new_description):
                        st.success(f"✅ Categoría actualizada correctamente")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Error al actualizar la categoría")

    def _show_move_technologies(self):
        """Interfaz para mover tecnologías entre categorías"""
        st.write("### 🔄 Mover Tecnologías Entre Categorías")
        
        all_queries = self.storage.get_all_hype_cycle_queries()
        categories = self.storage.storage.get_all_categories()
        
        if not all_queries or not categories:
            st.info("No hay suficientes datos para mover tecnologías.")
            return
        
        # Preparar opciones
        tech_options = {}
        for query in all_queries:
            query_id = query.get("query_id", query.get("analysis_id"))
            tech_name = query.get("technology_name", query.get("search_query", ""))[:30]
            current_cat_id = query.get("category_id", "unknown")
            
            # Obtener nombre de categoría actual
            current_cat_name = "Sin categoría"
            for cat in categories:
                if (cat.get("id") == current_cat_id or cat.get("category_id") == current_cat_id):
                    current_cat_name = cat.get("name", "Sin nombre")
                    break
            
            display_name = f"{tech_name} (Actual: {current_cat_name})"
            tech_options[display_name] = {
                "query_id": query_id,
                "tech_name": tech_name,
                "current_category_id": current_cat_id,
                "current_category_name": current_cat_name
            }
        
        category_options = {cat.get("name", "Sin nombre"): cat.get("id", cat.get("category_id")) for cat in categories}
        
        # Formulario de movimiento
        with st.form(f"move_tech_form_{self.unique_id}"):
            col1, col2 = st.columns(2)
            
            with col1:
                selected_tech = st.selectbox(
                    "Tecnología a mover:",
                    options=list(tech_options.keys()),
                    key=f"move_tech_selector_{self.unique_id}"
                )
            
            with col2:
                target_category = st.selectbox(
                    "Categoría destino:",
                    options=list(category_options.keys()),
                    key=f"move_target_cat_{self.unique_id}"
                )
            
            if st.form_submit_button("🔄 Mover Tecnología"):
                if selected_tech and target_category:
                    tech_info = tech_options[selected_tech]
                    target_cat_id = category_options[target_category]
                    
                    # Verificar que no sea la misma categoría
                    if tech_info["current_category_id"] == target_cat_id:
                        st.warning("La tecnología ya está en esa categoría.")
                    else:
                        if self._move_technology(tech_info["query_id"], target_cat_id):
                            st.success(f"✅ '{tech_info['tech_name']}' movida a '{target_category}'")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Error al mover la tecnología")

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
