# src/hype_cycle.py - VERSIÓN FINAL LIMPIA SIN LOGGING EN FRONTEND
import streamlit as st
import pandas as pd
import time
import logging
from datetime import datetime
from analysis import NewsAnalyzer, QueryBuilder
from category_admin import CategoryAdminInterface
from config import CONFIG

# Importar el nuevo sistema de almacenamiento
from hype_cycle_storage import (
    HypeCycleStorage, 
    HypeCycleHistoryInterface, 
    initialize_hype_cycle_storage,
    create_hype_cycle_interface
)
from data_storage import initialize_database

# Configurar logging real (no visible en frontend)
logger = logging.getLogger(__name__)

def run_hype_cycle_analysis():
    """Ejecuta el análisis del Hype Cycle con DynamoDB únicamente - VERSIÓN LIMPIA"""
    st.markdown('<p class="tab-subheader">📈 Análisis del Hype Cycle</p>', unsafe_allow_html=True)
    
    # Verificar configuración AWS
    aws_configured = (
        st.session_state.get('aws_access_key_id') and 
        st.session_state.get('aws_secret_access_key') and 
        st.session_state.get('aws_region')
    )
    
    if not aws_configured:
        st.error("❌ Se requieren credenciales de AWS para usar DynamoDB")
        st.info("Configura las credenciales en el panel lateral")
        return
    
    # Pestañas principales
    tab_analysis, tab_history, tab_admin = st.tabs([
        "🔍 Nuevo Análisis", 
        "📚 Historial",
        "🏷️ Administrar Categorías"
    ])
    
    with tab_analysis:
        _show_analysis_interface()
    
    with tab_history:
        _show_history_interface()
    
    with tab_admin:  
        _show_admin_interface()

def _get_dynamodb_instance():
    """Obtiene una instancia de DynamoDB configurada"""
    try:
        return initialize_database(
            "dynamodb",
            region_name=st.session_state.aws_region,
            aws_access_key_id=st.session_state.aws_access_key_id,
            aws_secret_access_key=st.session_state.aws_secret_access_key
        )
    except Exception as e:
        logger.error(f"Error conectando a DynamoDB: {str(e)}")
        return None

def _show_admin_interface():
    """Interfaz para administrar categorías y tecnologías"""
    try:
        db = _get_dynamodb_instance()
        
        if db:
            hype_storage = initialize_hype_cycle_storage(db.storage)
            stable_context = "hype_admin_main"
            admin_interface = CategoryAdminInterface(hype_storage, stable_context)
            admin_interface.show_admin_interface()
        else:
            st.error("No se pudo inicializar el sistema de almacenamiento DynamoDB")
            
    except Exception as e:
        st.error(f"Error en la interfaz de administración: {str(e)}")

def _show_analysis_interface():
    """VERSIÓN LIMPIA: Interfaz para realizar nuevos análisis"""
    st.write("""
    Esta herramienta te permite analizar tecnologías usando el modelo del Hype Cycle de Gartner.
    **Almacenamiento:** Todos los datos se guardan en DynamoDB en la nube.
    """)
    
    # Inicializar DynamoDB
    db = _get_dynamodb_instance()
    hype_storage = initialize_hype_cycle_storage(db.storage) if db else None
    
    if not hype_storage:
        st.error("❌ No se pudo conectar a DynamoDB. Verifica tu configuración.")
        return
    
    # Estado base estable
    STATE_PREFIX = "hype_analysis_main"
    
    # Verificar consulta reutilizada
    reuse_query = st.session_state.get('hype_reuse_query')
    if reuse_query:
        st.info("🔄 **Consulta cargada desde historial**")
        st.code(reuse_query['search_query'])
        
        if st.button("Limpiar consulta cargada", key=f"{STATE_PREFIX}_clear_reused"):
            del st.session_state.hype_reuse_query
            st.rerun()
    
    # === CONFIGURACIÓN DE ALMACENAMIENTO ===
    st.write("### 📂 Configuración de Almacenamiento")
    col1, col2 = st.columns(2)
    
    with col1:
        st.success(f"🗄️ **DynamoDB** - Región: {st.session_state.get('aws_region')}")
        auto_save = st.checkbox(
            "Guardar automáticamente", 
            value=True,
            key=f"{STATE_PREFIX}_auto_save"
        )
    
    with col2:
        try:
            categories = hype_storage.storage.get_all_categories()
            category_options = {cat.get("name", "Sin nombre"): cat.get("category_id") for cat in categories}
            
            selected_category_name = st.selectbox(
                "Categoría para guardar",
                options=list(category_options.keys()),
                key=f"{STATE_PREFIX}_category_selector"
            )
            selected_category_id = category_options[selected_category_name]
            
        except Exception as e:
            logger.error(f"Error cargando categorías: {str(e)}")
            selected_category_id = "default"

    # === INFORMACIÓN DE TECNOLOGÍA ===
    with st.expander("📝 Información de la tecnología (opcional)", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            technology_name = st.text_input(
                "Nombre de la tecnología",
                placeholder="ej: Inteligencia Artificial, Blockchain",
                key=f"{STATE_PREFIX}_tech_name"
            )
        
        with col2:
            technology_description = st.text_area(
                "Descripción (opcional)",
                height=60,
                key=f"{STATE_PREFIX}_tech_desc"
            )

    # === TÉRMINOS DE BÚSQUEDA SIMPLIFICADOS ===
    st.write("### 🎯 Términos de búsqueda")
    topics = _show_simple_topics_interface(STATE_PREFIX, reuse_query)
    
    # === OPCIONES AVANZADAS ===
    with st.expander("⚙️ Opciones avanzadas", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_year = st.number_input(
                "Año mínimo",
                min_value=2010,
                max_value=2025,
                value=2015,
                key=f"{STATE_PREFIX}_min_year"
            )
            
            max_results = st.number_input(
                "Máximo resultados",
                min_value=50,
                max_value=1000,
                value=200,
                key=f"{STATE_PREFIX}_max_results"
            )
        
        with col2:
            sources_filter = st.multiselect(
                "Filtrar fuentes",
                options=["Tech News", "Business News", "Academic Sources", "Blogs"],
                default=["Tech News", "Business News"],
                key=f"{STATE_PREFIX}_sources"
            )
            
            analysis_notes = st.text_input(
                "Notas del análisis",
                placeholder="Ej: Análisis Q1 2025",
                key=f"{STATE_PREFIX}_notes"
            )

    # === VALIDACIÓN Y ANÁLISIS ===
    valid_topics = [t for t in topics if t.get('value', '').strip()]
    can_analyze = len(valid_topics) > 0 and st.session_state.get('serp_api_key')
    
    # Mostrar estado actual
    if valid_topics:
        equation = _build_search_equation(valid_topics)
        st.write("### 📝 Consulta a ejecutar")
        st.code(equation)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Términos válidos", len(valid_topics))
        with col2:
            st.metric("Año mínimo", min_year)
        with col3:
            st.metric("Fuentes", len(sources_filter))
    
    # Mensajes de validación
    if not valid_topics:
        st.warning("⚠️ Añade al menos un término válido para continuar")
    elif not st.session_state.get('serp_api_key'):
        st.warning("⚠️ Configura tu API key de SerpAPI en el panel lateral")
    
    # === BOTÓN DE ANÁLISIS ===
    if st.button(
        "📊 Analizar Hype Cycle", 
        type="primary", 
        disabled=not can_analyze,
        key=f"{STATE_PREFIX}_analyze_btn"
    ):
        search_parameters = {
            "min_year": min_year,
            "sources_filter": sources_filter,
            "max_results": max_results,
            "auto_save": auto_save,
            "category_id": selected_category_id
        }
        
        _execute_clean_analysis(
            topics=valid_topics,
            search_parameters=search_parameters,
            analysis_notes=analysis_notes,
            technology_name=technology_name,
            technology_description=technology_description,
            selected_category_id=selected_category_id,
            auto_save=auto_save,
            hype_storage=hype_storage,
            state_prefix=STATE_PREFIX
        )
    
    elif not can_analyze:
        # Mostrar guía si no puede analizar
        st.info("""
        ### 📝 Guía del Hype Cycle
        
        El Hype Cycle de Gartner representa 5 fases en la evolución de una tecnología:
        
        1. **Innovation Trigger** - Primer interés, pruebas de concepto iniciales
        2. **Peak of Inflated Expectations** - Máximo entusiasmo, expectativas poco realistas  
        3. **Trough of Disillusionment** - Desencanto cuando la realidad no alcanza las expectativas
        4. **Slope of Enlightenment** - Comprensión realista de beneficios y limitaciones
        5. **Plateau of Productivity** - Adopción estable y generalizada
        
        Para comenzar, añade términos de búsqueda y configura tu API key de SerpAPI.
        """)

def _show_simple_topics_interface(prefix, reuse_query=None):
    """Interfaz simplificada para términos de búsqueda - SIN RERUNS PROBLEMÁTICOS"""
    
    # Key estable para el estado
    topics_key = f"{prefix}_topics_simple"
    
    # Inicializar estado
    if topics_key not in st.session_state:
        if reuse_query and reuse_query.get('search_terms'):
            st.session_state[topics_key] = reuse_query['search_terms']
        else:
            st.session_state[topics_key] = [{'value': '', 'operator': 'AND', 'exact_match': False}]
    
    # Mostrar términos actuales
    topics_list = st.session_state[topics_key]
    
    # Contenedor para todos los términos
    for i, topic in enumerate(topics_list):
        col1, col2, col3, col4 = st.columns([4, 1.5, 1.5, 1])
        
        with col1:
            # Input para el término
            value_key = f"{prefix}_term_{i}_value"
            new_value = st.text_input(
                f"Término {i+1}",
                value=topic.get('value', ''),
                key=value_key,
                placeholder="ej: artificial intelligence"
            )
            # Actualizar inmediatamente en el estado
            st.session_state[topics_key][i]['value'] = new_value
        
        with col2:
            # Selector de operador
            op_key = f"{prefix}_term_{i}_operator"
            new_operator = st.selectbox(
                "Operador",
                options=['AND', 'OR', 'NOT'],
                index=['AND', 'OR', 'NOT'].index(topic.get('operator', 'AND')),
                key=op_key
            )
            st.session_state[topics_key][i]['operator'] = new_operator
        
        with col3:
            # Checkbox para exacto
            exact_key = f"{prefix}_term_{i}_exact"
            new_exact = st.checkbox(
                "Exacto",
                value=topic.get('exact_match', False),
                key=exact_key
            )
            st.session_state[topics_key][i]['exact_match'] = new_exact
        
        with col4:
            # Botón eliminar (solo si hay más de uno)
            if len(topics_list) > 1:
                remove_key = f"{prefix}_remove_{i}"
                if st.button("❌", key=remove_key, help="Eliminar término"):
                    st.session_state[topics_key].pop(i)
                    st.rerun()
    
    # Botones de control
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Añadir término", key=f"{prefix}_add_term"):
            st.session_state[topics_key].append({
                'value': '', 
                'operator': 'AND', 
                'exact_match': False
            })
            st.rerun()
    
    with col2:
        if st.button("🧹 Limpiar todo", key=f"{prefix}_clear_terms"):
            st.session_state[topics_key] = [{'value': '', 'operator': 'AND', 'exact_match': False}]
            st.rerun()
    
    with col3:
        # Mostrar estado actual
        valid_count = len([t for t in topics_list if t.get('value', '').strip()])
        st.caption(f"📊 {len(topics_list)} término(s) | {valid_count} válido(s)")
    
    return st.session_state[topics_key]

def _execute_clean_analysis(topics, search_parameters, analysis_notes, technology_name, 
                          technology_description, selected_category_id, auto_save, 
                          hype_storage, state_prefix):
    """Ejecuta el análisis de forma limpia sin logging en frontend"""
    
    # Contenedor para progreso limpio
    progress_placeholder = st.empty()
    results_container = st.container()
    
    try:
        # Progreso simple
        with progress_placeholder:
            with st.spinner("🔄 Ejecutando análisis del Hype Cycle..."):
                
                # Inicializar componentes
                news_analyzer = NewsAnalyzer()
                query_builder = QueryBuilder()
                processed_topics = _process_topics(topics)
                
                # Construir query
                google_query = query_builder.build_google_query(
                    processed_topics,
                    search_parameters["min_year"],
                    include_patents=True
                )
                
                # Realizar búsqueda
                serp_success, serp_results = news_analyzer.perform_news_search(
                    serp_api_key=st.session_state.serp_api_key,
                    query=google_query
                )
                
                if not serp_success or not serp_results:
                    st.error("❌ No se pudieron obtener datos de SerpAPI")
                    return
                
                # Analizar Hype Cycle
                hype_data = news_analyzer.analyze_hype_cycle(serp_results)
                
                if not hype_data:
                    st.error("❌ No se pudo generar el análisis del Hype Cycle")
                    return
        
        # Limpiar progreso
        progress_placeholder.empty()
        
        # Mostrar resultados
        with results_container:
            # Resultado principal
            st.success(f"**Fase Detectada:** {hype_data['phase']} (Confianza: {hype_data['confidence']:.2f})")
            
            # Descripción de la fase
            _show_phase_description(hype_data['phase'])
            
            # Gráfico principal
            fig = news_analyzer.plot_hype_cycle(hype_data, topics)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de inflexión
            st.subheader("📊 Análisis de Puntos de Inflexión")
            inflection_points = news_analyzer.analyze_gartner_points(hype_data['yearly_stats'])
            inflection_fig = news_analyzer.plot_gartner_analysis(hype_data['yearly_stats'], inflection_points)
            if inflection_fig:
                st.plotly_chart(inflection_fig, use_container_width=True)
            
            # Guardar si está habilitado
            if auto_save and hype_storage:
                with st.spinner("💾 Guardando en DynamoDB..."):
                    try:
                        query_id = hype_storage.save_hype_cycle_query(
                            search_query=google_query,
                            search_terms=processed_topics,
                            hype_analysis_results=hype_data,
                            news_results=serp_results,
                            category_id=selected_category_id,
                            search_parameters=search_parameters,
                            notes=analysis_notes,
                            technology_name=technology_name,
                            technology_description=technology_description
                        )
                        
                        if query_id:
                            st.success(f"✅ Análisis guardado con ID: {query_id}")
                            
                            if st.button("📚 Ver en Historial", key=f"{state_prefix}_view_history_{query_id}"):
                                st.session_state.hype_show_query_id = query_id
                                st.rerun()
                        else:
                            st.warning("⚠️ Hubo problemas al guardar en DynamoDB")
                            
                    except Exception as save_error:
                        logger.error(f"Error guardando: {str(save_error)}")
                        st.error("❌ Error guardando en DynamoDB")
            
            # Análisis detallado
            query_info = {
                'google_query': google_query,
                'search_query': google_query,
                'time_range': f"{search_parameters['min_year']}-{datetime.now().year}"
            }
            
            news_analyzer.display_advanced_analysis(serp_results, query_info, st)
            news_analyzer.display_results(serp_results, st)
            
    except Exception as e:
        logger.error(f"Error en análisis: {str(e)}")
        st.error(f"❌ Error durante el análisis: {str(e)}")

def _show_phase_description(phase):
    """Muestra descripción de la fase detectada"""
    descriptions = {
        "Innovation Trigger": "🚀 **Fase inicial** - Alto interés y especulación, pocos casos reales",
        "Peak of Inflated Expectations": "📈 **Máximo entusiasmo** - Cobertura mediática máxima, expectativas altas",
        "Trough of Disillusionment": "📉 **Fase de desilusión** - Disminución del interés, reevaluación de expectativas",
        "Slope of Enlightenment": "📊 **Maduración** - Casos de uso definidos, beneficios comprobados",
        "Plateau of Productivity": "✅ **Madurez estable** - Adopción generalizada, beneficios demostrados"
    }
    
    description = descriptions.get(phase, "📝 Descripción no disponible")
    st.info(description)

def _show_history_interface():
    """Interfaz para mostrar el historial de consultas"""
    try:
        db = _get_dynamodb_instance()
        
        if db:
            hype_storage = initialize_hype_cycle_storage(db.storage)
            stable_context = "hype_history_main"
            history_interface = create_hype_cycle_interface(hype_storage, stable_context)
            
            history_interface.show_history_interface()
            
            # Mostrar consulta específica si se solicita
            show_query_id = st.session_state.get('hype_show_query_id')
            if show_query_id:
                st.info(f"Mostrando detalles de la consulta: {show_query_id}")
                query = hype_storage.get_query_by_id(show_query_id)
                if query:
                    st.json(query)
                else:
                    st.error("No se encontró la consulta especificada")
                
                if st.button("Volver al historial", key=f"back_to_history_{stable_context}_{show_query_id}"):
                    del st.session_state.hype_show_query_id
                    st.rerun()
        else:
            st.error("No se pudo inicializar el sistema de almacenamiento DynamoDB")
            
    except Exception as e:
        st.error(f"Error en la interfaz de historial: {str(e)}")

# ===== FUNCIONES AUXILIARES =====

def _build_search_equation(topics):
    """Construye la ecuación de búsqueda"""
    if not topics:
        return ""
    
    parts = []
    for i, topic in enumerate(topics):
        value = topic.get('value', '').strip()
        if not value:
            continue
            
        if topic.get('exact_match', False):
            term = f'"{value}"'
        else:
            term = value
        
        parts.append(term)
        
        # Añadir operador si no es el último
        remaining = [t for t in topics[i+1:] if t.get('value', '').strip()]
        if remaining:
            parts.append(f" {topic.get('operator', 'AND')} ")
    
    return "".join(parts)

def _process_topics(topics_data):
    """Procesa los topics antes de construir la query"""
    processed = []
    
    for topic in topics_data:
        value = topic.get('value', '').strip()
        if value:
            processed_topic = {
                'value': value,
                'operator': topic.get('operator', 'AND'),
                'exact_match': topic.get('exact_match', False)
            }
            
            # Si ya tiene comillas, no marcar como exact_match
            if value.startswith('"') and value.endswith('"'):
                processed_topic['exact_match'] = False
            
            processed.append(processed_topic)
    
    return processed