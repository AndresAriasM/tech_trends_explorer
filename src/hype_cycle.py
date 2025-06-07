# src/hype_cycle.py - ACTUALIZADO con sistema de almacenamiento
import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
from analysis import NewsAnalyzer, QueryBuilder
from config import CONFIG

# Importar el nuevo sistema de almacenamiento
from hype_cycle_storage import (
    HypeCycleStorage, 
    HypeCycleHistoryInterface, 
    initialize_hype_cycle_storage,
    create_hype_cycle_interface
)
from data_storage import initialize_database

def run_hype_cycle_analysis():
    """
    Ejecuta el análisis del Hype Cycle utilizando la API de SerpAPI
    Con sistema de almacenamiento automático en DynamoDB
    """
    st.markdown('<p class="tab-subheader">📈 Análisis del Hype Cycle</p>', unsafe_allow_html=True)
    
    # Pestañas para análisis y historial
    tab_analysis, tab_history = st.tabs(["🔍 Nuevo Análisis", "📚 Historial"])
    
    with tab_analysis:
        _show_analysis_interface()
    
    with tab_history:
        _show_history_interface()

def _show_analysis_interface():
    """Interfaz para realizar nuevos análisis"""
    st.write("""
    Esta herramienta te permite analizar tecnologías usando el modelo del Hype Cycle de Gartner.
    El análisis del Hype Cycle ayuda a entender en qué fase de expectativas y adopción se 
    encuentra una tecnología.
    """)
    
    # Inicializar sistema de almacenamiento
    storage_mode = st.session_state.get('hype_storage_mode', 'local')
    try:
        if storage_mode == 'local':
            db = initialize_database("local")
        else:
            # Usar DynamoDB si está configurado
            aws_configured = (
                st.session_state.get('aws_access_key_id') and 
                st.session_state.get('aws_secret_access_key') and 
                st.session_state.get('aws_region')
            )
            
            if aws_configured:
                db = initialize_database(
                    "dynamodb",
                    region_name=st.session_state.aws_region,
                    aws_access_key_id=st.session_state.aws_access_key_id,
                    aws_secret_access_key=st.session_state.aws_secret_access_key
                )
            else:
                st.warning("⚠️ DynamoDB no configurado. Usando almacenamiento local.")
                db = initialize_database("local")
        
        hype_storage = initialize_hype_cycle_storage(db.storage) if db else None
    except Exception as e:
        st.error(f"Error inicializando almacenamiento: {str(e)}")
        hype_storage = None
    
    # Verificar si hay una consulta para reutilizar
    reuse_query = st.session_state.get('hype_reuse_query')
    if reuse_query:
        st.info("🔄 **Consulta cargada desde historial**")
        st.code(reuse_query['search_query'])
        if st.button("Limpiar consulta cargada", key="hype_clear_reused_query_btn"):
            del st.session_state.hype_reuse_query
            st.rerun()
    
    # Configuración de categoría para guardar
    st.write("### 📂 Configuración de Almacenamiento")
    with st.expander("⚙️ Opciones de guardado", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Modo de almacenamiento - KEY ÚNICA AÑADIDA
            storage_options = ["local", "dynamodb"]
            current_mode = st.selectbox(
                "Modo de almacenamiento",
                options=storage_options,
                index=storage_options.index(storage_mode),
                help="Selecciona dónde guardar los resultados del análisis",
                key="hype_analysis_storage_mode_selectbox"  # ← KEY ÚNICA AÑADIDA
            )
            st.session_state.hype_storage_mode = current_mode
            
            # Auto-guardar - KEY ÚNICA AÑADIDA
            auto_save = st.checkbox(
                "Guardar automáticamente", 
                value=True,
                help="Guarda automáticamente cada análisis realizado",
                key="hype_analysis_auto_save_checkbox"  # ← KEY ÚNICA AÑADIDA
            )
        
        with col2:
            # Selector de categoría
            if hype_storage:
                try:
                    categories = hype_storage.storage.get_all_categories()
                    category_options = {cat.get("name", "Sin nombre"): cat.get("id", cat.get("category_id")) for cat in categories}
                    
                    # Selectbox para categoría - KEY ÚNICA AÑADIDA
                    selected_category_name = st.selectbox(
                        "Categoría para guardar",
                        options=list(category_options.keys()),
                        help="Selecciona la categoría donde guardar este análisis",
                        key="hype_analysis_category_selectbox"  # ← KEY ÚNICA AÑADIDA
                    )
                    
                    selected_category_id = category_options[selected_category_name]
                    
                    # Opción para crear nueva categoría - KEY ÚNICA AÑADIDA
                    if st.checkbox("Crear nueva categoría", key="hype_analysis_new_category_checkbox"):
                        new_cat_name = st.text_input("Nombre de la nueva categoría", key="hype_analysis_new_cat_name_input")
                        new_cat_desc = st.text_area("Descripción (opcional)", height=60, key="hype_analysis_new_cat_desc_textarea")
                        
                        if st.button("Crear Categoría", key="hype_analysis_create_category_btn") and new_cat_name:
                            try:
                                new_cat_id = hype_storage.storage.add_category(new_cat_name, new_cat_desc)
                                if new_cat_id:
                                    st.success(f"✅ Categoría '{new_cat_name}' creada")
                                    selected_category_id = new_cat_id
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error creando categoría: {str(e)}")
                
                except Exception as e:
                    st.warning(f"Error cargando categorías: {str(e)}")
                    selected_category_id = "default"
            else:
                selected_category_id = "default"
                st.info("Sistema de almacenamiento no disponible")
    
    # Configuración de búsqueda
    st.write("### 🎯 Define los términos para el análisis")
    
    # Si hay consulta reutilizada, cargar los términos
    if reuse_query and reuse_query.get('search_terms'):
        # Pre-cargar términos de la consulta reutilizada
        topics = reuse_query['search_terms']
        st.write("**Términos cargados desde historial:**")
        for i, term in enumerate(topics):
            st.write(f"{i+1}. {term.get('value', '')} ({term.get('operator', 'AND')})")
        
        if st.button("Modificar términos", key="hype_modify_terms_btn"):
            # Permite editar los términos
            topics = manage_topics("hype", preset_topics=topics)
    else:
        # Gestión normal de topics
        topics = manage_topics("hype")
    
    # Configuración adicional del Hype Cycle
    with st.expander("⚙️ Opciones avanzadas", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_year = st.number_input(
                "Año mínimo",
                min_value=2010,
                max_value=2025,
                value=reuse_query.get('search_parameters', {}).get('min_year', 2015) if reuse_query else 2015,
                help="Año desde el cual buscar resultados",
                key="hype_analysis_min_year_input"  # ← KEY ÚNICA AÑADIDA
            )
        with col2:
            # Multiselect para fuentes - KEY ÚNICA AÑADIDA
            sources_filter = st.multiselect(
                "Filtrar fuentes",
                options=["Tech News", "Business News", "Academic Sources", "Blogs"],
                default=reuse_query.get('search_parameters', {}).get('sources_filter', ["Tech News", "Business News"]) if reuse_query else ["Tech News", "Business News"],
                help="Tipos de fuentes a incluir en el análisis",
                key="hype_analysis_sources_filter_multiselect"  # ← KEY ÚNICA AÑADIDA
            )
        
        # Configuración adicional para almacenamiento
        col3, col4 = st.columns(2)
        with col3:
            max_results = st.number_input(
                "Máximo resultados API",
                min_value=50,
                max_value=1000,
                value=200,
                help="Número máximo de resultados a obtener de la API",
                key="hype_analysis_max_results_input"  # ← KEY ÚNICA AÑADIDA
            )
        with col4:
            analysis_notes = st.text_input(
                "Notas del análisis",
                placeholder="Ej: Análisis para Q1 2025, investigación de mercado...",
                help="Notas que se guardarán con el análisis",
                key="hype_analysis_notes_input"  # ← KEY ÚNICA AÑADIDA
            )
    
    # Mostrar información de consulta actual
    if topics:
        current_query = build_search_equation(topics)
        st.write("### 📝 Consulta actual")
        st.code(current_query)
    
    # Botón de análisis - KEY ÚNICA AÑADIDA
    if st.button("📊 Analizar Hype Cycle", type="primary", key="hype_analyze_main_btn"):
        if not topics:
            st.error("Por favor, ingresa al menos un tema para analizar")
            return
            
        if not st.session_state.serp_api_key:
            st.error("Se requiere una API key de SerpAPI. Por favor, configúrala en el panel lateral.")
            return
            
        # Almacenar parámetros de búsqueda
        search_parameters = {
            "min_year": min_year,
            "sources_filter": sources_filter,
            "max_results": max_results,
            "auto_save": auto_save,
            "category_id": selected_category_id
        }
        
        with st.spinner("🔄 Analizando el Hype Cycle..."):
            start_time = time.time()
            
            # Inicializar analizador
            news_analyzer = NewsAnalyzer()
            
            # Construir query
            query_builder = QueryBuilder()
            processed_topics = process_topics(topics)
            google_query = query_builder.build_google_query(
                processed_topics,
                min_year,
                include_patents=True
            )
            
            # Crear objeto de información de queries
            query_info = {
                'google_query': google_query,
                'scopus_query': query_builder.build_scopus_query(topics, min_year),
                'search_query': google_query,
                'time_range': f"{min_year}-{datetime.now().year}"
            }
            
            # Realizar búsqueda en SerpAPI
            serp_success, serp_results = news_analyzer.perform_news_search(
                serp_api_key=st.session_state.serp_api_key,
                query=google_query
            )
            
            if serp_success and serp_results:
                # Análisis del Hype Cycle
                hype_data = news_analyzer.analyze_hype_cycle(serp_results)
                
                if hype_data:
                    processing_time = time.time() - start_time
                    
                    # Mostrar resultados
                    st.success(f"**Fase Actual:** {hype_data['phase']} (Confianza: {hype_data['confidence']:.2f})")
                    
                    # Descripción de la fase
                    phase_descriptions = {
                        "Innovation Trigger": """
                            La tecnología está en su fase inicial de innovación. Se caracteriza por:
                            - Alto nivel de interés y especulación
                            - Pocos casos de implementación real
                            - Gran potencial percibido
                        """,
                        "Peak of Inflated Expectations": """
                            La tecnología está en su punto máximo de expectativas. Se observa:
                            - Máxima cobertura mediática
                            - Altas expectativas de mercado
                            - Posible sobreestimación de capacidades
                        """,
                        "Trough of Disillusionment": """
                            La tecnología está atravesando una fase de desilusión. Caracterizada por:
                            - Disminución del interés inicial
                            - Identificación de limitaciones reales
                            - Reevaluación de expectativas
                        """,
                        "Slope of Enlightenment": """
                            La tecnología está madurando hacia una comprensión realista. Se observa:
                            - Casos de uso bien definidos
                            - Beneficios comprobados
                            - Adopción más estratégica
                        """,
                        "Plateau of Productivity": """
                            La tecnología ha alcanzado un nivel de madurez estable. Caracterizada por:
                            - Adopción generalizada
                            - Beneficios claramente demostrados
                            - Implementación sistemática
                        """
                    }
                    
                    st.info(phase_descriptions.get(hype_data['phase'], "Descripción no disponible"))
                    
                    # Mostrar gráfico del Hype Cycle
                    fig = news_analyzer.plot_hype_cycle(hype_data, topics)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)

                    # Análisis de puntos de inflexión
                    st.write("### 📊 Análisis de Puntos de Inflexión")
                    inflection_points = news_analyzer.analyze_gartner_points(hype_data['yearly_stats'])
                    inflection_fig = news_analyzer.plot_gartner_analysis(
                        hype_data['yearly_stats'], 
                        inflection_points
                    )
                    if inflection_fig:
                        st.plotly_chart(inflection_fig, use_container_width=True)
                    
                    # Guardar automáticamente si está habilitado
                    if auto_save and hype_storage:
                        with st.spinner("💾 Guardando análisis..."):
                            try:
                                query_id = hype_storage.save_hype_cycle_query(
                                    search_query=google_query,
                                    search_terms=processed_topics,
                                    hype_analysis_results=hype_data,
                                    news_results=serp_results,
                                    category_id=selected_category_id,
                                    search_parameters=search_parameters,
                                    notes=analysis_notes
                                )
                                
                                if query_id:
                                    st.success(f"✅ Análisis guardado automáticamente con ID: {query_id}")
                                    
                                    # Opción para ver en historial
                                    if st.button("📚 Ver en Historial", key="hype_view_in_history_btn"):
                                        st.session_state.hype_show_query_id = query_id
                                        st.rerun()
                                        
                            except Exception as e:
                                st.error(f"❌ Error guardando análisis: {str(e)}")
                    
                    # Opción manual de guardado si auto-save está deshabilitado
                    elif hype_storage and not auto_save:
                        st.write("### 💾 Guardar Análisis")
                        
                        if st.button("Guardar este análisis", type="secondary", key="hype_manual_save_btn"):
                            with st.spinner("💾 Guardando análisis..."):
                                try:
                                    query_id = hype_storage.save_hype_cycle_query(
                                        search_query=google_query,
                                        search_terms=processed_topics,
                                        hype_analysis_results=hype_data,
                                        news_results=serp_results,
                                        category_id=selected_category_id,
                                        search_parameters=search_parameters,
                                        notes=analysis_notes
                                    )
                                    
                                    if query_id:
                                        st.success(f"✅ Análisis guardado con ID: {query_id}")
                                        
                                except Exception as e:
                                    st.error(f"❌ Error guardando análisis: {str(e)}")
                    
                    # Mostrar análisis detallado
                    news_analyzer.display_advanced_analysis(serp_results, query_info, st)
                    news_analyzer.display_results(serp_results, st)
                      
                else:
                    st.warning("No se pudo generar el análisis del Hype Cycle con los datos disponibles")
            else:
                st.error("No se pudieron obtener datos de SerpAPI para el análisis del Hype Cycle")
    else:
        # Mostrar guía cuando no hay análisis
        st.info("""
        ### 📝 Guía del Hype Cycle
        
        El Hype Cycle de Gartner representa 5 fases en la evolución de una tecnología:
        
        1. **Innovation Trigger**: Primer interés, pruebas de concepto iniciales
        2. **Peak of Inflated Expectations**: Máximo entusiasmo, expectativas poco realistas
        3. **Trough of Disillusionment**: Desencanto cuando la realidad no alcanza las expectativas
        4. **Slope of Enlightenment**: Comprensión realista de beneficios y limitaciones
        5. **Plateau of Productivity**: Adopción estable y generalizada
        
        Para comenzar, ingresa los términos de búsqueda en el formulario superior y haz clic en "Analizar Hype Cycle".
        """)

def _show_history_interface():
    """Interfaz para mostrar el historial de consultas"""
    try:
        # Inicializar sistema de almacenamiento
        storage_mode = st.session_state.get('hype_storage_mode', 'local')
        
        if storage_mode == 'local':
            db = initialize_database("local")
        else:
            aws_configured = (
                st.session_state.get('aws_access_key_id') and 
                st.session_state.get('aws_secret_access_key') and 
                st.session_state.get('aws_region')
            )
            
            if aws_configured:
                db = initialize_database(
                    "dynamodb",
                    region_name=st.session_state.aws_region,
                    aws_access_key_id=st.session_state.aws_access_key_id,
                    aws_secret_access_key=st.session_state.aws_secret_access_key
                )
            else:
                st.warning("⚠️ DynamoDB no configurado. Usando almacenamiento local.")
                db = initialize_database("local")
        
        if db:
            hype_storage = initialize_hype_cycle_storage(db.storage)
            history_interface = create_hype_cycle_interface(hype_storage)
            
            # Mostrar consulta específica si se solicita
            show_query_id = st.session_state.get('hype_show_query_id')
            if show_query_id:
                st.info(f"Mostrando detalles de la consulta: {show_query_id}")
                query = hype_storage.get_query_by_id(show_query_id)
                if query:
                    history_interface._display_query_details(query)
                else:
                    st.error("No se encontró la consulta especificada")
                
                if st.button("Volver al historial", key="hype_back_to_history_btn"):
                    del st.session_state.hype_show_query_id
                    st.rerun()
            else:
                # Mostrar interfaz completa de historial
                history_interface.show_history_interface()
        else:
            st.error("No se pudo inicializar el sistema de almacenamiento")
            
    except Exception as e:
        st.error(f"Error en la interfaz de historial: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# Funciones auxiliares reutilizadas del script principal
def manage_topics(prefix="hype", preset_topics=None):
    """Maneja la adición y eliminación de topics con opciones avanzadas."""
    # Usar un estado específico para este módulo
    state_key = f"{prefix}_topics_data"
    
    # Si hay preset_topics, cargarlos
    if preset_topics and state_key not in st.session_state:
        st.session_state[state_key] = preset_topics
    elif state_key not in st.session_state:
        st.session_state[state_key] = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    
    # Mostrar guía de búsqueda
    with st.expander("📖 Guía de Búsqueda Avanzada", expanded=False):
        st.markdown("""
        ### Operadores de Búsqueda
        - **AND**: Encuentra resultados que contienen TODOS los términos (Por defecto)
        - **OR**: Encuentra resultados que contienen ALGUNO de los términos
        
        ### Opciones Adicionales
        - **Coincidencia exacta** ("..."): Busca la frase exacta
        - **Exclusión** (-): Excluye términos específicos
        - **Comodín** (*): Busca variaciones de palabras
        
        ### Ejemplos
        - `"machine learning" AND robotics`: Encuentra resultados que contengan exactamente "machine learning" Y también "robotics"
        - `AI OR "artificial intelligence"`: Encuentra resultados que contengan "AI" O "artificial intelligence"
        - `blockchain -crypto`: Encuentra resultados sobre blockchain pero excluye los que mencionan crypto
        """)
    
    topics = []
    topics_to_remove = []

    st.write("### 🔍 Construye tu búsqueda")

    # Crear columnas para cada topic con opciones avanzadas
    for topic in st.session_state[state_key]:  # Usar el state_key específico
        col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
        
        with col1:
            value = st.text_input(
                f"Término {topic['id'] + 1}",
                value=topic.get('value', ''),
                key=f"{prefix}_topic_{topic['id']}",
                placeholder="Ej: 'artificial intelligence' OR robot*"
            )
            topic['value'] = value
            topics.append({
                'value': value,
                'operator': topic.get('operator', 'AND'),
                'exact_match': topic.get('exact_match', False)
            })
        
        with col2:
            operator = st.selectbox(
                "Operador",
                options=['AND', 'OR', 'NOT'],
                index=['AND', 'OR', 'NOT'].index(topic['operator']),
                key=f"{prefix}_operator_{topic['id']}"
            )
            topic['operator'] = operator  # Actualizamos el operador en el topic directamente
        
        with col3:
            exact_match = st.checkbox(
                "Coincidencia exacta",
                value=topic.get('exact_match', False),
                key=f"{prefix}_exact_{topic['id']}"
            )
            topic['exact_match'] = exact_match
        
        with col4:
            if len(st.session_state[state_key]) > 1:  # Usar el state_key específico
                if st.button('❌', key=f"{prefix}_remove_{topic['id']}"):
                    topics_to_remove.append(topic['id'])

    # Remover topics marcados para eliminación
    if topics_to_remove:
        st.session_state[state_key] = [  # Usar el state_key específico
            topic for topic in st.session_state[state_key] 
            if topic['id'] not in topics_to_remove
        ]
        st.rerun()

    # Botón para añadir nuevo topic
    if st.button("➕ Añadir otro término", key=f"{prefix}_add_topic"):
        new_id = max([t['id'] for t in st.session_state[state_key]]) + 1  # Usar el state_key específico
        st.session_state[state_key].append({  # Usar el state_key específico
            'id': new_id,
            'value': '',
            'operator': 'AND',
            'exact_match': False
        })
        st.rerun()

    # Construir y mostrar la ecuación final
    if topics:
        equation = build_search_equation(topics)
        st.write("### 📝 Ecuación de búsqueda")
        st.code(equation)

    return topics

def build_search_equation(topics):
    """Construye la ecuación de búsqueda en base a los términos y operadores, sin operador al final."""
    equation = ""
    for i, topic in enumerate(topics):
        # Escapamos los espacios si es una coincidencia exacta
        if topic['exact_match']:
            term = f'"{topic["value"]}"'
        else:
            term = topic['value']
        
        if i == len(topics) - 1:  # Si es el último término, no añadimos operador
            equation += f"{term}"
        else:  # Para todos los demás términos, agregamos operador
            equation += f"{term} {topic['operator']} "
    
    return equation

def process_topics(topics_data):
    """
    Procesa los topics antes de construir la query
    
    Args:
        topics_data: Lista de topics con sus operadores y opciones
    Returns:
        Lista de diccionarios procesados
    """
    processed_topics = []
    
    for topic in topics_data:
        if topic['value'].strip():  # Solo procesar topics no vacíos
            processed_topic = {
                'value': topic['value'].strip(),
                'operator': topic.get('operator', 'AND'),
                'exact_match': topic.get('exact_match', False)
            }
            
            # Si ya tiene comillas, desactivar exact_match
            if processed_topic['value'].startswith('"') and processed_topic['value'].endswith('"'):
                processed_topic['exact_match'] = False
            
            processed_topics.append(processed_topic)
    
    return processed_topics