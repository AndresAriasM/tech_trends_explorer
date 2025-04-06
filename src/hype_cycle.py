# src/hype_cycle.py
import streamlit as st
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
from analysis import NewsAnalyzer, QueryBuilder  # Añadida la importación de QueryBuilder
from config import CONFIG

def run_hype_cycle_analysis():
    """
    Ejecuta el análisis del Hype Cycle utilizando la API de SerpAPI
    """
    st.markdown('<p class="tab-subheader">📈 Análisis del Hype Cycle</p>', unsafe_allow_html=True)
    
    st.write("""
    Esta herramienta te permite analizar tecnologías usando el modelo del Hype Cycle de Gartner.
    El análisis del Hype Cycle ayuda a entender en qué fase de expectativas y adopción se 
    encuentra una tecnología.
    """)
    
    # Configuración de búsqueda
    st.write("### 🎯 Define los términos para el análisis")
    
    # Reutilizamos la gestión de topics del main original 
    topics = manage_topics("hype")  # Pasamos un prefijo único para este módulo
    
    # Configuración adicional del Hype Cycle
    with st.expander("⚙️ Opciones avanzadas", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            min_year = st.number_input(
                "Año mínimo",
                min_value=2010,
                max_value=2025,
                value=2015,
                help="Año desde el cual buscar resultados"
            )
        with col2:
            sources_filter = st.multiselect(
                "Filtrar fuentes",
                options=["Tech News", "Business News", "Academic Sources", "Blogs"],
                default=["Tech News", "Business News"],
                help="Tipos de fuentes a incluir en el análisis"
            )
    
    # Botón de análisis
    if st.button("📊 Analizar Hype Cycle", type="primary"):
        if not topics:
            st.error("Por favor, ingresa al menos un tema para analizar")
            return
            
        if not st.session_state.serp_api_key:
            st.error("Se requiere una API key de SerpAPI. Por favor, configúrala en el panel lateral.")
            return
            
        with st.spinner("🔄 Analizando el Hype Cycle..."):
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
                    # Mostrar fase actual
                    st.success(f"**Fase Actual:** {hype_data['phase']}")
                    
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
                    
                    st.info(phase_descriptions[hype_data['phase']])
                    
                    # Mostrar gráfico del Hype Cycle
                    fig = news_analyzer.plot_hype_cycle(hype_data, topics)
                    st.plotly_chart(fig, use_container_width=True)

                    # Análisis de puntos de inflexión
                    st.write("### 📊 Análisis de Puntos de Inflexión")
                    inflection_points = news_analyzer.analyze_gartner_points(hype_data['yearly_stats'])
                    inflection_fig = news_analyzer.plot_gartner_analysis(
                        hype_data['yearly_stats'], 
                        inflection_points
                    )
                    st.plotly_chart(inflection_fig, use_container_width=True)
                    
                    # Mostrar resultados detallados
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
        
        # Mostrar imagen del modelo de Gartner
        st.image("https://www.gartner.com/imagesrv/newsroom/images/hype-cycle-pr.png", 
                caption="Modelo del Hype Cycle de Gartner", width=600)

# Funciones auxiliares reutilizadas del script principal
def manage_topics(prefix="hype"):
    """Maneja la adición y eliminación de topics con opciones avanzadas."""
    # Usar un estado específico para este módulo
    state_key = f"{prefix}_topics_data"
    if state_key not in st.session_state:
        st.session_state[state_key] = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    
    # El resto de la función sigue igual, pero usando el state_key específico
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