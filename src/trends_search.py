# src/trends_search.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from googleapiclient.discovery import build
from analysis import QueryBuilder, ResultAnalyzer  # Aseguramos que QueryBuilder esté importado correctamente
from config import CONFIG

def run_trend_search():
    """
    Ejecuta la funcionalidad de búsqueda de tendencias usando Google Custom Search
    """
    st.markdown('<p class="tab-subheader">🔍 Búsqueda de Tendencias</p>', unsafe_allow_html=True)
    
    st.write("""
    Esta herramienta te permite analizar tendencias tecnológicas utilizando Google Custom Search.
    Los resultados se analizan para identificar patrones, fuentes principales, y distribución temporal.
    """)
    
    # Gestión de topics
    st.write("### 🎯 Define tus términos de búsqueda")
    topics = manage_topics("trends")  # Pasamos un prefijo único para este módulo
    
    # Configuración de búsqueda en el sidebar
    with st.expander("⚙️ Opciones de búsqueda", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.number_input(
                "Número de resultados",
                min_value=10,
                max_value=1000,
                value=50,
                step=10,
                help="Número total de resultados a obtener"
            )
        with col2:
            min_year = st.number_input(
                "Año mínimo",
                min_value=1970,
                max_value=2025,
                value=2014,
                help="Año desde el cual buscar resultados"
            )
        
        st.write("### 📑 Tipos de Contenido")
        col1, col2 = st.columns(2)
        with col1:
            academic = st.checkbox("Artículos Académicos", value=True, 
                                  help="Incluir resultados de Google Scholar y otros repositorios académicos")
            news = st.checkbox("Noticias", value=True, 
                              help="Incluir artículos de noticias y medios")
        with col2:
            pdfs = st.checkbox("PDFs", value=True, 
                              help="Incluir documentos PDF")
            patents = st.checkbox("Patentes", value=True, 
                                 help="Incluir patentes relacionadas")
        
        content_types = {
            'academic': academic,
            'news': news,
            'pdfs': pdfs,
            'patents': patents
        }
    
    # Botón de análisis
    if st.button("🔍 Analizar Tendencias", type="primary", key="trends_analyze_button"):  # Key única
        if not topics:
            st.error("Por favor, ingresa al menos un tema para buscar")
            return
            
        if not st.session_state.google_api_key or not st.session_state.search_engine_id:
            st.error("Se requiere Google API Key y Search Engine ID. Por favor, configúralos en el panel lateral.")
            return
            
        with st.spinner("🔄 Analizando tendencias tecnológicas..."):
            # Inicializar analizadores
            query_builder = QueryBuilder()
            
            # Construir query principal
            processed_topics = process_topics(topics)
            google_query = query_builder.build_google_query(
                processed_topics,
                min_year,
                content_types.get('patents', True)
            )
            
            # Crear objeto de información de queries
            query_info = {
                'google_query': google_query,
                'scopus_query': query_builder.build_scopus_query(topics, min_year),
                'search_query': google_query,
                'time_range': f"{min_year}-{datetime.now().year}"
            }
            
            # Realizar búsqueda general con Google API
            success, general_results = perform_search(
                st.session_state.google_api_key,
                st.session_state.search_engine_id,
                google_query,
                content_types,
                max_results=max_results
            )
            
            if success:
                # Mostrar resultados generales
                show_analysis_results(
                    results=general_results,
                    query_info=query_info,
                    search_topics=topics,
                    content_types=content_types
                )
            else:
                st.error("Error en la búsqueda general de tendencias")
    else:
        # Mostrar mensaje informativo cuando no hay búsqueda
        st.info("""
        ### 🚀 Cómo comenzar:
        
        1. Define tus términos de búsqueda en el formulario superior
        2. Configura las opciones de búsqueda según tus necesidades
        3. Haz clic en "Analizar Tendencias" para iniciar el análisis
        
        Los resultados te mostrarán:
        - Distribución por tipo de contenido
        - Tendencia temporal
        - Palabras clave más frecuentes
        - Resultados detallados con enlaces
        """)

# Funciones auxiliares importadas del script principal
# Modificación para manage_topics() en trends_search.py

def manage_topics(prefix="trends"):
    """Maneja la adición y eliminación de topics con opciones avanzadas."""
    # Usar un estado específico para este módulo
    state_key = f"{prefix}_topics_data"
    if state_key not in st.session_state:
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

def perform_search(api_key, search_engine_id, query, content_types, max_results=100):
    """
    Realiza búsquedas en Google Custom Search respetando el límite de resultados especificado
    """
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        all_results = []
        
        # Construir filtros basados en los tipos de contenido seleccionados
        type_filters = []
        base_query = query

        if content_types.get('academic', False):
            type_filters.append('site:scholar.google.com OR site:sciencedirect.com OR site:springer.com OR site:ieee.org')
        if content_types.get('news', False):
            type_filters.append('site:news.google.com OR site:reuters.com OR site:bloomberg.com')
        if content_types.get('pdfs', False):
            type_filters.append('filetype:pdf')
        if not content_types.get('patents', False):
            base_query += ' -patent -uspto.gov -espacenet'

        # Combinar query con filtros
        if type_filters:
            final_query = f"({base_query}) AND ({' OR '.join(type_filters)})"
        else:
            final_query = base_query

        # Calcular número total de páginas necesarias
        total_pages = (max_results + 9) // 10
        
        # Debug: Mostrar información de la búsqueda
        st.write(f"Buscando hasta {max_results} resultados en {total_pages} páginas")
        
        # Crear barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for page in range(total_pages):
            start_index = (page * 10) + 1
            
            # Actualizar estado
            progress = (page + 1) / total_pages
            progress_bar.progress(progress)
            status_text.text(f"Obteniendo resultados... Página {page + 1} de {total_pages} ({len(all_results)} resultados encontrados)")
            
            try:
                result = service.cse().list(
                    q=final_query,
                    cx=search_engine_id,
                    num=min(10, max_results - len(all_results)),  # No pedir más de los necesarios
                    start=start_index
                ).execute()
                
                items = result.get('items', [])
                if not items:
                    break
                    
                all_results.extend(items)
                
                # Verificar si ya tenemos suficientes resultados
                if len(all_results) >= max_results:
                    all_results = all_results[:max_results]
                    break
                
            except Exception as e:
                st.warning(f"Error en página {page + 1}: {str(e)}")
                break  # Si hay error, detenemos la búsqueda
        
        # Limpiar indicadores de progreso
        progress_bar.empty()
        status_text.empty()
        
        # Informar resultados obtenidos
        st.info(f"Búsqueda completada: {len(all_results)} resultados encontrados")
        
        return True, all_results
    except Exception as e:
        st.error(f"Error en la búsqueda: {str(e)}")
        return False, str(e)

def show_analysis_results(results, query_info, search_topics, content_types, hype_data=None, hype_figures=None):
    """
    Muestra los resultados del análisis de tendencias
    """
    # Crear instancia del analizador
    analyzer = ResultAnalyzer()
    processed_results, stats = analyzer.analyze_results(results, search_topics)
    
    # Función para determinar el tipo de resultado
    def get_result_type(result):
        url = result['link'].lower()
        title = result['title'].lower()
        
        if any(term in url for term in ['.pdf', '/pdf/']):
            return 'pdf'
        elif any(term in url for term in ['patent', 'uspto.gov', 'espacenet']):
            return 'patent'
        elif any(term in url for term in ['scholar.google', 'sciencedirect', 'springer', 'ieee']):
            return 'academic'
        elif any(term in url for term in ['news.google', 'reuters', 'bloomberg']):
            return 'news'
        return 'web'

    # Filtrar resultados según las selecciones del sidebar
    filtered_results = []
    for result in processed_results:
        result_type = get_result_type(result)
        
        # Aplicar filtros según el tipo
        if (
            (result_type == 'pdf' and content_types.get('pdfs', True)) or
            (result_type == 'academic' and content_types.get('academic', True)) or
            (result_type == 'news' and content_types.get('news', True)) or
            (result_type == 'patent' and content_types.get('patents', True)) or
            (result_type == 'web')  # Web siempre se muestra
        ):
            filtered_results.append(result)

    # Estados iniciales para la búsqueda en resultados
    if 'trends_search_query' not in st.session_state:  # Usar nombre específico para evitar conflictos
        st.session_state.trends_search_query = ''
    if 'trends_search_results' not in st.session_state:  # Usar nombre específico para evitar conflictos
        st.session_state.trends_search_results = filtered_results

    # Mostrar ecuaciones de búsqueda
    with st.expander("📝 Ver ecuaciones de búsqueda", expanded=True):
        st.write("##### Ecuación de búsqueda en Google:")
        st.code(query_info['google_query'])
        st.write("##### Ecuación de búsqueda en Scopus:")
        st.code(query_info['scopus_query'])

    # Mostrar conteo de resultados
    total_results = len(processed_results)
    filtered_count = len(filtered_results)
    search_count = len(st.session_state.trends_search_results)

    if filtered_count < total_results:
        st.info(f"Mostrando {filtered_count} de {total_results} resultados (filtrados por tipo de contenido)")
    
    if st.session_state.trends_search_query:
        st.info(f"Encontrados {search_count} resultados que coinciden con la búsqueda")
    
    # Calcular distribución de tipos de contenido
    content_distribution = {}
    for result in filtered_results:
        result_type = get_result_type(result).upper()
        content_distribution[result_type] = content_distribution.get(result_type, 0) + 1

    # Gráficos de distribución
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 📊 Distribución por tipo de contenido")
        fig_type = px.pie(
            values=list(content_distribution.values()),
            names=list(content_distribution.keys()),
            title=f"Distribución de {filtered_count} resultados por tipo"
        )
        fig_type.update_traces(
            textposition='inside',
            textinfo='percent+label+value',
            marker_colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Colores personalizados
        )
        fig_type.update_layout(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_type, use_container_width=True)
        
        # Mostrar tabla de distribución bajo el gráfico
        st.write("#### Detalle de distribución:")
        distribution_data = pd.DataFrame(
            {
                'Tipo': list(content_distribution.keys()),
                'Cantidad': list(content_distribution.values()),
                'Porcentaje': [f"{(v/filtered_count)*100:.1f}%" for v in content_distribution.values()]
            }
        )
        st.dataframe(distribution_data, hide_index=True)
    
    with col2:
        st.write("### 📈 Tendencia temporal")
        if stats['by_year']:
            fig_year = px.bar(
                x=list(stats['by_year'].keys()),
                y=list(stats['by_year'].values()),
                title="Distribución por año"
            )
            fig_year.update_xaxes(title="Año")
            fig_year.update_yaxes(title="Cantidad")
            st.plotly_chart(fig_year, use_container_width=True)
    
    # Gráfico de palabras clave
    st.write("### 🏷️ Palabras clave más frecuentes")
    if stats['common_keywords']:
        keywords_df = pd.DataFrame(stats['common_keywords'], columns=['Keyword', 'Count'])
        fig_keywords = px.bar(
            keywords_df.head(10),
            x='Keyword',
            y='Count',
            title="Términos más frecuentes en los resultados"
        )
        fig_keywords.update_layout(
            xaxis_title="Palabras clave",
            yaxis_title="Frecuencia",
            showlegend=False
        )
        st.plotly_chart(fig_keywords, use_container_width=True)
    
    # Botón de descarga
    st.write("### 📥 Exportar Resultados")
    
    # Combinar todas las figuras
    all_figures = {
        "Distribución por Tipo": fig_type,
        "Tendencia Temporal": fig_year,
        "Palabras Clave": fig_keywords
    }
    
    # Agregar figuras del Hype Cycle si existen
    if hype_figures:
        all_figures.update(hype_figures)
    
    # Usar session state para evitar recargas
    if 'trends_pdf_generated' not in st.session_state:  # Usar nombre específico para evitar conflictos
        st.session_state.trends_pdf_generated = False
    
    download_col1, download_col2 = st.columns([3, 1])
    
    with download_col1:
        try:
            pdf_data = export_to_pdf(
                results=results,
                query_info=query_info,
                hype_data=hype_data,
                figures=all_figures
            )
            
            if st.download_button(
                label="📥 Descargar Informe Completo (PDF)",
                data=pdf_data,
                file_name=f"analisis_tendencias_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key=f"trends_download_pdf_{datetime.now().timestamp()}",  # Key única con prefijo
                on_click=lambda: setattr(st.session_state, 'trends_pdf_generated', True)
            ):
                pass  # No hacemos nada aquí para evitar recargas
                
        except Exception as e:
            st.error(f"Error al generar el PDF: {str(e)}")
    
    with download_col2:
        if st.session_state.trends_pdf_generated:
            st.success("✅ Listo!")

    # Mostrar resultados
    st.write("### 📑 Resultados detallados")
    results_to_show = st.session_state.trends_search_results
    
    if not results_to_show:
        st.warning("No se encontraron resultados que coincidan con los filtros actuales")
    else:
        for i, result in enumerate(results_to_show):
            with st.expander(f"📄 {result['title']}", expanded=False):
                col1, col2 = st.columns([2,1])
                with col1:
                    st.markdown("**Descripción:**")
                    st.write(result['snippet'])
                    st.markdown(f"🔗 [Ver documento completo]({result['link']})")
                with col2:
                    st.markdown("**Detalles:**")
                    st.markdown(f"📅 **Año:** {result.get('year', 'No especificado')}")
                    st.markdown(f"🌍 **País:** {result.get('country', 'No especificado')}")
                    tipo = get_result_type(result).upper()
                    st.markdown(f"📊 **Tipo:** {tipo}")

def export_to_pdf(results, query_info, hype_data, figures, news_results=None):
    """Exporta los resultados a un archivo PDF"""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from io import BytesIO
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Estilos personalizados
    styles.add(ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    ))
    
    styles.add(ParagraphStyle(
        'SectionTitle',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=15,
        spaceAfter=10
    ))

    # Título y fecha
    story.append(Paragraph('Análisis de Tendencias Tecnológicas', styles['CustomTitle']))
    story.append(Paragraph(f'Generado el {datetime.now().strftime("%d-%m-%Y %H:%M")}', styles['Normal']))
    story.append(Spacer(1, 20))

    # Ecuaciones de búsqueda
    story.append(Paragraph('Ecuaciones de Búsqueda', styles['SectionTitle']))
    story.append(Paragraph(f'<b>Google:</b> {query_info["google_query"]}', styles['Normal']))
    story.append(Paragraph(f'<b>Scopus:</b> {query_info["scopus_query"]}', styles['Normal']))
    story.append(Spacer(1, 20))

    # Métricas y estadísticas
    story.append(Paragraph('Estadísticas Generales', styles['SectionTitle']))
    story.append(Paragraph(f'Total de resultados encontrados: {len(results)}', styles['Normal']))
    story.append(Spacer(1, 12))

    # Gráficos de análisis general
    for name, fig in figures.items():
        story.append(Paragraph(name, styles['SectionTitle']))
        img_bytes = BytesIO()
        fig.write_image(img_bytes, format='png', width=600, height=400)
        img = Image(img_bytes, width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 12))

    # En la función export_to_pdf, después de los gráficos generales
    if hype_data and isinstance(hype_data, dict):
        story.append(Paragraph('Análisis del Hype Cycle de Gartner', styles['SectionTitle']))
        story.append(Paragraph(f'<b>Fase Actual:</b> {hype_data["phase"]}', styles['Normal']))
        
        # Agregar todos los gráficos relacionados con Gartner
        for name, fig in figures.items():
            if any(term in name.lower() for term in ['hype', 'menciones', 'sentimiento']):
                story.append(Paragraph(name, styles['SectionTitle']))
                img_bytes = BytesIO()
                fig.write_image(img_bytes, format='png', width=600, height=400)
                img = Image(img_bytes, width=6*inch, height=4*inch)
                story.append(img)
                story.append(Spacer(1, 12))
                
    # Análisis del Hype Cycle
    if hype_data:
        story.append(Paragraph('Análisis del Hype Cycle de Gartner', styles['SectionTitle']))
        story.append(Paragraph(f'<b>Fase Actual:</b> {hype_data["phase"]}', styles['Normal']))
        
        # Descripción de la fase
        phase_descriptions = {
            "Innovation Trigger": """La tecnología está en su fase inicial de innovación, caracterizada por:
            • Alto nivel de interés y especulación
            • Pocos casos de implementación real
            • Gran potencial percibido""",
            "Peak of Inflated Expectations": """La tecnología está en su punto máximo de expectativas, donde se observa:
            • Máxima cobertura mediática
            • Altas expectativas de mercado
            • Posible sobreestimación de capacidades""",
            "Trough of Disillusionment": """La tecnología está atravesando una fase de desilusión, caracterizada por:
            • Disminución del interés inicial
            • Identificación de limitaciones reales
            • Reevaluación de expectativas""",
            "Slope of Enlightenment": """La tecnología está madurando hacia una comprensión realista, donde se observa:
            • Casos de uso bien definidos
            • Beneficios comprobados
            • Adopción más estratégica""",
            "Plateau of Productivity": """La tecnología ha alcanzado un nivel de madurez estable, caracterizada por:
            • Adopción generalizada
            • Beneficios claramente demostrados
            • Implementación sistemática"""
        }
        story.append(Paragraph(phase_descriptions[hype_data['phase']], styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Métricas del Hype Cycle
        if 'yearly_stats' in hype_data:
            yearly_stats = hype_data['yearly_stats']
            story.append(Paragraph('Métricas de Análisis', styles['SectionTitle']))
            story.append(Paragraph(f"<b>Total de Menciones:</b> {yearly_stats['mention_count'].sum()}", styles['Normal']))
            avg_sentiment = yearly_stats['sentiment_mean'].mean()
            sentiment_label = "Positivo" if avg_sentiment > 0 else "Negativo"
            story.append(Paragraph(f"<b>Sentimiento Promedio:</b> {avg_sentiment:.2f} ({sentiment_label})", styles['Normal']))
            
            trend = yearly_stats['mention_count'].pct_change().mean()
            trend_label = "Creciente" if trend > 0 else "Decreciente"
            story.append(Paragraph(f"<b>Tendencia:</b> {trend_label} ({trend:.1%})", styles['Normal']))

    # Resultados detallados
    story.append(Paragraph('Resultados Detallados', styles['SectionTitle']))
    for result in results:
        story.append(Paragraph(result['title'], styles['Heading3']))
        story.append(Paragraph(result.get('snippet', 'No hay descripción disponible'), styles['Normal']))
        story.append(Paragraph(f'<link href="{result["link"]}">{result["link"]}</link>', styles['Normal']))
        if result.get('year'):
            story.append(Paragraph(f'<b>Año:</b> {result["year"]}', styles['Normal']))
        story.append(Spacer(1, 12))

    # Construir PDF
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()