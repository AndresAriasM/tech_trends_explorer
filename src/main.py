# src/main.py
import streamlit as st
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import base64
import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO
from analysis import QueryBuilder, ResultAnalyzer, NewsAnalyzer

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Tech Trends Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 2rem;
        }
        .stButton button {
            width: 100%;
        }
        .results-container {
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.5rem;
        }
        .metric-card {
            background-color: white;
            padding: 1rem;
            border-radius: 0.3rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    </style>
""", unsafe_allow_html=True)

def test_api_connection(api_key, search_engine_id):
    try:
        print(f"Probando con API Key: {api_key[:10]}... y Search Engine ID: {search_engine_id}")
        service = build("customsearch", "v1", developerKey=api_key)
        print("Servicio creado exitosamente")
        
        result = service.cse().list(
            q="test",
            cx=search_engine_id,
            num=1
        ).execute()
        print("Búsqueda ejecutada exitosamente")
        print(f"Resultados obtenidos: {result.get('searchInformation', {}).get('totalResults', 'N/A')}")
        
        return True, "Conexión exitosa"
    except HttpError as e:
        error_details = str(e)
        print(f"Error HTTP: {error_details}")
        
        if e.resp.status == 403:
            return False, f"Error de autenticación (403): {error_details}"
        elif e.resp.status == 400:
            return False, f"Error en la configuración (400): {error_details}"
        else:
            return False, f"Error HTTP {e.resp.status}: {error_details}"
    except Exception as e:
        print(f"Error inesperado: {str(e)}")
        return False, f"Error inesperado: {str(e)}"

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
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ''
    if 'search_results' not in st.session_state:
        st.session_state.search_results = filtered_results

    # Función para actualizar resultados de búsqueda
    def update_search():
        if st.session_state.search_query:
            st.session_state.search_results = [
                result for result in filtered_results
                if st.session_state.search_query.lower() in result['title'].lower()
            ]
        else:
            st.session_state.search_results = filtered_results

    # Mostrar ecuaciones de búsqueda
    with st.expander("📝 Ver ecuaciones de búsqueda", expanded=True):
        st.write("##### Ecuación de búsqueda en Google:")
        st.code(query_info['google_query'])
        st.write("##### Ecuación de búsqueda en Scopus:")
        st.code(query_info['scopus_query'])


    # Mostrar conteo de resultados
    total_results = len(processed_results)
    filtered_count = len(filtered_results)
    search_count = len(st.session_state.search_results)

    if filtered_count < total_results:
        st.info(f"Mostrando {filtered_count} de {total_results} resultados (filtrados por tipo de contenido)")
    
    if st.session_state.search_query:
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


    # Métricas y gráficos...
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
    if 'pdf_generated' not in st.session_state:
        st.session_state.pdf_generated = False
    
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
                key=f"download_pdf_{datetime.now().timestamp()}",  # Key única
                on_click=lambda: setattr(st.session_state, 'pdf_generated', True)
            ):
                pass  # No hacemos nada aquí para evitar recargas
                
        except Exception as e:
            st.error(f"Error al generar el PDF: {str(e)}")
    
    with download_col2:
        if st.session_state.pdf_generated:
            st.success("✅ Listo!")

    # Mostrar resultados
    st.write("### 📑 Resultados detallados")
    results_to_show = st.session_state.search_results
    
    if not results_to_show:
        st.warning("No se encontraron resultados que coincidan con los filtros actuales")
    else:
        for result in results_to_show:
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

def sidebar_config():
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # API Keys
        api_key = st.text_input(
            "Google API Key",
            value=os.getenv('GOOGLE_API_KEY', ''),
            type="password"
        )
        search_engine_id = st.text_input(
            "Search Engine ID",
            value=os.getenv('SEARCH_ENGINE_ID', ''),
            type="password"
        )
        
        # Botón de prueba
        if st.button("🔄 Probar conexión"):
            with st.spinner("Probando conexión..."):
                success, message = test_api_connection(api_key, search_engine_id)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        st.divider()
        
        # Configuración de búsqueda
        st.subheader("🔍 Configuración de Búsqueda")
        
        # Número de resultados con mayor rango
        col1, col2 = st.columns(2)
        with col1:
            max_results = st.number_input(
                "Número de resultados",
                min_value=10,
                max_value=1000,
                value=10,
                step=10,
                help="Número total de resultados a obtener"
            )
        with col2:
            st.info(f"Consultas API: {(max_results + 9) // 10}")
        
        st.caption("⚠️ Cada 10 resultados = 1 consulta API")
        
        # Filtros de contenido
        st.subheader("📑 Tipos de Contenido")
        content_types = {
            'academic': st.checkbox("Artículos Académicos", value=True, help="Incluir resultados de Google Scholar y otros repositorios académicos"),
            'news': st.checkbox("Noticias", value=True, help="Incluir artículos de noticias y medios"),
            'pdfs': st.checkbox("PDFs", value=True, help="Incluir documentos PDF"),
            'patents': st.checkbox("Patentes", value=True, help="Incluir patentes relacionadas")
        }
        
        # Filtros temporales
        st.divider()
        st.subheader("📅 Filtros Temporales")
        
        min_year = st.number_input(
            "Año mínimo",
            min_value=2000,
            max_value=2024,
            value=2019,
            help="Año desde el cual buscar resultados"
        )
        
        st.divider()
        api_usage = st.expander("ℹ️ Información de uso de API")
        with api_usage:
            st.write("""
            - Límite diario gratuito: 100 consultas
            - Cada 10 resultados = 1 consulta
            - Máximo 100 resultados por consulta
            - Los resultados se obtienen en páginas de 10
            """)
        
        return {
            'api_key': api_key,
            'search_engine_id': search_engine_id,
            'min_year': min_year,
            'content_types': content_types,
            'max_results': max_results
        }
        

def perform_advanced_search(api_key, search_engine_id, query, config):
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        all_results = []
        seen_urls = set()  # Para controlar duplicados
        desired_results = config['desired_results']
        
        # Modificar query según tipos de contenido permitidos
        content_filters = []
        if config['content_types']['academic']:
            content_filters.extend(['site:scholar.google.com OR site:sciencedirect.com OR site:springer.com OR site:ieee.org'])
        if config['content_types']['news']:
            content_filters.extend(['site:news.google.com OR site:reuters.com OR site:bloomberg.com'])
        if config['content_types']['pdfs']:
            content_filters.append('filetype:pdf')
        if not config['content_types']['patents']:
            content_filters.append('-patent')
        
        # Agregar filtros al query original
        if content_filters:
            query = f"({query}) AND ({' OR '.join(content_filters)})"
        
        # Iniciar barra de progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        page = 1
        total_requested = 0
        
        while len(all_results) < desired_results:
            try:
                start_index = ((page - 1) * 10) + 1
                
                status_text.text(f"Buscando resultados... (Página {page})")
                
                result = service.cse().list(
                    q=query,
                    cx=search_engine_id,
                    num=10,
                    start=start_index
                ).execute()
                
                items = result.get('items', [])
                if not items:
                    break
                
                # Filtrar duplicados y procesar resultados
                for item in items:
                    url = item.get('link')
                    if url not in seen_urls:
                        # Verificar tipo de contenido
                        content_type = classify_content_type(item)
                        if should_include_content(content_type, config['content_types']):
                            seen_urls.add(url)
                            all_results.append(item)
                
                # Actualizar progreso
                progress = min(len(all_results) / desired_results, 1.0)
                progress_bar.progress(progress)
                
                page += 1
                total_requested += 10
                
                # Evitar exceder límites de API
                if total_requested >= 100:  # Límite de 100 resultados por consulta
                    status_text.warning("Se alcanzó el límite de resultados por consulta")
                    break
                
            except Exception as e:
                status_text.error(f"Error en página {page}: {str(e)}")
                break
        
        status_text.text(f"Búsqueda completada. Se encontraron {len(all_results)} resultados únicos.")
        return True, all_results
        
    except Exception as e:
        return False, str(e)

def classify_content_type(item):
    """Clasificar el tipo de contenido basado en la URL y metadata"""
    url = item.get('link', '').lower()
    title = item.get('title', '').lower()
    
    if 'pdf' in url or any(mime in item.get('mime', '') for mime in ['pdf', 'application/pdf']):
        return 'PDF'
    elif any(domain in url for domain in ['scholar.google', 'sciencedirect', 'springer', 'ieee']):
        return 'Academic'
    elif any(domain in url for domain in ['news.google', 'reuters', 'bloomberg']):
        return 'News'
    elif any(term in url for term in ['patent', 'uspto.gov', 'espacenet']):
        return 'Patent'
    else:
        return 'Web Page'

def should_include_content(content_type, content_types):
    """Determinar si un tipo de contenido debe ser incluido según la configuración"""
    mapping = {
        'Web Page': 'web_pages',
        'Academic': 'academic',
        'News': 'news',
        'PDF': 'pdfs',
        'Patent': 'patents'
    }
    return content_types.get(mapping.get(content_type, 'web_pages'), False)

def main():    
    # Cargar configuración
    config = sidebar_config()
    
    # Encabezado principal
    st.markdown('<p class="main-header">🔍 Tech Trends Explorer</p>', unsafe_allow_html=True)
    
    # Verificar configuración
    if not config['api_key'] or not config['search_engine_id']:
        st.warning("⚠️ Por favor, configura las credenciales de API en el sidebar")
        return
    
    # Área de búsqueda
    st.write("### 🎯 Define tus términos de búsqueda")
    
    # Topics dinámicos
    if 'num_topics' not in st.session_state:
        st.session_state.num_topics = 1
    
    topics = []
    for i in range(st.session_state.num_topics):
        topic = st.text_input(
            f"Tema {i+1}",
            key=f"topic_{i}",
            placeholder="Ej: Artificial Intelligence, Blockchain, IoT..."
        )
        topics.append(topic)
    
    if st.button("+ Añadir otro tema"):
        st.session_state.num_topics += 1
        st.rerun()
    
    # Botón de búsqueda
    if st.button("🔍 Analizar Tendencias", type="primary"):
        if not any(topics):
            st.error("Por favor, ingresa al menos un tema para buscar")
            return
            
        with st.spinner("🔄 Analizando tendencias tecnológicas..."):
            # Construir queries
            query_builder = QueryBuilder()
            google_query = query_builder.build_google_query(
                topics, 
                config['min_year'],
                config['content_types']['patents']
            )
            scopus_query = query_builder.build_scopus_query(
                topics, 
                config['min_year']
            )
            
            # Crear el objeto query_info
            query_info = {
                'google_query': google_query,
                'scopus_query': scopus_query,
                'topics': topics,
                'min_year': config['min_year'],
                'content_types': config['content_types']
            }
            
            # Realizar búsqueda general
            success, results = perform_search(
                config['api_key'], 
                config['search_engine_id'], 
                google_query,
                config['content_types'],
                max_results=config['max_results']
            )
            
            # Realizar búsqueda específica de noticias para Hype Cycle
            news_analyzer = NewsAnalyzer()
            news_success, news_results = news_analyzer.perform_news_search(
                config['api_key'],
                config['search_engine_id'],
                google_query
            )
    
            if not success:
                st.error(f"Error al realizar la búsqueda: {results}")
                return
            
            if not results:
                st.warning("No se encontraron resultados para tu búsqueda")
                return
            
            # Después de realizar las búsquedas y antes de crear las pestañas
            if news_success and news_results:
                hype_data = news_analyzer.analyze_hype_cycle(news_results)
                hype_figures = {}
                
                # Crear las figuras del Hype Cycle
                hype_figures['Hype Cycle'] = news_analyzer.plot_hype_cycle(hype_data) 
                
                # Crear figuras de análisis temporal
                yearly_stats = hype_data['yearly_stats']

                inflection_points = news_analyzer.analyze_gartner_points(yearly_stats)
                
                hype_figures['Análisis de Puntos de Inflexión'] = news_analyzer.plot_gartner_analysis(yearly_stats, inflection_points)
                
                # Figura de menciones
                mentions_fig = px.bar(
                    yearly_stats,
                    x='year',
                    y='mention_count',
                    title="Evolución de Menciones por Año"
                )
                mentions_fig.update_layout(
                    xaxis_title="Año",
                    yaxis_title="Número de Menciones",
                    showlegend=True
                )
                hype_figures['Menciones por Año'] = mentions_fig
                
                # Figura de sentimiento
                sentiment_fig = px.line(
                    yearly_stats,
                    x='year',
                    y='sentiment_mean',
                    title="Evolución del Sentimiento"
                )
                sentiment_fig.update_layout(
                    xaxis_title="Año",
                    yaxis_title="Sentimiento Promedio",
                    showlegend=True
                )
                hype_figures['Evolución del Sentimiento'] = sentiment_fig

            # Crear pestañas
            tab1, tab2 = st.tabs(["📊 Análisis General", "📈 Análisis Hype Cycle"])

            with tab1:
                # Mostrar análisis general con botón de descarga
                show_analysis_results(
                    results=results, 
                    query_info=query_info, 
                    search_topics=topics,  # Cambio 'topics' por 'search_topics'
                    content_types=config['content_types'],
                    hype_data=hype_data if news_success else None,
                    hype_figures=hype_figures if news_success else {}
                )
            
            with tab2:
                st.write("### 📈 Análisis del Hype Cycle de Gartner")
                
                # Explicación del Hype Cycle
                with st.expander("ℹ️ ¿Qué es el Hype Cycle?", expanded=True):
                    st.markdown("""
                    El **Hype Cycle de Gartner** es una representación gráfica de la madurez y adopción de tecnologías específicas. 
                    Se compone de cinco fases principales:
                    
                    1. **Innovation Trigger (Disparador de Innovación)**
                       - Primera aparición pública de la tecnología
                       - Alto nivel de especulación
                       - Poca implementación práctica
                    
                    2. **Peak of Inflated Expectations (Pico de Expectativas Infladas)**
                       - Máxima publicidad y expectativas
                       - Muchas historias de éxito y fracaso
                       - Alto interés mediático
                    
                    3. **Trough of Disillusionment (Valle de la Desilusión)**
                       - Disminución del interés
                       - Fracasos y desafíos documentados
                       - Menor cobertura mediática
                    
                    4. **Slope of Enlightenment (Pendiente de la Iluminación)**
                       - Comprensión más realista
                       - Beneficios reales documentados
                       - Mejores prácticas establecidas
                    
                    5. **Plateau of Productivity (Meseta de la Productividad)**
                       - Adopción generalizada
                       - Relevancia y rol claramente establecidos
                       - Tecnología madura y estable
                    """)
                
                if news_success and news_results:
                    # Análisis del Hype Cycle
                    hype_data = news_analyzer.analyze_hype_cycle(news_results)
                    
                    # Fase actual con explicación detallada
                    st.write("#### 🎯 Fase Actual")
                    current_phase = hype_data['phase']
                    
                    # Descripción específica según la fase
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
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.success(f"**Fase Actual:** {current_phase}")
                    with col2:
                        st.info(phase_descriptions[current_phase])
                    
                    # Gráfico del Hype Cycle
                    st.write("#### 📊 Visualización del Hype Cycle")
                    fig = news_analyzer.plot_hype_cycle(hype_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Métricas clave
                    st.write("#### 📈 Métricas de Análisis")
                    col1, col2, col3 = st.columns(3)
                    
                    yearly_stats = hype_data['yearly_stats']
                    with col1:
                        total_mentions = yearly_stats['mention_count'].sum()
                        st.metric("Total de Menciones", total_mentions)
                    with col2:
                        avg_sentiment = yearly_stats['sentiment_mean'].mean()
                        sentiment_label = "Positivo" if avg_sentiment > 0 else "Negativo"
                        st.metric("Sentimiento Promedio", f"{avg_sentiment:.2f} ({sentiment_label})")
                    with col3:
                        trend = yearly_stats['mention_count'].pct_change().mean()
                        trend_label = "Creciente" if trend > 0 else "Decreciente"
                        st.metric("Tendencia", f"{trend_label} ({trend:.1%})")
                    
                    # Análisis de puntos de inflexión
                    st.write("#### 📊 Análisis de Puntos de Inflexión")
                    inflection_points = news_analyzer.analyze_gartner_points(yearly_stats)
                    fig_inflection = news_analyzer.plot_gartner_analysis(yearly_stats, inflection_points)
                    st.plotly_chart(fig_inflection, use_container_width=True)
                    
                    # Mostrar detalles de los puntos de inflexión
                    st.write("#### 📋 Detalles de los Puntos de Inflexión")
                    for phase, point in inflection_points.items():
                        if point:
                            with st.expander(f"💡 {phase.replace('_', ' ').title()}"):
                                st.write(f"**Año:** {point['year']}")
                                st.write(f"**Menciones:** {point['mentions']}")
                                st.write(f"**Sentimiento:** {point['sentiment']:.2f}")
                                
                                # Agregar explicación según la fase
                                if phase == 'innovation_trigger':
                                    st.info("Primera aparición significativa en medios")
                                elif phase == 'peak':
                                    st.info("Punto de máxima atención mediática")
                                elif phase == 'trough':
                                    st.info("Punto de menor interés post-pico")
                    
                    # Análisis temporal detallado
                    st.write("#### 📅 Análisis Temporal")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        mentions_fig = px.bar(
                            yearly_stats,
                            x='year',
                            y='mention_count',
                            title="Evolución de Menciones por Año"
                        )
                        mentions_fig.update_layout(
                            xaxis_title="Año",
                            yaxis_title="Número de Menciones",
                            showlegend=True
                        )
                        st.plotly_chart(mentions_fig, use_container_width=True)
                    
                    with col2:
                        sentiment_fig = px.line(
                            yearly_stats,
                            x='year',
                            y='sentiment_mean',
                            title="Evolución del Sentimiento"
                        )
                        sentiment_fig.update_layout(
                            xaxis_title="Año",
                            yaxis_title="Sentimiento Promedio",
                            showlegend=True
                        )
                        st.plotly_chart(sentiment_fig, use_container_width=True)
                    
                    # Evidencia de noticias
                    st.write("#### 📰 Evidencia en Medios")
                    st.write("Noticias más relevantes que respaldan el análisis:")
                    
                    for result in hype_data['results'][:5]:
                        with st.expander(f"📄 {result['title']}", expanded=False):
                            st.markdown(f"**Resumen:** {result['text']}")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.markdown(f"🔗 [Leer noticia completa]({result['link']})")
                                st.write(f"📅 Año: {result['year']}")
                            with col2:
                                sentiment = result['sentiment']
                                sentiment_color = 'green' if sentiment > 0 else 'red'
                                st.markdown(f"💭 Sentimiento: <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", 
                                          unsafe_allow_html=True)
                    
                    # Conclusiones y recomendaciones
                    st.write("#### 🎯 Conclusiones y Recomendaciones")
                    conclusions = {
                        "Innovation Trigger": """
                            - **Oportunidad:** Momento ideal para investigación y desarrollo
                            - **Riesgo:** Alto nivel de incertidumbre
                            - **Recomendación:** Monitorear avances y casos de uso pioneros
                        """,
                        "Peak of Inflated Expectations": """
                            - **Oportunidad:** Alta visibilidad y interés del mercado
                            - **Riesgo:** Posibles expectativas irrealistas
                            - **Recomendación:** Evaluar casos de uso específicos y ROI
                        """,
                        "Trough of Disillusionment": """
                            - **Oportunidad:** Evaluación realista de beneficios
                            - **Riesgo:** Pérdida de interés y apoyo
                            - **Recomendación:** Focalizarse en casos de uso probados
                        """,
                        "Slope of Enlightenment": """
                            - **Oportunidad:** Implementación con beneficios claros
                            - **Riesgo:** Competencia creciente
                            - **Recomendación:** Desarrollar estrategias de adopción
                        """,
                        "Plateau of Productivity": """
                            - **Oportunidad:** Tecnología probada y estable
                            - **Riesgo:** Commoditización
                            - **Recomendación:** Optimizar implementación y costos
                        """
                    }
                    st.markdown(conclusions[current_phase])
                    
                else:
                    st.warning("No se encontraron suficientes datos para realizar el análisis del Hype Cycle")
    
if __name__ == "__main__":
    main()