# src/main.py - ACTUALIZADO con sistema de almacenamiento para Hype Cycle
import streamlit as st
import os
import json
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from config import CONFIG
from utils.api_helpers import test_api_connection, load_config_from_file
from trends_search import run_trend_search
from hype_cycle import run_hype_cycle_analysis
from s_curve import run_s_curve_analysis

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
        .welcome-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #4e8df5;
        }
        .tab-subheader {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            color: #1f77b4;
        }
        .api-config {
            background-color: #f0f8ff;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #cce5ff;
        }
        .aws-config {
            background-color: #fff3cd;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #ffeaa7;
        }
        .hype-storage-config {
            background-color: #e8f5e8;
            padding: 12px;
            border-radius: 8px;
            margin-bottom: 15px;
            border: 1px solid #c3e6c3;
        }
    </style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """
    Inicializa variables en el session state
    """
    # Credenciales de API
    if 'google_api_key' not in st.session_state:
        st.session_state.google_api_key = os.getenv('GOOGLE_API_KEY', '')
    if 'search_engine_id' not in st.session_state:
        st.session_state.search_engine_id = os.getenv('SEARCH_ENGINE_ID', '')
    if 'serp_api_key' not in st.session_state:
        st.session_state.serp_api_key = os.getenv('SERP_API_KEY', '')
    if 'scopus_api_key' not in st.session_state:
        st.session_state.scopus_api_key = os.getenv('SCOPUS_API_KEY', '')
    
    # Credenciales de AWS DynamoDB
    if 'aws_access_key_id' not in st.session_state:
        st.session_state.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', '')
    if 'aws_secret_access_key' not in st.session_state:
        st.session_state.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    if 'aws_region' not in st.session_state:
        st.session_state.aws_region = os.getenv('AWS_DEFAULT_REGION', 'us-east-2')
    
    # Configuración específica para Hype Cycle Storage
    if 'hype_storage_mode' not in st.session_state:
        st.session_state.hype_storage_mode = 'local'  # 'local' o 'dynamodb'
    if 'hype_auto_save' not in st.session_state:
        st.session_state.hype_auto_save = True
    if 'hype_default_category' not in st.session_state:
        st.session_state.hype_default_category = 'default'
    
    # Estado para tópicos de búsqueda - específicos para cada módulo
    if 'hype_topics_data' not in st.session_state:
        st.session_state.hype_topics_data = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    if 'trends_topics_data' not in st.session_state:
        st.session_state.trends_topics_data = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    if 'scurve_topics_data' not in st.session_state:
        st.session_state.scurve_topics_data = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    
    # Para compatibilidad con código existente
    if 'topics_data' not in st.session_state:
        st.session_state.topics_data = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    
    # Estado para resultados actuales
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'current_query_info' not in st.session_state:
        st.session_state.current_query_info = None
    
    # Estados específicos para Hype Cycle
    if 'hype_reuse_query' not in st.session_state:
        st.session_state.hype_reuse_query = None
    if 'hype_show_query_id' not in st.session_state:
        st.session_state.hype_show_query_id = None

def show_welcome_screen():
    """Muestra la pantalla de bienvenida"""
    st.markdown('<p class="main-header">🔍 Tech Trends Explorer</p>', unsafe_allow_html=True)
    
    # Tarjeta de bienvenida
    st.markdown("""
    <div class="welcome-card">
        <h2>Bienvenido al Explorador de Tendencias Tecnológicas</h2>
        <p>Esta herramienta te permite analizar y visualizar tendencias tecnológicas emergentes utilizando 
        múltiples fuentes de datos. Selecciona una funcionalidad en las pestañas de arriba para comenzar.</p>
        
        <h3>🆕 Nuevas Características v2.1:</h3>
        <ul>
            <li>✅ <strong>Almacenamiento automático de consultas Hype Cycle</strong></li>
            <li>✅ <strong>Sistema de categorías para organizar análisis</strong></li>
            <li>✅ <strong>Historial de consultas sin gastar tokens</strong></li>
            <li>✅ <strong>Reutilización de análisis previos</strong></li>
            <li>✅ <strong>Comparación entre fases del Hype Cycle</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Descripción de funcionalidades
    st.markdown("### 🛠️ Funcionalidades Principales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Búsqueda de Tendencias**
        
        Analiza tendencias tecnológicas usando Google Custom Search. Visualiza la distribución de resultados,
        palabras clave frecuentes y evolución temporal.
        
        *Requiere: Google API Key & Search Engine ID*
        """)
    
    with col2:
        st.markdown("""
        **📈 Análisis del Hype Cycle** ⭐ *MEJORADO*
        
        Determina la posición de tecnologías dentro del ciclo de sobreexpectación de Gartner.
        **Ahora con almacenamiento automático y historial completo.**
        
        *Requiere: SerpAPI Key*
        """)
    
    with col3:
        st.markdown("""
        **📉 Curvas en S**
        
        Analiza el ciclo de vida de tecnologías utilizando datos de publicaciones 
        académicas de Scopus. Visualiza curvas S y determina la fase de adopción.
        
        *Requiere: Scopus API Key*
        """)
    
    # Instrucciones de inicio
    st.markdown("### 🚀 Cómo Comenzar")
    st.markdown("""
    1. **Configura tus API keys** en la barra lateral
    2. **Selecciona el modo de almacenamiento** (Local o AWS DynamoDB)
    3. **Crea categorías** para organizar tus análisis
    4. **Realiza análisis** - se guardarán automáticamente
    5. **Consulta el historial** para reutilizar análisis sin gastar tokens
    
    **💡 Tip**: Las consultas del Hype Cycle se guardan automáticamente con toda la información de la API, 
    incluyendo el punto exacto del ciclo donde se encuentra cada tecnología.
    """)
    
    # Ejemplo de visualización
    if st.checkbox("Mostrar ejemplo de análisis guardado", key="main_show_example_checkbox"):
        st.markdown("### 📊 Ejemplo de Datos Guardados")
        
        # Crear datos de ejemplo
        example_data = pd.DataFrame({
            'Consulta': ['AI AND Agriculture', 'Blockchain OR Finance', 'Quantum Computing', 'IoT AND Healthcare'],
            'Fase': ['Slope of Enlightenment', 'Plateau of Productivity', 'Innovation Trigger', 'Peak of Expectations'],
            'Confianza': [0.85, 0.92, 0.76, 0.88],
            'Total Menciones': [1250, 3400, 890, 2100],
            'Fecha': ['2025-06-01', '2025-06-03', '2025-06-05', '2025-06-07']
        })
        
        st.dataframe(example_data, use_container_width=True)
        
        # Gráfico de ejemplo
        import plotly.express as px
        fig = px.scatter(
            example_data,
            x='Confianza',
            y='Total Menciones',
            color='Fase',
            hover_data=['Consulta'],
            title="Ejemplo: Comparación de Consultas Guardadas"
        )
        st.plotly_chart(fig, use_container_width=True)

def sidebar_config():
    """Configuración del sidebar con opciones para todas las APIs y almacenamiento"""
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Opción de carga de archivo
        st.subheader("📁 Cargar Configuración")
        uploaded_file = st.file_uploader(
            "Cargar archivo de configuración",
            type=['json'],
            help="Sube un archivo JSON con tus credenciales de API y AWS"
        )
        
        if uploaded_file is not None:
            config_data = load_config_from_file(uploaded_file)
            if config_data:
                # Actualizar session state con las claves de API
                if 'GOOGLE_API_KEY' in config_data:
                    st.session_state.google_api_key = config_data['GOOGLE_API_KEY']
                if 'SEARCH_ENGINE_ID' in config_data:
                    st.session_state.search_engine_id = config_data['SEARCH_ENGINE_ID']
                if 'SERP_API_KEY' in config_data:
                    st.session_state.serp_api_key = config_data['SERP_API_KEY']
                if 'SCOPUS_API_KEY' in config_data:
                    st.session_state.scopus_api_key = config_data['SCOPUS_API_KEY']
                
                # Actualizar credenciales de AWS
                if 'AWS_ACCESS_KEY_ID' in config_data:
                    st.session_state.aws_access_key_id = config_data['AWS_ACCESS_KEY_ID']
                if 'AWS_SECRET_ACCESS_KEY' in config_data:
                    st.session_state.aws_secret_access_key = config_data['AWS_SECRET_ACCESS_KEY']
                if 'AWS_DEFAULT_REGION' in config_data:
                    st.session_state.aws_region = config_data['AWS_DEFAULT_REGION']
                
                st.success("✅ Configuración cargada exitosamente")
        
        st.divider()
        
        # Configuración específica para Hype Cycle Storage
        with st.expander("🗄️ Configuración Hype Cycle Storage", expanded=True):
            st.markdown('<div class="hype-storage-config">', unsafe_allow_html=True)
            st.write("**Configuración para almacenamiento de análisis Hype Cycle**")
            
            # Modo de almacenamiento - KEY ÚNICA AÑADIDA
            storage_mode = st.radio(
                "Modo de almacenamiento",
                options=["local", "dynamodb"],
                index=0 if st.session_state.hype_storage_mode == "local" else 1,
                help="Local: archivos JSON | DynamoDB: base de datos en la nube",
                key="main_hype_storage_mode_radio_unique"  # ← KEY ÚNICA AÑADIDA
            )
            st.session_state.hype_storage_mode = storage_mode
            
            # Auto-guardar - KEY ÚNICA AÑADIDA
            auto_save = st.checkbox(
                "Guardar automáticamente",
                value=st.session_state.hype_auto_save,
                help="Guardar cada análisis automáticamente sin preguntar",
                key="main_hype_auto_save_checkbox_unique"  # ← KEY ÚNICA AÑADIDA
            )
            st.session_state.hype_auto_save = auto_save
            
            # Mostrar estado actual
            if storage_mode == "local":
                st.info("💾 Los análisis se guardarán en archivos locales")
                data_path = os.path.abspath("./data")
                st.caption(f"Ubicación: {data_path}")
            else:
                aws_configured = (
                    st.session_state.aws_access_key_id and 
                    st.session_state.aws_secret_access_key
                )
                if aws_configured:
                    st.success("☁️ DynamoDB configurado y listo")
                else:
                    st.warning("⚠️ Configura AWS DynamoDB abajo")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Sección de Google API
        with st.expander("🔍 Google API (Búsqueda de Tendencias)", expanded=False):
            st.markdown('<div class="api-config">', unsafe_allow_html=True)
            google_api_key = st.text_input(
                "Google API Key",
                value=st.session_state.google_api_key,
                type="password",
                help="Necesaria para la búsqueda de tendencias"
            )
            st.session_state.google_api_key = google_api_key
            
            search_engine_id = st.text_input(
                "Search Engine ID",
                value=st.session_state.search_engine_id,
                type="password",
                help="ID del motor de búsqueda personalizado"
            )
            st.session_state.search_engine_id = search_engine_id
            
            if st.button("🔄 Probar conexión Google", key="test_google_unique"):
                with st.spinner("Probando conexión con Google API..."):
                    success, message = test_api_connection("google", google_api_key, search_engine_id)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sección de SerpAPI
        with st.expander("📈 SerpAPI (Hype Cycle)", expanded=False):
            st.markdown('<div class="api-config">', unsafe_allow_html=True)
            serp_api_key = st.text_input(
                "SerpAPI Key",
                value=st.session_state.serp_api_key,
                type="password",
                help="Necesaria para el análisis del Hype Cycle"
            )
            st.session_state.serp_api_key = serp_api_key
            
            if st.button("🔄 Probar conexión SerpAPI", key="test_serp_unique"):
                with st.spinner("Probando conexión con SerpAPI..."):
                    success, message = test_api_connection("serp", serp_api_key)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Sección de Scopus API
        with st.expander("📉 Scopus API (Curvas en S)", expanded=False):
            st.markdown('<div class="api-config">', unsafe_allow_html=True)
            scopus_api_key = st.text_input(
                "Scopus API Key",
                value=st.session_state.scopus_api_key,
                type="password",
                help="Necesaria para el análisis de curvas en S"
            )
            st.session_state.scopus_api_key = scopus_api_key
            
            if st.button("🔄 Probar conexión Scopus", key="test_scopus_unique"):
                with st.spinner("Probando conexión con Scopus API..."):
                    success, message = test_api_connection("scopus", scopus_api_key)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Sección de AWS DynamoDB
        with st.expander("🗄️ AWS DynamoDB (Almacenamiento)", expanded=False):
            st.markdown('<div class="aws-config">', unsafe_allow_html=True)
            st.write("**Configuración de AWS para almacenamiento en DynamoDB**")
            
            aws_access_key = st.text_input(
                "AWS Access Key ID",
                value=st.session_state.aws_access_key_id,
                type="password",
                help="Tu Access Key ID de AWS"
            )
            st.session_state.aws_access_key_id = aws_access_key
            
            aws_secret_key = st.text_input(
                "AWS Secret Access Key",
                value=st.session_state.aws_secret_access_key,
                type="password",
                help="Tu Secret Access Key de AWS"
            )
            st.session_state.aws_secret_access_key = aws_secret_key
            
            # AWS Region selectbox - KEY ÚNICA AÑADIDA
            aws_region = st.selectbox(
                "AWS Region",
                options=['us-east-2', 'us-west-2', 'eu-west-1', 'ap-southeast-1'],
                index=0,
                help="Región donde están creadas las tablas DynamoDB",
                key="main_aws_region_selectbox_unique"  # ← KEY ÚNICA AÑADIDA
            )
            st.session_state.aws_region = aws_region
            
            if st.button("🔄 Probar conexión AWS", key="test_aws_unique"):
                with st.spinner("Probando conexión con DynamoDB..."):
                    try:
                        from data_storage import DynamoDBStorage
                        dynamo_storage = DynamoDBStorage(
                            region_name=aws_region,
                            aws_access_key_id=aws_access_key,
                            aws_secret_access_key=aws_secret_key
                        )
                        st.success("✅ Conexión exitosa con DynamoDB")
                    except Exception as e:
                        st.error(f"❌ Error de conexión: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Acerca de la aplicación
        with st.expander("ℹ️ Acerca de", expanded=False):
            st.write("""
            **Tech Trends Explorer v2.1**
            
            **🆕 Nuevas características:**
            - ✅ Almacenamiento automático de consultas Hype Cycle
            - ✅ Sistema de categorías jerárquico
            - ✅ Historial de consultas sin gastar tokens
            - ✅ Reutilización de análisis previos
            - ✅ Comparación entre fases del Hype Cycle
            - ✅ Dashboard de métricas y estadísticas
            
            **Características existentes:**
            - ✅ Almacenamiento en AWS DynamoDB
            - ✅ Mejor gestión de datos
            - ✅ Acceso desde cualquier dispositivo
            - ✅ Análisis de tendencias múltiples
            
            Esta aplicación te permite analizar tendencias tecnológicas 
            utilizando múltiples fuentes de datos y métodos de análisis.
            
            © 2025 Todos los derechos reservados
            """)

def main():
    """
    Función principal que maneja la interfaz y flujo de la aplicación
    """
    # Inicializar estado de la sesión
    initialize_session_state()
    
    # Configuración del sidebar
    sidebar_config()
    
    # Pestañas principales
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Inicio", 
        "🔍 Búsqueda de Tendencias", 
        "📈 Análisis del Hype Cycle",
        "📉 Curvas en S",
        "🗄️ Datos Guardados"
    ])
    
    with tab1:
        show_welcome_screen()
    
    with tab2:
        if not st.session_state.google_api_key or not st.session_state.search_engine_id:
            st.warning("⚠️ Para usar esta funcionalidad, configura tu Google API Key y Search Engine ID en el panel lateral")
        else:
            run_trend_search()
    
    with tab3:
        if not st.session_state.serp_api_key:
            st.warning("⚠️ Para usar esta funcionalidad, configura tu SerpAPI Key en el panel lateral")
        else:
            run_hype_cycle_analysis()
    
    with tab4:
        if not st.session_state.scopus_api_key:
            st.warning("⚠️ Para usar esta funcionalidad, configura tu Scopus API Key en el panel lateral")
        else:
            run_s_curve_analysis()
    
    with tab5:
        st.title("🗄️ Gestión de Datos")
        
        # Subtabs para diferentes tipos de datos
        subtab1, subtab2 = st.tabs(["📈 Datos del Hype Cycle", "📊 Otros Análisis"])
        
        with subtab1:
            _show_hype_cycle_data_management()
        
        with subtab2:
            _show_general_data_management()

def _show_hype_cycle_data_management():
    """Gestión específica de datos del Hype Cycle"""
    st.header("📈 Gestión de Datos del Hype Cycle")
    
    # Información sobre el estado del almacenamiento
    storage_mode = st.session_state.hype_storage_mode
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if storage_mode == "local":
            st.info("""
            **Modo Local Activo**: Los análisis de Hype Cycle se guardan en archivos JSON locales.
            
            📂 Ubicación: `./data/`
            
            ⚠️ Estos datos solo están disponibles en tu computadora actual.
            """)
        else:
            aws_configured = (
                st.session_state.aws_access_key_id and 
                st.session_state.aws_secret_access_key
            )
            
            if aws_configured:
                st.success(f"""
                **Modo DynamoDB Activo**: Los análisis se guardan en AWS DynamoDB.
                
                ☁️ Región: {st.session_state.aws_region}
                
                ✅ Los datos están disponibles desde cualquier dispositivo.
                """)
            else:
                st.warning("""
                **DynamoDB no configurado**: Usando almacenamiento local como fallback.
                
                Para usar DynamoDB, configura las credenciales de AWS en el panel lateral.
                """)
    
    with col2:
        # Métricas rápidas
        try:
            from data_storage import initialize_database
            from hype_cycle_storage import initialize_hype_cycle_storage
            
            # Inicializar storage
            if storage_mode == "local":
                db = initialize_database("local")
            else:
                aws_configured = (
                    st.session_state.aws_access_key_id and 
                    st.session_state.aws_secret_access_key
                )
                if aws_configured:
                    db = initialize_database(
                        "dynamodb",
                        region_name=st.session_state.aws_region,
                        aws_access_key_id=st.session_state.aws_access_key_id,
                        aws_secret_access_key=st.session_state.aws_secret_access_key
                    )
                else:
                    db = initialize_database("local")
            
            if db:
                hype_storage = initialize_hype_cycle_storage(db.storage)
                queries = hype_storage.get_all_hype_cycle_queries()
                
                st.metric("Total Consultas", len(queries))
                
                if queries:
                    # Fase más común
                    phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in queries]
                    most_common = max(set(phases), key=phases.count) if phases else "N/A"
                    st.metric("Fase Más Común", most_common)
                    
                    # Consultas esta semana
                    from datetime import datetime, timedelta
                    week_ago = (datetime.now() - timedelta(days=7)).isoformat()
                    recent = [q for q in queries if q.get("execution_date", "") > week_ago]
                    st.metric("Esta Semana", len(recent))
                
        except Exception as e:
            st.error(f"Error obteniendo métricas: {str(e)}")
    
    st.divider()
    
    # Interfaz de historial específica para Hype Cycle
    try:
        from data_storage import initialize_database
        from hype_cycle_storage import initialize_hype_cycle_storage, create_hype_cycle_interface
        
        # Inicializar storage según configuración
        if storage_mode == "local":
            db = initialize_database("local")
        else:
            aws_configured = (
                st.session_state.aws_access_key_id and 
                st.session_state.aws_secret_access_key
            )
            if aws_configured:
                db = initialize_database(
                    "dynamodb",
                    region_name=st.session_state.aws_region,
                    aws_access_key_id=st.session_state.aws_access_key_id,
                    aws_secret_access_key=st.session_state.aws_secret_access_key
                )
            else:
                st.warning("DynamoDB no configurado. Usando almacenamiento local.")
                db = initialize_database("local")
        
        if db:
            hype_storage = initialize_hype_cycle_storage(db.storage)
            history_interface = create_hype_cycle_interface(hype_storage, "main_data_mgmt")  # ← CONTEXTO ÚNICO
            
            # Mostrar interfaz de historial
            history_interface.show_history_interface()
        else:
            st.error("No se pudo inicializar el sistema de almacenamiento")
            
    except Exception as e:
        st.error(f"Error en la interfaz de historial: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

def _show_general_data_management():
    """Gestión de datos generales (S-curves, trends, etc.)"""
    st.header("📊 Otros Análisis Guardados")
    
    # Configuración para datos generales
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Selector para modo de almacenamiento general - KEY ÚNICA AÑADIDA
        storage_mode = st.radio(
            "Modo de almacenamiento",
            options=["Local", "AWS DynamoDB"],
            index=0,
            help="Selecciona dónde están guardados los otros análisis",
            key="main_general_storage_mode_radio_unique"  # ← KEY ÚNICA AÑADIDA
        )
    
    with col2:
        # Mostrar información sobre el modo seleccionado
        if storage_mode == "Local":
            st.info("""
            **Modo local activo**: Los datos se guardan en archivos JSON locales.
            
            📂 Ubicación: `./data/`
            
            ⚠️ Estos datos solo están disponibles en tu computadora actual.
            """)
            
            if st.button("📂 Abrir carpeta de datos", key="main_open_data_folder_btn_unique"):
                data_path = os.path.abspath("./data")
                os.makedirs(data_path, exist_ok=True)
                st.success(f"📁 Datos guardados en: {data_path}")
        else:
            # Configuración de DynamoDB
            aws_configured = (
                st.session_state.aws_access_key_id and 
                st.session_state.aws_secret_access_key and 
                st.session_state.aws_region
            )
            
            if not aws_configured:
                st.warning("⚠️ Configuración de AWS incompleta")
                st.info("""
                Para usar DynamoDB:
                1. Configura las credenciales en el panel lateral
                2. Asegúrate de que las tablas estén creadas:
                   - tech-trends-analyses
                   - tech-trends-categories
                """)
            else:
                st.success("✅ Configuración de AWS disponible")
                st.info(f"""
                **Región**: {st.session_state.aws_region}
                
                ☁️ Los datos se guardarán en DynamoDB y estarán disponibles desde cualquier dispositivo.
                """)
    
    # Inicializar sistema de almacenamiento
    from data_storage import initialize_database
    
    if storage_mode == "Local":
        db = initialize_database("local")
    else:
        if aws_configured:
            try:
                db = initialize_database(
                    "dynamodb",
                    region_name=st.session_state.aws_region,
                    aws_access_key_id=st.session_state.aws_access_key_id,
                    aws_secret_access_key=st.session_state.aws_secret_access_key
                )
            except Exception as e:
                st.error(f"❌ Error al conectar con DynamoDB: {str(e)}")
                st.info("Usando almacenamiento local como fallback")
                db = initialize_database("local")
        else:
            st.warning("Configuración de AWS incompleta. Usando almacenamiento local.")
            db = initialize_database("local")
    
    # Cargar interfaz de gestión general
    if db is not None:
        try:
            from database_manager import run_database_manager
            run_database_manager(db)
        except Exception as e:
            st.error(f"Error al inicializar el gestor de base de datos: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.warning("No se pudo inicializar el sistema de almacenamiento.")

if __name__ == "__main__":
    main()