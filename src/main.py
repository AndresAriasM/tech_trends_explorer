# src/main.py
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
    
    # Estado para tópicos de búsqueda - específicos para cada módulo
    if 'hype_topics_data' not in st.session_state:
        st.session_state.hype_topics_data = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    if 'trends_topics_data' not in st.session_state:
        st.session_state.trends_topics_data = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    if 'scurve_topics_data' not in st.session_state:
        st.session_state.scurve_topics_data = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    
    # Para compatibilidad con código existente, mantenemos 'topics_data'
    if 'topics_data' not in st.session_state:
        st.session_state.topics_data = [{'id': 0, 'value': '', 'operator': 'AND', 'exact_match': False}]
    
    # Estado para resultados actuales
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    if 'current_query_info' not in st.session_state:
        st.session_state.current_query_info = None

def show_welcome_screen():
    """Muestra la pantalla de bienvenida"""
    st.markdown('<p class="main-header">🔍 Tech Trends Explorer</p>', unsafe_allow_html=True)
    
    # Tarjeta de bienvenida
    st.markdown("""
    <div class="welcome-card">
        <h2>Bienvenido al Explorador de Tendencias Tecnológicas</h2>
        <p>Esta herramienta te permite analizar y visualizar tendencias tecnológicas emergentes utilizando 
        múltiples fuentes de datos. Selecciona una funcionalidad en las pestañas de arriba para comenzar.</p>
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
        **📈 Análisis del Hype Cycle**
        
        Determina la posición de tecnologías dentro del ciclo de sobreexpectación de Gartner.
        Identifica fases como "Disparador de Innovación" o "Meseta de Productividad".
        
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
    1. Configura tus API keys en la barra lateral
    2. Selecciona una funcionalidad en las pestañas superiores
    3. Sigue las instrucciones específicas de cada módulo
    
    Si es tu primera vez, te recomendamos revisar la configuración de API para asegurarte 
    de que todas las funcionalidades estén disponibles.
    """)
    
    # Ejemplo de visualización
    if st.checkbox("Mostrar ejemplo de visualización"):
        import plotly.express as px
        
        # Datos de ejemplo
        years = list(range(2010, 2025))
        values = [5, 8, 12, 20, 35, 65, 120, 180, 220, 250, 270, 285, 290, 292, 293]
        
        # Crear un gráfico de ejemplo
        fig = px.line(
            x=years, 
            y=values,
            markers=True,
            labels={"x": "Año", "y": "Número de Publicaciones"},
            title="Ejemplo: Adopción de Inteligencia Artificial (2010-2024)"
        )
        
        # Personalizar el gráfico
        fig.update_layout(
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            plot_bgcolor="white"
        )
        
        # Mostrar el gráfico
        st.plotly_chart(fig, use_container_width=True)

def sidebar_config():
    """Configuración del sidebar con opciones para todas las APIs"""
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Opción de carga de archivo
        st.subheader("📁 Cargar Configuración")
        uploaded_file = st.file_uploader(
            "Cargar archivo de configuración",
            type=['json'],
            help="Sube un archivo JSON con tus credenciales de API"
        )
        
        if uploaded_file is not None:
            config_data = load_config_from_file(uploaded_file)
            if config_data:
                # Actualizar session state con las claves disponibles
                if 'GOOGLE_API_KEY' in config_data:
                    st.session_state.google_api_key = config_data['GOOGLE_API_KEY']
                if 'SEARCH_ENGINE_ID' in config_data:
                    st.session_state.search_engine_id = config_data['SEARCH_ENGINE_ID']
                if 'SERP_API_KEY' in config_data:
                    st.session_state.serp_api_key = config_data['SERP_API_KEY']
                if 'SCOPUS_API_KEY' in config_data:
                    st.session_state.scopus_api_key = config_data['SCOPUS_API_KEY']
                
                st.success("✅ Configuración cargada exitosamente")
        
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
            
            if st.button("🔄 Probar conexión Google", key="test_google"):
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
            
            if st.button("🔄 Probar conexión SerpAPI", key="test_serp"):
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
            
            if st.button("🔄 Probar conexión Scopus", key="test_scopus"):
                with st.spinner("Probando conexión con Scopus API..."):
                    success, message = test_api_connection("scopus", scopus_api_key)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Acerca de la aplicación
        with st.expander("ℹ️ Acerca de", expanded=False):
            st.write("""
            **Tech Trends Explorer**
            
            Versión: 2.0
            
            Desarrollado por: Tu Nombre/Equipo
            
            Esta aplicación te permite analizar tendencias tecnológicas 
            utilizando múltiples fuentes de datos y métodos de análisis.
            
            © 2025 Todos los derechos reservados
            """)

def check_github_config():
    """
    Verifica la configuración de GitHub y devuelve un estado y mensaje.
    Maneja correctamente la ausencia de archivos secrets.toml.
    """
    # Verificar variables de entorno primero
    github_token = os.environ.get('GITHUB_TOKEN')
    repo_owner = os.environ.get('GITHUB_REPO_OWNER')
    repo_name = os.environ.get('GITHUB_REPO_NAME')
    
    # Intentar obtener de secrets solo si la verificación anterior falla
    if not github_token or not repo_owner or not repo_name:
        try:
            # Verificar si hay un archivo de secretos disponible antes de intentar acceder
            github_token = github_token or st.secrets.get("github_token", "")
            repo_owner = repo_owner or st.secrets.get("github_repo_owner", "")
            repo_name = repo_name or st.secrets.get("github_repo_name", "")
        except FileNotFoundError:
            # No hay archivo de secretos, continuar con los valores actuales
            pass
        except Exception as e:
            # Otro tipo de error al acceder a secretos
            st.warning(f"Error al acceder a secretos: {str(e)}")
    
    # Determinar si la configuración es válida
    is_valid = bool(github_token and repo_owner and repo_name)
    
    return is_valid, github_token, repo_owner, repo_name

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
        # Mostrar pantalla de bienvenida en la pestaña de inicio
        show_welcome_screen()
    
    with tab2:
        # Verificar configuración antes de ejecutar la búsqueda de tendencias
        if not st.session_state.google_api_key or not st.session_state.search_engine_id:
            st.warning("⚠️ Para usar esta funcionalidad, configura tu Google API Key y Search Engine ID en el panel lateral")
        else:
            # Ejecutar el módulo de búsqueda de tendencias
            run_trend_search()
    
    with tab3:
        # Verificar configuración antes de ejecutar el análisis del Hype Cycle
        if not st.session_state.serp_api_key:
            st.warning("⚠️ Para usar esta funcionalidad, configura tu SerpAPI Key en el panel lateral")
        else:
            # Ejecutar el módulo de análisis del Hype Cycle
            run_hype_cycle_analysis()
    
    with tab4:
        # Verificar configuración antes de ejecutar el análisis de curvas en S
        if not st.session_state.scopus_api_key:
            st.warning("⚠️ Para usar esta funcionalidad, configura tu Scopus API Key en el panel lateral")
        else:
            # Ejecutar el módulo de análisis de curvas en S
            run_s_curve_analysis()
    
    with tab5:
        st.title("🗄️ Datos Guardados")
        
        # Configuración para datos guardados
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Selector para modo de almacenamiento
            storage_mode = st.radio(
                "Modo de almacenamiento",
                options=["Local", "GitHub"],
                index=0,  # Por defecto usar almacenamiento local
                help="Selecciona dónde guardar los datos de análisis"
            )
        
        with col2:
            # Mostrar información sobre el modo seleccionado
            if storage_mode == "Local":
                st.info("""
                **Modo local activo**: Los datos se guardarán en archivos JSON locales.
                
                📂 Ubicación: `./data/`
                
                ⚠️ Estos datos solo están disponibles en tu computadora actual.
                """)
                
                # Opcional: Botón para abrir carpeta local
                if st.button("Abrir carpeta de datos", key="open_data_folder"):
                    data_path = os.path.abspath("./data")
                    os.makedirs(data_path, exist_ok=True)  # Crear directorio si no existe
                    
                    # Intentar abrir carpeta según el sistema operativo
                    try:
                        if os.name == 'nt':  # Windows
                            os.startfile(data_path)
                        elif os.name == 'posix':  # macOS y Linux
                            import subprocess
                            subprocess.call(('open', data_path) if os.uname().sysname == 'Darwin' else ('xdg-open', data_path))
                    except Exception as e:
                        st.error(f"No se pudo abrir la carpeta: {str(e)}")
            else:
                # Configuración de GitHub
                github_configured = (
                    os.environ.get('GITHUB_TOKEN') or 
                    st.session_state.get('github_token') or
                    (hasattr(st, 'secrets') and st.secrets.get("github_token"))
                )
                
                if not github_configured:
                    st.warning("⚠️ Configuración de GitHub incompleta")
                    
                    # Formulario para configuración temporal
                    with st.form("github_config_form"):
                        temp_token = st.text_input("Token de GitHub (temporal)", type="password")
                        temp_owner = st.text_input("Usuario/Organización de GitHub", value="tu-usuario-github")
                        temp_repo = st.text_input("Nombre del repositorio", value="tech-trends-explorer")
                        
                        submit_config = st.form_submit_button("Usar configuración temporal")
                        
                        if submit_config and temp_token:
                            # Guardar en session_state
                            st.session_state.github_token = temp_token
                            st.session_state.github_repo_owner = temp_owner
                            st.session_state.github_repo_name = temp_repo
                            
                            # Establecer variables de entorno
                            os.environ['GITHUB_TOKEN'] = temp_token
                            os.environ['GITHUB_REPO_OWNER'] = temp_owner
                            os.environ['GITHUB_REPO_NAME'] = temp_repo
                            
                            st.success("✅ Configuración temporal aplicada. Recarga esta pestaña para continuar.")
                            github_configured = True
                else:
                    st.success("✅ Configuración de GitHub disponible")
        
        # Inicializar sistema de almacenamiento según modo seleccionado
        from data_storage import initialize_github_db
        
        if storage_mode == "Local":
            # Usar almacenamiento local
            db = initialize_github_db(use_local=True)
        else:
            # Usar GitHub con configuración disponible
            repo_owner = os.environ.get('GITHUB_REPO_OWNER', st.session_state.get("github_repo_owner", ""))
            repo_name = os.environ.get('GITHUB_REPO_NAME', st.session_state.get("github_repo_name", ""))
            
            # Verificar si hay credenciales suficientes
            if not (github_configured and repo_owner and repo_name):
                st.error("❌ Faltan datos de configuración para GitHub")
                db = None
            else:
                db = initialize_github_db(repo_owner, repo_name)
        
        # Dividir la interfaz con pestañas
        if db is not None:
            # Cargar el gestor de base de datos
            try:
                from database_manager import run_database_manager
                run_database_manager(db)
            except Exception as e:
                st.error(f"Error al inicializar el gestor de base de datos: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
        else:
            st.warning("No se pudo inicializar el sistema de almacenamiento. Verifica la configuración.")

if __name__ == "__main__":
    main()

