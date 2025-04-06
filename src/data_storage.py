# src/data_storage.py
import json
import os
import pandas as pd
from datetime import datetime
import streamlit as st
import base64
import requests
import time
import traceback

# Intentar importar PyGithub
try:
    from github import Github, GithubException
    GITHUB_AVAILABLE = True
except ImportError:
    st.warning("📦 Se requiere instalar PyGithub. Ejecuta: pip install PyGithub")
    GITHUB_AVAILABLE = False

def load_credentials_from_file(file_path="config.json"):
    """
    Carga credenciales desde un archivo local.
    
    Args:
        file_path: Ruta al archivo JSON de configuración
        
    Returns:
        tuple: (token, owner, repo) o (None, None, None) si hay error
    """
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
            return config.get("github_token"), config.get("github_repo_owner"), config.get("github_repo_name")
    except Exception as e:
        st.error(f"Error al cargar configuración: {str(e)}")
        return None, None, None

# Añade esta clase a data_storage.py

class LocalStorage:
    """
    Clase para gestionar el almacenamiento persistente de datos
    usando archivos JSON locales.
    """
    
    def __init__(self, base_path="./data"):
        """
        Inicializa el sistema de almacenamiento local.
        
        Args:
            base_path: Ruta base donde se guardarán los archivos
        """
        self.base_path = base_path
        self.db_path = os.path.join(base_path, "s_curve_db.json")
        self.categories_path = os.path.join(base_path, "categories.json")
        
        # Asegurar que el directorio existe
        os.makedirs(base_path, exist_ok=True)
        
        # Cargar datos al inicializar
        self.data = self._load_data()
        self.categories = self._load_categories()
    
    def _load_data(self):
        """
        Carga los datos desde archivo local o crea estructura vacía.
        """
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"searches": [], "last_update": datetime.now().isoformat()}
        except Exception as e:
            st.warning(f"Error al cargar datos locales: {str(e)}. Se creará nueva estructura.")
            return {"searches": [], "last_update": datetime.now().isoformat()}
    
    def _load_categories(self):
        """
        Carga las categorías desde archivo local o crea estructura vacía.
        """
        try:
            if os.path.exists(self.categories_path):
                with open(self.categories_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}
        except Exception as e:
            st.warning(f"Error al cargar categorías locales: {str(e)}. Se crearán predeterminadas.")
            return {"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}
    
    def save_data(self):
        """
        Guarda los datos en archivo local.
        """
        try:
            self.data["last_update"] = datetime.now().isoformat()
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error al guardar datos: {str(e)}")
            return False
    
    def save_categories(self):
        """
        Guarda las categorías en archivo local.
        """
        try:
            with open(self.categories_path, 'w', encoding='utf-8') as f:
                json.dump(self.categories, f, indent=2)
            return True
        except Exception as e:
            st.error(f"Error al guardar categorías: {str(e)}")
            return False
    
    def add_search_result(self, search_data, category_id="default"):
        """
        Añade un resultado de búsqueda a la base de datos.
        
        Args:
            search_data: Datos de la búsqueda
            category_id: ID de la categoría (por defecto 'default')
            
        Returns:
            str: ID del registro guardado
        """
        # Generar ID único para este registro
        search_id = f"search_{int(time.time())}_{hash(search_data.get('query', ''))%1000}"
        
        # Añadir metadatos
        search_data["id"] = search_id
        search_data["timestamp"] = datetime.now().isoformat()
        search_data["category_id"] = category_id
        
        # Añadir a la lista de búsquedas
        self.data["searches"].append(search_data)
        
        # Guardar cambios
        self.save_data()
        
        return search_id
    
    def get_search_by_id(self, search_id):
        """
        Obtiene un resultado de búsqueda por su ID.
        
        Args:
            search_id: ID del registro a buscar
            
        Returns:
            dict: Datos de la búsqueda o None si no se encuentra
        """
        for search in self.data["searches"]:
            if search["id"] == search_id:
                return search
        return None
    
    def get_searches_by_category(self, category_id):
        """
        Obtiene todos los resultados de búsqueda de una categoría.
        
        Args:
            category_id: ID de la categoría
            
        Returns:
            list: Lista de búsquedas en la categoría
        """
        return [search for search in self.data["searches"] if search["category_id"] == category_id]
    
    def get_all_searches(self):
        """
        Obtiene todos los resultados de búsqueda.
        
        Returns:
            list: Lista de todas las búsquedas
        """
        return self.data["searches"]
    
    def add_category(self, name, description=""):
        """
        Añade una nueva categoría.
        
        Args:
            name: Nombre de la categoría
            description: Descripción de la categoría
            
        Returns:
            str: ID de la categoría creada
        """
        # Generar ID para la categoría
        category_id = f"cat_{int(time.time())}_{hash(name)%1000}"
        
        # Crear nueva categoría
        new_category = {
            "id": category_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        # Añadir a la lista de categorías
        self.categories["categories"].append(new_category)
        
        # Guardar cambios
        self.save_categories()
        
        return category_id
    
    def get_all_categories(self):
        """
        Obtiene todas las categorías disponibles.
        
        Returns:
            list: Lista de todas las categorías
        """
        return self.categories["categories"]
    
    def get_category_by_id(self, category_id):
        """
        Obtiene una categoría por su ID.
        
        Args:
            category_id: ID de la categoría
            
        Returns:
            dict: Datos de la categoría o None si no se encuentra
        """
        for category in self.categories["categories"]:
            if category["id"] == category_id:
                return category
        return None
        
class GitHubStorage:
    """
    Clase para gestionar el almacenamiento persistente de datos
    usando archivos JSON en un repositorio de GitHub.
    """
    
    def __init__(self, repo_owner, repo_name, branch="main"):
        """
        Inicializa el sistema de almacenamiento en GitHub.
        
        Args:
            repo_owner: Propietario del repositorio
            repo_name: Nombre del repositorio
            branch: Rama del repositorio (por defecto 'main')
        """
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.branch = branch
        self.db_path = "data/s_curve_db.json"
        self.categories_path = "data/categories.json"
        
        # Usar token de GitHub si está disponible
        self.github_token = os.environ.get('GITHUB_TOKEN')
        self.github_api = None
        
        # Verificar si podemos usar la API de GitHub
        if self.github_token and GITHUB_AVAILABLE:
            try:
                self.github_api = Github(self.github_token)
                # Probar la conexión
                self.github_api.get_user()
            except Exception as e:
                st.warning(f"⚠️ Error al conectar con API de GitHub: {str(e)}")
                self.github_api = None
        
        # Intentar cargar datos al inicializar
        self.data = self._load_data()
        self.categories = self._load_categories()
    
    def _load_data(self):
        """
        Carga los datos de la base desde GitHub o crea una estructura vacía.
        
        Returns:
            dict: Datos de la base
        """
        try:
            if self.github_api:
                # Usar la API de GitHub para acceder a los datos
                repo = self.github_api.get_repo(f"{self.repo_owner}/{self.repo_name}")
                try:
                    file_content = repo.get_contents(self.db_path, ref=self.branch)
                    content = base64.b64decode(file_content.content).decode('utf-8')
                    return json.loads(content)
                except Exception:
                    # Si el archivo no existe, crear estructura base
                    return {"searches": [], "last_update": datetime.now().isoformat()}
            else:
                # Alternativa: obtener vía GET request directo
                url = f"https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/{self.branch}/{self.db_path}"
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    # Si el archivo no existe, crear estructura base
                    return {"searches": [], "last_update": datetime.now().isoformat()}
        except Exception as e:
            st.warning(f"No se pudo cargar la base de datos: {str(e)}. Se creará una nueva.")
            return {"searches": [], "last_update": datetime.now().isoformat()}
    
    def _load_categories(self):
        """
        Carga las categorías desde GitHub o crea una estructura vacía.
        
        Returns:
            dict: Datos de categorías
        """
        try:
            if self.github_api:
                # Usar la API de GitHub
                repo = self.github_api.get_repo(f"{self.repo_owner}/{self.repo_name}")
                try:
                    file_content = repo.get_contents(self.categories_path, ref=self.branch)
                    content = base64.b64decode(file_content.content).decode('utf-8')
                    return json.loads(content)
                except Exception:
                    # Si el archivo no existe, crear estructura base
                    return {"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}
            else:
                # Alternativa: obtener vía GET request directo
                url = f"https://raw.githubusercontent.com/{self.repo_owner}/{self.repo_name}/{self.branch}/{self.categories_path}"
                response = requests.get(url)
                if response.status_code == 200:
                    return response.json()
                else:
                    # Si el archivo no existe, crear estructura base
                    return {"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}
        except Exception as e:
            st.warning(f"No se pudieron cargar las categorías: {str(e)}. Se crearán predeterminadas.")
            return {"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}
    
    def save_data(self):
        """
        Guarda los datos actualizados en GitHub.
        
        Returns:
            bool: True si se guardaron correctamente, False en caso contrario
        """
        if not self.github_token or not GITHUB_AVAILABLE:
            st.error("❌ No se puede guardar: Token de GitHub no configurado o PyGithub no disponible.")
            return False
        
        try:
            # Actualizar timestamp
            self.data["last_update"] = datetime.now().isoformat()
            
            # Convertir a JSON
            json_content = json.dumps(self.data, indent=2)
            
            # Guardar en GitHub
            repo = self.github_api.get_repo(f"{self.repo_owner}/{self.repo_name}")
            
            try:
                # Verificar si el archivo existe
                file_content = repo.get_contents(self.db_path, ref=self.branch)
                # Si existe, actualizar
                repo.update_file(
                    self.db_path,
                    f"Actualización de datos de curvas S: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    json_content,
                    file_content.sha,
                    branch=self.branch
                )
            except Exception:
                # Si no existe, crear
                repo.create_file(
                    self.db_path,
                    f"Creación de base de datos de curvas S: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    json_content,
                    branch=self.branch
                )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error al guardar datos en GitHub: {str(e)}")
            st.error(traceback.format_exc())
            return False
    
    def save_categories(self):
        """
        Guarda las categorías actualizadas en GitHub.
        
        Returns:
            bool: True si se guardaron correctamente, False en caso contrario
        """
        if not self.github_token or not GITHUB_AVAILABLE:
            st.error("❌ No se puede guardar: Token de GitHub no configurado o PyGithub no disponible.")
            return False
        
        try:
            # Convertir a JSON
            json_content = json.dumps(self.categories, indent=2)
            
            # Guardar en GitHub
            repo = self.github_api.get_repo(f"{self.repo_owner}/{self.repo_name}")
            
            try:
                # Verificar si el archivo existe
                file_content = repo.get_contents(self.categories_path, ref=self.branch)
                # Si existe, actualizar
                repo.update_file(
                    self.categories_path,
                    f"Actualización de categorías: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    json_content,
                    file_content.sha,
                    branch=self.branch
                )
            except Exception:
                # Si no existe, crear
                repo.create_file(
                    self.categories_path,
                    f"Creación de categorías: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    json_content,
                    branch=self.branch
                )
            
            return True
            
        except Exception as e:
            st.error(f"❌ Error al guardar categorías en GitHub: {str(e)}")
            st.error(traceback.format_exc())
            return False
    
    def add_search_result(self, search_data, category_id="default"):
        """
        Añade un resultado de búsqueda a la base de datos.
        """
        try:
            # Generar ID único para este registro
            query_hash = hash(str(search_data.get('query', ''))) % 1000
            search_id = f"search_{int(time.time())}_{query_hash}"
            
            # Añadir metadatos
            search_data["id"] = search_id
            search_data["timestamp"] = datetime.now().isoformat()
            search_data["category_id"] = category_id
            
            # Añadir a la lista de búsquedas
            self.data["searches"].append(search_data)
            
            # Guardar cambios
            success = self.save_data()
            
            if not success:
                st.error("❌ No se pudo guardar el análisis en el almacenamiento local.")
                return None
                
            return search_id
            
        except Exception as e:
            import traceback
            st.error(f"❌ Error en add_search_result: {str(e)}")
            st.code(traceback.format_exc())
            return None
    
    def get_search_by_id(self, search_id):
        """
        Obtiene un resultado de búsqueda por su ID.
        
        Args:
            search_id: ID del registro a buscar
            
        Returns:
            dict: Datos de la búsqueda o None si no se encuentra
        """
        for search in self.data["searches"]:
            if search["id"] == search_id:
                return search
        return None
    
    def get_searches_by_category(self, category_id):
        """
        Obtiene todos los resultados de búsqueda de una categoría.
        
        Args:
            category_id: ID de la categoría
            
        Returns:
            list: Lista de búsquedas en la categoría
        """
        return [search for search in self.data["searches"] if search["category_id"] == category_id]
    
    def get_all_searches(self):
        """
        Obtiene todos los resultados de búsqueda.
        
        Returns:
            list: Lista de todas las búsquedas
        """
        return self.data["searches"]
    
    def add_category(self, name, description=""):
        """
        Añade una nueva categoría.
        
        Args:
            name: Nombre de la categoría
            description: Descripción de la categoría
            
        Returns:
            str: ID de la categoría creada
        """
        # Generar ID para la categoría
        category_id = f"cat_{int(time.time())}_{hash(name)%1000}"
        
        # Crear nueva categoría
        new_category = {
            "id": category_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        # Añadir a la lista de categorías
        self.categories["categories"].append(new_category)
        
        # Guardar cambios
        self.save_categories()
        
        return category_id
    
    def get_all_categories(self):
        """
        Obtiene todas las categorías disponibles.
        
        Returns:
            list: Lista de todas las categorías
        """
        return self.categories["categories"]
    
    def get_category_by_id(self, category_id):
        """
        Obtiene una categoría por su ID.
        
        Args:
            category_id: ID de la categoría
            
        Returns:
            dict: Datos de la categoría o None si no se encuentra
        """
        for category in self.categories["categories"]:
            if category["id"] == category_id:
                return category
        return None


class SCurveDatabase:
    """
    Clase específica para gestionar datos de análisis de curvas S,
    utilizando GitHubStorage como backend.
    """
    
    def __init__(self, github_storage):
        """
        Inicializa la base de datos de curvas S.
        
        Args:
            github_storage: Instancia de GitHubStorage
        """
        self.storage = github_storage
    
    def save_s_curve_analysis(self, 
                          query, 
                          paper_data=None, 
                          patent_data=None, 
                          paper_metrics=None, 
                          patent_metrics=None,
                          category_id="default",
                          analysis_name=None):
        """
        Guarda un análisis completo de curva S.
        """
        try:
            # Convertir datos a formato serializable
            # Esto soluciona problemas con objetos numpy o no serializables
            if paper_data:
                paper_data = {str(k): float(v) for k, v in paper_data.items()}
            
            if patent_data:
                patent_data = {str(k): float(v) for k, v in patent_data.items()}
            
            # Limpiar métricas para asegurar que sean serializables
            if paper_metrics:
                # Eliminar campos no serializables si existen
                paper_metrics = {k: v for k, v in paper_metrics.items() 
                            if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
                # Convertir valores numpy a Python nativos
                paper_metrics = {k: float(v) if hasattr(v, 'item') else v 
                            for k, v in paper_metrics.items()}
                
            if patent_metrics:
                # Eliminar campos no serializables si existen
                patent_metrics = {k: v for k, v in patent_metrics.items() 
                            if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
                # Convertir valores numpy a Python nativos
                patent_metrics = {k: float(v) if hasattr(v, 'item') else v 
                                for k, v in patent_metrics.items()}
            
            # Crear estructura de datos para guardar
            if not analysis_name:
                # Generar nombre basado en la consulta
                terms = str(query).replace('TITLE(', '').replace(')', '').replace('"', '')
                analysis_name = f"Análisis de {terms[:30]}..." if len(terms) > 30 else f"Análisis de {terms}"
            
            analysis_data = {
                "name": analysis_name,
                "query": str(query),
                "execution_date": datetime.now().isoformat(),
                "paper_data": paper_data,
                "patent_data": patent_data,
                "paper_metrics": paper_metrics,
                "patent_metrics": patent_metrics
            }
            
            # Guardar en la base de datos
            try:
                search_id = self.storage.add_search_result(analysis_data, category_id)
                return search_id
            except Exception as e:
                import traceback
                st.error(f"❌ Error al guardar análisis: {str(e)}")
                st.code(traceback.format_exc())
                return None
                
        except Exception as e:
            import traceback
            st.error(f"❌ Error al preparar datos para guardar: {str(e)}")
            st.code(traceback.format_exc())
            return None
    
    def get_category_analysis(self, category_id):
        """
        Obtiene todos los análisis de una categoría.
        
        Args:
            category_id: ID de la categoría
            
        Returns:
            list: Lista de análisis en la categoría
        """
        return self.storage.get_searches_by_category(category_id)
    
    def get_analysis_by_id(self, analysis_id):
        """
        Obtiene un análisis específico por su ID.
        
        Args:
            analysis_id: ID del análisis
            
        Returns:
            dict: Datos del análisis o None si no se encuentra
        """
        return self.storage.get_search_by_id(analysis_id)
    
    def create_category(self, name, description=""):
        """
        Crea una nueva categoría para agrupar análisis.
        
        Args:
            name: Nombre de la categoría
            description: Descripción de la categoría
            
        Returns:
            str: ID de la categoría creada
        """
        return self.storage.add_category(name, description)
    
    def get_all_categories(self):
        """
        Obtiene todas las categorías disponibles.
        
        Returns:
            list: Lista de todas las categorías
        """
        return self.storage.get_all_categories()


def initialize_github_db(repo_owner=None, repo_name=None, token=None, use_local=False):
    """
    Inicializa y devuelve una instancia de la base de datos de curvas S.
    Intenta múltiples fuentes para obtener credenciales.
    
    Args:
        repo_owner: Propietario del repositorio (opcional)
        repo_name: Nombre del repositorio (opcional)
        token: Token de GitHub (opcional)
        use_local: Si es True, usa almacenamiento local en lugar de GitHub
        
    Returns:
        SCurveDatabase: Instancia de la base de datos o None si hay error
    """
    try:
        # Opción para usar almacenamiento local
        if use_local:
            # Crear directorio data si no existe
            os.makedirs("./data", exist_ok=True)
            
            # Inicializar almacenamiento local
            local_storage = LocalStorage()
            return SCurveDatabase(local_storage)
        
        # Continuar con lógica de GitHub
        # 1. Usar parámetros pasados directamente
        github_token = token
        
        # 2. Verificar session_state (credenciales ya cargadas en sesión actual)
        if not github_token and hasattr(st, 'session_state'):
            github_token = st.session_state.get('github_token')
        
        # 3. Intentar cargar desde archivo local
        if not github_token:
            file_token, file_owner, file_repo = load_credentials_from_file()
            github_token = file_token
            repo_owner = repo_owner or file_owner
            repo_name = repo_name or file_repo
        
        # 4. Usar variables de entorno
        github_token = github_token or os.environ.get('GITHUB_TOKEN')
        repo_owner = repo_owner or os.environ.get('GITHUB_REPO_OWNER')
        repo_name = repo_name or os.environ.get('GITHUB_REPO_NAME')
        
        # 5. Intentar secrets de Streamlit (con manejo de excepciones)
        if not (github_token and repo_owner and repo_name):
            try:
                github_token = github_token or st.secrets.get("github_token", "")
                repo_owner = repo_owner or st.secrets.get("github_repo_owner", "")
                repo_name = repo_name or st.secrets.get("github_repo_name", "")
            except (FileNotFoundError, AttributeError):
                pass
            except Exception as e:
                st.warning(f"Error al acceder a secretos: {str(e)}")
        
        # Verificar si hay configuración suficiente
        if not (repo_owner and repo_name and github_token):
            st.warning("⚠️ Configuración de GitHub incompleta. Se requiere token, propietario y nombre del repositorio.")
            return None
        
        # Si llegamos aquí, hay suficiente información para intentar crear la instancia
        try:
            # Establecer la variable de entorno para PyGithub
            os.environ['GITHUB_TOKEN'] = github_token
            
            # Crear las instancias
            github_storage = GitHubStorage(repo_owner, repo_name)
            return SCurveDatabase(github_storage)
        except Exception as e:
            st.error(f"❌ Error al inicializar almacenamiento: {str(e)}")
            st.error(traceback.format_exc())
            return None
            
    except Exception as e:
        st.error(f"❌ Error no controlado: {str(e)}")
        st.error(traceback.format_exc())
        return None