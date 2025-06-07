# src/data_storage.py
import json
import os
import pandas as pd
from datetime import datetime
import streamlit as st
import time
import traceback
import uuid

# AWS DynamoDB
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    st.warning("📦 Se requiere instalar boto3. Ejecuta: pip install boto3")
    BOTO3_AVAILABLE = False

def load_credentials_from_file(file_path="config.json"):
    """
    Carga credenciales desde un archivo local.
    
    Args:
        file_path: Ruta al archivo JSON de configuración
        
    Returns:
        dict: Configuración cargada o diccionario vacío si hay error
    """
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
            return config
    except Exception as e:
        st.error(f"Error al cargar configuración: {str(e)}")
        return {}

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
        
        # Inicializar archivos con estructuras básicas si no existen
        self._init_files()
        
        # Cargar datos al inicializar
        self.data = self._load_data()
        self.categories = self._load_categories()
    
    def _init_files(self):
        """
        Inicializa los archivos con estructuras básicas si no existen.
        """
        # Inicializar archivo de base de datos
        if not os.path.exists(self.db_path):
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump({"searches": [], "last_update": datetime.now().isoformat()}, f, indent=2)
                
        # Inicializar archivo de categorías
        if not os.path.exists(self.categories_path):
            with open(self.categories_path, 'w', encoding='utf-8') as f:
                json.dump({"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}, f, indent=2)
    
    def _load_data(self):
        """
        Carga los datos desde archivo local o crea estructura vacía.
        """
        try:
            if os.path.exists(self.db_path):
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
                    else:
                        return {"searches": [], "last_update": datetime.now().isoformat()}
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
                    content = f.read().strip()
                    if content:
                        return json.loads(content)
                    else:
                        return {"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}
            return {"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}
        except Exception as e:
            st.warning(f"Error al cargar categorías locales: {str(e)}. Se crearán predeterminadas.")
            return {"categories": [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]}
    
    def save_data(self):
        """
        Guarda los datos en archivo local.
        """
        try:
            os.makedirs(self.base_path, exist_ok=True)
            self.data["last_update"] = datetime.now().isoformat()
            
            temp_path = f"{self.db_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            
            if os.path.exists(temp_path):
                if os.path.exists(self.db_path):
                    os.remove(self.db_path)
                os.rename(temp_path, self.db_path)
                return True
            else:
                st.error("No se pudo crear el archivo temporal para guardar datos")
                return False
        except Exception as e:
            st.error(f"Error al guardar datos: {str(e)}")
            st.error(traceback.format_exc())
            return False
    
    def save_categories(self):
        """
        Guarda las categorías en archivo local.
        """
        try:
            os.makedirs(self.base_path, exist_ok=True)
            
            temp_path = f"{self.categories_path}.tmp"
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.categories, f, ensure_ascii=False, indent=2)
            
            if os.path.exists(temp_path):
                if os.path.exists(self.categories_path):
                    os.remove(self.categories_path)
                os.rename(temp_path, self.categories_path)
                return True
            else:
                st.error("No se pudo crear el archivo temporal para guardar categorías")
                return False
        except Exception as e:
            st.error(f"Error al guardar categorías: {str(e)}")
            st.error(traceback.format_exc())
            return False
    
    def add_search_result(self, search_data, category_id="default"):
        """
        Añade un resultado de búsqueda a la base de datos.
        """
        try:
            # Verificar que los datos son serializables
            try:
                json.dumps(search_data)
            except TypeError as e:
                st.error(f"Los datos no son serializables: {str(e)}")
                search_data = self._sanitize_data(search_data)
            
            # Generar ID único
            query_str = str(search_data.get('query', ''))
            search_id = f"search_{int(time.time())}_{abs(hash(query_str))%1000}"
            
            # Añadir metadatos
            search_data["id"] = search_id
            search_data["timestamp"] = datetime.now().isoformat()
            search_data["category_id"] = category_id
            
            if "searches" not in self.data:
                self.data["searches"] = []
            
            self.data["searches"].append(search_data)
            
            success = self.save_data()
            
            if success:
                st.success(f"✅ Análisis guardado correctamente con ID: {search_id}")
                return search_id
            else:
                st.error("❌ Error al guardar datos.")
                return None
                
        except Exception as e:
            st.error(f"❌ Error en add_search_result: {str(e)}")
            st.error(traceback.format_exc())
            return None
    
    def _sanitize_data(self, data):
        """
        Sanitiza recursivamente datos para asegurarse de que sean serializables.
        """
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, (int, float, str, bool)) or data is None:
            return data
        elif hasattr(data, 'item'):  # Objetos numpy
            return data.item()
        else:
            return str(data)
    
    def get_search_by_id(self, search_id):
        """
        Obtiene un resultado de búsqueda por su ID.
        """
        for search in self.data["searches"]:
            if search["id"] == search_id:
                return search
        return None
    
    def get_searches_by_category(self, category_id):
        """
        Obtiene todos los resultados de búsqueda de una categoría.
        """
        return [search for search in self.data["searches"] if search["category_id"] == category_id]
    
    def get_all_searches(self):
        """
        Obtiene todos los resultados de búsqueda.
        """
        return self.data["searches"]
    
    def add_category(self, name, description=""):
        """
        Añade una nueva categoría.
        """
        category_id = f"cat_{int(time.time())}_{hash(name)%1000}"
        
        new_category = {
            "id": category_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        
        self.categories["categories"].append(new_category)
        self.save_categories()
        
        return category_id
    
    def get_all_categories(self):
        """
        Obtiene todas las categorías disponibles.
        """
        return self.categories["categories"]
    
    def get_category_by_id(self, category_id):
        """
        Obtiene una categoría por su ID.
        """
        for category in self.categories["categories"]:
            if category["id"] == category_id:
                return category
        return None


class DynamoDBStorage:
    """
    Clase para gestionar el almacenamiento persistente de datos
    usando AWS DynamoDB.
    """
    
    def __init__(self, region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
        """
        Inicializa el sistema de almacenamiento en DynamoDB.
        
        Args:
            region_name: Región de AWS
            aws_access_key_id: Access Key ID de AWS
            aws_secret_access_key: Secret Access Key de AWS
        """
        self.region_name = region_name or os.environ.get('AWS_DEFAULT_REGION')
        self.analyses_table_name = 'tech-trends-analyses'
        self.categories_table_name = 'tech-trends-categories'
        
        # Configurar credenciales
        session_kwargs = {}
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key,
                'region_name': self.region_name
            })
        elif os.environ.get('AWS_ACCESS_KEY_ID') and os.environ.get('AWS_SECRET_ACCESS_KEY'):
            session_kwargs.update({
                'aws_access_key_id': os.environ.get('AWS_ACCESS_KEY_ID'),
                'aws_secret_access_key': os.environ.get('AWS_SECRET_ACCESS_KEY'),
                'region_name': self.region_name
            })
        else:
            # Usar credenciales por defecto (perfil AWS, rol IAM, etc.)
            session_kwargs.update({'region_name': self.region_name})
        
        try:
            # Crear cliente DynamoDB
            self.dynamodb = boto3.resource('dynamodb', **session_kwargs)
            self.analyses_table = self.dynamodb.Table(self.analyses_table_name)
            self.categories_table = self.dynamodb.Table(self.categories_table_name)
            
            # Verificar conectividad
            self._verify_connection()
            
        except Exception as e:
            st.error(f"❌ Error al conectar con DynamoDB: {str(e)}")
            raise e
    
    def _verify_connection(self):
        """
        Verifica la conexión con DynamoDB.
        """
        try:
            # Verificar que las tablas existen
            self.analyses_table.table_status
            self.categories_table.table_status
            
            # Inicializar categoría por defecto si no existe
            self._ensure_default_category()
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ResourceNotFoundException':
                st.error("❌ Las tablas de DynamoDB no existen. Por favor, créalas primero.")
                st.info("📝 Tablas requeridas: tech-trends-analyses, tech-trends-categories")
            else:
                st.error(f"❌ Error de DynamoDB: {str(e)}")
            raise e
        except NoCredentialsError:
            st.error("❌ Credenciales de AWS no configuradas correctamente.")
            raise
    
    def _ensure_default_category(self):
        """
        Asegura que exista la categoría por defecto.
        """
        try:
            response = self.categories_table.get_item(Key={'category_id': 'default'})
            if 'Item' not in response:
                # Crear categoría por defecto
                self.categories_table.put_item(
                    Item={
                        'category_id': 'default',
                        'name': 'Sin categoría',
                        'description': 'Categoría por defecto',
                        'created_at': datetime.now().isoformat()
                    }
                )
        except Exception as e:
            st.warning(f"No se pudo verificar/crear categoría por defecto: {str(e)}")
    
    def add_search_result(self, search_data, category_id="default"):
        """
        Añade un resultado de búsqueda a DynamoDB.
        """
        try:
            # Generar ID único
            analysis_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Sanitizar datos
            sanitized_data = self._sanitize_data(search_data)
            
            # Preparar item para DynamoDB
            item = {
                'analysis_id': analysis_id,
                'timestamp': timestamp,
                'category_id': category_id,
                **sanitized_data
            }
            
            # Guardar en DynamoDB
            self.analyses_table.put_item(Item=item)
            
            st.success(f"✅ Análisis guardado correctamente en DynamoDB con ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            st.error(f"❌ Error al guardar en DynamoDB: {str(e)}")
            st.error(traceback.format_exc())
            return None
    
    def get_search_by_id(self, analysis_id):
        """
        Obtiene un análisis por su ID.
        """
        try:
            # DynamoDB requiere conocer tanto PK como SK para get_item eficiente
            # Como no conocemos el timestamp, usamos scan con filtro
            response = self.analyses_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('analysis_id').eq(analysis_id)
            )
            
            items = response.get('Items', [])
            if items:
                return items[0]  # Debería ser único
            return None
            
        except Exception as e:
            st.error(f"Error al obtener análisis: {str(e)}")
            return None
    
    def get_searches_by_category(self, category_id):
        """
        Obtiene todos los análisis de una categoría.
        """
        try:
            response = self.analyses_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('category_id').eq(category_id)
            )
            
            return response.get('Items', [])
            
        except Exception as e:
            st.error(f"Error al obtener análisis por categoría: {str(e)}")
            return []
    
    def get_all_searches(self):
        """
        Obtiene todos los análisis.
        """
        try:
            response = self.analyses_table.scan()
            analyses = response.get('Items', [])
            
            # Manejar paginación si hay muchos items
            while 'LastEvaluatedKey' in response:
                response = self.analyses_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                analyses.extend(response.get('Items', []))
            
            # Ordenar por timestamp (más reciente primero)
            analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return analyses
            
        except Exception as e:
            st.error(f"Error al obtener todos los análisis: {str(e)}")
            return []
    
    def add_category(self, name, description=""):
        """
        Añade una nueva categoría.
        """
        try:
            category_id = f"cat_{int(time.time())}_{abs(hash(name))%1000}"
            
            item = {
                'category_id': category_id,
                'name': name,
                'description': description,
                'created_at': datetime.now().isoformat()
            }
            
            self.categories_table.put_item(Item=item)
            
            return category_id
            
        except Exception as e:
            st.error(f"Error al crear categoría: {str(e)}")
            return None
    
    def get_all_categories(self):
        """
        Obtiene todas las categorías.
        """
        try:
            response = self.categories_table.scan()
            categories = response.get('Items', [])
            
            # Manejar paginación
            while 'LastEvaluatedKey' in response:
                response = self.categories_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                categories.extend(response.get('Items', []))
            
            # Ordenar por nombre
            categories.sort(key=lambda x: x.get('name', ''))
            
            return categories
            
        except Exception as e:
            st.error(f"Error al obtener categorías: {str(e)}")
            return [{"id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]
    
    def get_category_by_id(self, category_id):
        """
        Obtiene una categoría por su ID.
        """
        try:
            response = self.categories_table.get_item(Key={'category_id': category_id})
            return response.get('Item')
            
        except Exception as e:
            st.error(f"Error al obtener categoría: {str(e)}")
            return None
    
    def save_data(self):
        """
        Método de compatibilidad (no necesario en DynamoDB).
        """
        return True
    
    def save_categories(self):
        """
        Método de compatibilidad (no necesario en DynamoDB).
        """
        return True
    
    def _sanitize_data(self, data):
        """
        Sanitiza datos para DynamoDB.
        """
        if isinstance(data, dict):
            return {k: self._sanitize_data(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        elif isinstance(data, (int, float, str, bool)) or data is None:
            return data
        elif hasattr(data, 'item'):  # Objetos numpy
            return data.item()
        else:
            return str(data)


class SCurveDatabase:
    """
    Clase específica para gestionar datos de análisis de curvas S,
    compatible tanto con LocalStorage como con DynamoDBStorage.
    """
    
    def __init__(self, storage):
        """
        Inicializa la base de datos de curvas S.
        
        Args:
            storage: Instancia de LocalStorage o DynamoDBStorage
        """
        self.storage = storage
    
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
            if paper_data:
                paper_data = {str(k): float(v) for k, v in paper_data.items()}
            
            if patent_data:
                patent_data = {str(k): float(v) for k, v in patent_data.items()}
            
            # Limpiar métricas
            if paper_metrics:
                paper_metrics = {k: v for k, v in paper_metrics.items() 
                            if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
                paper_metrics = {k: float(v) if hasattr(v, 'item') else v 
                            for k, v in paper_metrics.items()}
                
            if patent_metrics:
                patent_metrics = {k: v for k, v in patent_metrics.items() 
                            if isinstance(v, (str, int, float, bool, list, dict)) or v is None}
                patent_metrics = {k: float(v) if hasattr(v, 'item') else v 
                                for k, v in patent_metrics.items()}
            
            # Crear estructura de datos
            if not analysis_name:
                terms = str(query).replace('TITLE(', '').replace(')', '').replace('"', '')
                analysis_name = f"Análisis de {terms[:30]}..." if len(terms) > 30 else f"Análisis de {terms}"
            
            analysis_data = {
                "name": analysis_name,
                "query": str(query),
                "execution_date": datetime.now().isoformat(),
                "analysis_type": "s_curve",
                "paper_data": paper_data,
                "patent_data": patent_data,
                "paper_metrics": paper_metrics,
                "patent_metrics": patent_metrics
            }
            
            # Guardar usando el storage configurado
            search_id = self.storage.add_search_result(analysis_data, category_id)
            return search_id
                
        except Exception as e:
            st.error(f"❌ Error al preparar datos para guardar: {str(e)}")
            st.error(traceback.format_exc())
            return None
    
    def get_category_analysis(self, category_id):
        """
        Obtiene todos los análisis de una categoría.
        """
        return self.storage.get_searches_by_category(category_id)
    
    def get_analysis_by_id(self, analysis_id):
        """
        Obtiene un análisis específico por su ID.
        """
        return self.storage.get_search_by_id(analysis_id)
    
    def create_category(self, name, description=""):
        """
        Crea una nueva categoría.
        """
        return self.storage.add_category(name, description)
    
    def get_all_categories(self):
        """
        Obtiene todas las categorías disponibles.
        """
        return self.storage.get_all_categories()


def save_analysis_direct(analysis_name, query, paper_data=None, patent_data=None, paper_metrics=None, patent_metrics=None):
    """
    Función directa para guardar análisis sin pasar por el sistema de almacenamiento.
    """
    try:
        os.makedirs("./data", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./data/analysis_{timestamp}.json"
        
        # Sanitizar datos
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        if paper_data:
            paper_data = {str(k): float(v) for k, v in paper_data.items()}
        
        if patent_data:
            patent_data = {str(k): float(v) for k, v in patent_data.items()}
        
        if paper_metrics:
            paper_metrics = sanitize(paper_metrics)
        
        if patent_metrics:
            patent_metrics = sanitize(patent_metrics)
        
        data = {
            "name": analysis_name,
            "query": str(query),
            "timestamp": datetime.now().isoformat(),
            "paper_data": paper_data,
            "patent_data": patent_data,
            "paper_metrics": paper_metrics,
            "patent_metrics": patent_metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        
        st.success(f"✅ Análisis guardado directamente en: {filename}")
        return filename
    
    except Exception as e:
        st.error(f"❌ Error al guardar directamente: {str(e)}")
        st.error(traceback.format_exc())
        return None


def initialize_database(storage_type="local", **kwargs):
    """
    Inicializa y devuelve una instancia de la base de datos.
    
    Args:
        storage_type: "local" o "dynamodb"
        **kwargs: Argumentos adicionales para el storage
        
    Returns:
        SCurveDatabase: Instancia de la base de datos o None si hay error
    """
    try:
        if storage_type == "local":
            # Usar almacenamiento local
            local_storage = LocalStorage()
            return SCurveDatabase(local_storage)
        
        elif storage_type == "dynamodb":
            if not BOTO3_AVAILABLE:
                st.error("❌ boto3 no está disponible. Instala con: pip install boto3")
                return None
            
            # Usar DynamoDB
            dynamo_storage = DynamoDBStorage(**kwargs)
            return SCurveDatabase(dynamo_storage)
        
        else:
            st.error(f"❌ Tipo de almacenamiento no soportado: {storage_type}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error al inicializar base de datos: {str(e)}")
        st.error(traceback.format_exc())
        return None


# Función de compatibilidad con código existente
def initialize_github_db(repo_owner=None, repo_name=None, token=None, use_local=False):
    """
    Función de compatibilidad. Ahora redirige a initialize_database.
    """
    if use_local:
        return initialize_database("local")
    else:
        # Si se especificaron parámetros de GitHub, ignorarlos y usar local
        st.warning("⚠️ El almacenamiento en GitHub ya no está soportado. Usando almacenamiento local.")
        return initialize_database("local")