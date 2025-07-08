# src/data_storage.py - SOLO DYNAMODB
import json
import streamlit as st
import time
import traceback
import uuid
from datetime import datetime, timezone
from decimal import Decimal

# AWS DynamoDB
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    st.error("📦 Se requiere instalar boto3. Ejecuta: pip install boto3")
    BOTO3_AVAILABLE = False

class DynamoDBStorage:
    """
    Clase para gestionar el almacenamiento persistente de datos
    usando AWS DynamoDB únicamente.
    """
    
    def __init__(self, region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
        """
        Inicializa el sistema de almacenamiento en DynamoDB.
        """
        self.region_name = region_name
        self.analyses_table_name = 'tech-trends-analyses'
        self.categories_table_name = 'tech-trends-categories'
        
        # Configurar credenciales
        session_kwargs = {
            'region_name': self.region_name
        }
        
        if aws_access_key_id and aws_secret_access_key:
            session_kwargs.update({
                'aws_access_key_id': aws_access_key_id,
                'aws_secret_access_key': aws_secret_access_key
            })
        
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
        """Verifica la conexión con DynamoDB."""
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
        """Asegura que exista la categoría por defecto."""
        try:
            response = self.categories_table.get_item(Key={'category_id': 'default'})
            if 'Item' not in response:
                self.categories_table.put_item(
                    Item={
                        'category_id': 'default',
                        'name': 'Sin categoría',
                        'description': 'Categoría por defecto',
                        'created_at': datetime.now(timezone.utc).isoformat()
                    }
                )
        except Exception as e:
            st.warning(f"No se pudo verificar/crear categoría por defecto: {str(e)}")
    
    def _generate_unique_id(self, prefix="analysis"):
        """Genera un ID único garantizado."""
        timestamp = int(time.time() * 1000)  # Milisegundos para mayor unicidad
        unique_part = str(uuid.uuid4())[:8]
        return f"{prefix}_{timestamp}_{unique_part}"
    
    def _sanitize_for_dynamodb(self, data):
        """Sanitiza datos para DynamoDB removiendo NaN e Infinity."""
        import math
        
        def clean_value(value):
            if isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(item) for item in value]
            elif isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    return 0.0
                return value
            elif hasattr(value, 'item'):  # numpy types
                try:
                    float_val = float(value.item())
                    if math.isnan(float_val) or math.isinf(float_val):
                        return 0.0
                    return float_val
                except:
                    return 0.0
            else:
                return value
        
        return clean_value(data)
    
    def _convert_floats_to_decimal(self, obj):
        """Convierte floats a Decimal para DynamoDB."""
        import math
        
        if isinstance(obj, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimal(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return Decimal('0')
            try:
                return Decimal(str(round(obj, 6)))
            except:
                return Decimal('0')
        elif isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        elif hasattr(obj, 'item'):  # numpy types
            try:
                value = float(obj.item())
                if math.isnan(value) or math.isinf(value):
                    return Decimal('0')
                return Decimal(str(round(value, 6)))
            except:
                return Decimal('0')
        else:
            return str(obj)
    
    def _convert_decimals_to_float(self, obj):
        """Convierte Decimals de vuelta a float al leer."""
        if isinstance(obj, dict):
            return {k: self._convert_decimals_to_float(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_decimals_to_float(item) for item in obj]
        elif isinstance(obj, Decimal):
            return float(obj)
        else:
            return obj
    
    def add_search_result(self, search_data, category_id="default"):
        """Añade un resultado de búsqueda a DynamoDB."""
        try:
            # Generar ID único garantizado
            analysis_id = self._generate_unique_id("analysis")
            timestamp = datetime.now(timezone.utc).isoformat()
            
            # Sanitizar datos
            sanitized_data = self._sanitize_for_dynamodb(search_data)
            sanitized_data = self._convert_floats_to_decimal(sanitized_data)
            
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
            return None
    
    def get_search_by_id(self, analysis_id):
        """Obtiene un análisis por su ID."""
        try:
            response = self.analyses_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('analysis_id').eq(analysis_id)
            )
            
            items = response.get('Items', [])
            if items:
                return self._convert_decimals_to_float(items[0])
            return None
            
        except Exception as e:
            st.error(f"Error al obtener análisis: {str(e)}")
            return None
    
    def get_searches_by_category(self, category_id):
        """Obtiene todos los análisis de una categoría."""
        try:
            response = self.analyses_table.scan(
                FilterExpression=boto3.dynamodb.conditions.Attr('category_id').eq(category_id)
            )
            
            items = response.get('Items', [])
            return [self._convert_decimals_to_float(item) for item in items]
            
        except Exception as e:
            st.error(f"Error al obtener análisis por categoría: {str(e)}")
            return []
    
    def get_all_searches(self):
        """Obtiene todos los análisis."""
        try:
            response = self.analyses_table.scan()
            analyses = response.get('Items', [])
            
            # Manejar paginación
            while 'LastEvaluatedKey' in response:
                response = self.analyses_table.scan(
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                analyses.extend(response.get('Items', []))
            
            # Convertir Decimals y ordenar
            analyses = [self._convert_decimals_to_float(item) for item in analyses]
            analyses.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
            
            return analyses
            
        except Exception as e:
            st.error(f"Error al obtener todos los análisis: {str(e)}")
            return []
    
    def add_category(self, name, description=""):
        """Añade una nueva categoría."""
        try:
            category_id = self._generate_unique_id("cat")
            
            item = {
                'category_id': category_id,
                'name': name,
                'description': description,
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            self.categories_table.put_item(Item=item)
            return category_id
            
        except Exception as e:
            st.error(f"Error al crear categoría: {str(e)}")
            return None
    
    def get_all_categories(self):
        """Obtiene todas las categorías."""
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
            return [{"category_id": "default", "name": "Sin categoría", "description": "Categoría por defecto"}]
    
    def get_category_by_id(self, category_id):
        """Obtiene una categoría por su ID."""
        try:
            response = self.categories_table.get_item(Key={'category_id': category_id})
            return response.get('Item')
            
        except Exception as e:
            st.error(f"Error al obtener categoría: {str(e)}")
            return None
    
    def delete_item(self, analysis_id, timestamp):
        """Elimina un item específico."""
        try:
            self.analyses_table.delete_item(
                Key={
                    'analysis_id': analysis_id,
                    'timestamp': timestamp
                }
            )
            return True
        except Exception as e:
            st.error(f"Error eliminando item: {str(e)}")
            return False
    
    def update_item(self, analysis_id, timestamp, updates):
        """Actualiza un item específico."""
        try:
            # Preparar la expresión de actualización
            update_expression = "SET "
            expression_attribute_values = {}
            
            for key, value in updates.items():
                update_expression += f"{key} = :{key}, "
                expression_attribute_values[f":{key}"] = self._convert_floats_to_decimal(value)
            
            # Remover la última coma
            update_expression = update_expression.rstrip(", ")
            
            self.analyses_table.update_item(
                Key={
                    'analysis_id': analysis_id,
                    'timestamp': timestamp
                },
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values
            )
            return True
        except Exception as e:
            st.error(f"Error actualizando item: {str(e)}")
            return False


class SCurveDatabase:
    """Clase específica para gestionar datos de análisis de curvas S con DynamoDB."""
    
    def __init__(self, storage):
        """Inicializa la base de datos de curvas S."""
        self.storage = storage
    
    def save_s_curve_analysis(self, 
                          query, 
                          paper_data=None, 
                          patent_data=None, 
                          paper_metrics=None, 
                          patent_metrics=None,
                          category_id="default",
                          analysis_name=None):
        """Guarda un análisis completo de curva S."""
        try:
            # Limpiar y sanitizar datos
            if paper_data:
                paper_data = {str(k): float(v) for k, v in paper_data.items()}
            
            if patent_data:
                patent_data = {str(k): float(v) for k, v in patent_data.items()}
            
            if paper_metrics:
                paper_metrics = self.storage._sanitize_for_dynamodb(paper_metrics)
                
            if patent_metrics:
                patent_metrics = self.storage._sanitize_for_dynamodb(patent_metrics)
            
            # Crear estructura de datos
            if not analysis_name:
                terms = str(query).replace('TITLE(', '').replace(')', '').replace('"', '')
                analysis_name = f"Análisis de {terms[:30]}..." if len(terms) > 30 else f"Análisis de {terms}"
            
            analysis_data = {
                "name": analysis_name,
                "query": str(query),
                "execution_date": datetime.now(timezone.utc).isoformat(),
                "analysis_type": "s_curve",
                "paper_data": paper_data,
                "patent_data": patent_data,
                "paper_metrics": paper_metrics,
                "patent_metrics": patent_metrics
            }
            
            return self.storage.add_search_result(analysis_data, category_id)
                
        except Exception as e:
            st.error(f"❌ Error al preparar datos para guardar: {str(e)}")
            return None
    
    def get_category_analysis(self, category_id):
        """Obtiene todos los análisis de una categoría."""
        return self.storage.get_searches_by_category(category_id)
    
    def get_analysis_by_id(self, analysis_id):
        """Obtiene un análisis específico por su ID."""
        return self.storage.get_search_by_id(analysis_id)
    
    def create_category(self, name, description=""):
        """Crea una nueva categoría."""
        return self.storage.add_category(name, description)
    
    def get_all_categories(self):
        """Obtiene todas las categorías disponibles."""
        return self.storage.get_all_categories()


def initialize_database(storage_type="dynamodb", **kwargs):
    """
    Inicializa y devuelve una instancia de la base de datos.
    Solo soporta DynamoDB.
    """
    try:
        if storage_type != "dynamodb":
            st.error("❌ Solo se soporta DynamoDB. Cambiando a DynamoDB automáticamente.")
        
        if not BOTO3_AVAILABLE:
            st.error("❌ boto3 no está disponible. Instala con: pip install boto3")
            return None
        
        # Usar DynamoDB
        dynamo_storage = DynamoDBStorage(**kwargs)
        return SCurveDatabase(dynamo_storage)
            
    except Exception as e:
        st.error(f"❌ Error al inicializar base de datos: {str(e)}")
        return None