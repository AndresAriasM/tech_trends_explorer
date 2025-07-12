# src/hype_cycle_storage.py - VERSIÓN OPTIMIZADA PARA RENDIMIENTO Y CORRECCIÓN DE BUGS
import streamlit as st
import pandas as pd
import time
import json
import uuid
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, asdict
from enum import Enum
import math

# IMPORTAR BOTO3 CONDITIONS CORRECTAMENTE
try:
    import boto3
    from boto3.dynamodb.conditions import Attr, Key
    BOTO3_CONDITIONS_AVAILABLE = True
except ImportError:
    BOTO3_CONDITIONS_AVAILABLE = False

# Configurar logging real (no visible en frontend)
logger = logging.getLogger(__name__)

class HypeCyclePhase(Enum):
    """Enum para las fases del Hype Cycle"""
    INNOVATION_TRIGGER = "Innovation Trigger"
    PEAK_EXPECTATIONS = "Peak of Inflated Expectations"
    TROUGH_DISILLUSIONMENT = "Trough of Disillusionment"
    SLOPE_ENLIGHTENMENT = "Slope of Enlightenment"
    PLATEAU_PRODUCTIVITY = "Plateau of Productivity"
    PRE_INNOVATION = "Pre-Innovation Trigger"

@dataclass
class HypeCycleMetrics:
    """Métricas específicas del análisis de Hype Cycle"""
    phase: str
    confidence: float
    total_mentions: int
    peak_mentions: int
    latest_year: int
    sentiment_avg: float
    sentiment_trend: float
    hype_cycle_position_x: float = 50.0
    hype_cycle_position_y: float = 50.0
    time_to_plateau: str = "N/A"
    innovation_trigger_year: Optional[int] = None
    peak_year: Optional[int] = None
    trough_year: Optional[int] = None
    inflection_points: Optional[Dict] = None

@dataclass
class HypeCycleQuery:
    """Estructura para almacenar consultas de Hype Cycle"""
    query_id: str
    category_id: str
    search_query: str
    search_terms: List[Dict]
    execution_date: str
    api_usage: Dict
    hype_metrics: HypeCycleMetrics
    yearly_stats: List[Dict]
    news_results: List[Dict]
    search_parameters: Dict
    data_quality: Dict
    processing_time: float
    technology_name: str = ""
    category_name: str = ""
    last_updated: str = ""
    is_active: bool = True
    technology_description: str = ""
    created_by: str = "system"
    version: str = "1.0"
    notes: str = ""

class HypeCycleStorage:
    """Clase especializada para gestionar almacenamiento de consultas Hype Cycle - VERSIÓN OPTIMIZADA"""
    
    def __init__(self, db_storage):
        """Inicializa con el storage de DynamoDB"""
        self.storage = db_storage
        
        # Verificar que es DynamoDB
        if not hasattr(db_storage, 'dynamodb'):
            raise ValueError("HypeCycleStorage requiere DynamoDB storage")
        
        # CACHE para evitar consultas repetitivas
        self._cache = {}
        self._cache_timeout = 300  # 5 minutos
        self._last_cache_time = {}
    
    def _get_cache_key(self, operation: str, params: str = "") -> str:
        """Genera clave de cache"""
        return f"{operation}_{params}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica si el cache es válido"""
        if cache_key not in self._cache:
            return False
        
        last_time = self._last_cache_time.get(cache_key, 0)
        return (time.time() - last_time) < self._cache_timeout
    
    def _set_cache(self, cache_key: str, data: Any):
        """Establece cache"""
        self._cache[cache_key] = data
        self._last_cache_time[cache_key] = time.time()
    
    def _get_cache(self, cache_key: str) -> Any:
        """Obtiene datos del cache"""
        return self._cache.get(cache_key)
    
    def _invalidate_cache(self):
        """Invalida todo el cache"""
        self._cache.clear()
        self._last_cache_time.clear()
    
    def _generate_unique_query_id(self):
        """Genera un ID único garantizado para queries de Hype Cycle"""
        timestamp = int(time.time() * 1000)
        unique_part = str(uuid.uuid4())[:12]
        return f"hype_{timestamp}_{unique_part}"
    
    def save_hype_cycle_query(self, 
                            search_query: str,
                            search_terms: List[Dict],
                            hype_analysis_results: Dict,
                            news_results: List[Dict],
                            category_id: str = "default",
                            search_parameters: Dict = None,
                            notes: str = "",
                            technology_name: str = None,
                            technology_description: str = "") -> str:
        """
        VERSIÓN OPTIMIZADA: Guarda una consulta completa de Hype Cycle
        """
        try:
            # Importar el positioner
            from hype_cycle_positioning import HypeCyclePositioner
            positioner = HypeCyclePositioner()
            
            # Generar ID único
            query_id = self._generate_unique_query_id()
            logger.info(f"Generando query con ID: {query_id}")
            
            # Procesar métricas del Hype Cycle
            hype_metrics = self._extract_hype_metrics(hype_analysis_results)
            
            # Calcular posición en la gráfica
            pos_x, pos_y = positioner.calculate_position(
                hype_metrics.phase, 
                hype_metrics.confidence,
                hype_metrics.total_mentions
            )
            hype_metrics.hype_cycle_position_x = pos_x
            hype_metrics.hype_cycle_position_y = pos_y
            hype_metrics.time_to_plateau = positioner.estimate_time_to_plateau(
                hype_metrics.phase, 
                hype_metrics.confidence
            )
            
            # Obtener información de categoría
            category_name = self._get_category_name(category_id)
            
            # Generar nombre de tecnología si no se proporciona
            if not technology_name:
                technology_name = self._extract_technology_name(search_query, search_terms)
            
            # Crear timestamp
            execution_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Crear estructura simplificada para DynamoDB
            item = {
                # Claves requeridas
                'analysis_id': query_id,
                'timestamp': execution_timestamp,
                
                # CAMPO CRÍTICO: query_id también como campo separado
                'query_id': query_id,
                
                # Datos básicos
                'analysis_type': 'hype_cycle',
                'category_id': str(category_id),  # Asegurar que sea string
                'category_name': category_name,
                'search_query': search_query,
                'technology_name': technology_name,
                'technology_description': technology_description,
                'notes': notes,
                'execution_date': execution_timestamp,
                'last_updated': execution_timestamp,
                'is_active': True,
                'version': '1.0',
                'created_by': 'hype_cycle_analyzer',
                
                # Métricas del Hype Cycle
                'hype_metrics': {
                    'phase': hype_metrics.phase,
                    'confidence': self._safe_float(hype_metrics.confidence),
                    'total_mentions': self._safe_int(hype_metrics.total_mentions),
                    'peak_mentions': self._safe_int(hype_metrics.peak_mentions),
                    'latest_year': self._safe_int(hype_metrics.latest_year),
                    'sentiment_avg': self._safe_float(hype_metrics.sentiment_avg),
                    'sentiment_trend': self._safe_float(hype_metrics.sentiment_trend),
                    'hype_cycle_position_x': self._safe_float(hype_metrics.hype_cycle_position_x),
                    'hype_cycle_position_y': self._safe_float(hype_metrics.hype_cycle_position_y),
                    'time_to_plateau': hype_metrics.time_to_plateau
                },
                
                # Términos de búsqueda
                'search_terms': self._clean_search_terms(search_terms),
                
                # Información de la API
                'api_usage': {
                    'total_results': len(news_results),
                    'search_timestamp': execution_timestamp,
                    'api_provider': 'SerpAPI'
                },
                
                # Parámetros de búsqueda
                'search_parameters': search_parameters or {},
                
                # Calidad de datos
                'data_quality': {
                    'total_results': len(news_results),
                    'quality_score': self._calculate_simple_quality(news_results)
                },
                
                # Sample de noticias (limitado)
                'news_sample': self._create_news_sample(news_results[:5]),
                
                # Tiempo de procesamiento
                'processing_time': 0.0
            }
            
            # Convertir a Decimal para DynamoDB
            final_item = self.storage._convert_floats_to_decimal(item)
            
            # Guardar en DynamoDB
            self.storage.analyses_table.put_item(Item=final_item)
            logger.info(f"Query {query_id} guardado exitosamente")
            
            # Invalidar cache después de guardar
            self._invalidate_cache()
            
            return query_id
            
        except Exception as e:
            logger.error(f"Error guardando query: {str(e)}")
            return None
    
    def _safe_float(self, value, default=0.0):
        """Convierte valor a float de forma segura"""
        try:
            if isinstance(value, Decimal):
                return float(value)
            elif isinstance(value, (int, float)):
                if math.isnan(value) or math.isinf(value):
                    return default
                return float(value)
            elif isinstance(value, str):
                return float(value.replace(',', ''))
            else:
                return default
        except:
            return default
    
    def _safe_int(self, value, default=0):
        """Convierte valor a int de forma segura"""
        try:
            if isinstance(value, Decimal):
                return int(value)
            elif isinstance(value, (int, float)):
                return int(value)
            elif isinstance(value, str):
                return int(float(value.replace(',', '')))
            else:
                return default
        except:
            return default
    
    def _clean_search_terms(self, search_terms):
        """Limpia términos de búsqueda para DynamoDB"""
        cleaned = []
        for term in search_terms:
            if isinstance(term, dict) and term.get('value'):
                cleaned.append({
                    'value': str(term.get('value', '')),
                    'operator': str(term.get('operator', 'AND')),
                    'exact_match': bool(term.get('exact_match', False))
                })
        return cleaned
    
    def _calculate_simple_quality(self, news_results):
        """Calcula calidad simple de datos"""
        if not news_results:
            return 0.0
        
        total = len(news_results)
        with_date = sum(1 for r in news_results if r.get('date'))
        return (with_date / total) if total > 0 else 0.0
    
    def _create_news_sample(self, news_results):
        """Crea muestra limitada de noticias"""
        sample = []
        for result in news_results:
            if isinstance(result, dict):
                sample.append({
                    'title': str(result.get('title', ''))[:100],
                    'date': str(result.get('date', '')),
                    'source': str(result.get('source', ''))[:50]
                })
        return sample
    
    def _get_category_name(self, category_id: str) -> str:
        """Obtiene el nombre de la categoría de forma segura"""
        try:
            category = self.storage.get_category_by_id(category_id)
            return category.get("name") if category else "Sin categoría"
        except:
            return "Sin categoría"
    
    def _extract_hype_metrics(self, hype_results: Dict) -> HypeCycleMetrics:
        """Extrae métricas específicas del análisis de Hype Cycle"""
        try:
            metrics_data = hype_results.get('metrics', {})
            
            return HypeCycleMetrics(
                phase=hype_results.get('phase', 'Unknown'),
                confidence=self._safe_float(hype_results.get('confidence', 0.0)),
                total_mentions=self._safe_int(metrics_data.get('total_mentions', 0)),
                peak_mentions=self._safe_int(metrics_data.get('peak_mentions', 0)),
                latest_year=self._safe_int(metrics_data.get('latest_year', datetime.now().year)),
                sentiment_avg=self._safe_float(hype_results.get('sentiment_avg', 0.0)),
                sentiment_trend=0.0
            )
            
        except Exception as e:
            logger.warning(f"Error extrayendo métricas: {str(e)}")
            return HypeCycleMetrics(
                phase="Unknown",
                confidence=0.0,
                total_mentions=0,
                peak_mentions=0,
                latest_year=datetime.now().year,
                sentiment_avg=0.0,
                sentiment_trend=0.0
            )
    
    def _extract_technology_name(self, search_query: str, search_terms: List[Dict]) -> str:
        """Extrae un nombre de tecnología limpio de la consulta"""
        for term in search_terms:
            value = term.get('value', '').strip().strip('"')
            if len(value) > 2 and value.lower() not in ['and', 'or', 'not']:
                return value.title()
        
        # Fallback: limpiar la query
        clean_query = search_query.replace('"', '').replace(' AND ', ' ').replace(' OR ', ' ')
        clean_query = clean_query.replace(' NOT ', ' ').replace('after:', '').replace('before:', '')
        words = [w for w in clean_query.split() if not w.isdigit() and len(w) > 2]
        return ' '.join(words[:2]).title() if words else "Tecnología"

    # ===== MÉTODOS DE CONSULTA OPTIMIZADOS CON CACHE =====
    
    @st.cache_data(ttl=300)  # Cache de Streamlit por 5 minutos
    def get_queries_by_category(_self, category_id: str) -> List[Dict]:
        """OPTIMIZADO: Obtiene todas las consultas de Hype Cycle de una categoría con cache"""
        try:
            cache_key = _self._get_cache_key("queries_by_category", str(category_id))
            
            # Verificar cache
            if _self._is_cache_valid(cache_key):
                return _self._get_cache(cache_key)
            
            if not BOTO3_CONDITIONS_AVAILABLE:
                result = _self._get_queries_by_category_fallback(category_id)
                _self._set_cache(cache_key, result)
                return result
            
            # Usar condiciones de boto3 correctamente
            items = []
            
            # Primera consulta
            response = _self.storage.analyses_table.scan(
                FilterExpression=Attr('category_id').eq(str(category_id)) & Attr('analysis_type').eq('hype_cycle')
            )
            
            items.extend(response.get('Items', []))
            
            # Manejar paginación
            while 'LastEvaluatedKey' in response:
                response = _self.storage.analyses_table.scan(
                    FilterExpression=Attr('category_id').eq(str(category_id)) & Attr('analysis_type').eq('hype_cycle'),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            # Convertir decimales y retornar
            converted_items = [_self.storage._convert_decimals_to_float(item) for item in items]
            
            # Filtro adicional en memoria por si acaso
            filtered_items = []
            for item in converted_items:
                item_category = str(item.get('category_id', ''))
                item_type = item.get('analysis_type', '')
                
                if item_category == str(category_id) and item_type == 'hype_cycle':
                    filtered_items.append(item)
            
            # Guardar en cache
            _self._set_cache(cache_key, filtered_items)
            
            logger.info(f"Categoría {category_id}: encontrados {len(filtered_items)} items")
            return filtered_items
                
        except Exception as e:
            logger.error(f"Error obteniendo consultas por categoría: {str(e)}")
            # Fallback
            return _self._get_queries_by_category_fallback(category_id)
    
    def _get_queries_by_category_fallback(self, category_id: str) -> List[Dict]:
        """Método fallback sin usar condiciones boto3"""
        try:
            # Obtener todos los items y filtrar en memoria
            all_items = self.get_all_hype_cycle_queries()
            
            filtered_items = []
            for item in all_items:
                item_category = str(item.get('category_id', ''))
                if item_category == str(category_id):
                    filtered_items.append(item)
            
            logger.info(f"Fallback - Categoría {category_id}: {len(filtered_items)} items")
            return filtered_items
            
        except Exception as e:
            logger.error(f"Error en fallback: {str(e)}")
            return []

    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        """CORREGIDO: Obtiene una consulta específica por ID con búsqueda robusta"""
        try:
            cache_key = self._get_cache_key("query_by_id", query_id)
            
            # Verificar cache
            if self._is_cache_valid(cache_key):
                return self._get_cache(cache_key)
            
            result = None
            
            # PASO 1: Búsqueda por scan con condiciones (igual que en delete y move)
            if not BOTO3_CONDITIONS_AVAILABLE:
                result = self._get_query_by_id_fallback_robust(query_id)
            else:
                try:
                    # Buscar por analysis_id o query_id
                    response = self.storage.analyses_table.scan(
                        FilterExpression=(
                            Attr('analysis_id').eq(query_id) | 
                            Attr('query_id').eq(query_id)
                        ) & Attr('analysis_type').eq('hype_cycle')
                    )
                    
                    items = response.get('Items', [])
                    if items:
                        result = self.storage._convert_decimals_to_float(items[0])
                        logger.info(f"Query encontrado por scan exacto: {result.get('analysis_id')}")
                    
                except Exception as e:
                    logger.error(f"Error en scan para get_query_by_id: {str(e)}")
            
            # PASO 2: Si no se encontró, búsqueda exhaustiva con coincidencias parciales
            if not result:
                logger.info(f"Query {query_id} no encontrado con scan, intentando búsqueda exhaustiva...")
                all_items = self.get_all_hype_cycle_queries()
                
                for item in all_items:
                    item_query_id = item.get('query_id', '')
                    item_analysis_id = item.get('analysis_id', '')
                    
                    # Búsqueda exacta primero
                    if item_query_id == query_id or item_analysis_id == query_id:
                        result = item
                        logger.info(f"Query encontrado por coincidencia exacta: query_id={item_query_id}, analysis_id={item_analysis_id}")
                        break
                    
                    # Búsqueda parcial como fallback
                    if (query_id in item_query_id or query_id in item_analysis_id or
                        item_query_id.startswith(query_id) or item_analysis_id.startswith(query_id)):
                        result = item
                        logger.info(f"Query encontrado por coincidencia parcial: query_id={item_query_id}, analysis_id={item_analysis_id}")
                        break
            
            # Guardar en cache
            self._set_cache(cache_key, result)
            return result
                
        except Exception as e:
            logger.error(f"Error obteniendo consulta por ID: {str(e)}")
            return self._get_query_by_id_fallback_robust(query_id)
    
    def _get_query_by_id_fallback_robust(self, query_id: str) -> Optional[Dict]:
        """Método fallback robusto para buscar por ID"""
        try:
            all_items = self.get_all_hype_cycle_queries()
            
            # Búsqueda exacta primero
            for item in all_items:
                if (item.get('analysis_id') == query_id or 
                    item.get('query_id') == query_id):
                    return item
            
            # Búsqueda parcial como fallback
            for item in all_items:
                query_id_partial = item.get('query_id', '')
                analysis_id_partial = item.get('analysis_id', '')
                
                if (query_id in query_id_partial or query_id in analysis_id_partial or
                    query_id_partial.startswith(query_id) or analysis_id_partial.startswith(query_id)):
                    return item
            
            return None
            
        except Exception as e:
            logger.error(f"Error en fallback robusto: {str(e)}")
            return None
    
    def _get_query_by_id_fallback(self, query_id: str) -> Optional[Dict]:
        """Método fallback para buscar por ID"""
        try:
            all_items = self.get_all_hype_cycle_queries()
            
            for item in all_items:
                if (item.get('analysis_id') == query_id or 
                    item.get('query_id') == query_id):
                    return item
            
            return None
            
        except Exception as e:
            logger.error(f"Error en fallback búsqueda por ID: {str(e)}")
            return None

    @st.cache_data(ttl=300)  # Cache de Streamlit por 5 minutos
    def get_all_hype_cycle_queries(_self) -> List[Dict]:
        """OPTIMIZADO: Obtiene todas las consultas de Hype Cycle con cache"""
        try:
            cache_key = _self._get_cache_key("all_queries")
            
            # Verificar cache
            if _self._is_cache_valid(cache_key):
                return _self._get_cache(cache_key)
            
            if not BOTO3_CONDITIONS_AVAILABLE:
                result = _self._get_all_hype_cycle_queries_fallback()
                _self._set_cache(cache_key, result)
                return result
            
            items = []
            
            # Primera consulta
            response = _self.storage.analyses_table.scan(
                FilterExpression=Attr('analysis_type').eq('hype_cycle')
            )
            
            items.extend(response.get('Items', []))
            
            # Manejar paginación
            while 'LastEvaluatedKey' in response:
                response = _self.storage.analyses_table.scan(
                    FilterExpression=Attr('analysis_type').eq('hype_cycle'),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            # Convertir decimales
            converted_items = [_self.storage._convert_decimals_to_float(item) for item in items]
            
            # Ordenar por fecha (más reciente primero)
            converted_items.sort(key=lambda x: x.get('execution_date', ''), reverse=True)
            
            # Guardar en cache
            _self._set_cache(cache_key, converted_items)
            
            logger.info(f"Total Hype Cycle queries encontradas: {len(converted_items)}")
            return converted_items
                
        except Exception as e:
            logger.error(f"Error obteniendo todas las consultas: {str(e)}")
            return _self._get_all_hype_cycle_queries_fallback()
    
    def _get_all_hype_cycle_queries_fallback(self) -> List[Dict]:
        """Método fallback para obtener todas las consultas"""
        try:
            # Usar el método base de storage
            all_analyses = self.storage.get_all_searches()
            
            # Filtrar solo hype_cycle
            hype_queries = []
            for analysis in all_analyses:
                if analysis.get('analysis_type') == 'hype_cycle':
                    hype_queries.append(analysis)
            
            logger.info(f"Fallback - Total Hype Cycle queries: {len(hype_queries)}")
            return hype_queries
            
        except Exception as e:
            logger.error(f"Error en fallback obtener todas: {str(e)}")
            return []
    
    # ===== MÉTODOS DE GESTIÓN CORREGIDOS =====
    
    def delete_query(self, query_id: str) -> bool:
        """CORREGIDO: Elimina una consulta específica con mejor manejo de IDs inconsistentes"""
        try:
            # Invalidar cache primero
            self._invalidate_cache()
            
            logger.info(f"Intentando eliminar query con ID: {query_id}")
            
            # PASO 1: Intentar encontrar el item usando múltiples métodos
            target_item = None
            
            # Método A: Buscar por query_id o analysis_id usando scan
            if not BOTO3_CONDITIONS_AVAILABLE:
                target_item = self._find_item_for_deletion_fallback(query_id)
            else:
                try:
                    response = self.storage.analyses_table.scan(
                        FilterExpression=(
                            Attr('analysis_id').eq(query_id) | 
                            Attr('query_id').eq(query_id)
                        ) & Attr('analysis_type').eq('hype_cycle')
                    )
                    
                    items = response.get('Items', [])
                    if items:
                        target_item = items[0]
                        logger.info(f"Item encontrado usando scan: {target_item.get('analysis_id')}")
                    
                except Exception as e:
                    logger.error(f"Error en scan: {str(e)}")
            
            # Método B: Si no se encontró, buscar en todos los items
            if not target_item:
                logger.info("Item no encontrado con scan, intentando búsqueda exhaustiva...")
                all_items = self.get_all_hype_cycle_queries()
                
                for item in all_items:
                    item_query_id = item.get('query_id', '')
                    item_analysis_id = item.get('analysis_id', '')
                    
                    if item_query_id == query_id or item_analysis_id == query_id:
                        target_item = item
                        logger.info(f"Item encontrado en búsqueda exhaustiva: query_id={item_query_id}, analysis_id={item_analysis_id}")
                        break
                    
                    # También buscar coincidencias parciales (por si hay truncamiento)
                    if (query_id in item_query_id or query_id in item_analysis_id or
                        item_query_id.startswith(query_id) or item_analysis_id.startswith(query_id)):
                        target_item = item
                        logger.info(f"Item encontrado por coincidencia parcial: query_id={item_query_id}, analysis_id={item_analysis_id}")
                        break
            
            # PASO 2: Si no se encontró el item, reportar error detallado
            if not target_item:
                logger.error(f"No se encontró item con ID {query_id} después de búsqueda exhaustiva")
                return False
            
            # PASO 3: Extraer las claves correctas para DynamoDB
            analysis_id = target_item.get('analysis_id')
            timestamp = target_item.get('timestamp')
            
            if not analysis_id or not timestamp:
                logger.error(f"Item encontrado pero faltan claves: analysis_id={analysis_id}, timestamp={timestamp}")
                return False
            
            logger.info(f"Eliminando item con claves: analysis_id={analysis_id}, timestamp={timestamp}")
            
            # PASO 4: Ejecutar eliminación
            delete_response = self.storage.analyses_table.delete_item(
                Key={
                    'analysis_id': analysis_id,
                    'timestamp': timestamp
                },
                ReturnValues='ALL_OLD'  # Para confirmar que se eliminó
            )
            
            # PASO 5: Verificar que se eliminó
            deleted_item = delete_response.get('Attributes')
            if deleted_item:
                logger.info(f"Item {query_id} eliminado exitosamente")
                return True
            else:
                logger.warning(f"Eliminación ejecutada pero no se retornó item eliminado")
                return True  # Asumir éxito si no hay error
            
        except Exception as e:
            logger.error(f"Error eliminando consulta {query_id}: {str(e)}")
            return False
    
    def _find_item_for_deletion_fallback(self, query_id: str) -> Optional[Dict]:
        """Método fallback para encontrar items para eliminación"""
        try:
            # Obtener todos los items y buscar manualmente
            all_queries = self.get_all_hype_cycle_queries()
            
            for query in all_queries:
                if (query.get('analysis_id') == query_id or 
                    query.get('query_id') == query_id):
                    return query
                    
                # Buscar coincidencias parciales
                query_id_partial = query.get('query_id', '')
                analysis_id_partial = query.get('analysis_id', '')
                
                if (query_id in query_id_partial or query_id in analysis_id_partial or
                    query_id_partial.startswith(query_id) or analysis_id_partial.startswith(query_id)):
                    return query
            
            return None
            
        except Exception as e:
            logger.error(f"Error en fallback de búsqueda: {str(e)}")
            return None
    
    def debug_query_ids(self, partial_id: str) -> Dict:
        """NUEVO: Método de debug para investigar problemas de IDs"""
        try:
            debug_info = {
                'searched_id': partial_id,
                'matching_items': [],
                'all_ids_sample': [],
                'total_items': 0
            }
            
            all_queries = self.get_all_hype_cycle_queries()
            debug_info['total_items'] = len(all_queries)
            
            # Mostrar muestra de todos los IDs
            for i, query in enumerate(all_queries[:10]):
                debug_info['all_ids_sample'].append({
                    'index': i,
                    'query_id': query.get('query_id', 'N/A'),
                    'analysis_id': query.get('analysis_id', 'N/A'),
                    'tech_name': query.get('technology_name', query.get('search_query', ''))[:30]
                })
            
            # Buscar coincidencias
            for query in all_queries:
                query_id = query.get('query_id', '')
                analysis_id = query.get('analysis_id', '')
                
                if (partial_id in query_id or partial_id in analysis_id or
                    query_id.startswith(partial_id) or analysis_id.startswith(partial_id) or
                    query_id == partial_id or analysis_id == partial_id):
                    
                    debug_info['matching_items'].append({
                        'query_id': query_id,
                        'analysis_id': analysis_id,
                        'timestamp': query.get('timestamp', 'N/A'),
                        'tech_name': query.get('technology_name', query.get('search_query', ''))[:50],
                        'exact_match': (query_id == partial_id or analysis_id == partial_id)
                    })
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}
    
    def _delete_query_fallback(self, query_id: str) -> bool:
        """Método fallback para eliminación sin usar condiciones boto3"""
        try:
            # Obtener todas las consultas y encontrar la que coincida
            all_queries = self.get_all_hype_cycle_queries()
            
            target_query = None
            for query in all_queries:
                if (query.get('analysis_id') == query_id or 
                    query.get('query_id') == query_id):
                    target_query = query
                    break
            
            if not target_query:
                logger.warning(f"Fallback: No se encontró query {query_id}")
                return False
            
            # Eliminar usando storage base
            analysis_id = target_query.get('analysis_id', query_id)
            timestamp = target_query.get('timestamp', '')
            
            if not timestamp:
                logger.error(f"Fallback: No se encontró timestamp para {query_id}")
                return False
            
            return self.storage.delete_item(analysis_id, timestamp)
            
        except Exception as e:
            logger.error(f"Error en fallback eliminación: {str(e)}")
            return False
    
    def update_query(self, query_id: str, updates: Dict) -> bool:
        """CORREGIDO: Actualiza una consulta específica con mejor manejo"""
        try:
            # Invalidar cache primero
            self._invalidate_cache()
            
            # Buscar la consulta completa para obtener las claves
            query = self.get_query_by_id(query_id)
            if not query:
                logger.error(f"No se encontró query {query_id} para actualizar")
                return False
            
            analysis_id = query.get('analysis_id', query_id)
            timestamp = query.get('timestamp')
            
            if not timestamp:
                logger.error(f"No se encontró timestamp para query {query_id}")
                return False
            
            # Preparar las actualizaciones
            updates["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Construir la expresión de actualización
            update_expression_parts = []
            expression_attribute_values = {}
            
            for key, value in updates.items():
                update_expression_parts.append(f"{key} = :{key}")
                expression_attribute_values[f":{key}"] = self.storage._convert_floats_to_decimal(value)
            
            update_expression = "SET " + ", ".join(update_expression_parts)
            
            # Ejecutar la actualización
            self.storage.analyses_table.update_item(
                Key={
                    'analysis_id': analysis_id,
                    'timestamp': timestamp
                },
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values
            )
            
            logger.info(f"Query {query_id} actualizado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando consulta {query_id}: {str(e)}")
            return False
    
    def move_technology_to_category(self, query_id: str, target_category_id: str) -> bool:
        """CORREGIDO: Mueve una tecnología a otra categoría con búsqueda robusta de IDs"""
        try:
            logger.info(f"Iniciando movimiento de {query_id} a categoría {target_category_id}")
            
            # PASO 1: Buscar la consulta usando el método robusto (igual que en delete_query)
            current_query = None
            
            # Método A: Búsqueda por scan con condiciones
            if not BOTO3_CONDITIONS_AVAILABLE:
                current_query = self._find_item_for_deletion_fallback(query_id)
            else:
                try:
                    response = self.storage.analyses_table.scan(
                        FilterExpression=(
                            Attr('analysis_id').eq(query_id) | 
                            Attr('query_id').eq(query_id)
                        ) & Attr('analysis_type').eq('hype_cycle')
                    )
                    
                    items = response.get('Items', [])
                    if items:
                        current_query = self.storage._convert_decimals_to_float(items[0])
                        logger.info(f"Item encontrado para mover: {current_query.get('analysis_id')}")
                    
                except Exception as e:
                    logger.error(f"Error en scan para movimiento: {str(e)}")
            
            # Método B: Búsqueda exhaustiva si no se encontró
            if not current_query:
                logger.info("Item no encontrado con scan, buscando exhaustivamente...")
                all_items = self.get_all_hype_cycle_queries()
                
                for item in all_items:
                    item_query_id = item.get('query_id', '')
                    item_analysis_id = item.get('analysis_id', '')
                    
                    if (item_query_id == query_id or item_analysis_id == query_id or
                        query_id in item_query_id or query_id in item_analysis_id or
                        item_query_id.startswith(query_id) or item_analysis_id.startswith(query_id)):
                        current_query = item
                        logger.info(f"Item encontrado exhaustivamente: query_id={item_query_id}, analysis_id={item_analysis_id}")
                        break
            
            # PASO 2: Verificar que se encontró la consulta
            if not current_query:
                logger.warning(f"No se encontró query {query_id} después de búsqueda exhaustiva")
                return False
            
            # PASO 3: Obtener IDs reales del item encontrado
            real_query_id = current_query.get('query_id', query_id)
            real_analysis_id = current_query.get('analysis_id', query_id)
            current_category_id = str(current_query.get("category_id", "default"))
            target_category_id = str(target_category_id)
            
            logger.info(f"IDs reales - query_id: {real_query_id}, analysis_id: {real_analysis_id}")
            
            # PASO 4: Verificar que la categoría destino existe
            target_category = self.storage.get_category_by_id(target_category_id)
            if not target_category:
                logger.warning(f"No se encontró categoría destino {target_category_id}")
                return False
            
            # PASO 5: Verificar si ya está en la categoría correcta
            if current_category_id == target_category_id:
                logger.info(f"Query {real_query_id} ya está en categoría {target_category_id}")
                return True
            
            # PASO 6: Preparar los datos de actualización
            new_category_name = target_category.get("name", "Sin nombre")
            
            updates = {
                "category_id": target_category_id,
                "category_name": new_category_name
            }
            
            # PASO 7: Ejecutar la actualización usando los IDs reales
            success = self._update_query_robust(real_analysis_id, current_query.get('timestamp'), updates)
            
            if success:
                logger.info(f"Query {real_query_id} movido exitosamente de {current_category_id} a {target_category_id}")
                
                # Invalidar el cache específicamente para las categorías afectadas
                self._invalidate_cache()
                
                # También limpiar el cache de Streamlit
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
                
            return success
                
        except Exception as e:
            logger.error(f"Error moviendo tecnología: {str(e)}")
            return False
    
    def _update_query_robust(self, analysis_id: str, timestamp: str, updates: Dict) -> bool:
        """Actualiza una consulta de forma robusta"""
        try:
            if not timestamp:
                logger.error(f"No se encontró timestamp para {analysis_id}")
                return False
            
            # Preparar las actualizaciones
            updates["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Construir la expresión de actualización
            update_expression_parts = []
            expression_attribute_values = {}
            
            for key, value in updates.items():
                update_expression_parts.append(f"{key} = :{key}")
                expression_attribute_values[f":{key}"] = self.storage._convert_floats_to_decimal(value)
            
            update_expression = "SET " + ", ".join(update_expression_parts)
            
            # Ejecutar la actualización
            self.storage.analyses_table.update_item(
                Key={
                    'analysis_id': analysis_id,
                    'timestamp': timestamp
                },
                UpdateExpression=update_expression,
                ExpressionAttributeValues=expression_attribute_values,
                ReturnValues='UPDATED_NEW'  # Para confirmar que se actualizó
            )
            
            logger.info(f"Query {analysis_id} actualizado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error actualizando query {analysis_id}: {str(e)}")
            return False

    # ===== NUEVOS MÉTODOS PARA GESTIÓN DE DUPLICADOS =====
    
    def find_duplicates(self) -> List[Dict]:
        """Encuentra consultas duplicadas basadas en search_query"""
        try:
            all_queries = self.get_all_hype_cycle_queries()
            
            # Agrupar por search_query
            query_groups = {}
            for query in all_queries:
                search_query = query.get('search_query', '').strip().lower()
                if search_query:
                    if search_query not in query_groups:
                        query_groups[search_query] = []
                    query_groups[search_query].append(query)
            
            # Encontrar grupos con más de un elemento
            duplicates = []
            for search_query, queries in query_groups.items():
                if len(queries) > 1:
                    # Ordenar por fecha (más reciente primero)
                    queries.sort(key=lambda x: x.get('execution_date', ''), reverse=True)
                    
                    duplicates.append({
                        'search_query': search_query,
                        'total_count': len(queries),
                        'queries': queries,
                        'keep_query': queries[0],  # Mantener el más reciente
                        'delete_queries': queries[1:]  # Eliminar los más antiguos
                    })
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Error encontrando duplicados: {str(e)}")
            return []
    
    def delete_duplicates(self, duplicates_to_delete: List[str]) -> Dict[str, bool]:
        """Elimina consultas duplicadas especificadas"""
        results = {}
        
        for query_id in duplicates_to_delete:
            try:
                success = self.delete_query(query_id)
                results[query_id] = success
                logger.info(f"Duplicado {query_id}: {'eliminado' if success else 'falló'}")
            except Exception as e:
                results[query_id] = False
                logger.error(f"Error eliminando duplicado {query_id}: {str(e)}")
        
        return results
    
    def batch_delete_queries(self, query_ids: List[str]) -> Dict[str, bool]:
        """Elimina múltiples consultas en lote"""
        results = {}
        
        for query_id in query_ids:
            try:
                success = self.delete_query(query_id)
                results[query_id] = success
                time.sleep(0.1)  # Pequeña pausa para evitar throttling
            except Exception as e:
                results[query_id] = False
                logger.error(f"Error en eliminación en lote {query_id}: {str(e)}")
        
        return results

    # ===== MÉTODOS DE DEBUG MEJORADOS =====
    
    def debug_category_queries(self, category_id: str) -> Dict:
        """MEJORADO: Método de debug para investigar problemas con categorías"""
        try:
            # Obtener todos los registros sin filtrar
            all_items_response = self.storage.analyses_table.scan()
            all_items = all_items_response.get('Items', [])
            
            # Estadísticas detalladas
            total_items = len(all_items)
            hype_cycle_items = [item for item in all_items if item.get('analysis_type') == 'hype_cycle']
            category_items = [item for item in hype_cycle_items if str(item.get('category_id', '')) == str(category_id)]
            
            # Categorías únicas
            unique_categories = set()
            for item in hype_cycle_items:
                unique_categories.add(str(item.get('category_id', 'None')))
            
            # Análisis de duplicados
            query_counts = {}
            for item in hype_cycle_items:
                search_query = item.get('search_query', '').strip().lower()
                if search_query:
                    query_counts[search_query] = query_counts.get(search_query, 0) + 1
            
            duplicated_queries = {k: v for k, v in query_counts.items() if v > 1}
            
            debug_info = {
                'total_items_in_table': total_items,
                'hype_cycle_items': len(hype_cycle_items),
                'items_in_category': len(category_items),
                'unique_categories': list(unique_categories),
                'category_id_searched': str(category_id),
                'duplicated_queries_count': len(duplicated_queries),
                'total_duplicates': sum(duplicated_queries.values()) - len(duplicated_queries),
                'cache_status': {
                    'cache_keys': list(self._cache.keys()),
                    'cache_size': len(self._cache)
                },
                'sample_items': []
            }
            
            # Muestras de items para debug
            for item in category_items[:3]:
                debug_info['sample_items'].append({
                    'analysis_id': item.get('analysis_id'),
                    'query_id': item.get('query_id'),
                    'category_id': str(item.get('category_id')),
                    'analysis_type': item.get('analysis_type'),
                    'search_query': item.get('search_query', '')[:50] + '...',
                    'execution_date': item.get('execution_date', '')
                })
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}

# ===== RESTO DE CLASES SIN CAMBIOS PERO OPTIMIZADAS =====

class HypeCycleHistoryInterface:
    """Interfaz OPTIMIZADA para gestionar el historial de consultas de Hype Cycle"""
    
    def __init__(self, hype_storage, context_prefix: str = "default"):
        self.storage = hype_storage
        self.context_prefix = context_prefix if context_prefix != "default" else "hype_history"
        self._state_key_base = f"hype_history_state_{self.context_prefix}"
        self._init_stable_states()
    
    def _init_stable_states(self):
        """Inicializa estados estables que persisten entre reruns"""
        state_keys = [
            f"{self._state_key_base}_selected_category",
            f"{self._state_key_base}_selected_query", 
            f"{self._state_key_base}_filter_category",
            f"{self._state_key_base}_move_source_tech",
            f"{self._state_key_base}_move_target_cat",
            f"{self._state_key_base}_duplicates_found",
            f"{self._state_key_base}_selected_duplicates"
        ]
        
        for key in state_keys:
            if key not in st.session_state:
                if "selected_category" in key:
                    st.session_state[key] = ""
                elif "selected_query" in key:
                    st.session_state[key] = ""
                elif "filter_category" in key:
                    st.session_state[key] = "Todas"
                elif "move_source_tech" in key:
                    st.session_state[key] = ""
                elif "move_target_cat" in key:
                    st.session_state[key] = ""
                elif "duplicates_found" in key:
                    st.session_state[key] = []
                elif "selected_duplicates" in key:
                    st.session_state[key] = []
    
    def _safe_format_value(self, value, format_type="float", format_str=".2f", default="0.00"):
        """Formatea un valor de forma segura para evitar errores de tipo"""
        try:
            if value is None:
                return default
            
            if isinstance(value, Decimal):
                numeric_value = float(value)
            elif isinstance(value, str):
                clean_value = value.replace(',', '').replace('%', '').strip()
                numeric_value = float(clean_value) if clean_value else 0.0
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                return str(value)
            
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                return default
            
            if format_type == "percent":
                return f"{numeric_value * 100:.1f}%"
            elif format_type == "int":
                return str(int(numeric_value))
            else:
                return f"{numeric_value:{format_str}}"
                
        except (ValueError, TypeError, decimal.InvalidOperation):
            return default
    
    def show_history_interface(self):
        """Muestra la interfaz completa de historial OPTIMIZADA"""
        st.header("📚 Historial de Consultas de Hype Cycle")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🔍 Explorar por Categoría", 
            "📊 Vista de Resumen", 
            "⚙️ Gestionar Consultas",
            "🔄 Mover Tecnologías",
            "🧹 Gestionar Duplicados",
            "🛠️ Debug"
        ])
        
        with tab1:
            self._show_category_explorer()
        
        with tab2:
            self._show_summary_dashboard()
        
        with tab3:
            self._show_query_manager()
        
        with tab4:
            self._show_move_technologies_optimized()
        
        with tab5:
            self._show_duplicates_manager()
        
        with tab6:
            self._show_debug_tab()
    
    def _show_move_technologies_optimized(self):
        """CORREGIDA: Interfaz para mover tecnologías sin botones bloqueados"""
        st.subheader("🔄 Mover Tecnologías Entre Categorías")
        
        st.write("Mueve tecnologías entre diferentes categorías en DynamoDB.")
        
        # Obtener datos una sola vez
        with st.spinner("Cargando datos..."):
            all_queries = self.storage.get_all_hype_cycle_queries()
            categories = self.storage.storage.get_all_categories()
        
        if not all_queries:
            st.info("No hay tecnologías para mover.")
            return
        
        if len(categories) < 2:
            st.warning("Se necesitan al menos 2 categorías para mover tecnologías.")
            return
        
        # USAR FORMULARIO SIN DISABLED PARA EVITAR BLOQUEOS
        with st.form(key=f"{self._state_key_base}_move_form_optimized", clear_on_submit=False):
            st.write("### 📋 Seleccionar Movimiento")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("#### 🔬 Tecnología a Mover")
                
                # Crear opciones simplificadas
                tech_options = []
                tech_data = {}
                
                for query in all_queries:
                    query_id = query.get("query_id", query.get("analysis_id"))
                    tech_name = (
                        query.get("technology_name") or 
                        query.get("search_query", "")[:30] or 
                        "Sin nombre"
                    )
                    
                    # Obtener categoría actual
                    current_cat_id = query.get("category_id", "unknown")
                    current_cat_name = "Sin categoría"
                    for cat in categories:
                        if cat.get("category_id") == current_cat_id:
                            current_cat_name = cat.get("name", "Sin nombre")
                            break
                    
                    phase = query.get("hype_metrics", {}).get("phase", "Unknown")
                    display_name = f"{tech_name} | {current_cat_name} | {phase}"
                    
                    tech_options.append(display_name)
                    tech_data[display_name] = {
                        "query_id": query_id,
                        "tech_name": tech_name,
                        "current_cat_id": current_cat_id,
                        "current_cat_name": current_cat_name
                    }
                
                selected_tech_display = st.selectbox(
                    "Selecciona la tecnología:",
                    options=tech_options,
                    help=f"{len(tech_options)} tecnologías disponibles"
                )
                
                if selected_tech_display:
                    selected_tech_info = tech_data[selected_tech_display]
                    
                    st.info(f"""
                    **Tecnología seleccionada:**
                    - 🔬 Nombre: {selected_tech_info['tech_name']}
                    - 📁 Categoría actual: {selected_tech_info['current_cat_name']}
                    - 🆔 ID: {selected_tech_info['query_id'][:12]}...
                    """)
            
            with col2:
                st.write("#### 🎯 Categoría Destino")
                
                if selected_tech_display:
                    current_cat_id = selected_tech_info["current_cat_id"]
                    
                    # Filtrar categorías disponibles (excluir la actual)
                    available_categories = []
                    category_data = {}
                    
                    for cat in categories:
                        if cat.get("category_id") != current_cat_id:
                            cat_name = cat.get("name", "Sin nombre")
                            available_categories.append(cat_name)
                            category_data[cat_name] = cat.get("category_id")
                    
                    if available_categories:
                        target_category_name = st.selectbox(
                            "Nueva categoría:",
                            options=available_categories,
                            help=f"{len(available_categories)} categorías disponibles"
                        )
                        
                        target_category_id = category_data[target_category_name]
                        
                        st.success(f"""
                        **Movimiento programado:**
                        📁 **De:** {selected_tech_info['current_cat_name']}
                        📁 **A:** {target_category_name}
                        """)
                    else:
                        st.warning("No hay otras categorías disponibles.")
                        target_category_name = None
                        target_category_id = None
                else:
                    target_category_name = None
                    target_category_id = None
            
            # Controles del formulario
            st.write("---")
            st.write("### ⚙️ Confirmar Operación")
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if target_category_id:
                    confirm_move = st.checkbox(
                        f"✅ Confirmar movimiento a '{target_category_name}'",
                        help="Esta acción modificará la categoría en DynamoDB"
                    )
                    
                    if not confirm_move:
                        st.warning("⚠️ Marca la casilla para habilitar el movimiento")
                else:
                    confirm_move = False
                    st.info("Selecciona una tecnología y categoría destino")
            
            with col2:
                # BOTÓN SIN DISABLED - Validación después del submit
                submitted = st.form_submit_button(
                    "🔄 EJECUTAR MOVIMIENTO", 
                    type="primary"
                )
            
            # Procesar cuando se envía el formulario
            if submitted:
                if not selected_tech_display:
                    st.error("❌ Selecciona una tecnología")
                elif not target_category_id:
                    st.error("❌ Selecciona una categoría destino")
                elif not confirm_move:
                    st.error("❌ Debes confirmar el movimiento marcando la casilla")
                else:
                    progress_container = st.container()
                    
                    with progress_container:
                        with st.spinner(f"Moviendo '{selected_tech_info['tech_name']}' a '{target_category_name}'..."):
                            # MÉTODO MEJORADO: Primero verificar que existe usando debug
                            query_to_move = selected_tech_info['query_id']
                            
                            # Debug: Verificar que el item existe
                            debug_info = self.storage.debug_query_ids(query_to_move)
                            
                            if debug_info.get('matching_items'):
                                # El item existe, proceder con movimiento
                                matching_item = debug_info['matching_items'][0]
                                st.info(f"🔍 Item encontrado: {matching_item['tech_name']}")
                                
                                success = self.storage.move_technology_to_category(
                                    query_to_move, 
                                    target_category_id
                                )
                                
                                if success:
                                    st.success(f"✅ '{selected_tech_info['tech_name']}' movida exitosamente a '{target_category_name}'!")
                                    st.balloons()
                                    
                                    # Limpiar cache
                                    if hasattr(st, 'cache_data'):
                                        st.cache_data.clear()
                                    
                                    # Pequeña pausa antes de rerun para mostrar el mensaje
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.error("❌ Error moviendo la tecnología. Revisa los logs de DynamoDB.")
                                    
                                    with st.expander("🔍 Información de debug"):
                                        st.write("**Item que se intentó mover:**")
                                        st.json(matching_item)
                                        
                                        st.write("**Información del movimiento:**")
                                        st.write(f"- Query ID usado: {query_to_move}")
                                        st.write(f"- Categoría origen: {selected_tech_info['current_cat_id']}")
                                        st.write(f"- Categoría destino: {target_category_id}")
                                        
                                        st.write("**Posibles causas:**")
                                        st.write("- Permisos insuficientes en DynamoDB")
                                        st.write("- Error en la actualización de las claves primary/sort")
                                        st.write("- Categoría destino no válida")
                                        
                                        # Verificar que la categoría destino existe
                                        try:
                                            target_cat_info = self.storage.storage.get_category_by_id(target_category_id)
                                            if target_cat_info:
                                                st.write("✅ Categoría destino existe en BD")
                                            else:
                                                st.write("❌ Categoría destino NO existe en BD")
                                        except:
                                            st.write("❌ Error verificando categoría destino")
                            else:
                                # El item no existe
                                st.error(f"❌ No se encontró el item con ID: {query_to_move}")
                                
                                with st.expander("🔍 Información de Debug - Item No Encontrado"):
                                    st.write(f"**ID buscado:** {query_to_move}")
                                    st.write(f"**Total items en base:** {debug_info.get('total_items', 0)}")
                                    
                                    if debug_info.get('all_ids_sample'):
                                        st.write("**Muestra de IDs existentes:**")
                                        for sample in debug_info['all_ids_sample'][:3]:
                                            st.write(f"- query_id: {sample['query_id']}")
                                            st.write(f"  analysis_id: {sample['analysis_id']}")
                                            st.write(f"  tecnología: {sample['tech_name']}")
                                            st.write("---")
                                    
                                    st.write("**Posibles soluciones:**")
                                    st.write("1. Limpiar cache y recargar")
                                    st.write("2. El item pudo haber sido eliminado")
                                    st.write("3. Usar Debug de IDs en la pestaña Debug")
                                    
                                    # Botón para limpiar cache
                                    if st.button("🔄 Limpiar Cache", key=f"{self._state_key_base}_clean_cache_move"):
                                        if hasattr(st, 'cache_data'):
                                            st.cache_data.clear()
                                        self.storage._invalidate_cache()
                                        st.rerun()
    
    def _show_duplicates_manager(self):
        """NUEVA: Gestión avanzada de duplicados"""
        st.subheader("🧹 Gestionar Consultas Duplicadas")
        
        st.write("""
        Esta herramienta te ayuda a identificar y eliminar consultas duplicadas 
        basadas en el texto de búsqueda similar.
        """)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔍 Buscar Duplicados", type="primary", key=f"{self._state_key_base}_find_duplicates"):
                with st.spinner("Analizando consultas duplicadas..."):
                    duplicates = self.storage.find_duplicates()
                    st.session_state[f"{self._state_key_base}_duplicates_found"] = duplicates
        
        with col2:
            if st.button("🧹 Limpiar Cache", key=f"{self._state_key_base}_clear_cache"):
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
                self.storage._invalidate_cache()
                st.success("✅ Cache limpiado")
        
        # Mostrar duplicados encontrados
        duplicates = st.session_state.get(f"{self._state_key_base}_duplicates_found", [])
        
        if duplicates:
            st.write(f"### 📊 Duplicados Encontrados: {len(duplicates)} grupos")
            
            total_duplicates = sum(len(dup['delete_queries']) for dup in duplicates)
            st.metric("Total de consultas duplicadas para eliminar", total_duplicates)
            
            # Mostrar cada grupo de duplicados
            selected_for_deletion = []
            
            for i, duplicate_group in enumerate(duplicates):
                with st.expander(f"Grupo {i+1}: {duplicate_group['search_query'][:60]}... ({duplicate_group['total_count']} duplicados)", expanded=False):
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write("**Consulta duplicada:**")
                        st.code(duplicate_group['search_query'])
                        
                        st.write("**Consulta a MANTENER (más reciente):**")
                        keep_query = duplicate_group['keep_query']
                        st.write(f"- ID: {keep_query.get('query_id', 'N/A')}")
                        st.write(f"- Fecha: {keep_query.get('execution_date', 'N/A')[:19]}")
                        st.write(f"- Fase: {keep_query.get('hype_metrics', {}).get('phase', 'N/A')}")
                    
                    with col2:
                        st.write("**Consultas a ELIMINAR:**")
                        
                        for del_query in duplicate_group['delete_queries']:
                            query_id = del_query.get('query_id', 'N/A')
                            
                            if st.checkbox(
                                f"Eliminar {query_id[:12]}...",
                                key=f"{self._state_key_base}_delete_{query_id}",
                                value=True  # Por defecto seleccionado
                            ):
                                selected_for_deletion.append(query_id)
                            
                            st.caption(f"Fecha: {del_query.get('execution_date', 'N/A')[:19]}")
            
            # Botón para eliminar seleccionados
            if selected_for_deletion:
                st.write("---")
                st.write(f"### 🗑️ Eliminar {len(selected_for_deletion)} consultas duplicadas")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    confirm_deletion = st.checkbox(
                        "Confirmar eliminación masiva",
                        help="Esta acción no se puede deshacer"
                    )
                
                with col2:
                    if confirm_deletion:
                        safety_text = st.text_input(
                            "Escribe 'ELIMINAR' para confirmar:",
                            placeholder="ELIMINAR"
                        )
                        safety_confirmed = safety_text.upper().strip() == "ELIMINAR"
                    else:
                        safety_confirmed = False
                
                with col3:
                    if confirm_deletion and safety_confirmed:
                        if st.button(
                            f"🗑️ ELIMINAR {len(selected_for_deletion)} DUPLICADOS",
                            type="secondary",
                            key=f"{self._state_key_base}_execute_mass_delete"
                        ):
                            with st.spinner(f"Eliminando {len(selected_for_deletion)} consultas duplicadas..."):
                                results = self.storage.batch_delete_queries(selected_for_deletion)
                                
                                successful = sum(1 for success in results.values() if success)
                                failed = len(results) - successful
                                
                                if successful > 0:
                                    st.success(f"✅ {successful} consultas eliminadas exitosamente")
                                
                                if failed > 0:
                                    st.error(f"❌ {failed} consultas no pudieron eliminarse")
                                    
                                    with st.expander("Ver errores"):
                                        for query_id, success in results.items():
                                            if not success:
                                                st.write(f"- Error eliminando: {query_id}")
                                
                                # Limpiar estado y cache
                                st.session_state[f"{self._state_key_base}_duplicates_found"] = []
                                self.storage._invalidate_cache()
                                if hasattr(st, 'cache_data'):
                                    st.cache_data.clear()
                                
                                time.sleep(2)
                                st.rerun()
        else:
            st.info("Haz clic en 'Buscar Duplicados' para encontrar consultas duplicadas.")
    
    def _show_category_explorer(self):
        """OPTIMIZADA: Muestra explorador por categorías con cache"""
        st.subheader("Explorar Consultas por Categoría")
        
        # Cache para categorías
        @st.cache_data(ttl=300)
        def get_categories():
            try:
                return self.storage.storage.get_all_categories()
            except:
                return [{"category_id": "default", "name": "Sin categoría"}]
        
        categories = get_categories()
        
        if not categories:
            st.warning("No hay categorías disponibles.")
            return
        
        category_options = {cat.get("name", "Sin nombre"): cat.get("category_id") for cat in categories}
        
        category_selector_key = f"{self._state_key_base}_category_explorer_selector"
        
        # Mantener selección previa
        saved_category = st.session_state.get(f"{self._state_key_base}_selected_category", "")
        try:
            if saved_category and saved_category in category_options.keys():
                default_index = list(category_options.keys()).index(saved_category)
            else:
                default_index = 0
        except:
            default_index = 0
        
        selected_category_name = st.selectbox(
            "Selecciona una categoría",
            options=list(category_options.keys()),
            index=default_index,
            key=category_selector_key
        )
        
        st.session_state[f"{self._state_key_base}_selected_category"] = selected_category_name
        
        selected_category_id = category_options[selected_category_name]
        
        # Obtener consultas con cache
        with st.spinner("Cargando consultas..."):
            queries = self.storage.get_queries_by_category(selected_category_id)
        
        # Mostrar estadísticas
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Consultas en categoría", len(queries))
        with col2:
            if queries:
                phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in queries]
                most_common = max(set(phases), key=phases.count) if phases else "N/A"
                st.metric("Fase más común", most_common)
        with col3:
            if queries:
                recent_count = len([q for q in queries if q.get("execution_date", "").startswith(datetime.now().strftime("%Y-%m"))])
                st.metric("Este mes", recent_count)
        
        if not queries:
            st.info(f"No hay consultas guardadas en la categoría '{selected_category_name}'")
            return
        
        st.write(f"**{len(queries)} consultas encontradas en '{selected_category_name}'**")
        
        # Mostrar consultas de forma optimizada
        for i, query in enumerate(queries[:10]):  # Limitar para mejorar rendimiento
            self._display_query_card_optimized(query, i)
        
        if len(queries) > 10:
            st.info(f"Mostrando las primeras 10 de {len(queries)} consultas. Usa filtros para refinar la búsqueda.")
    
    def _display_query_card_optimized(self, query: Dict, index: int):
        """OPTIMIZADA: Muestra una tarjeta de consulta con menos información para mejorar rendimiento"""
        query_id = query.get('query_id', query.get('analysis_id', 'unknown'))
        
        with st.expander(
            f"🔍 {query.get('search_query', 'Sin consulta')[:60]}... - "
            f"**{query.get('hype_metrics', {}).get('phase', 'Unknown')}**",
            expanded=False
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.code(query.get('search_query', 'No disponible'))
                
                # Mostrar solo información esencial
                tech_name = query.get('technology_name') or query.get('search_query', '')[:30]
                st.write(f"**Tecnología:** {tech_name}")
                
                try:
                    date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                    st.write(f"**Fecha:** {date.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.write("**Fecha:** No disponible")
            
            with col2:
                hype_metrics = query.get('hype_metrics', {})
                
                st.metric("Fase", hype_metrics.get('phase', 'Unknown'))
                
                confidence = self._safe_format_value(hype_metrics.get('confidence', 0), "float", ".2f")
                st.metric("Confianza", confidence)
                
                mentions = self._safe_format_value(hype_metrics.get('total_mentions', 0), "int")
                st.metric("Menciones", mentions)
            
            # Botones con keys únicos
            col1, col2, col3 = st.columns(3)
            
            with col1:
                reuse_button_key = f"{self._state_key_base}_reuse_btn_{query_id}_{index}"
                if st.button(f"🔄 Reutilizar", key=reuse_button_key, help="Reutilizar esta consulta"):
                    self._reuse_query(query)
            
            with col2:
                view_button_key = f"{self._state_key_base}_view_btn_{query_id}_{index}"
                if st.button(f"👁️ Ver Detalles", key=view_button_key, help="Ver información completa"):
                    with st.expander("Detalles completos", expanded=True):
                        st.json(query)
            
            with col3:
                delete_button_key = f"{self._state_key_base}_del_btn_{query_id}_{index}"
                if st.button(f"🗑️ Eliminar", key=delete_button_key, type="secondary", help="Eliminar esta consulta"):
                    if st.checkbox(f"Confirmar eliminación de {query_id[:12]}...", key=f"{self._state_key_base}_confirm_del_{query_id}_{index}"):
                        if self.storage.delete_query(query_id):
                            st.success("✅ Consulta eliminada")
                            st.rerun()
                        else:
                            st.error("❌ Error eliminando")
    
    # Mantener otros métodos sin cambios pero optimizados...
    def _show_summary_dashboard(self):
        """Dashboard de resumen optimizado"""
        st.subheader("Dashboard de Resumen")
        
        with st.spinner("Cargando estadísticas..."):
            all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay consultas de Hype Cycle guardadas")
            return
        
        # Métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Consultas", len(all_queries))
        
        with col2:
            phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in all_queries]
            most_common_phase = max(set(phases), key=phases.count) if phases else "N/A"
            st.metric("Fase Más Común", most_common_phase)
        
        with col3:
            confidences = []
            for q in all_queries:
                conf_raw = q.get("hype_metrics", {}).get("confidence", 0)
                conf_numeric = float(self._safe_format_value(conf_raw, "float", "", "0"))
                confidences.append(conf_numeric)
            
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            avg_conf_formatted = self._safe_format_value(avg_confidence, "float", ".2f")
            st.metric("Confianza Promedio", avg_conf_formatted)
        
        with col4:
            current_month = datetime.now().strftime("%Y-%m")
            recent_queries = [q for q in all_queries if q.get("execution_date", "").startswith(current_month)]
            st.metric("Consultas Este Mes", len(recent_queries))
        
        # Gráfico de distribución de fases
        st.subheader("Distribución de Fases del Hype Cycle")
        
        phase_counts = {}
        for query in all_queries:
            phase = query.get("hype_metrics", {}).get("phase", "Unknown")
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        if phase_counts:
            fig_phases = px.pie(
                values=list(phase_counts.values()),
                names=list(phase_counts.keys()),
                title="Distribución de Tecnologías por Fase del Hype Cycle"
            )
            st.plotly_chart(fig_phases, use_container_width=True)
    
    def _show_query_manager(self):
        """OPTIMIZADA: Interfaz para gestionar consultas"""
        st.subheader("Gestionar Consultas")
        
        with st.spinner("Cargando consultas..."):
            all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay consultas para gestionar")
            return
        
        # Crear tabla resumida para mejor rendimiento
        query_data = []
        for query in all_queries[:50]:  # Limitar para mejor rendimiento
            try:
                date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                formatted_date = date.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_date = "Fecha inválida"
            
            confidence_raw = query.get('hype_metrics', {}).get('confidence', 0)
            confidence_formatted = self._safe_format_value(confidence_raw, "float", ".2f")
            
            query_data.append({
                "ID": query.get("query_id", query.get("analysis_id", "Unknown"))[:12] + "...",
                "Consulta": query.get("search_query", "")[:50] + "...",
                "Fase": query.get("hype_metrics", {}).get("phase", "Unknown"),
                "Fecha": formatted_date,
                "Confianza": confidence_formatted,
                "Resultados": query.get("api_usage", {}).get("total_results", 0)
            })
        
        df_queries = pd.DataFrame(query_data)
        st.dataframe(df_queries, use_container_width=True)
        
        if len(all_queries) > 50:
            st.info(f"Mostrando las primeras 50 de {len(all_queries)} consultas para mejor rendimiento")
    
    def _show_debug_tab(self):
        """MEJORADA: Pestaña de debug con investigación de IDs"""
        st.subheader("🛠️ Debug - Investigar Problemas")
        
        st.write("Esta herramienta ayuda a diagnosticar problemas de rendimiento y datos.")
        
        # Información de cache
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 💾 Estado del Cache")
            cache_keys = list(self.storage._cache.keys())
            st.metric("Entradas en cache", len(cache_keys))
            
            if cache_keys:
                with st.expander("Ver keys de cache"):
                    for key in cache_keys:
                        cache_time = self.storage._last_cache_time.get(key, 0)
                        age = time.time() - cache_time
                        st.write(f"- {key}: {age:.1f}s")
        
        with col2:
            st.write("### 🔄 Operaciones de Limpieza")
            
            if st.button("🧹 Limpiar Cache Interno", key=f"{self._state_key_base}_clear_internal_cache"):
                self.storage._invalidate_cache()
                st.success("✅ Cache interno limpiado")
            
            if st.button("🧹 Limpiar Cache Streamlit", key=f"{self._state_key_base}_clear_streamlit_cache"):
                if hasattr(st, 'cache_data'):
                    st.cache_data.clear()
                st.success("✅ Cache de Streamlit limpiado")
        
        # Debug de categorías
        st.write("### 🏷️ Debug de Categorías")
        
        try:
            categories = self.storage.storage.get_all_categories()
            category_options = {cat.get("name", "Sin nombre"): cat.get("category_id") for cat in categories}
            
            selected_category_name = st.selectbox(
                "Categoría a investigar:",
                options=list(category_options.keys()),
                key=f"{self._state_key_base}_debug_category"
            )
            
            selected_category_id = category_options[selected_category_name]
            
            if st.button("🔍 Investigar Categoría", key=f"{self._state_key_base}_debug_btn"):
                with st.spinner("Investigando..."):
                    debug_info = self.storage.debug_category_queries(selected_category_id)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total items en tabla", debug_info.get('total_items_in_table', 0))
                    
                    with col2:
                        st.metric("Items Hype Cycle", debug_info.get('hype_cycle_items', 0))
                    
                    with col3:
                        st.metric("Items en categoría", debug_info.get('items_in_category', 0))
                    
                    # Información de duplicados
                    st.write("### 🔄 Información de Duplicados")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Consultas duplicadas", debug_info.get('duplicated_queries_count', 0))
                    
                    with col2:
                        st.metric("Total duplicados", debug_info.get('total_duplicates', 0))
                    
                    # Estado del cache
                    cache_info = debug_info.get('cache_status', {})
                    st.write("### 💾 Estado del Cache")
                    st.write(f"- Cache keys: {len(cache_info.get('cache_keys', []))}")
                    st.write(f"- Cache size: {cache_info.get('cache_size', 0)}")
                    
                    with st.expander("Ver información completa de debug"):
                        st.json(debug_info)
        
        except Exception as e:
            st.error(f"Error en debug: {str(e)}")
        
        # NUEVA FUNCIÓN: Debug de IDs específicos
        st.write("### 🆔 Debug de IDs Específicos")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            debug_id = st.text_input(
                "ID a investigar (completo o parcial):",
                placeholder="ej: hype_1752031872541_5cf5514f-80b",
                key=f"{self._state_key_base}_debug_id_input"
            )
        
        with col2:
            if st.button("🔍 Investigar ID", key=f"{self._state_key_base}_debug_id_btn"):
                if debug_id:
                    with st.spinner(f"Investigando ID: {debug_id}"):
                        debug_info = self.storage.debug_query_ids(debug_id)
                        
                        st.write("### 📊 Resultados de la Investigación")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Items coincidentes", len(debug_info.get('matching_items', [])))
                        
                        with col2:
                            st.metric("Total items en BD", debug_info.get('total_items', 0))
                        
                        with col3:
                            exact_matches = len([item for item in debug_info.get('matching_items', []) if item.get('exact_match')])
                            st.metric("Coincidencias exactas", exact_matches)
                        
                        # Mostrar items coincidentes
                        if debug_info.get('matching_items'):
                            st.write("### ✅ Items Encontrados")
                            
                            for i, item in enumerate(debug_info['matching_items']):
                                match_type = "🎯 Exacta" if item.get('exact_match') else "🔍 Parcial"
                                
                                with st.expander(f"{match_type} - {item['tech_name']}", expanded=item.get('exact_match', False)):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.write("**Identificadores:**")
                                        st.code(f"query_id: {item['query_id']}")
                                        st.code(f"analysis_id: {item['analysis_id']}")
                                        st.code(f"timestamp: {item['timestamp']}")
                                    
                                    with col2:
                                        st.write("**Información:**")
                                        st.write(f"**Tecnología:** {item['tech_name']}")
                                        st.write(f"**Tipo de coincidencia:** {match_type}")
                                        
                                        # Botón para intentar eliminar este item específico
                                        if st.button(f"🗑️ Eliminar Este Item", key=f"{self._state_key_base}_delete_debug_{i}"):
                                            success = self.storage.delete_query(item['analysis_id'])
                                            if success:
                                                st.success("✅ Item eliminado exitosamente")
                                                time.sleep(1)
                                                st.rerun()
                                            else:
                                                st.error("❌ Error eliminando item")
                        else:
                            st.write("### ❌ No se encontraron coincidencias")
                            
                            st.write("**Muestra de IDs existentes en la base de datos:**")
                            for sample in debug_info.get('all_ids_sample', []):
                                st.write(f"**{sample['index'] + 1}.** {sample['tech_name']}")
                                st.code(f"query_id: {sample['query_id']}")
                                st.code(f"analysis_id: {sample['analysis_id']}")
                                st.write("---")
                else:
                    st.warning("Ingresa un ID para investigar")
    
    def _reuse_query(self, query: Dict):
        """Permite reutilizar una consulta existente"""
        st.info("**Consulta seleccionada para reutilizar:**")
        st.code(query.get('search_query', 'No disponible'))
        
        st.session_state.hype_reuse_query = {
            'search_query': query.get('search_query', ''),
            'search_terms': query.get('search_terms', []),
            'search_parameters': query.get('search_parameters', {})
        }
        
        st.success("✅ Consulta cargada. Ve a la pestaña 'Nuevo Análisis' para ejecutarla nuevamente o modificarla.")

def initialize_hype_cycle_storage(db_storage):
    """Inicializa el sistema de almacenamiento de Hype Cycle"""
    return HypeCycleStorage(db_storage)

def create_hype_cycle_interface(hype_storage, context_prefix: str = "default"):
    """Crea la interfaz completa de historial de Hype Cycle"""
    return HypeCycleHistoryInterface(hype_storage, context_prefix)