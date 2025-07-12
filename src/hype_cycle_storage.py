# src/hype_cycle_storage.py - VERSIÓN CORREGIDA PARA FILTROS DYNAMODB
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
    """Clase especializada para gestionar almacenamiento de consultas Hype Cycle - VERSIÓN CORREGIDA"""
    
    def __init__(self, db_storage):
        """Inicializa con el storage de DynamoDB"""
        self.storage = db_storage
        
        # Verificar que es DynamoDB
        if not hasattr(db_storage, 'dynamodb'):
            raise ValueError("HypeCycleStorage requiere DynamoDB storage")
    
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
        VERSIÓN LIMPIA: Guarda una consulta completa de Hype Cycle sin logging en frontend
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

    # ===== MÉTODOS DE CONSULTA CORREGIDOS =====
    
    def get_queries_by_category(self, category_id: str) -> List[Dict]:
        """CORREGIDO: Obtiene todas las consultas de Hype Cycle de una categoría con paginación"""
        try:
            if not BOTO3_CONDITIONS_AVAILABLE:
                # Fallback sin condiciones
                return self._get_queries_by_category_fallback(category_id)
            
            # Usar condiciones de boto3 correctamente
            items = []
            
            # Primera consulta
            response = self.storage.analyses_table.scan(
                FilterExpression=Attr('category_id').eq(str(category_id)) & Attr('analysis_type').eq('hype_cycle')
            )
            
            items.extend(response.get('Items', []))
            
            # Manejar paginación
            while 'LastEvaluatedKey' in response:
                response = self.storage.analyses_table.scan(
                    FilterExpression=Attr('category_id').eq(str(category_id)) & Attr('analysis_type').eq('hype_cycle'),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            # Debug logging
            logger.info(f"Categoría {category_id}: encontrados {len(items)} items")
            
            # Convertir decimales y retornar
            converted_items = [self.storage._convert_decimals_to_float(item) for item in items]
            
            # Filtro adicional en memoria por si acaso
            filtered_items = []
            for item in converted_items:
                item_category = str(item.get('category_id', ''))
                item_type = item.get('analysis_type', '')
                
                if item_category == str(category_id) and item_type == 'hype_cycle':
                    filtered_items.append(item)
            
            logger.info(f"Categoría {category_id}: después de filtro adicional {len(filtered_items)} items")
            return filtered_items
                
        except Exception as e:
            logger.error(f"Error obteniendo consultas por categoría: {str(e)}")
            # Fallback
            return self._get_queries_by_category_fallback(category_id)
    
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
        """CORREGIDO: Obtiene una consulta específica por ID"""
        try:
            if not BOTO3_CONDITIONS_AVAILABLE:
                return self._get_query_by_id_fallback(query_id)
            
            # Buscar por analysis_id o query_id
            response = self.storage.analyses_table.scan(
                FilterExpression=(
                    Attr('analysis_id').eq(query_id) | 
                    Attr('query_id').eq(query_id)
                ) & Attr('analysis_type').eq('hype_cycle')
            )
            
            items = response.get('Items', [])
            if items:
                return self.storage._convert_decimals_to_float(items[0])
            return None
                
        except Exception as e:
            logger.error(f"Error obteniendo consulta por ID: {str(e)}")
            return self._get_query_by_id_fallback(query_id)
    
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

    def get_all_hype_cycle_queries(self) -> List[Dict]:
        """CORREGIDO: Obtiene todas las consultas de Hype Cycle con paginación"""
        try:
            if not BOTO3_CONDITIONS_AVAILABLE:
                return self._get_all_hype_cycle_queries_fallback()
            
            items = []
            
            # Primera consulta
            response = self.storage.analyses_table.scan(
                FilterExpression=Attr('analysis_type').eq('hype_cycle')
            )
            
            items.extend(response.get('Items', []))
            
            # Manejar paginación
            while 'LastEvaluatedKey' in response:
                response = self.storage.analyses_table.scan(
                    FilterExpression=Attr('analysis_type').eq('hype_cycle'),
                    ExclusiveStartKey=response['LastEvaluatedKey']
                )
                items.extend(response.get('Items', []))
            
            # Debug total
            logger.info(f"Total Hype Cycle queries encontradas: {len(items)}")
            
            # Convertir decimales
            converted_items = [self.storage._convert_decimals_to_float(item) for item in items]
            
            # Ordenar por fecha (más reciente primero)
            converted_items.sort(key=lambda x: x.get('execution_date', ''), reverse=True)
            
            return converted_items
                
        except Exception as e:
            logger.error(f"Error obteniendo todas las consultas: {str(e)}")
            return self._get_all_hype_cycle_queries_fallback()
    
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
    
    # ===== MÉTODOS DE GESTIÓN =====
    
    def delete_query(self, query_id: str) -> bool:
        """CORREGIDO: Elimina una consulta específica usando múltiples métodos"""
        try:
            # Método 1: Buscar el item completo primero para obtener las claves exactas
            if not BOTO3_CONDITIONS_AVAILABLE:
                return self._delete_query_fallback(query_id)
            
            # Buscar el item usando filtros
            response = self.storage.analyses_table.scan(
                FilterExpression=(
                    Attr('analysis_id').eq(query_id) | 
                    Attr('query_id').eq(query_id)
                ) & Attr('analysis_type').eq('hype_cycle')
            )
            
            items = response.get('Items', [])
            if not items:
                logger.warning(f"No se encontró item con ID {query_id}")
                return False
            
            # Tomar el primer item encontrado
            item = items[0]
            
            # Eliminar usando las claves exactas del item
            delete_response = self.storage.analyses_table.delete_item(
                Key={
                    'analysis_id': item['analysis_id'],
                    'timestamp': item['timestamp']
                }
            )
            
            logger.info(f"Item {query_id} eliminado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error eliminando consulta {query_id}: {str(e)}")
            return self._delete_query_fallback(query_id)
    
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
        """Actualiza una consulta específica"""
        try:
            query = self.get_query_by_id(query_id)
            if query:
                updates["last_updated"] = datetime.now(timezone.utc).isoformat()
                return self.storage.update_item(query_id, query["timestamp"], updates)
            return False
            
        except Exception as e:
            logger.error(f"Error actualizando consulta: {str(e)}")
            return False
    
    def move_technology_to_category(self, query_id: str, target_category_id: str) -> bool:
        """Mueve una tecnología a otra categoría en DynamoDB"""
        try:
            current_query = self.get_query_by_id(query_id)
            
            if not current_query:
                logger.warning(f"No se encontró query {query_id}")
                return False
            
            current_category_id = current_query.get("category_id", "default")
            
            target_category = self.storage.get_category_by_id(target_category_id)
            if not target_category:
                logger.warning(f"No se encontró categoría destino {target_category_id}")
                return False
            
            if str(current_category_id) == str(target_category_id):
                logger.info(f"Query {query_id} ya está en categoría {target_category_id}")
                return True  # Ya está en la categoría correcta
            
            new_category_name = target_category.get("name", "Sin nombre")
            
            updates = {
                "category_id": str(target_category_id),
                "category_name": new_category_name,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            success = self.update_query(query_id, updates)
            if success:
                logger.info(f"Query {query_id} movido de {current_category_id} a {target_category_id}")
            
            return success
                
        except Exception as e:
            logger.error(f"Error moviendo tecnología: {str(e)}")
            return False

    # ===== MÉTODOS DE DEBUG =====
    
    def debug_category_queries(self, category_id: str) -> Dict:
        """Método de debug para investigar problemas con categorías"""
        try:
            # Obtener todos los registros sin filtrar
            all_items_response = self.storage.analyses_table.scan()
            all_items = all_items_response.get('Items', [])
            
            # Estadísticas
            total_items = len(all_items)
            hype_cycle_items = [item for item in all_items if item.get('analysis_type') == 'hype_cycle']
            category_items = [item for item in hype_cycle_items if str(item.get('category_id', '')) == str(category_id)]
            
            # Categorías únicas
            unique_categories = set()
            for item in hype_cycle_items:
                unique_categories.add(str(item.get('category_id', 'None')))
            
            debug_info = {
                'total_items_in_table': total_items,
                'hype_cycle_items': len(hype_cycle_items),
                'items_in_category': len(category_items),
                'unique_categories': list(unique_categories),
                'category_id_searched': str(category_id),
                'sample_items': []
            }
            
            # Muestras de items para debug
            for item in category_items[:3]:
                debug_info['sample_items'].append({
                    'analysis_id': item.get('analysis_id'),
                    'category_id': str(item.get('category_id')),
                    'analysis_type': item.get('analysis_type'),
                    'search_query': item.get('search_query', '')[:50] + '...'
                })
            
            return debug_info
            
        except Exception as e:
            return {'error': str(e)}

# ===== RESTO DE CLASES SIN CAMBIOS =====

class HypeCycleHistoryInterface:
    """Interfaz para gestionar el historial de consultas de Hype Cycle con estados estables"""
    
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
            f"{self._state_key_base}_move_target_cat"
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
        """Muestra la interfaz completa de historial"""
        st.header("📚 Historial de Consultas de Hype Cycle")
        
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Explorar por Categoría", 
            "📊 Vista de Resumen", 
            "⚙️ Gestionar Consultas",
            "🔄 Mover Tecnologías",
            "🛠️ Debug"
        ])
        
        with tab1:
            self._show_category_explorer()
        
        with tab2:
            self._show_summary_dashboard()
        
        with tab3:
            self._show_query_manager()
        
        with tab4:
            self._show_move_technologies()
        
        with tab5:
            self._show_debug_tab()
    
    def _show_debug_tab(self):
        """NUEVA: Pestaña de debug para investigar problemas"""
        st.subheader("🛠️ Debug - Investigar Problemas de Consultas")
        
        st.write("Esta pestaña te ayuda a investigar por qué no aparecen todas las consultas.")
        
        # Obtener categorías
        try:
            categories = self.storage.storage.get_all_categories()
        except:
            categories = [{"category_id": "default", "name": "Sin categoría"}]
        
        # Selector de categoría para debug
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
                
                st.write("### 📊 Información de Debug")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total items en tabla", debug_info.get('total_items_in_table', 0))
                
                with col2:
                    st.metric("Items Hype Cycle", debug_info.get('hype_cycle_items', 0))
                
                with col3:
                    st.metric("Items en esta categoría", debug_info.get('items_in_category', 0))
                
                st.write("### 🏷️ Categorías encontradas en la tabla:")
                unique_cats = debug_info.get('unique_categories', [])
                for cat in unique_cats:
                    if cat == str(selected_category_id):
                        st.write(f"- **{cat}** ← Esta es la categoría buscada")
                    else:
                        st.write(f"- {cat}")
                
                st.write(f"### 🔍 Búsqueda realizada para: `{debug_info.get('category_id_searched')}`")
                
                if debug_info.get('sample_items'):
                    st.write("### 📋 Muestra de items encontrados:")
                    for item in debug_info['sample_items']:
                        st.write(f"- **ID:** {item['analysis_id']}")
                        st.write(f"  - **Categoría:** {item['category_id']}")
                        st.write(f"  - **Tipo:** {item['analysis_type']}")
                        st.write(f"  - **Query:** {item['search_query']}")
                        st.write("---")
                
                if debug_info.get('error'):
                    st.error(f"Error en debug: {debug_info['error']}")
        
        # Test de métodos de consulta
        st.write("### 🧪 Test de Métodos de Consulta")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Test get_all_hype_cycle_queries", key=f"{self._state_key_base}_test_all"):
                all_queries = self.storage.get_all_hype_cycle_queries()
                st.metric("Total consultas encontradas", len(all_queries))
                
                if all_queries:
                    st.write("**Primeras 3 consultas:**")
                    for i, query in enumerate(all_queries[:3]):
                        st.write(f"{i+1}. {query.get('search_query', 'Sin query')[:50]}... (Cat: {query.get('category_id')})")
        
        with col2:
            if st.button("Test get_queries_by_category", key=f"{self._state_key_base}_test_cat"):
                cat_queries = self.storage.get_queries_by_category(selected_category_id)
                st.metric("Consultas en categoría", len(cat_queries))
                
                if cat_queries:
                    st.write("**Consultas en esta categoría:**")
                    for i, query in enumerate(cat_queries):
                        st.write(f"{i+1}. {query.get('search_query', 'Sin query')[:50]}...")
    
    def _show_category_explorer(self):
        """Muestra explorador por categorías con estados estables"""
        st.subheader("Explorar Consultas por Categoría")
        
        try:
            categories = self.storage.storage.get_all_categories()
        except:
            categories = [{"category_id": "default", "name": "Sin categoría"}]
        
        category_options = {cat.get("name", "Sin nombre"): cat.get("category_id") for cat in categories}
        
        category_selector_key = f"{self._state_key_base}_category_explorer_selector"
        
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
        queries = self.storage.get_queries_by_category(selected_category_id)
        
        # Mostrar información de debug
        debug_info = self.storage.debug_category_queries(selected_category_id)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Consultas mostradas", len(queries))
        with col2:
            st.metric("Total en tabla", debug_info.get('hype_cycle_items', 0))
        with col3:
            st.metric("En esta categoría", debug_info.get('items_in_category', 0))
        
        if len(queries) != debug_info.get('items_in_category', 0):
            st.warning(f"⚠️ Discrepancia detectada: se muestran {len(queries)} pero deberían ser {debug_info.get('items_in_category', 0)}")
        
        if not queries:
            st.info(f"No hay consultas guardadas en la categoría '{selected_category_name}'")
            
            # Mostrar información de debug cuando no hay resultados
            if debug_info.get('items_in_category', 0) > 0:
                st.error("⚠️ **Problema detectado**: Hay consultas en la base de datos pero no se están mostrando.")
                with st.expander("Ver información de debug"):
                    st.json(debug_info)
            
            return
        
        st.write(f"**{len(queries)} consultas encontradas en '{selected_category_name}'**")
        
        for i, query in enumerate(queries):
            self._display_query_card(query, i)
    
    def _display_query_card(self, query: Dict, index: int):
        """Muestra una tarjeta de consulta con keys estables"""
        query_id = query.get('query_id', query.get('analysis_id', 'unknown'))
        
        with st.expander(
            f"🔍 {query.get('search_query', 'Sin consulta')[:60]}... - "
            f"**{query.get('hype_metrics', {}).get('phase', 'Unknown')}**",
            expanded=False
        ):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Consulta completa:**")
                st.code(query.get('search_query', 'No disponible'))
                
                search_terms = query.get('search_terms', [])
                if search_terms:
                    st.write("**Términos de búsqueda:**")
                    for term in search_terms:
                        st.write(f"- {term.get('value', '')} ({term.get('operator', 'AND')})")
                
            with col2:
                st.write("**Métricas del Hype Cycle:**")
                hype_metrics = query.get('hype_metrics', {})
                
                st.metric("Fase", hype_metrics.get('phase', 'Unknown'))
                
                confidence = hype_metrics.get('confidence', 0)
                confidence_formatted = self._safe_format_value(confidence, "float", ".2f")
                st.metric("Confianza", confidence_formatted)
                
                mentions = hype_metrics.get('total_mentions', 0)
                mentions_formatted = self._safe_format_value(mentions, "int")
                st.metric("Total Menciones", mentions_formatted)
                
                try:
                    date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                    st.write(f"**Fecha:** {date.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.write("**Fecha:** No disponible")
            
            reuse_button_key = f"{self._state_key_base}_reuse_btn_{query_id}"
            
            if st.button(f"🔄 Reutilizar Consulta", key=reuse_button_key):
                self._reuse_query(query)
    
    def _show_summary_dashboard(self):
        """Muestra dashboard de resumen"""
        st.subheader("Dashboard de Resumen")
        
        all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay consultas de Hype Cycle guardadas")
            return
        
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
        """Interfaz para gestionar consultas con estados estables"""
        st.subheader("Gestionar Consultas")
        
        all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay consultas para gestionar")
            return
        
        query_data = []
        for query in all_queries:
            try:
                date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                formatted_date = date.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_date = "Fecha inválida"
            
            confidence_raw = query.get('hype_metrics', {}).get('confidence', 0)
            confidence_formatted = self._safe_format_value(confidence_raw, "float", ".2f")
            
            query_data.append({
                "ID": query.get("query_id", query.get("analysis_id", "Unknown")),
                "Consulta": query.get("search_query", "")[:50] + "...",
                "Fase": query.get("hype_metrics", {}).get("phase", "Unknown"),
                "Fecha": formatted_date,
                "Confianza": confidence_formatted,
                "Resultados": query.get("api_usage", {}).get("total_results", 0)
            })
        
        df_queries = pd.DataFrame(query_data)
        st.dataframe(df_queries, use_container_width=True)
        
        query_options = {}
        for i, query in enumerate(all_queries):
            query_id = query.get("query_id", query.get("analysis_id", f"query_{i}"))
            query_name = f"{query.get('search_query', '')[:30]}... ({query.get('hype_metrics', {}).get('phase', 'Unknown')})"
            query_options[query_name] = query_id
        
        if query_options:
            query_manager_selector_key = f"{self._state_key_base}_query_manager_selector"
            
            saved_query = st.session_state.get(f"{self._state_key_base}_selected_query", "")
            try:
                if saved_query and saved_query in query_options.values():
                    saved_name = None
                    for name, qid in query_options.items():
                        if qid == saved_query:
                            saved_name = name
                            break
                    
                    if saved_name and saved_name in query_options.keys():
                        default_index = list(query_options.keys()).index(saved_name)
                    else:
                        default_index = 0
                else:
                    default_index = 0
            except:
                default_index = 0
            
            selected_query_name = st.selectbox(
                "Selecciona una consulta para gestionar:",
                options=list(query_options.keys()),
                index=default_index,
                key=query_manager_selector_key
            )
            
            selected_query_id = query_options[selected_query_name]
            st.session_state[f"{self._state_key_base}_selected_query"] = selected_query_id
            
            selected_query = next((q for q in all_queries if q.get("query_id") == selected_query_id or q.get("analysis_id") == selected_query_id), None)
            
            if selected_query:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    view_details_key = f"{self._state_key_base}_view_details_{selected_query_id}"
                    if st.button("📊 Ver Detalles", key=view_details_key):
                        st.json(selected_query)
                
                with col2:
                    reuse_query_key = f"{self._state_key_base}_reuse_query_{selected_query_id}"
                    if st.button("🔄 Reutilizar Consulta", key=reuse_query_key):
                        self._reuse_query(selected_query)
                
                with col3:
                    delete_key = f"{self._state_key_base}_delete_{selected_query_id}"
                    confirm_delete_key = f"{self._state_key_base}_confirm_delete_{selected_query_id}"
                    
                    if st.button("🗑️ Eliminar", type="secondary", key=delete_key):
                        if st.checkbox("Confirmar eliminación", key=confirm_delete_key):
                            if self.storage.delete_query(selected_query_id):
                                st.success(f"Consulta {selected_query_id} eliminada")
                                if f"{self._state_key_base}_selected_query" in st.session_state:
                                    del st.session_state[f"{self._state_key_base}_selected_query"]
                                st.rerun()
                            else:
                                st.error(f"Error eliminando consulta {selected_query_id}")

    def _show_move_technologies(self):
        """Interfaz para mover tecnologías con estados estables"""
        st.subheader("🔄 Mover Tecnologías Entre Categorías")
        
        st.write("Mueve tecnologías entre diferentes categorías en DynamoDB.")
        
        all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay tecnologías para mover.")
            return
        
        try:
            categories = self.storage.storage.get_all_categories()
        except:
            categories = [{"category_id": "default", "name": "Sin categoría"}]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("#### 🔬 Seleccionar Tecnología")
            
            # Crear opciones de tecnologías
            tech_options = {}
            for query in all_queries:
                query_id = query.get("query_id", query.get("analysis_id"))
                tech_name = (
                    query.get("technology_name") or 
                    query.get("name") or 
                    query.get("search_query", "")[:30]
                )
                
                current_cat_id = query.get("category_id", "unknown")
                current_cat_name = "Sin categoría"
                for cat in categories:
                    if cat.get("category_id") == current_cat_id:
                        current_cat_name = cat.get("name", "Sin nombre")
                        break
                
                phase = query.get("hype_metrics", {}).get("phase", "Unknown")
                display_name = f"{tech_name} | {current_cat_name} | {phase}"
                tech_options[display_name] = {
                    "query_id": query_id,
                    "query": query,
                    "tech_name": tech_name,
                    "current_cat_name": current_cat_name
                }
            
            tech_selector_key = f"{self._state_key_base}_move_tech_selector"
            
            selected_tech_display = st.selectbox(
                f"Tecnología a mover ({len(tech_options)} disponibles):",
                options=list(tech_options.keys()),
                key=tech_selector_key
            )
            
            selected_tech_info = tech_options[selected_tech_display]
            selected_query = selected_tech_info["query"]
        
        with col2:
            st.write("#### 🎯 Categoría Destino")
            
            current_cat_id = selected_query.get("category_id")
            available_categories = {
                cat.get("name", "Sin nombre"): cat.get("category_id") 
                for cat in categories 
                if cat.get("category_id") != current_cat_id
            }
            
            if not available_categories:
                st.warning("No hay otras categorías disponibles.")
                return
            
            target_category_key = f"{self._state_key_base}_move_target_category"
            
            target_category_name = st.selectbox(
                "Mover a categoría:",
                options=list(available_categories.keys()),
                key=target_category_key
            )
            
            target_category_id = available_categories[target_category_name]
            
            st.info(f"**Movimiento:** '{selected_tech_info['tech_name']}' → '{target_category_name}'")
            
            confirm_key = f"{self._state_key_base}_confirm_move_{selected_tech_info['query_id']}"
            confirm_move = st.checkbox("Confirmar movimiento", key=confirm_key)
            
            move_button_key = f"{self._state_key_base}_execute_move_{selected_tech_info['query_id']}"
            
            if confirm_move and st.button("🔄 MOVER TECNOLOGÍA", type="primary", key=move_button_key):
                with st.spinner("Moviendo tecnología..."):
                    success = self.storage.move_technology_to_category(
                        selected_tech_info['query_id'], 
                        target_category_id
                    )
                    
                    if success:
                        st.success(f"✅ '{selected_tech_info['tech_name']}' movida exitosamente")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("❌ Error moviendo la tecnología")

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