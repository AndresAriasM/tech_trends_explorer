# src/hype_cycle_storage.py
import streamlit as st
import pandas as pd
import time
import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, asdict
from enum import Enum

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
    
    # NUEVOS CAMPOS para posicionamiento en gráfica
    hype_cycle_position_x: float = 50.0  # Posición X en la curva (0-100)
    hype_cycle_position_y: float = 50.0  # Posición Y en la curva (expectativas)
    time_to_plateau: str = "N/A"         # Tiempo estimado hasta plateau
    
    # Campos existentes
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
    search_terms: List[Dict]  # Lista de términos con operadores
    execution_date: str
    api_usage: Dict  # Información sobre uso de API (tokens, requests, etc.)
    
    # Resultados del análisis
    hype_metrics: HypeCycleMetrics
    yearly_stats: List[Dict]  # Datos anuales de menciones y sentimiento
    news_results: List[Dict]  # Resultados completos de noticias
    
    # Metadatos adicionales
    search_parameters: Dict  # Parámetros de búsqueda (año mínimo, filtros, etc.)
    data_quality: Dict  # Métricas de calidad de datos
    processing_time: float  # Tiempo de procesamiento
    
    # NUEVOS CAMPOS para administración
    technology_name: str = ""     # Nombre simplificado para mostrar en gráfica
    category_name: str = ""       # Nombre de la categoría (para referencia rápida)
    last_updated: str = ""        # Última actualización
    is_active: bool = True        # Si debe mostrarse en gráficas activas
    technology_description: str = ""  # Descripción opcional
    
    # Para versionado y auditoría
    created_by: str = "system"
    version: str = "1.0"
    notes: str = ""

class HypeCycleStorage:
    """Clase especializada para gestionar almacenamiento de consultas Hype Cycle"""
    
    def __init__(self, db_storage):
        """
        Inicializa con el storage de DynamoDB existente
        """
        self.storage = db_storage
        self.table_name = "hype-cycle-queries"
        
        # Si es DynamoDB, asegurar que existe la tabla específica
        if hasattr(db_storage, 'dynamodb'):
            self._ensure_hype_cycle_table()
    
    def _convert_floats_to_decimal(self, obj):
        """
        Convierte recursivamente todos los floats a Decimal para DynamoDB
        
        Args:
            obj: Objeto a convertir (dict, list, float, etc.)
            
        Returns:
            Objeto con floats convertidos a Decimal
        """
        if isinstance(obj, dict):
            return {k: self._convert_floats_to_decimal(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_floats_to_decimal(item) for item in obj]
        elif isinstance(obj, float):
            # Convertir float a Decimal con precisión limitada
            return Decimal(str(round(obj, 6)))
        elif isinstance(obj, (int, str, bool)) or obj is None:
            return obj
        elif hasattr(obj, 'item'):  # Objetos numpy
            return Decimal(str(round(float(obj.item()), 6)))
        else:
            # Para otros tipos, intentar convertir a string
            return str(obj)
    
    def _convert_dataclass_to_dict(self, dataclass_obj):
        """
        Convierte dataclass a diccionario y maneja floats
        
        Args:
            dataclass_obj: Objeto dataclass
            
        Returns:
            Diccionario con valores convertidos
        """
        if hasattr(dataclass_obj, '__dict__'):
            # Es un dataclass o objeto similar
            obj_dict = {}
            for key, value in dataclass_obj.__dict__.items():
                obj_dict[key] = self._convert_floats_to_decimal(value)
            return obj_dict
        else:
            # No es un dataclass, convertir directamente
            return self._convert_floats_to_decimal(dataclass_obj)
    
    def _ensure_hype_cycle_table(self):
        """Asegura que existe la tabla específica para Hype Cycle en DynamoDB"""
        try:
            # Intentar acceder a la tabla
            table = self.storage.dynamodb.Table(self.table_name)
            table.table_status
        except Exception:
            # Si no existe, usar la tabla general de análisis
            self.table_name = self.storage.analyses_table_name
            st.info(f"Usando tabla general: {self.table_name}")
    
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
        Guarda una consulta completa de Hype Cycle con soporte completo para DynamoDB
        """
        try:
            # Importar el positioner
            from hype_cycle_positioning import HypeCyclePositioner
            positioner = HypeCyclePositioner()
            
            # Generar ID único
            query_id = f"hype_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            
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
            
            # Estimar tiempo al plateau
            hype_metrics.time_to_plateau = positioner.estimate_time_to_plateau(
                hype_metrics.phase, 
                hype_metrics.confidence
            )
            
            # Procesar estadísticas anuales
            yearly_stats = self._process_yearly_stats(hype_analysis_results.get('yearly_stats', []))
            
            # Calcular métricas de calidad de datos
            data_quality = self._calculate_data_quality(news_results, yearly_stats)
            
            # Estimar uso de API
            api_usage = {
                "estimated_requests": len(news_results) // 10 + 1,
                "total_results": len(news_results),
                "search_timestamp": datetime.now(timezone.utc).isoformat(),
                "api_provider": "SerpAPI"
            }
            
            # Obtener información de categoría
            try:
                category = self.storage.get_category_by_id(category_id)
                category_name = category.get("name") if category else "Sin categoría"
            except:
                category_name = "Sin categoría"
            
            # Generar nombre de tecnología si no se proporciona
            if not technology_name:
                technology_name = self._extract_technology_name(search_query, search_terms)
            
            # Crear objeto de consulta
            hype_query = HypeCycleQuery(
                query_id=query_id,
                category_id=category_id,
                search_query=search_query,
                search_terms=search_terms,
                execution_date=datetime.now(timezone.utc).isoformat(),
                api_usage=api_usage,
                hype_metrics=hype_metrics,
                yearly_stats=yearly_stats,
                news_results=self._sanitize_news_results(news_results),
                search_parameters=search_parameters or {},
                data_quality=data_quality,
                processing_time=time.time(),
                notes=notes,
                technology_name=technology_name,
                category_name=category_name,
                last_updated=datetime.now(timezone.utc).isoformat(),
                is_active=True,
                technology_description=technology_description
            )
            
            # Convertir a diccionario y preparar para almacenamiento
            query_dict = self._prepare_for_storage(hype_query)
            
            # NUEVO: Convertir floats a Decimal para DynamoDB
            if hasattr(self.storage, 'analyses_table'):
                # Es DynamoDB - convertir floats a Decimal
                query_dict = self._convert_floats_to_decimal(query_dict)
            
            # Guardar en el storage
            if hasattr(self.storage, 'analyses_table'):
                # DynamoDB
                self.storage.analyses_table.put_item(Item=query_dict)
            else:
                # Storage local - los floats están bien aquí
                if "hype_cycle_queries" not in self.storage.data:
                    self.storage.data["hype_cycle_queries"] = []
                self.storage.data["hype_cycle_queries"].append(query_dict)
                self.storage.save_data()
            
            st.success(f"✅ Consulta de Hype Cycle guardada con ID: {query_id}")
            return query_id
            
        except Exception as e:
            st.error(f"❌ Error al guardar consulta de Hype Cycle: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None
    
    def _extract_hype_metrics(self, hype_results: Dict) -> HypeCycleMetrics:
        """Extrae métricas específicas del análisis de Hype Cycle"""
        try:
            metrics_data = hype_results.get('metrics', {})
            inflection_points = hype_results.get('inflection_points', {})
            
            # Extraer años de puntos de inflexión
            innovation_year = None
            peak_year = None
            trough_year = None
            
            if inflection_points:
                if inflection_points.get('innovation_trigger'):
                    innovation_year = inflection_points['innovation_trigger'].get('year')
                if inflection_points.get('peak'):
                    peak_year = inflection_points['peak'].get('year')
                if inflection_points.get('trough'):
                    trough_year = inflection_points['trough'].get('year')
            
            return HypeCycleMetrics(
                phase=hype_results.get('phase', 'Unknown'),
                confidence=float(hype_results.get('confidence', 0.0)),
                total_mentions=int(metrics_data.get('total_mentions', 0)),
                peak_mentions=int(metrics_data.get('peak_mentions', 0)),
                latest_year=int(metrics_data.get('latest_year', datetime.now().year)),
                sentiment_avg=float(hype_results.get('yearly_stats', {}).get('sentiment_mean', {}).mean() if hasattr(hype_results.get('yearly_stats', {}), 'mean') else 0.0),
                sentiment_trend=0.0,  # Se calculará más adelante
                innovation_trigger_year=innovation_year,
                peak_year=peak_year,
                trough_year=trough_year,
                inflection_points=inflection_points
            )
            
        except Exception as e:
            st.warning(f"Error extrayendo métricas: {str(e)}")
            return HypeCycleMetrics(
                phase="Unknown",
                confidence=0.0,
                total_mentions=0,
                peak_mentions=0,
                latest_year=datetime.now().year,
                sentiment_avg=0.0,
                sentiment_trend=0.0
            )
    
    def _process_yearly_stats(self, yearly_stats) -> List[Dict]:
        """Procesa estadísticas anuales para almacenamiento"""
        try:
            if hasattr(yearly_stats, 'to_dict'):
                # Si es un DataFrame de pandas
                records = yearly_stats.to_dict('records')
            elif isinstance(yearly_stats, list):
                records = yearly_stats
            else:
                records = []
            
            # Asegurar que todos los valores son serializables
            processed_records = []
            for record in records:
                processed_record = {}
                for key, value in record.items():
                    if hasattr(value, 'item'):  # numpy types
                        processed_record[key] = value.item()
                    elif isinstance(value, (int, float, str, bool)) or value is None:
                        processed_record[key] = value
                    else:
                        processed_record[key] = str(value)
                processed_records.append(processed_record)
            
            return processed_records
            
        except Exception as e:
            st.warning(f"Error procesando estadísticas anuales: {str(e)}")
            return []
    
    def _sanitize_news_results(self, news_results: List[Dict]) -> List[Dict]:
        """Sanitiza resultados de noticias para almacenamiento"""
        sanitized = []
        
        for result in news_results:
            sanitized_result = {}
            for key, value in result.items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    sanitized_result[key] = value
                elif isinstance(value, list):
                    sanitized_result[key] = [str(item) for item in value]
                else:
                    sanitized_result[key] = str(value)
            sanitized.append(sanitized_result)
        
        return sanitized
    
    def _calculate_data_quality(self, news_results: List[Dict], yearly_stats: List[Dict]) -> Dict:
        """Calcula métricas de calidad de los datos"""
        try:
            # Calcular completitud de datos
            total_results = len(news_results)
            results_with_date = sum(1 for r in news_results if r.get('date'))
            results_with_sentiment = sum(1 for r in news_results if r.get('sentiment') is not None)
            results_with_country = sum(1 for r in news_results if r.get('country'))
            
            # Calcular distribución temporal
            years_covered = len(yearly_stats) if yearly_stats else 0
            
            return {
                "total_results": total_results,
                "completeness_score": {
                    "date_coverage": (results_with_date / total_results) if total_results > 0 else 0,
                    "sentiment_coverage": (results_with_sentiment / total_results) if total_results > 0 else 0,
                    "geography_coverage": (results_with_country / total_results) if total_results > 0 else 0
                },
                "temporal_coverage": {
                    "years_covered": years_covered,
                    "year_range": f"{min(s.get('year', 0) for s in yearly_stats)}-{max(s.get('year', 0) for s in yearly_stats)}" if yearly_stats else "No data"
                },
                "quality_score": ((results_with_date + results_with_sentiment + results_with_country) / (total_results * 3)) if total_results > 0 else 0
            }
            
        except Exception as e:
            st.warning(f"Error calculando calidad de datos: {str(e)}")
            return {"quality_score": 0, "error": str(e)}
    
    def _prepare_for_storage(self, hype_query: HypeCycleQuery) -> Dict:
        """Prepara el objeto para almacenamiento - VERSIÓN MEJORADA"""
        
        # Convertir dataclass a diccionario manejando floats
        if hasattr(hype_query, '__dict__'):
            query_dict = {}
            for key, value in hype_query.__dict__.items():
                if hasattr(value, '__dict__') and hasattr(value, '__dataclass_fields__'):
                    # Es un dataclass (como HypeCycleMetrics)
                    query_dict[key] = self._convert_dataclass_to_dict(value)
                elif isinstance(value, (list, dict)):
                    # Es una lista o diccionario
                    query_dict[key] = self._convert_floats_to_decimal(value)
                else:
                    # Es un valor simple
                    query_dict[key] = self._convert_floats_to_decimal(value)
        else:
            # No es un objeto con __dict__, usar asdict
            from dataclasses import asdict
            query_dict = asdict(hype_query)
        
        # Agregar campos necesarios para compatibilidad
        query_dict.update({
            "analysis_id": hype_query.query_id,
            "timestamp": hype_query.execution_date,
            "analysis_type": "hype_cycle",
            "query": hype_query.search_query,
            "name": f"Hype Cycle: {hype_query.technology_name or hype_query.search_query[:50]}..."
        })
        
        # Actualizar tiempo de procesamiento
        processing_start = query_dict.get("processing_time", time.time())
        query_dict["processing_time"] = time.time() - processing_start
        
        return query_dict
    
    def get_queries_by_category(self, category_id: str) -> List[Dict]:
        """Obtiene todas las consultas de Hype Cycle de una categoría"""
        try:
            if hasattr(self.storage, 'analyses_table'):
                # DynamoDB
                response = self.storage.analyses_table.scan(
                    FilterExpression="category_id = :cat_id AND analysis_type = :type",
                    ExpressionAttributeValues={
                        ":cat_id": category_id,
                        ":type": "hype_cycle"
                    }
                )
                return response.get('Items', [])
            else:
                # Storage local
                queries = self.storage.data.get("hype_cycle_queries", [])
                return [q for q in queries if q.get("category_id") == category_id]
                
        except Exception as e:
            st.error(f"Error obteniendo consultas: {str(e)}")
            return []
    
    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        """Obtiene una consulta específica por ID"""
        try:
            if hasattr(self.storage, 'analyses_table'):
                # DynamoDB
                response = self.storage.analyses_table.scan(
                    FilterExpression="analysis_id = :id OR query_id = :id",
                    ExpressionAttributeValues={":id": query_id}
                )
                items = response.get('Items', [])
                return items[0] if items else None
            else:
                # Storage local
                queries = self.storage.data.get("hype_cycle_queries", [])
                for query in queries:
                    if query.get("query_id") == query_id or query.get("analysis_id") == query_id:
                        return query
                return None
                
        except Exception as e:
            st.error(f"Error obteniendo consulta: {str(e)}")
            return None
    
    def get_all_hype_cycle_queries(self) -> List[Dict]:
        """Obtiene todas las consultas de Hype Cycle"""
        try:
            if hasattr(self.storage, 'analyses_table'):
                # DynamoDB
                response = self.storage.analyses_table.scan(
                    FilterExpression="analysis_type = :type",
                    ExpressionAttributeValues={":type": "hype_cycle"}
                )
                return response.get('Items', [])
            else:
                # Storage local
                return self.storage.data.get("hype_cycle_queries", [])
                
        except Exception as e:
            st.error(f"Error obteniendo todas las consultas: {str(e)}")
            return []
    
    def delete_query(self, query_id: str) -> bool:
        """Elimina una consulta específica"""
        try:
            if hasattr(self.storage, 'analyses_table'):
                # DynamoDB - necesitamos el timestamp también
                query = self.get_query_by_id(query_id)
                if query:
                    self.storage.analyses_table.delete_item(
                        Key={
                            "analysis_id": query_id,
                            "timestamp": query["timestamp"]
                        }
                    )
                    return True
            else:
                # Storage local
                queries = self.storage.data.get("hype_cycle_queries", [])
                self.storage.data["hype_cycle_queries"] = [
                    q for q in queries 
                    if q.get("query_id") != query_id and q.get("analysis_id") != query_id
                ]
                self.storage.save_data()
                return True
            
            return False
            
        except Exception as e:
            st.error(f"Error eliminando consulta: {str(e)}")
            return False
    def _extract_technology_name(self, search_query: str, search_terms: List[Dict]) -> str:
        """Extrae un nombre de tecnología limpio de la consulta"""
        # Priorizar el primer término más significativo
        for term in search_terms:
            value = term.get('value', '').strip().strip('"')
            if len(value) > 2 and value.lower() not in ['and', 'or', 'not']:
                return value.title()
        
        # Fallback: limpiar la query
        clean_query = search_query.replace('"', '').replace(' AND ', ' ').replace(' OR ', ' ')
        clean_query = clean_query.replace(' NOT ', ' ').replace('after:', '').replace('before:', '')
        words = [w for w in clean_query.split() if not w.isdigit() and len(w) > 2]
        return ' '.join(words[:2]).title() if words else "Tecnología"

    def _extract_hype_metrics(self, hype_results: Dict) -> HypeCycleMetrics:
        """Extrae métricas específicas del análisis de Hype Cycle - VERSIÓN SIN FLOATS"""
        try:
            metrics_data = hype_results.get('metrics', {})
            inflection_points = hype_results.get('inflection_points', {})
            
            # Extraer años de puntos de inflexión
            innovation_year = None
            peak_year = None
            trough_year = None
            
            if inflection_points:
                if inflection_points.get('innovation_trigger'):
                    innovation_year = inflection_points['innovation_trigger'].get('year')
                if inflection_points.get('peak'):
                    peak_year = inflection_points['peak'].get('year')
                if inflection_points.get('trough'):
                    trough_year = inflection_points['trough'].get('year')
            
            # Calcular sentiment promedio
            sentiment_avg = 0.0
            yearly_stats = hype_results.get('yearly_stats')
            if yearly_stats is not None:
                try:
                    if hasattr(yearly_stats, 'sentiment_mean'):
                        sentiment_avg = float(yearly_stats['sentiment_mean'].mean())
                    elif isinstance(yearly_stats, list):
                        sentiments = [float(stat.get('sentiment_mean', 0)) for stat in yearly_stats]
                        sentiment_avg = sum(sentiments) / len(sentiments) if sentiments else 0.0
                except:
                    sentiment_avg = 0.0
            
            # Crear métricas sin floats (se convertirán después)
            return HypeCycleMetrics(
                phase=hype_results.get('phase', 'Unknown'),
                confidence=float(hype_results.get('confidence', 0.0)),
                total_mentions=int(metrics_data.get('total_mentions', 0)),
                peak_mentions=int(metrics_data.get('peak_mentions', 0)),
                latest_year=int(metrics_data.get('latest_year', datetime.now().year)),
                sentiment_avg=sentiment_avg,
                sentiment_trend=0.0,
                innovation_trigger_year=innovation_year,
                peak_year=peak_year,
                trough_year=trough_year,
                inflection_points=inflection_points,
                hype_cycle_position_x=50.0,  # Se actualizará después
                hype_cycle_position_y=50.0,  # Se actualizará después
                time_to_plateau="N/A"        # Se actualizará después
            )
            
        except Exception as e:
            st.warning(f"Error extrayendo métricas: {str(e)}")
            return HypeCycleMetrics(
                phase="Unknown",
                confidence=0.0,
                total_mentions=0,
                peak_mentions=0,
                latest_year=datetime.now().year,
                sentiment_avg=0.0,
                sentiment_trend=0.0,
                hype_cycle_position_x=50.0,
                hype_cycle_position_y=50.0,
                time_to_plateau="N/A"
            )

    def _prepare_for_storage(self, hype_query: HypeCycleQuery) -> Dict:
        """Prepara el objeto para almacenamiento en DynamoDB - VERSIÓN ACTUALIZADA"""
        # Convertir a diccionario
        query_dict = asdict(hype_query)
        
        # Agregar campos necesarios para DynamoDB
        query_dict.update({
            "analysis_id": hype_query.query_id,  # Para compatibilidad con tabla general
            "timestamp": hype_query.execution_date,
            "analysis_type": "hype_cycle",
            "query": hype_query.search_query,
            "name": f"Hype Cycle: {hype_query.technology_name or hype_query.search_query[:50]}..."
        })
        
        # Actualizar tiempo de procesamiento
        query_dict["processing_time"] = time.time() - query_dict["processing_time"]
        
        return query_dict

# REEMPLAZAR la clase HypeCycleHistoryInterface en hype_cycle_storage.py

class HypeCycleHistoryInterface:
    """Interfaz para gestionar el historial de consultas de Hype Cycle"""
    
    def __init__(self, hype_storage, context_prefix: str = "default"):
        self.storage = hype_storage
        self.context_prefix = context_prefix  # Añadir prefijo de contexto
        
        # GENERAR TIMESTAMP ÚNICO para evitar conflictos
        import time
        self.unique_id = str(int(time.time()))[-6:]  # Últimos 6 dígitos del timestamp
    
    def show_history_interface(self):
        """Muestra la interfaz completa de historial"""
        st.header("📚 Historial de Consultas de Hype Cycle")
        
        # Pestañas para diferentes vistas - KEYS ÚNICAS
        tab1, tab2, tab3 = st.tabs([
            "🔍 Explorar por Categoría", 
            "📊 Vista de Resumen", 
            "⚙️ Gestionar Consultas"
        ])
        
        with tab1:
            self._show_category_explorer()
        
        with tab2:
            self._show_summary_dashboard()
        
        with tab3:
            self._show_query_manager()
    
    def _show_category_explorer(self):
        """Muestra explorador por categorías"""
        st.subheader("Explorar Consultas por Categoría")
        
        # Obtener categorías disponibles
        try:
            categories = self.storage.storage.get_all_categories()
        except:
            categories = [{"id": "default", "name": "Sin categoría"}]
        
        # Selector de categoría - KEY ÚNICA CON TIMESTAMP
        category_options = {cat.get("name", "Sin nombre"): cat.get("id", cat.get("category_id")) for cat in categories}
        
        selected_category_name = st.selectbox(
            "Selecciona una categoría",
            options=list(category_options.keys()),
            key=f"{self.context_prefix}_history_category_explorer_{self.unique_id}"  # ← KEY ÚNICA
        )
        
        selected_category_id = category_options[selected_category_name]
        
        # Obtener consultas de la categoría
        queries = self.storage.get_queries_by_category(selected_category_id)
        
        if not queries:
            st.info(f"No hay consultas guardadas en la categoría '{selected_category_name}'")
            return
        
        st.write(f"**{len(queries)} consultas encontradas en '{selected_category_name}'**")
        
        # Mostrar consultas
        for i, query in enumerate(queries):
            self._display_query_card(query, i)  # Pasar índice para keys únicas
    
    def _display_query_card(self, query: Dict, index: int):
        """Muestra una tarjeta de consulta"""
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
                
                # Mostrar términos de búsqueda si están disponibles
                search_terms = query.get('search_terms', [])
                if search_terms:
                    st.write("**Términos de búsqueda:**")
                    for term in search_terms:
                        st.write(f"- {term.get('value', '')} ({term.get('operator', 'AND')})")
                
                # Información de calidad de datos
                data_quality = query.get('data_quality', {})
                if data_quality:
                    quality_score = data_quality.get('quality_score', 0)
                    st.write(f"**Calidad de datos:** {quality_score:.2%}")
                
            with col2:
                st.write("**Métricas del Hype Cycle:**")
                hype_metrics = query.get('hype_metrics', {})
                
                st.metric("Fase", hype_metrics.get('phase', 'Unknown'))
                st.metric("Confianza", f"{hype_metrics.get('confidence', 0):.2f}")
                st.metric("Total Menciones", hype_metrics.get('total_mentions', 0))
                
                # Fecha de ejecución
                try:
                    date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                    st.write(f"**Fecha:** {date.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.write("**Fecha:** No disponible")
            
            # Botón para reutilizar consulta - KEY ÚNICA
            if st.button(f"🔄 Reutilizar Consulta", key=f"{self.context_prefix}_reuse_query_{self.unique_id}_{index}"):
                self._reuse_query(query)
    
    def _show_summary_dashboard(self):
        """Muestra dashboard de resumen"""
        st.subheader("Dashboard de Resumen")
        
        # Obtener todas las consultas
        all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay consultas de Hype Cycle guardadas")
            return
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Consultas", len(all_queries))
        
        with col2:
            # Fases más comunes
            phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in all_queries]
            most_common_phase = max(set(phases), key=phases.count) if phases else "N/A"
            st.metric("Fase Más Común", most_common_phase)
        
        with col3:
            # Promedio de confianza
            confidences = [q.get("hype_metrics", {}).get("confidence", 0) for q in all_queries]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            st.metric("Confianza Promedio", f"{avg_confidence:.2f}")
        
        with col4:
            # Consultas este mes
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
            import plotly.express as px
            fig_phases = px.pie(
                values=list(phase_counts.values()),
                names=list(phase_counts.keys()),
                title="Distribución de Tecnologías por Fase del Hype Cycle"
            )
            st.plotly_chart(fig_phases, use_container_width=True)
        
        # Timeline de consultas
        st.subheader("Timeline de Consultas")
        
        # Preparar datos para timeline
        timeline_data = []
        for query in all_queries:
            try:
                date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                timeline_data.append({
                    "Fecha": date,
                    "Consulta": query.get("search_query", "")[:30] + "...",
                    "Fase": query.get("hype_metrics", {}).get("phase", "Unknown"),
                    "Confianza": query.get("hype_metrics", {}).get("confidence", 0)
                })
            except:
                continue
        
        if timeline_data:
            import pandas as pd
            import plotly.express as px
            df_timeline = pd.DataFrame(timeline_data)
            df_timeline = df_timeline.sort_values("Fecha")
            
            fig_timeline = px.scatter(
                df_timeline,
                x="Fecha",
                y="Confianza",
                color="Fase",
                hover_data=["Consulta"],
                title="Timeline de Consultas por Confianza y Fase"
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

    def _show_query_manager(self):
        """Interfaz para gestionar consultas"""
        st.subheader("Gestionar Consultas")
        
        # Obtener todas las consultas
        all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay consultas para gestionar")
            return
        
        # Tabla de consultas para gestión
        query_data = []
        for query in all_queries:
            try:
                date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                formatted_date = date.strftime("%Y-%m-%d %H:%M")
            except:
                formatted_date = "Fecha inválida"
            
            query_data.append({
                "ID": query.get("query_id", query.get("analysis_id", "Unknown")),
                "Consulta": query.get("search_query", "")[:50] + "...",
                "Fase": query.get("hype_metrics", {}).get("phase", "Unknown"),
                "Fecha": formatted_date,
                "Confianza": f"{query.get('hype_metrics', {}).get('confidence', 0):.2f}",
                "Resultados": query.get("api_usage", {}).get("total_results", 0)
            })
        
        import pandas as pd
        df_queries = pd.DataFrame(query_data)
        
        # Mostrar tabla con selección
        st.write("Selecciona consultas para gestionar:")
        
        # Mostrar tabla simple (sin selection ya que puede causar problemas)
        st.dataframe(df_queries, use_container_width=True)
        
        # Selector manual para gestión
        query_options = {}
        for i, query in enumerate(all_queries):
            query_id = query.get("query_id", query.get("analysis_id", f"query_{i}"))
            query_name = f"{query.get('search_query', '')[:30]}... ({query.get('hype_metrics', {}).get('phase', 'Unknown')})"
            query_options[query_name] = query_id
        
        if query_options:
            selected_query_name = st.selectbox(
                "Selecciona una consulta para gestionar:",
                options=list(query_options.keys()),
                key=f"query_manager_selectbox_{self.unique_id}"  
            )
            
            selected_query_id = query_options[selected_query_name]
            selected_query = next((q for q in all_queries if q.get("query_id") == selected_query_id or q.get("analysis_id") == selected_query_id), None)
            
            if selected_query:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 Ver Detalles", key=f"view_details_{self.unique_id}"):
                        self._display_query_details(selected_query)
                
                with col2:
                    if st.button("🔄 Reutilizar Consulta", key=f"reuse_query_{self.unique_id}"):
                        self._reuse_query(selected_query)
                
                with col3:
                    if st.button("🗑️ Eliminar", type="secondary", key=f"delete_{self.unique_id}"):
                        if st.checkbox("Confirmar eliminación", key=f"confirm_delete_{self.unique_id}"):
                            if self.storage.delete_query(selected_query_id):
                                st.success(f"Consulta {selected_query_id} eliminada")
                                st.rerun()
                            else:
                                st.error(f"Error eliminando consulta {selected_query_id}")

    def _display_query_details(self, query: Dict):
        """Muestra detalles completos de una consulta"""
        st.subheader(f"Detalles de Consulta: {query.get('search_query', 'Sin nombre')[:50]}...")
        
        # Información básica
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Información General**")
            st.write(f"ID: `{query.get('query_id', 'Unknown')}`")
            try:
                date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                st.write(f"Fecha: {date.strftime('%Y-%m-%d %H:%M:%S')}")
            except:
                st.write("Fecha: No disponible")
            st.write(f"Tiempo de procesamiento: {query.get('processing_time', 0):.2f}s")
        
        with col2:
            st.write("**Métricas del Hype Cycle**")
            hype_metrics = query.get('hype_metrics', {})
            for key, value in hype_metrics.items():
                if key not in ['inflection_points']:  # Excluir objetos complejos
                    st.write(f"{key}: {value}")
        
        with col3:
            st.write("**Uso de API**")
            api_usage = query.get('api_usage', {})
            for key, value in api_usage.items():
                st.write(f"{key}: {value}")
        
        # Datos anuales si están disponibles
        yearly_stats = query.get('yearly_stats', [])
        if yearly_stats:
            st.subheader("Estadísticas Anuales")
            import pandas as pd
            df_yearly = pd.DataFrame(yearly_stats)
            st.dataframe(df_yearly)
        
        # Muestra de resultados de noticias
        news_results = query.get('news_results', [])
        if news_results:
            st.subheader(f"Muestra de Resultados de Noticias ({len(news_results)} total)")
            for i, result in enumerate(news_results[:3]):  # Mostrar solo los primeros 3
                with st.expander(f"Noticia {i+1}: {result.get('title', 'Sin título')[:50]}..."):
                    st.write(f"**Fuente:** {result.get('source', 'No especificada')}")
                    st.write(f"**Fecha:** {result.get('date', 'No especificada')}")
                    st.write(f"**Sentimiento:** {result.get('sentiment', 'No calculado')}")
                    st.write(f"**Link:** {result.get('link', 'No disponible')}")

    def _reuse_query(self, query: Dict):
        """Permite reutilizar una consulta existente"""
        st.info("**Consulta seleccionada para reutilizar:**")
        st.code(query.get('search_query', 'No disponible'))
        
        # Guardar en session state para que esté disponible en la pestaña principal
        st.session_state.hype_reuse_query = {
            'search_query': query.get('search_query', ''),
            'search_terms': query.get('search_terms', []),
            'search_parameters': query.get('search_parameters', {})
        }
        
        st.success("✅ Consulta cargada. Ve a la pestaña 'Nuevo Análisis' para ejecutarla nuevamente o modificarla.")

# Función para integrar con el sistema existente
def initialize_hype_cycle_storage(db_storage):
    """Inicializa el sistema de almacenamiento de Hype Cycle"""
    return HypeCycleStorage(db_storage)

def create_hype_cycle_interface(hype_storage, context_prefix: str = "default"):
    """Crea la interfaz completa de historial de Hype Cycle"""
    return HypeCycleHistoryInterface(hype_storage, context_prefix)