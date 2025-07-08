# src/hype_cycle_storage.py - CORREGIDO CON FUNCIONES DE NUBE Y FORMATEO SEGURO
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
import math

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
    """Clase especializada para gestionar almacenamiento de consultas Hype Cycle"""
    
    def __init__(self, db_storage):
        """Inicializa con el storage de DynamoDB"""
        self.storage = db_storage
        
        # Verificar que es DynamoDB
        if not hasattr(db_storage, 'dynamodb'):
            raise ValueError("HypeCycleStorage requiere DynamoDB storage")
    
    def _generate_unique_query_id(self):
        """Genera un ID único garantizado para queries de Hype Cycle"""
        timestamp = int(time.time() * 1000)  # Milisegundos
        unique_part = str(uuid.uuid4())[:12]  # Más caracteres para unicidad
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
        Guarda una consulta completa de Hype Cycle - VERSIÓN ÚNICA Y LIMPIA
        """
        try:
            # Importar el positioner
            from hype_cycle_positioning import HypeCyclePositioner
            positioner = HypeCyclePositioner()
            
            # Generar ID único garantizado
            query_id = self._generate_unique_query_id()
            
            # Sanitizar datos ANTES de procesar
            cleaned_hype_results = self.storage._sanitize_for_dynamodb(hype_analysis_results)
            cleaned_news_results = self.storage._sanitize_for_dynamodb(news_results)
            
            # Procesar métricas del Hype Cycle
            hype_metrics = self._extract_hype_metrics(cleaned_hype_results)
            
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
            
            # Procesar estadísticas anuales
            yearly_stats = self._process_yearly_stats(cleaned_hype_results.get('yearly_stats', []))
            
            # Calcular métricas de calidad
            data_quality = self._calculate_data_quality(cleaned_news_results, yearly_stats)
            
            # Obtener información de categoría
            category_name = self._get_category_name(category_id)
            
            # Generar nombre de tecnología si no se proporciona
            if not technology_name:
                technology_name = self._extract_technology_name(search_query, search_terms)
            
            # Crear timestamp único
            execution_timestamp = datetime.now(timezone.utc).isoformat()
            
            # Crear objeto de consulta
            hype_query = HypeCycleQuery(
                query_id=query_id,
                category_id=category_id,
                search_query=search_query,
                search_terms=search_terms,
                execution_date=execution_timestamp,
                api_usage={
                    "estimated_requests": len(cleaned_news_results) // 10 + 1,
                    "total_results": len(cleaned_news_results),
                    "search_timestamp": execution_timestamp,
                    "api_provider": "SerpAPI"
                },
                hype_metrics=hype_metrics,
                yearly_stats=yearly_stats,
                news_results=self._sanitize_news_results(cleaned_news_results),
                search_parameters=search_parameters or {},
                data_quality=data_quality,
                processing_time=time.time(),
                notes=notes,
                technology_name=technology_name,
                category_name=category_name,
                last_updated=execution_timestamp,
                is_active=True,
                technology_description=technology_description
            )
            
            # Convertir a diccionario y preparar para DynamoDB
            query_dict = self._prepare_for_dynamodb(hype_query)
            
            # Guardar en DynamoDB
            self.storage.analyses_table.put_item(Item=query_dict)
            
            st.success(f"✅ Consulta de Hype Cycle guardada con ID único: {query_id}")
            return query_id
            
        except Exception as e:
            st.error(f"❌ Error al guardar consulta de Hype Cycle: {str(e)}")
            import traceback
            with st.expander("🔍 Ver detalles del error"):
                st.code(traceback.format_exc())
            return None
    
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
            
            # Calcular sentiment promedio de forma segura
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
                records = yearly_stats.to_dict('records')
            elif isinstance(yearly_stats, list):
                records = yearly_stats
            else:
                records = []
            
            processed_records = []
            for record in records:
                processed_record = {}
                for key, value in record.items():
                    if hasattr(value, 'item'):
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
            total_results = len(news_results)
            results_with_date = sum(1 for r in news_results if r.get('date'))
            results_with_sentiment = sum(1 for r in news_results if r.get('sentiment') is not None)
            results_with_country = sum(1 for r in news_results if r.get('country'))
            
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
    
    def _prepare_for_dynamodb(self, hype_query: HypeCycleQuery) -> Dict:
        """Prepara el objeto para almacenamiento en DynamoDB"""
        # Convertir dataclass a diccionario
        query_dict = asdict(hype_query)
        
        # Agregar campos requeridos para DynamoDB
        query_dict.update({
            "analysis_id": hype_query.query_id,  # Clave primaria
            "timestamp": hype_query.execution_date,  # Clave de ordenamiento
            "analysis_type": "hype_cycle",
            "query": hype_query.search_query,
            "name": f"Hype Cycle: {hype_query.technology_name or hype_query.search_query[:50]}..."
        })
        
        # Actualizar tiempo de procesamiento
        query_dict["processing_time"] = time.time() - query_dict["processing_time"]
        
        # Convertir floats a Decimal para DynamoDB
        return self.storage._convert_floats_to_decimal(query_dict)
    
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
    
    def get_queries_by_category(self, category_id: str) -> List[Dict]:
        """Obtiene todas las consultas de Hype Cycle de una categoría"""
        try:
            response = self.storage.analyses_table.scan(
                FilterExpression="category_id = :cat_id AND analysis_type = :type",
                ExpressionAttributeValues={
                    ":cat_id": category_id,
                    ":type": "hype_cycle"
                }
            )
            items = response.get('Items', [])
            return [self.storage._convert_decimals_to_float(item) for item in items]
                
        except Exception as e:
            st.error(f"Error obteniendo consultas por categoría: {str(e)}")
            return []

    def get_query_by_id(self, query_id: str) -> Optional[Dict]:
        """Obtiene una consulta específica por ID"""
        try:
            response = self.storage.analyses_table.scan(
                FilterExpression="analysis_id = :id OR query_id = :id",
                ExpressionAttributeValues={":id": query_id}
            )
            items = response.get('Items', [])
            if items:
                return self.storage._convert_decimals_to_float(items[0])
            return None
                
        except Exception as e:
            st.error(f"Error obteniendo consulta por ID: {str(e)}")
            return None

    def get_all_hype_cycle_queries(self) -> List[Dict]:
        """Obtiene todas las consultas de Hype Cycle"""
        try:
            response = self.storage.analyses_table.scan(
                FilterExpression="analysis_type = :type",
                ExpressionAttributeValues={":type": "hype_cycle"}
            )
            items = response.get('Items', [])
            return [self.storage._convert_decimals_to_float(item) for item in items]
                
        except Exception as e:
            st.error(f"Error obteniendo todas las consultas: {str(e)}")
            return []
    
    def delete_query(self, query_id: str) -> bool:
        """Elimina una consulta específica"""
        try:
            # Primero obtener el timestamp
            query = self.get_query_by_id(query_id)
            if query:
                return self.storage.delete_item(query_id, query["timestamp"])
            return False
            
        except Exception as e:
            st.error(f"Error eliminando consulta: {str(e)}")
            return False
    
    def update_query(self, query_id: str, updates: Dict) -> bool:
        """Actualiza una consulta específica"""
        try:
            # Primero obtener el timestamp
            query = self.get_query_by_id(query_id)
            if query:
                updates["last_updated"] = datetime.now(timezone.utc).isoformat()
                return self.storage.update_item(query_id, query["timestamp"], updates)
            return False
            
        except Exception as e:
            st.error(f"Error actualizando consulta: {str(e)}")
            return False
    
    def move_technology_to_category(self, query_id: str, target_category_id: str) -> bool:
        """
        Mueve una tecnología a otra categoría en DynamoDB
        
        Args:
            query_id: ID de la consulta/tecnología a mover
            target_category_id: ID de la categoría destino
            
        Returns:
            bool: True si se movió exitosamente, False en caso contrario
        """
        try:
            # 1. Obtener la tecnología actual
            current_query = self.get_query_by_id(query_id)
            
            if not current_query:
                st.error(f"❌ No se encontró la tecnología con ID: {query_id}")
                return False
            
            current_category_id = current_query.get("category_id", "default")
            
            # 2. Verificar que la categoría destino existe
            target_category = self.storage.get_category_by_id(target_category_id)
            if not target_category:
                st.error(f"❌ La categoría destino no existe: {target_category_id}")
                return False
            
            # 3. Verificar que no es la misma categoría
            if current_category_id == target_category_id:
                st.warning("⚠️ La tecnología ya está en esa categoría.")
                return False
            
            # 4. Obtener nombre de la nueva categoría
            new_category_name = target_category.get("name", "Sin nombre")
            
            # 5. Actualizar en DynamoDB
            updates = {
                "category_id": target_category_id,
                "category_name": new_category_name,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            success = self.update_query(query_id, updates)
            
            if success:
                tech_name = current_query.get("technology_name") or current_query.get("name") or "Tecnología"
                st.success(f"✅ '{tech_name}' movida exitosamente a '{new_category_name}'")
                return True
            else:
                st.error("❌ Error actualizando la tecnología en DynamoDB")
                return False
                
        except Exception as e:
            st.error(f"❌ Error moviendo tecnología: {str(e)}")
            import traceback
            st.error(traceback.format_exc())
            return False

class HypeCycleHistoryInterface:
    """Interfaz para gestionar el historial de consultas de Hype Cycle con formateo seguro"""
    
    def __init__(self, hype_storage, context_prefix: str = "default"):
        self.storage = hype_storage
        self.context_prefix = context_prefix
        self.unique_id = str(int(time.time()))[-6:]
    
    def _safe_format_value(self, value, format_type="float", format_str=".2f", default="0.00"):
        """
        Formatea un valor de forma segura para evitar errores de tipo
        
        Args:
            value: Valor a formatear
            format_type: Tipo de formato ("float", "int", "percent")
            format_str: String de formato
            default: Valor por defecto
            
        Returns:
            String formateado de forma segura
        """
        try:
            if value is None:
                return default
            
            # Convertir Decimal a float
            if isinstance(value, Decimal):
                numeric_value = float(value)
            elif isinstance(value, str):
                # Limpiar strings y convertir
                clean_value = value.replace(',', '').replace('%', '').strip()
                numeric_value = float(clean_value) if clean_value else 0.0
            elif isinstance(value, (int, float)):
                numeric_value = float(value)
            else:
                return str(value)
            
            # Verificar NaN e infinito
            if math.isnan(numeric_value) or math.isinf(numeric_value):
                return default
            
            # Aplicar formato según tipo
            if format_type == "percent":
                return f"{numeric_value * 100:.1f}%"
            elif format_type == "int":
                return str(int(numeric_value))
            else:  # float
                return f"{numeric_value:{format_str}}"
                
        except (ValueError, TypeError, decimal.InvalidOperation):
            return default
    
    def show_history_interface(self):
        """Muestra la interfaz completa de historial"""
        st.header("📚 Historial de Consultas de Hype Cycle")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🔍 Explorar por Categoría", 
            "📊 Vista de Resumen", 
            "⚙️ Gestionar Consultas",
            "🔄 Mover Tecnologías"  # Nueva pestaña
        ])
        
        with tab1:
            self._show_category_explorer()
        
        with tab2:
            self._show_summary_dashboard()
        
        with tab3:
            self._show_query_manager()
        
        with tab4:
            self._show_move_technologies()  # Nueva función
    
    def _show_category_explorer(self):
        """Muestra explorador por categorías"""
        st.subheader("Explorar Consultas por Categoría")
        
        try:
            categories = self.storage.storage.get_all_categories()
        except:
            categories = [{"category_id": "default", "name": "Sin categoría"}]
        
        category_options = {cat.get("name", "Sin nombre"): cat.get("category_id") for cat in categories}
        
        selected_category_name = st.selectbox(
            "Selecciona una categoría",
            options=list(category_options.keys()),
            key=f"{self.context_prefix}_history_category_explorer_{self.unique_id}"
        )
        
        selected_category_id = category_options[selected_category_name]
        queries = self.storage.get_queries_by_category(selected_category_id)
        
        if not queries:
            st.info(f"No hay consultas guardadas en la categoría '{selected_category_name}'")
            return
        
        st.write(f"**{len(queries)} consultas encontradas en '{selected_category_name}'**")
        
        for i, query in enumerate(queries):
            self._display_query_card(query, i)
    
    def _display_query_card(self, query: Dict, index: int):
        """Muestra una tarjeta de consulta con formateo seguro"""
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
                
                data_quality = query.get('data_quality', {})
                if data_quality:
                    quality_score = data_quality.get('quality_score', 0)
                    # FORMATEO SEGURO PARA PORCENTAJE
                    quality_formatted = self._safe_format_value(quality_score, "percent")
                    st.write(f"**Calidad de datos:** {quality_formatted}")
                
            with col2:
                st.write("**Métricas del Hype Cycle:**")
                hype_metrics = query.get('hype_metrics', {})
                
                st.metric("Fase", hype_metrics.get('phase', 'Unknown'))
                
                # FORMATEO SEGURO PARA CONFIANZA
                confidence = hype_metrics.get('confidence', 0)
                confidence_formatted = self._safe_format_value(confidence, "float", ".2f")
                st.metric("Confianza", confidence_formatted)
                
                # FORMATEO SEGURO PARA MENCIONES
                mentions = hype_metrics.get('total_mentions', 0)
                mentions_formatted = self._safe_format_value(mentions, "int")
                st.metric("Total Menciones", mentions_formatted)
                
                try:
                    date = datetime.fromisoformat(query.get("execution_date", "").replace('Z', '+00:00'))
                    st.write(f"**Fecha:** {date.strftime('%Y-%m-%d %H:%M')}")
                except:
                    st.write("**Fecha:** No disponible")
            
            if st.button(f"🔄 Reutilizar Consulta", key=f"{self.context_prefix}_reuse_query_{self.unique_id}_{index}"):
                self._reuse_query(query)
    
    def _show_summary_dashboard(self):
        """Muestra dashboard de resumen con formateo seguro"""
        st.subheader("Dashboard de Resumen")
        
        all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay consultas de Hype Cycle guardadas")
            return
        
        # Métricas generales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Consultas", len(all_queries))
        
        with col2:
            phases = [q.get("hype_metrics", {}).get("phase", "Unknown") for q in all_queries]
            most_common_phase = max(set(phases), key=phases.count) if phases else "N/A"
            st.metric("Fase Más Común", most_common_phase)
        
        with col3:
            # CONFIANZA PROMEDIO CON FORMATEO SEGURO
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
        """Interfaz para gestionar consultas"""
        st.subheader("Gestionar Consultas")
        
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
            
            # FORMATEO SEGURO PARA CONFIANZA
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
        
        # Selector para gestión
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
                        st.json(selected_query)
                
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

    def _show_move_technologies(self):
        """Interfaz para mover tecnologías entre categorías - VERSIÓN CLOUD"""
        st.subheader("🔄 Mover Tecnologías Entre Categorías")
        
        st.write("""
        Mueve tecnologías entre diferentes categorías en DynamoDB.
        Los cambios se guardan inmediatamente en la nube.
        """)
        
        # Obtener todas las consultas y categorías
        all_queries = self.storage.get_all_hype_cycle_queries()
        
        if not all_queries:
            st.info("No hay tecnologías para mover.")
            return
        
        try:
            categories = self.storage.storage.get_all_categories()
        except:
            categories = [{"category_id": "default", "name": "Sin categoría"}]
        
        # Interfaz de selección
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("#### 🔬 Seleccionar Tecnología")
            
            # Filtro por categoría actual
            category_options = {"Todas": "all"}
            category_options.update({cat.get("name", "Sin nombre"): cat.get("category_id") for cat in categories})
            
            filter_category = st.selectbox(
                "Filtrar por categoría actual:",
                options=list(category_options.keys()),
                key=f"move_filter_category_{self.unique_id}"
            )
            
            # Filtrar tecnologías
            if filter_category == "Todas":
                filtered_queries = all_queries
            else:
                filter_cat_id = category_options[filter_category]
                filtered_queries = [q for q in all_queries if q.get("category_id") == filter_cat_id]
            
            if not filtered_queries:
                st.info("No hay tecnologías en la categoría seleccionada.")
                return
            
            # Selector de tecnología
            tech_options = {}
            for query in filtered_queries:
                query_id = query.get("query_id", query.get("analysis_id"))
                tech_name = (
                    query.get("technology_name") or 
                    query.get("name") or 
                    query.get("search_query", "")[:30]
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
                tech_options[display_name] = {
                    "query_id": query_id,
                    "query": query,
                    "tech_name": tech_name,
                    "current_cat_name": current_cat_name
                }
            
            selected_tech_display = st.selectbox(
                f"Tecnología a mover ({len(tech_options)} disponibles):",
                options=list(tech_options.keys()),
                key=f"move_tech_selector_{self.unique_id}"
            )
            
            selected_tech_info = tech_options[selected_tech_display]
            selected_query = selected_tech_info["query"]
            
            # Mostrar información de la tecnología
            with st.expander("ℹ️ Información de la Tecnología", expanded=True):
                st.write(f"**Nombre:** {selected_tech_info['tech_name']}")
                st.write(f"**Categoría actual:** {selected_tech_info['current_cat_name']}")
                st.write(f"**ID:** {selected_tech_info['query_id']}")
                
                # Métricas con formateo seguro
                hype_metrics = selected_query.get('hype_metrics', {})
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Fase", hype_metrics.get('phase', 'Unknown'))
                    
                    confidence_raw = hype_metrics.get('confidence', 0)
                    confidence_formatted = self._safe_format_value(confidence_raw, "float", ".2f")
                    st.metric("Confianza", confidence_formatted)
                    
                with col_b:
                    mentions_raw = hype_metrics.get('total_mentions', 0)
                    mentions_formatted = self._safe_format_value(mentions_raw, "int")
                    st.metric("Menciones", mentions_formatted)
                    
                    try:
                        exec_date = selected_query.get("execution_date", "")
                        if exec_date:
                            date_obj = datetime.fromisoformat(exec_date.replace('Z', '+00:00'))
                            formatted_date = date_obj.strftime("%Y-%m-%d")
                            st.write(f"**Fecha:** {formatted_date}")
                    except:
                        st.write("**Fecha:** No disponible")
        
        with col2:
            st.write("#### 🎯 Categoría Destino")
            
            # Excluir la categoría actual de las opciones
            current_cat_id = selected_query.get("category_id")
            available_categories = {
                cat.get("name", "Sin nombre"): cat.get("category_id") 
                for cat in categories 
                if cat.get("category_id") != current_cat_id
            }
            
            if not available_categories:
                st.warning("No hay otras categorías disponibles para mover la tecnología.")
                return
            
            target_category_name = st.selectbox(
                "Mover a categoría:",
                options=list(available_categories.keys()),
                key=f"move_target_category_{self.unique_id}"
            )
            
            target_category_id = available_categories[target_category_name]
            
            # Mostrar información de la categoría destino
            try:
                target_queries = self.storage.get_queries_by_category(target_category_id)
                
                with st.expander("ℹ️ Información de Categoría Destino", expanded=True):
                    st.write(f"**Nombre:** {target_category_name}")
                    st.write(f"**Tecnologías actuales:** {len(target_queries)}")
                    
                    if target_queries:
                        # Distribución de fases en categoría destino
                        phase_dist = {}
                        for q in target_queries:
                            phase = q.get("hype_metrics", {}).get("phase", "Unknown")
                            phase_dist[phase] = phase_dist.get(phase, 0) + 1
                        
                        st.write("**Distribución por fases:**")
                        for phase, count in phase_dist.items():
                            st.write(f"• {phase}: {count}")
                    else:
                        st.write("• No hay tecnologías en esta categoría")
            except Exception as e:
                st.warning(f"Error obteniendo info de categoría destino: {str(e)}")
            
            # Zona de acción
            st.write("---")
            st.info(f"**Movimiento:** '{selected_tech_info['tech_name']}' → '{target_category_name}'")
            
            # Estados únicos para confirmación
            confirm_key = f"confirm_move_{selected_tech_info['query_id']}_{target_category_id}_{self.unique_id}"
            
            confirm_move = st.checkbox(
                f"Confirmar movimiento de tecnología",
                key=confirm_key
            )
            
            # Botones de acción
            col_a, col_b = st.columns(2)
            
            with col_a:
                move_button_key = f"execute_move_{selected_tech_info['query_id']}_{target_category_id}_{self.unique_id}"
                
                if confirm_move and st.button(
                    "🔄 MOVER TECNOLOGÍA", 
                    type="primary",
                    key=move_button_key
                ):
                    with st.spinner(f"Moviendo '{selected_tech_info['tech_name']}' a DynamoDB..."):
                        success = self.storage.move_technology_to_category(
                            selected_tech_info['query_id'], 
                            target_category_id
                        )
                        
                        if success:
                            # Limpiar estados para forzar refresh
                            st.session_state[f"move_filter_category_{self.unique_id}"] = "Todas"
                            
                            # Limpiar cualquier caché
                            for key in list(st.session_state.keys()):
                                if key.startswith('chart_cache_') or key.startswith('admin_state_'):
                                    del st.session_state[key]
                            
                            time.sleep(1)
                            st.rerun()
            
            with col_b:
                preview_button_key = f"preview_move_{selected_tech_info['query_id']}_{target_category_id}_{self.unique_id}"
                
                if st.button(
                    "👀 Preview", 
                    key=preview_button_key
                ):
                    st.write("### 📊 Preview del Movimiento")
                    
                    try:
                        current_count = len(target_queries) if 'target_queries' in locals() else 0
                        new_count = current_count + 1
                        
                        col_prev1, col_prev2 = st.columns(2)
                        
                        with col_prev1:
                            st.write("**Estado Actual:**")
                            st.metric("Tecnologías", current_count)
                        
                        with col_prev2:
                            st.write("**Después del Movimiento:**")
                            st.metric("Tecnologías", new_count, delta=1)
                            
                    except Exception as e:
                        st.warning(f"Error generando preview: {str(e)}")

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

# Funciones para integración
def initialize_hype_cycle_storage(db_storage):
    """Inicializa el sistema de almacenamiento de Hype Cycle"""
    return HypeCycleStorage(db_storage)

def create_hype_cycle_interface(hype_storage, context_prefix: str = "default"):
    """Crea la interfaz completa de historial de Hype Cycle"""
    return HypeCycleHistoryInterface(hype_storage, context_prefix)