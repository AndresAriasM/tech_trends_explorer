# src/hype_ai_analyzer.py - ANALIZADOR IA INDEPENDIENTE PARA HYPE CYCLE
import json
import time
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from decimal import Decimal
import math

# Dotenv para cargar variables de entorno
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    logging.warning("python-dotenv no disponible. Instala con: pip install python-dotenv")

# OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Configurar logging
logger = logging.getLogger(__name__)

class HypeAIAnalyzer:
    """Analizador IA independiente para generar insights inteligentes del Hype Cycle"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4"):
        """
        Inicializa el analizador IA
        
        Args:
            api_key: API key de OpenAI (opcional, se carga desde .env si no se proporciona)
            model: Modelo a usar (gpt-4, gpt-3.5-turbo, etc.)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("Se requiere instalar openai: pip install openai")
        
        # Cargar API key desde .env si no se proporciona
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
            
        if not api_key:
            raise ValueError(
                "No se encontró API key de OpenAI. "
                "Proporciona api_key como parámetro o configura OPENAI_API_KEY en tu archivo .env"
            )
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.api_key_source = "environment" if api_key == os.getenv("OPENAI_API_KEY") else "parameter"
        
        # Configuración de costos por modelo (por 1K tokens)
        self.pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
        }
    
    def analyze_category_hype(self, 
                            queries: List[Dict], 
                            category_name: str, 
                            analysis_depth: str = "Detallado",
                            custom_focus: str = None) -> Dict:
        """
        Función principal: Analiza una categoría completa del Hype Cycle
        
        Args:
            queries: Lista de consultas/tecnologías de la categoría
            category_name: Nombre de la categoría
            analysis_depth: Nivel de análisis (Ejecutivo, Detallado, Técnico)
            custom_focus: Enfoque personalizado (opcional)
        
        Returns:
            Dict con el análisis completo y metadatos
        """
        try:
            logger.info(f"Iniciando análisis IA para categoría: {category_name}")
            
            # Validar datos de entrada
            if not queries:
                return {
                    "success": False,
                    "error": "No hay datos para analizar",
                    "analysis": None
                }
            
            # Procesar datos para el análisis
            processed_data = self._process_category_data(queries, category_name)
            
            # Construir prompt específico
            system_prompt = self._get_system_prompt(analysis_depth)
            user_prompt = self._build_analysis_prompt(processed_data, category_name, analysis_depth, custom_focus)
            
            # Realizar llamada a OpenAI
            start_time = time.time()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,  # Consistente pero no robótico
                max_tokens=1500,  # Suficiente para análisis detallado
                top_p=0.9
            )
            
            processing_time = time.time() - start_time
            
            # Procesar respuesta
            analysis_text = response.choices[0].message.content
            usage = response.usage
            
            # Calcular costos
            cost_breakdown = self._calculate_detailed_cost(usage)
            
            # Formato final del resultado
            result = {
                "success": True,
                "analysis": analysis_text,
                "metadata": {
                    "category_name": category_name,
                    "technologies_analyzed": len(queries),
                    "analysis_depth": analysis_depth,
                    "model_used": self.model,
                    "processing_time": round(processing_time, 2),
                    "timestamp": datetime.now().isoformat()
                },
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens
                },
                "cost": cost_breakdown,
                "data_summary": processed_data["summary"]
            }
            
            logger.info(f"Análisis completado - Tokens: {usage.total_tokens}, Costo: ${cost_breakdown['total']:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error en análisis IA: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "analysis": None,
                "metadata": {"category_name": category_name, "error_time": datetime.now().isoformat()}
            }
    
    def _process_category_data(self, queries: List[Dict], category_name: str) -> Dict:
        """Procesa y estructura los datos de la categoría para el análisis"""
        
        # Estadísticas básicas
        total_technologies = len(queries)
        
        # Distribución de fases
        phases = []
        confidences = []
        mentions = []
        sentiments = []
        execution_dates = []
        
        technologies_by_phase = {}
        
        for query in queries:
            hype_metrics = query.get("hype_metrics", {})
            
            # Extraer datos de forma segura
            phase = hype_metrics.get("phase", "Unknown")
            confidence = self._safe_float(hype_metrics.get("confidence", 0))
            total_mentions = self._safe_int(hype_metrics.get("total_mentions", 0))
            sentiment = self._safe_float(hype_metrics.get("sentiment_avg", 0))
            
            phases.append(phase)
            confidences.append(confidence)
            mentions.append(total_mentions)
            sentiments.append(sentiment)
            
            # Agrupar por fase
            if phase not in technologies_by_phase:
                technologies_by_phase[phase] = []
            
            tech_info = {
                "name": query.get("technology_name") or query.get("search_query", "")[:30],
                "confidence": confidence,
                "mentions": total_mentions,
                "sentiment": sentiment,
                "time_to_plateau": hype_metrics.get("time_to_plateau", "N/A"),
                "execution_date": query.get("execution_date", ""),
                "search_query": query.get("search_query", "")[:50]  # Para contexto adicional
            }
            
            technologies_by_phase[phase].append(tech_info)
            
            # Fechas para análisis temporal
            exec_date = query.get("execution_date", "")
            if exec_date:
                execution_dates.append(exec_date)
        
        # Calcular estadísticas agregadas
        phase_distribution = {phase: len(techs) for phase, techs in technologies_by_phase.items()}
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        avg_mentions = sum(mentions) / len(mentions) if mentions else 0
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Identificar tendencias
        most_common_phase = max(phase_distribution.items(), key=lambda x: x[1])[0] if phase_distribution else "Unknown"
        
        # Análisis temporal básico
        recent_analyses = len([date for date in execution_dates if date.startswith(datetime.now().strftime("%Y-%m"))])
        
        # Datos procesados para el prompt
        processed_data = {
            "category_name": category_name,
            "total_technologies": total_technologies,
            "phase_distribution": phase_distribution,
            "technologies_by_phase": technologies_by_phase,
            "statistics": {
                "avg_confidence": round(avg_confidence, 3),
                "avg_mentions": round(avg_mentions, 1),
                "avg_sentiment": round(avg_sentiment, 3),
                "most_common_phase": most_common_phase,
                "recent_analyses": recent_analyses
            },
            "summary": {
                "total_technologies": total_technologies,
                "phases_represented": len(phase_distribution),
                "dominant_phase": most_common_phase,
                "avg_confidence": round(avg_confidence, 2),
                "analysis_freshness": f"{recent_analyses} análisis este mes"
            }
        }
        
        return processed_data
    
    def _get_system_prompt(self, analysis_depth: str) -> str:
        """Genera el prompt del sistema basado en el nivel de análisis"""
        
        base_system = """Eres un experto analista de tecnología especializado en el Hype Cycle de Gartner. 

Tu experiencia incluye:
- 15+ años analizando tecnologías emergentes
- Profundo conocimiento del modelo Hype Cycle
- Experiencia en consultoría estratégica para Fortune 500
- Capacidad de traducir datos técnicos en insights de negocio

Tu tarea es analizar datos del Hype Cycle y generar insights valiosos para tomadores de decisiones."""
        
        depth_instructions = {
            "Ejecutivo": """
AUDIENCIA: C-Level executives y tomadores de decisiones estratégicas
ESTILO: Conciso, enfocado en implicaciones de negocio y recomendaciones accionables
LONGITUD: 3-4 párrafos, máximo 400 palabras
ENFOQUE: Qué significa para la estrategia de la empresa, timing de inversiones, riesgos y oportunidades
""",
            "Detallado": """
AUDIENCIA: Gerentes de producto, directores de tecnología, analistas de mercado
ESTILO: Equilibrio entre insights técnicos y de negocio, con análisis contextual profundo
LONGITUD: 4-6 párrafos, 500-700 palabras
ENFOQUE: Análisis de patrones sectoriales, explicación contextual de posiciones, interrelaciones entre tecnologías

INSTRUCCIONES ESPECÍFICAS:
- Identifica WHY ciertas tecnologías están agrupadas en fases específicas
- Explica las características comunes dentro de cada fase
- Analiza factores técnicos, de mercado y de adopción que determinan las posiciones
- Evalúa la coherencia lógica de la distribución observada
- Proporciona insights sobre las dinámicas internas del sector
""",
            "Técnico": """
AUDIENCIA: CTOs, arquitectos de soluciones, investigadores técnicos
ESTILO: Detallado técnicamente, análisis profundo de datos, metodología clara
LONGITUD: 6-8 párrafos, 700-900 palabras
ENFOQUE: Análisis metodológico, confiabilidad de datos, predicciones técnicas, factores de adopción
"""
        }
        
        return base_system + "\n\n" + depth_instructions.get(analysis_depth, depth_instructions["Detallado"])
    
    def _build_analysis_prompt(self, processed_data: Dict, category_name: str, 
                             analysis_depth: str, custom_focus: str = None) -> str:
        """Construye el prompt específico del usuario para el análisis"""
        
        # Datos principales
        data = processed_data
        stats = data["statistics"]
        
        # Construir información detallada de tecnologías por fase
        phase_details = ""
        for phase, technologies in data["technologies_by_phase"].items():
            if technologies:
                tech_list = []
                for tech in technologies[:5]:  # Limitar a 5 por fase para no sobrecargar
                    confidence_pct = f"{tech['confidence']*100:.0f}%"
                    sentiment_desc = "positivo" if tech['sentiment'] > 0.1 else "negativo" if tech['sentiment'] < -0.1 else "neutral"
                    tech_list.append(f"  • {tech['name']} (confianza: {confidence_pct}, menciones: {tech['mentions']}, sentimiento: {sentiment_desc})")
                
                tech_details = "\n".join(tech_list)
                if len(technologies) > 5:
                    tech_details += f"\n  • ... y {len(technologies)-5} tecnologías más"
                
                phase_details += f"\n{phase} ({len(technologies)} tecnologías):\n{tech_details}\n"
        
        # Construir el prompt principal con enfoque mejorado
        prompt = f"""
ANÁLISIS REQUERIDO PARA: {category_name}

DATOS DE LA CATEGORÍA:
Total de tecnologías analizadas: {data['total_technologies']}
Distribución por fases del Hype Cycle: {json.dumps(data['phase_distribution'], indent=2)}
Fase dominante: {stats['most_common_phase']}

ESTADÍSTICAS AGREGADAS:
- Confianza promedio: {stats['avg_confidence']} (escala 0-1)
- Menciones promedio: {stats['avg_mentions']}
- Sentimiento promedio: {stats['avg_sentiment']} (escala -1 a +1)
- Análisis recientes: {stats['recent_analyses']} este mes

TECNOLOGÍAS POR FASE:
{phase_details}

TAREAS ESPECÍFICAS DE ANÁLISIS:
1. **Contexto Sectorial**: Analiza qué caracteriza al sector {category_name} y cómo se refleja en la distribución del Hype Cycle
2. **Patrones por Fase**: Explica qué TIPOS de tecnologías dominan cada fase y POR QUÉ están ahí (factores técnicos, de mercado, regulatorios)
3. **Características Comunes**: Identifica qué comparten en común las tecnologías dentro de cada fase - ¿hay patrones de madurez, complejidad, o adopción?
4. **Coherencia del Resultado**: Evalúa si la distribución tiene sentido dado el contexto actual del sector {category_name}
5. **Factores Diferenciales**: Explica qué hace que unas tecnologías avancen más rápido que otras en el ciclo
6. **Implicaciones Estratégicas**: Proporciona insights específicos sobre timing de inversión y adopción

ENFOQUE ANALÍTICO PROFUNDO:
- Conecta los datos cuantitativos (confianza, menciones, sentimiento) con el contexto cualitativo de cada tecnología
- Explica las RAZONES SUBYACENTES de por qué ciertas tecnologías están agrupadas en fases específicas
- Identifica si hay tecnologías "fuera de lugar" o sorpresas en la distribución
- Analiza la coherencia interna del sector: ¿las tecnologías maduras habilitan a las emergentes?
"""
        
        # Agregar enfoque personalizado si se proporciona
        if custom_focus:
            prompt += f"\n\nENFOQUE ESPECÍFICO SOLICITADO:\n{custom_focus}"
        
        # Agregar instrucciones específicas por profundidad
        depth_instructions = {
            "Ejecutivo": "\n\nENFOQUE EJECUTIVO: Prioriza implicaciones estratégicas, timing de mercado, y recomendaciones de inversión.",
            "Detallado": "\n\nENFOQUE DETALLADO: Incluye análisis de datos, comparaciones entre tecnologías, y explicación de tendencias.",
            "Técnico": "\n\nENFOQUE TÉCNICO: Profundiza en metodología, confiabilidad de datos, factores técnicos de adopción, y predicciones fundamentadas."
        }
        
        prompt += depth_instructions.get(analysis_depth, "")
        
        prompt += "\n\nGenera tu análisis basándote estrictamente en los datos proporcionados. Si detectas limitaciones en los datos, menciónalo apropiadamente."
        
        return prompt
    
    def _calculate_detailed_cost(self, usage) -> Dict:
        """Calcula el costo detallado del análisis"""
        
        model_pricing = self.pricing.get(self.model, self.pricing["gpt-4"])
        
        input_cost = (usage.prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (usage.completion_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_tokens": usage.prompt_tokens,
            "output_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
            "input_cost": round(input_cost, 4),
            "output_cost": round(output_cost, 4),
            "total": round(total_cost, 4),
            "model": self.model
        }
    
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
    
    def generate_comparative_analysis(self, categories_data: Dict[str, List[Dict]]) -> Dict:
        """
        FUNCIÓN BONUS: Genera análisis comparativo entre múltiples categorías
        
        Args:
            categories_data: Dict con {category_name: [queries]}
        
        Returns:
            Análisis comparativo entre categorías
        """
        try:
            # Procesar cada categoría
            processed_categories = {}
            
            for category_name, queries in categories_data.items():
                if queries:
                    processed_categories[category_name] = self._process_category_data(queries, category_name)
            
            if len(processed_categories) < 2:
                return {
                    "success": False,
                    "error": "Se necesitan al menos 2 categorías para comparación"
                }
            
            # Construir prompt comparativo
            comparison_prompt = self._build_comparison_prompt(processed_categories)
            system_prompt = self._get_comparison_system_prompt()
            
            # Ejecutar análisis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": comparison_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            usage = response.usage
            cost_breakdown = self._calculate_detailed_cost(usage)
            
            return {
                "success": True,
                "analysis": response.choices[0].message.content,
                "categories_compared": list(processed_categories.keys()),
                "usage": {
                    "total_tokens": usage.total_tokens,
                    "cost": cost_breakdown["total"]
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_comparison_prompt(self, processed_categories: Dict) -> str:
        """Construye prompt para análisis comparativo"""
        
        prompt = "ANÁLISIS COMPARATIVO ENTRE CATEGORÍAS:\n\n"
        
        for category_name, data in processed_categories.items():
            stats = data["statistics"]
            prompt += f"CATEGORÍA: {category_name}\n"
            prompt += f"- Tecnologías: {data['total_technologies']}\n"
            prompt += f"- Fase dominante: {stats['most_common_phase']}\n"
            prompt += f"- Confianza promedio: {stats['avg_confidence']}\n"
            prompt += f"- Distribución: {data['phase_distribution']}\n\n"
        
        prompt += """
ANÁLISIS REQUERIDO:
1. Compara el nivel de madurez entre categorías
2. Identifica qué categorías están más avanzadas/atrasadas
3. Analiza patrones comunes y diferencias significativas
4. Proporciona insights estratégicos sobre el panorama tecnológico general
5. Recomienda enfoques diferenciados por categoría
"""
        
        return prompt
    
    def _get_comparison_system_prompt(self) -> str:
        """Sistema prompt para análisis comparativo"""
        return """Eres un analista senior de tecnología especializado en análisis comparativo del Hype Cycle entre diferentes sectores.

Tu expertise incluye análisis cross-industry, identificación de patrones macro-tecnológicos, y desarrollo de estrategias de innovación multi-sector.

Genera un análisis comparativo profundo que ayude a entender el panorama tecnológico general y las oportunidades estratégicas entre sectores."""


# Funciones auxiliares para integración fácil

def get_openai_key_from_env() -> Optional[str]:
    """Obtiene la API key de OpenAI desde variables de entorno"""
    return os.getenv("OPENAI_API_KEY")

def validate_openai_key(api_key: str = None) -> Dict:
    """Valida que la API key de OpenAI funcione"""
    try:
        # Usar API key del .env si no se proporciona
        if api_key is None:
            api_key = get_openai_key_from_env()
            
        if not api_key:
            return {
                "valid": False,
                "error": "No API key provided",
                "message": "API key no encontrada en .env ni proporcionada como parámetro"
            }
        
        client = OpenAI(api_key=api_key)
        
        # Test simple
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=5
        )
        
        return {
            "valid": True,
            "test_tokens": response.usage.total_tokens,
            "message": "API key válida",
            "source": "environment" if api_key == os.getenv("OPENAI_API_KEY") else "parameter"
        }
        
    except Exception as e:
        return {
            "valid": False,
            "error": str(e),
            "message": "API key inválida o error de conexión"
        }

def check_env_setup() -> Dict:
    """Verifica el setup completo del entorno"""
    status = {
        "dotenv_available": DOTENV_AVAILABLE,
        "openai_available": OPENAI_AVAILABLE,
        "api_key_in_env": bool(get_openai_key_from_env()),
        "api_key_valid": False,
        "ready": False
    }
    
    # Verificar API key si está disponible
    if status["api_key_in_env"]:
        validation = validate_openai_key()
        status["api_key_valid"] = validation["valid"]
    
    # Determinar si está listo para usar
    status["ready"] = (
        status["openai_available"] and 
        status["api_key_in_env"] and 
        status["api_key_valid"]
    )
    
    return status

def estimate_analysis_cost(num_technologies: int, analysis_depth: str = "Detallado") -> Dict:
    """Estima el costo de un análisis antes de ejecutarlo"""
    
    # Estimaciones basadas en experiencia
    base_tokens = {
        "Ejecutivo": 800,
        "Detallado": 1200,
        "Técnico": 1600
    }
    
    tokens_per_tech = 50  # Aproximadamente
    
    estimated_tokens = base_tokens.get(analysis_depth, 1200) + (num_technologies * tokens_per_tech)
    
    # Costo estimado con GPT-4
    estimated_cost = (estimated_tokens / 1000) * 0.04  # Promedio input/output
    
    return {
        "estimated_tokens": estimated_tokens,
        "estimated_cost": round(estimated_cost, 4),
        "technologies": num_technologies,
        "depth": analysis_depth
    }