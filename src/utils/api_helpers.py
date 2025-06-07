# utils/api_helpers.py
import json
import requests
import streamlit as st
from typing import Tuple, Dict, Any
import os

def test_api_connection(api_type: str, *args) -> Tuple[bool, str]:
    """
    Prueba la conexión con diferentes APIs
    
    Args:
        api_type: Tipo de API ('google', 'serp', 'scopus', 'aws')
        *args: Argumentos específicos para cada API
        
    Returns:
        Tuple[bool, str]: (éxito, mensaje)
    """
    try:
        if api_type == "google":
            return _test_google_api(args[0], args[1])
        elif api_type == "serp":
            return _test_serp_api(args[0])
        elif api_type == "scopus":
            return _test_scopus_api(args[0])
        elif api_type == "aws":
            return _test_aws_connection(args[0], args[1], args[2])
        else:
            return False, f"Tipo de API no soportado: {api_type}"
            
    except Exception as e:
        return False, f"Error probando {api_type}: {str(e)}"

def _test_google_api(api_key: str, search_engine_id: str) -> Tuple[bool, str]:
    """Prueba la conexión con Google Custom Search API"""
    if not api_key or not search_engine_id:
        return False, "API Key y Search Engine ID son requeridos"
    
    try:
        from googleapiclient.discovery import build
        
        service = build("customsearch", "v1", developerKey=api_key)
        result = service.cse().list(
            q="test",
            cx=search_engine_id,
            num=1
        ).execute()
        
        if 'items' in result:
            return True, f"✅ Conexión exitosa - {len(result['items'])} resultado(s) de prueba"
        else:
            return True, "✅ Conexión exitosa - Sin resultados para query de prueba"
            
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower():
            return False, "❌ Cuota de API agotada - Verifica tu límite diario"
        elif "invalid" in error_msg.lower():
            return False, "❌ API Key o Search Engine ID inválidos"
        else:
            return False, f"❌ Error: {error_msg}"

def _test_serp_api(api_key: str) -> Tuple[bool, str]:
    """Prueba la conexión con SerpAPI"""
    if not api_key:
        return False, "SerpAPI Key es requerida"
    
    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={
                "q": "test",
                "api_key": api_key,
                "num": 1
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if "error" in data:
                return False, f"❌ Error de SerpAPI: {data['error']}"
            else:
                return True, "✅ Conexión exitosa con SerpAPI"
        elif response.status_code == 401:
            return False, "❌ API Key de SerpAPI inválida"
        elif response.status_code == 429:
            return False, "❌ Límite de rate excedido en SerpAPI"
        else:
            return False, f"❌ Error HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "❌ Timeout - SerpAPI no responde"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def _test_scopus_api(api_key: str) -> Tuple[bool, str]:
    """Prueba la conexión con Scopus API"""
    if not api_key:
        return False, "Scopus API Key es requerida"
    
    try:
        headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json"
        }
        
        response = requests.get(
            "https://api.elsevier.com/content/search/scopus",
            headers=headers,
            params={
                "query": "TITLE(test)",
                "count": 1
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            total_results = int(data.get("search-results", {}).get("opensearch:totalResults", 0))
            return True, f"✅ Conexión exitosa - {total_results} resultados disponibles"
        elif response.status_code == 401:
            return False, "❌ API Key de Scopus inválida"
        elif response.status_code == 429:
            return False, "❌ Límite de rate excedido en Scopus"
        else:
            return False, f"❌ Error HTTP {response.status_code}"
            
    except requests.exceptions.Timeout:
        return False, "❌ Timeout - Scopus API no responde"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def _test_aws_connection(access_key: str, secret_key: str, region: str) -> Tuple[bool, str]:
    """Prueba la conexión con AWS DynamoDB"""
    if not access_key or not secret_key:
        return False, "Access Key y Secret Key de AWS son requeridos"
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        
        # Crear cliente DynamoDB
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        
        # Listar tablas para verificar conexión
        client = boto3.client(
            'dynamodb',
            region_name=region,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key
        )
        
        response = client.list_tables()
        table_count = len(response.get('TableNames', []))
        
        return True, f"✅ Conexión exitosa - {table_count} tabla(s) encontrada(s) en {region}"
        
    except NoCredentialsError:
        return False, "❌ Credenciales de AWS inválidas"
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'UnrecognizedClientException':
            return False, "❌ Credenciales de AWS inválidas"
        else:
            return False, f"❌ Error de AWS: {error_code}"
    except Exception as e:
        return False, f"❌ Error: {str(e)}"

def load_config_from_file(uploaded_file) -> Dict[str, Any]:
    """
    Carga configuración desde un archivo JSON subido
    
    Args:
        uploaded_file: Archivo subido via st.file_uploader
        
    Returns:
        Dict con la configuración cargada
    """
    try:
        content = uploaded_file.read()
        config_data = json.loads(content)
        
        # Validar que el archivo tiene la estructura esperada
        expected_keys = [
            'GOOGLE_API_KEY', 'SEARCH_ENGINE_ID', 'SERP_API_KEY', 
            'SCOPUS_API_KEY', 'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY'
        ]
        
        found_keys = [key for key in expected_keys if key in config_data]
        
        if found_keys:
            st.success(f"✅ Configuración cargada - {len(found_keys)} clave(s) encontrada(s)")
            return config_data
        else:
            st.error("❌ El archivo no contiene las claves de configuración esperadas")
            return {}
            
    except json.JSONDecodeError:
        st.error("❌ Error: El archivo no es un JSON válido")
        return {}
    except Exception as e:
        st.error(f"❌ Error cargando archivo: {str(e)}")
        return {}

def create_config_template() -> str:
    """
    Crea un template de configuración JSON
    
    Returns:
        String con el template JSON
    """
    template = {
        "GOOGLE_API_KEY": "tu_google_api_key_aqui",
        "SEARCH_ENGINE_ID": "tu_search_engine_id_aqui", 
        "SERP_API_KEY": "tu_serp_api_key_aqui",
        "SCOPUS_API_KEY": "tu_scopus_api_key_aqui",
        "AWS_ACCESS_KEY_ID": "tu_aws_access_key_aqui",
        "AWS_SECRET_ACCESS_KEY": "tu_aws_secret_key_aqui",
        "AWS_DEFAULT_REGION": "us-east-2"
    }
    
    return json.dumps(template, indent=2)

def estimate_api_costs(api_type: str, num_requests: int) -> Dict[str, Any]:
    """
    Estima los costos de uso de APIs
    
    Args:
        api_type: Tipo de API
        num_requests: Número de requests estimados
        
    Returns:
        Dict con información de costos
    """
    cost_info = {
        "google": {
            "free_quota": 100,  # requests por día
            "cost_per_request": 0.005,  # USD por request adicional
            "currency": "USD"
        },
        "serp": {
            "free_quota": 100,  # requests por mes
            "cost_per_request": 0.025,  # USD por request adicional
            "currency": "USD"
        },
        "scopus": {
            "free_quota": 0,  # Sin cuota gratuita
            "cost_per_request": 0.0,  # Basado en suscripción institucional
            "currency": "USD",
            "note": "Requiere suscripción institucional"
        }
    }
    
    if api_type not in cost_info:
        return {"error": "API type not supported"}
    
    info = cost_info[api_type]
    
    if num_requests <= info["free_quota"]:
        estimated_cost = 0
        additional_requests = 0
    else:
        additional_requests = num_requests - info["free_quota"]
        estimated_cost = additional_requests * info["cost_per_request"]
    
    return {
        "api_type": api_type,
        "total_requests": num_requests,
        "free_requests": min(num_requests, info["free_quota"]),
        "additional_requests": additional_requests,
        "estimated_cost": estimated_cost,
        "currency": info["currency"],
        "note": info.get("note", "")
    }

def validate_api_response(api_type: str, response_data: Dict) -> Tuple[bool, str]:
    """
    Valida la respuesta de una API para detectar errores comunes
    
    Args:
        api_type: Tipo de API
        response_data: Datos de respuesta de la API
        
    Returns:
        Tuple[bool, str]: (es_válida, mensaje)
    """
    try:
        if api_type == "google":
            if "error" in response_data:
                error = response_data["error"]
                return False, f"Error de Google API: {error.get('message', 'Error desconocido')}"
            elif "items" not in response_data:
                return False, "No se encontraron resultados"
            else:
                return True, f"Respuesta válida - {len(response_data['items'])} resultados"
        
        elif api_type == "serp":
            if "error" in response_data:
                return False, f"Error de SerpAPI: {response_data['error']}"
            elif "organic_results" not in response_data and "news_results" not in response_data:
                return False, "No se encontraron resultados orgánicos o de noticias"
            else:
                result_count = len(response_data.get("organic_results", [])) + len(response_data.get("news_results", []))
                return True, f"Respuesta válida - {result_count} resultados"
        
        elif api_type == "scopus":
            if "service-error" in response_data:
                error = response_data["service-error"]
                return False, f"Error de Scopus: {error.get('status', {}).get('statusText', 'Error desconocido')}"
            elif "search-results" not in response_data:
                return False, "Formato de respuesta de Scopus inválido"
            else:
                total_results = response_data["search-results"].get("opensearch:totalResults", 0)
                return True, f"Respuesta válida - {total_results} resultados encontrados"
        
        else:
            return False, f"Tipo de API no soportado: {api_type}"
            
    except Exception as e:
        return False, f"Error validando respuesta: {str(e)}"

def get_rate_limit_info(api_type: str) -> Dict[str, Any]:
    """
    Obtiene información sobre límites de rate para cada API
    
    Args:
        api_type: Tipo de API
        
    Returns:
        Dict con información de rate limits
    """
    rate_limits = {
        "google": {
            "requests_per_day": 100,  # Gratis
            "requests_per_second": 10,
            "paid_requests_per_day": 10000,
            "recommended_delay": 0.1  # segundos entre requests
        },
        "serp": {
            "requests_per_month": 100,  # Plan gratuito
            "requests_per_second": 5,
            "paid_requests_per_month": 5000,  # Plan básico
            "recommended_delay": 0.2
        },
        "scopus": {
            "requests_per_week": 5000,  # Típico para instituciones
            "requests_per_second": 2,
            "recommended_delay": 0.5
        }
    }
    
    return rate_limits.get(api_type, {"error": "API type not supported"})

def save_api_usage_log(api_type: str, query: str, response_size: int, success: bool):
    """
    Guarda un log de uso de API para tracking
    
    Args:
        api_type: Tipo de API utilizada
        query: Query realizada
        response_size: Tamaño de respuesta (número de resultados)
        success: Si la request fue exitosa
    """
    try:
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "api_type": api_type,
            "query": query[:100],  # Truncar query larga
            "response_size": response_size,
            "success": success
        }
        
        # Crear directorio de logs si no existe
        os.makedirs("./logs", exist_ok=True)
        
        # Cargar log existente o crear nuevo
        log_file = f"./logs/api_usage_{api_type}.json"
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        # Mantener solo los últimos 1000 logs
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Guardar log actualizado
        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)
            
    except Exception as e:
        st.warning(f"No se pudo guardar log de API: {str(e)}")

def get_api_usage_stats(api_type: str) -> Dict[str, Any]:
    """
    Obtiene estadísticas de uso de una API
    
    Args:
        api_type: Tipo de API
        
    Returns:
        Dict con estadísticas de uso
    """
    try:
        log_file = f"./logs/api_usage_{api_type}.json"
        
        if not os.path.exists(log_file):
            return {"total_requests": 0, "successful_requests": 0, "error_rate": 0}
        
        with open(log_file, 'r') as f:
            logs = json.load(f)
        
        if not logs:
            return {"total_requests": 0, "successful_requests": 0, "error_rate": 0}
        
        total_requests = len(logs)
        successful_requests = sum(1 for log in logs if log.get("success", False))
        error_rate = (total_requests - successful_requests) / total_requests if total_requests > 0 else 0
        
        # Estadísticas por día
        today = pd.Timestamp.now().date()
        today_requests = sum(1 for log in logs if pd.Timestamp(log["timestamp"]).date() == today)
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "error_rate": error_rate,
            "today_requests": today_requests,
            "avg_response_size": sum(log.get("response_size", 0) for log in logs) / total_requests if total_requests > 0 else 0
        }
        
    except Exception as e:
        st.warning(f"Error obteniendo estadísticas de API: {str(e)}")
        return {"error": str(e)}