# src/utils/api_helpers.py
import streamlit as st
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def test_api_connection(api_type, api_key, secondary_key=None):
    """
    Prueba la conexión con diferentes APIs
    
    Args:
        api_type: Tipo de API a probar ('google', 'serp', 'scopus')
        api_key: Clave principal
        secondary_key: Clave secundaria (como search_engine_id)
    
    Returns:
        Tuple: (éxito, mensaje)
    """
    if api_type == "google":
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            result = service.cse().list(
                q="test",
                cx=secondary_key,
                num=1
            ).execute()
            return True, "Conexión exitosa con Google API"
        except HttpError as e:
            if e.resp.status == 403:
                return False, f"Error de autenticación (403): {str(e)}"
            else:
                return False, f"Error en Google API: {str(e)}"
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"
            
    elif api_type == "serp":
        try:
            params = {
                "api_key": api_key,
                "q": "test",
                "tbm": "nws",
                "num": 1
            }
            response = requests.get("https://serpapi.com/search", params=params)
            
            if response.status_code == 200:
                return True, "Conexión exitosa con SerpAPI"
            else:
                error_message = response.json().get('error', 'Error desconocido')
                return False, f"Error en SerpAPI ({response.status_code}): {error_message}"
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"
            
    elif api_type == "scopus":
        try:
            headers = {
                "X-ELS-APIKey": api_key,
                "Accept": "application/json"
            }
            response = requests.get(
                "https://api.elsevier.com/content/search/scopus",
                headers=headers,
                params={"query": "TITLE-ABS-KEY(test)", "count": 1}
            )
            
            if response.status_code == 200:
                return True, "Conexión exitosa con Scopus API"
            else:
                return False, f"Error en Scopus API ({response.status_code}): {response.text}"
        except Exception as e:
            return False, f"Error inesperado: {str(e)}"
    
    return False, "Tipo de API no reconocido"

def load_config_from_file(uploaded_file):
    """
    Carga la configuración desde un archivo subido
    
    Args:
        uploaded_file: Archivo JSON subido a través de st.file_uploader
        
    Returns:
        dict o None: Diccionario con la configuración o None si hay error
    """
    try:
        import json
        
        # Leer el contenido del archivo
        content = uploaded_file.read()
        
        # Decodificar el contenido
        if uploaded_file.type == "application/json":
            config_data = json.loads(content)
        else:
            st.error("Por favor, sube un archivo JSON válido")
            return None
            
        # Validar la estructura del archivo
        required_keys = ['GOOGLE_API_KEY', 'SEARCH_ENGINE_ID', 'SERP_API_KEY', 'SCOPUS_API_KEY']
        for key in required_keys:
            if key not in config_data:
                st.warning(f"Advertencia: El archivo no contiene la clave '{key}'")
        
        return config_data
        
    except json.JSONDecodeError:
        st.error("El archivo no es un JSON válido")
        return None
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None