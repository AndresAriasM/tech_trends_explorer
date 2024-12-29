# utils/api_test.py

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def test_api_connection(api_key, search_engine_id):
    try:
        print(f"Probando con API Key: {api_key[:10]}... y Search Engine ID: {search_engine_id}")
        # Crear el servicio
        service = build("customsearch", "v1", developerKey=api_key)
        print("Servicio creado exitosamente")
        
        # Hacer una búsqueda de prueba
        result = service.cse().list(
            q="test",
            cx=search_engine_id,
            num=1
        ).execute()
        print("Búsqueda ejecutada exitosamente")
        print(f"Resultados obtenidos: {result.get('searchInformation', {}).get('totalResults', 'N/A')}")
        
        return True, "Conexión exitosa"
    except HttpError as e:
        error_details = str(e)
        print(f"Error HTTP: {error_details}")
        
        if e.resp.status == 403:
            return False, f"Error de autenticación (403): {error_details}"
        elif e.resp.status == 400:
            return False, f"Error en la configuración (400): {error_details}"
        else:
            return False, f"Error HTTP {e.resp.status}: {error_details}"
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"