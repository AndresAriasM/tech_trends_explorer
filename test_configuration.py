#!/usr/bin/env python3
"""
Script de prueba de configuración - VERSIÓN CORREGIDA
Tech Trends Explorer v2.0

Verifica que todas las APIs y servicios estén configurados correctamente.
"""

import os
import json
import sys

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_status(message, status="info"):
    """Imprime mensaje con color según el estado"""
    if status == "success":
        print(f"{Colors.GREEN}✅ {message}{Colors.ENDC}")
    elif status == "error":
        print(f"{Colors.RED}❌ {message}{Colors.ENDC}")
    elif status == "warning":
        print(f"{Colors.YELLOW}⚠️ {message}{Colors.ENDC}")
    else:
        print(f"{Colors.BLUE}ℹ️ {message}{Colors.ENDC}")

def test_dependencies():
    """Verifica que las dependencias estén instaladas"""
    print_status("Verificando dependencias...", "info")
    
    # Lista de paquetes con sus nombres de importación
    required_packages = [
        ('streamlit', 'streamlit'), 
        ('pandas', 'pandas'), 
        ('plotly', 'plotly'), 
        ('requests', 'requests'),
        ('boto3', 'boto3'), 
        ('python-dotenv', 'dotenv'),  # Corregido aquí
        ('scipy', 'scipy'), 
        ('numpy', 'numpy')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
            print_status(f"{package_name}: instalado", "success")
        except ImportError:
            print_status(f"{package_name}: NO instalado", "error")
            missing_packages.append(package_name)
    
    if missing_packages:
        print_status(f"Instalar paquetes faltantes: uv pip install {' '.join(missing_packages)}", "warning")
        return False
    
    return True

def load_configuration():
    """Carga configuración desde múltiples fuentes"""
    print_status("Cargando configuración...", "info")
    
    # Cargar .env si está disponible
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print_status("Variables de entorno .env cargadas", "success")
    except ImportError:
        print_status("python-dotenv no disponible, usando solo variables de sistema", "warning")
    except Exception as e:
        print_status(f"Error cargando .env: {str(e)}", "warning")
    
    config = {}
    
    # Variables de entorno
    env_vars = [
        'GOOGLE_API_KEY', 'SEARCH_ENGINE_ID', 'SERP_API_KEY', 'SCOPUS_API_KEY',
        'AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_DEFAULT_REGION'
    ]
    
    for var in env_vars:
        value = os.getenv(var)
        if value:
            config[var] = value
            # Mostrar solo los primeros y últimos caracteres por seguridad
            masked_value = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "****"
            print_status(f"{var}: configurado ({masked_value})", "success")
        else:
            print_status(f"{var}: NO configurado", "warning")
    
    # Intentar cargar config.json
    if os.path.exists('config.json'):
        try:
            with open('config.json', 'r') as f:
                file_config = json.load(f)
                for key, value in file_config.items():
                    if key not in config and value and not key.startswith('_'):
                        config[key] = value
                        masked_value = f"{value[:4]}...{value[-4:]}" if len(str(value)) > 8 else "****"
                        print_status(f"{key}: cargado desde config.json ({masked_value})", "success")
        except Exception as e:
            print_status(f"Error cargando config.json: {str(e)}", "error")
    
    return config

def test_google_api(config):
    """Prueba la API de Google Custom Search"""
    print_status("Probando Google Custom Search API...", "info")
    
    api_key = config.get('GOOGLE_API_KEY')
    search_engine_id = config.get('SEARCH_ENGINE_ID')
    
    if not api_key or not search_engine_id:
        print_status("Credenciales de Google no configuradas", "error")
        return False
    
    try:
        from googleapiclient.discovery import build
        service = build("customsearch", "v1", developerKey=api_key)
        
        # Realizar búsqueda de prueba
        result = service.cse().list(
            q="test",
            cx=search_engine_id,
            num=1
        ).execute()
        
        if 'items' in result:
            print_status("Google Custom Search API: funcionando correctamente", "success")
            return True
        else:
            print_status("Google Custom Search API: respuesta vacía", "warning")
            return False
            
    except Exception as e:
        print_status(f"Error en Google API: {str(e)}", "error")
        return False

def test_serp_api(config):
    """Prueba SerpAPI"""
    print_status("Probando SerpAPI...", "info")
    
    api_key = config.get('SERP_API_KEY')
    
    if not api_key:
        print_status("API key de SerpAPI no configurada", "error")
        return False
    
    try:
        import requests
        
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
            print_status("SerpAPI: funcionando correctamente", "success")
            return True
        else:
            print_status(f"SerpAPI: error {response.status_code}", "error")
            return False
            
    except Exception as e:
        print_status(f"Error en SerpAPI: {str(e)}", "error")
        return False

def test_scopus_api(config):
    """Prueba la API de Scopus"""
    print_status("Probando Scopus API...", "info")
    
    api_key = config.get('SCOPUS_API_KEY')
    
    if not api_key:
        print_status("API key de Scopus no configurada", "error")
        return False
    
    try:
        import requests
        
        response = requests.get(
            "https://api.elsevier.com/content/search/scopus",
            headers={
                "X-ELS-APIKey": api_key,
                "Accept": "application/json"
            },
            params={
                "query": "TITLE(test)",
                "count": 1
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print_status("Scopus API: funcionando correctamente", "success")
            return True
        else:
            print_status(f"Scopus API: error {response.status_code}", "error")
            return False
            
    except Exception as e:
        print_status(f"Error en Scopus API: {str(e)}", "error")
        return False

def test_dynamodb(config):
    """Prueba la conexión con DynamoDB"""
    print_status("Probando AWS DynamoDB...", "info")
    
    aws_access_key = config.get('AWS_ACCESS_KEY_ID')
    aws_secret_key = config.get('AWS_SECRET_ACCESS_KEY')
    aws_region = config.get('AWS_DEFAULT_REGION', 'us-east-1')
    
    if not aws_access_key or not aws_secret_key:
        print_status("Credenciales de AWS no configuradas", "error")
        return False
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Crear cliente DynamoDB
        dynamodb = boto3.resource(
            'dynamodb',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # Verificar tablas
        analyses_table = dynamodb.Table('tech-trends-analyses')
        categories_table = dynamodb.Table('tech-trends-categories')
        
        # Comprobar estado de las tablas
        analyses_status = analyses_table.table_status
        categories_status = categories_table.table_status
        
        print_status(f"Tabla analyses: {analyses_status}", "success" if analyses_status == "ACTIVE" else "warning")
        print_status(f"Tabla categories: {categories_status}", "success" if categories_status == "ACTIVE" else "warning")
        
        if analyses_status == "ACTIVE" and categories_status == "ACTIVE":
            print_status("DynamoDB: funcionando correctamente", "success")
            return True
        else:
            print_status("DynamoDB: tablas no están activas", "warning")
            return False
            
    except ClientError as e:
        if e.response['Error']['Code'] == 'ResourceNotFoundException':
            print_status("DynamoDB: tablas no existen. Créalas en AWS Console", "error")
        else:
            print_status(f"Error en DynamoDB: {str(e)}", "error")
        return False
    except Exception as e:
        print_status(f"Error conectando con DynamoDB: {str(e)}", "error")
        return False

def test_file_system():
    """Verifica permisos del sistema de archivos"""
    print_status("Verificando sistema de archivos...", "info")
    
    try:
        # Crear directorio de datos si no existe
        os.makedirs("./data", exist_ok=True)
        print_status("Directorio ./data: accesible", "success")
        
        # Probar escritura
        test_file = "./data/test_write.txt"
        with open(test_file, 'w') as f:
            f.write("test")
        
        # Probar lectura
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Limpiar
        os.remove(test_file)
        
        print_status("Permisos de escritura/lectura: OK", "success")
        return True
        
    except Exception as e:
        print_status(f"Error en sistema de archivos: {str(e)}", "error")
        return False

def main():
    """Función principal de prueba"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 60)
    print("🔍 TECH TRENDS EXPLORER - VERIFICACIÓN DE CONFIGURACIÓN")
    print("=" * 60)
    print(f"{Colors.ENDC}")
    
    # Resultados de las pruebas
    results = {}
    
    # 1. Verificar dependencias
    print(f"\n{Colors.BOLD}1. DEPENDENCIAS{Colors.ENDC}")
    results['dependencies'] = test_dependencies()
    
    if not results['dependencies']:
        print_status("Instala las dependencias faltantes antes de continuar", "error")
        return
    
    # 2. Cargar configuración
    print(f"\n{Colors.BOLD}2. CONFIGURACIÓN{Colors.ENDC}")
    config = load_configuration()
    
    # 3. Verificar sistema de archivos
    print(f"\n{Colors.BOLD}3. SISTEMA DE ARCHIVOS{Colors.ENDC}")
    results['filesystem'] = test_file_system()
    
    # 4. Probar APIs
    print(f"\n{Colors.BOLD}4. APIs EXTERNAS{Colors.ENDC}")
    results['google'] = test_google_api(config)
    results['serp'] = test_serp_api(config)
    results['scopus'] = test_scopus_api(config)
    
    # 5. Probar DynamoDB
    print(f"\n{Colors.BOLD}5. AWS DYNAMODB{Colors.ENDC}")
    results['dynamodb'] = test_dynamodb(config)
    
    # Resumen final
    print(f"\n{Colors.BOLD}📊 RESUMEN DE VERIFICACIÓN{Colors.ENDC}")
    print("=" * 40)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper()}: {status}")
    
    print("=" * 40)
    print(f"TOTAL: {passed_tests}/{total_tests} pruebas pasaron")
    
    if passed_tests == total_tests:
        print_status("🎉 ¡Configuración completa y funcionando!", "success")
        print_status("Ya puedes ejecutar: streamlit run src/main.py", "info")
    elif passed_tests >= total_tests - 1:
        print_status("⚠️ Configuración casi completa. Revisa las pruebas fallidas.", "warning")
    else:
        print_status("❌ Configuración incompleta. Revisa las credenciales y servicios.", "error")
    
    print(f"\n{Colors.BLUE}💡 Para más ayuda, consulta la documentación{Colors.ENDC}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ Prueba interrumpida por el usuario{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error inesperado: {str(e)}{Colors.ENDC}")
        sys.exit(1)