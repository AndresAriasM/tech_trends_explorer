#!/usr/bin/env python3
"""
Script de diagnóstico específico para DynamoDB
Tech Trends Explorer v2.0

Prueba paso a paso la conectividad y configuración de AWS DynamoDB
"""

import os
import json
import sys
from datetime import datetime

# Colores para output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
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
    elif status == "info":
        print(f"{Colors.BLUE}ℹ️ {message}{Colors.ENDC}")
    elif status == "step":
        print(f"{Colors.PURPLE}🔄 {message}{Colors.ENDC}")
    else:
        print(f"{Colors.CYAN}📋 {message}{Colors.ENDC}")

def load_aws_credentials():
    """Carga las credenciales de AWS desde múltiples fuentes"""
    print_status("PASO 1: Cargando credenciales de AWS...", "step")
    
    # Intentar cargar .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        print_status("Archivo .env cargado", "success")
    except ImportError:
        print_status("python-dotenv no disponible", "warning")
    except Exception as e:
        print_status(f"Error cargando .env: {str(e)}", "warning")
    
    # Obtener credenciales
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_DEFAULT_REGION')
    
    # Verificar credenciales
    print_status("Verificando credenciales...", "info")
    
    if aws_access_key:
        masked_key = f"{aws_access_key[:8]}...{aws_access_key[-4:]}"
        print_status(f"AWS_ACCESS_KEY_ID: {masked_key}", "success")
    else:
        print_status("AWS_ACCESS_KEY_ID: NO encontrado", "error")
    
    if aws_secret_key:
        masked_secret = f"{aws_secret_key[:8]}...{aws_secret_key[-4:]}"
        print_status(f"AWS_SECRET_ACCESS_KEY: {masked_secret}", "success")
    else:
        print_status("AWS_SECRET_ACCESS_KEY: NO encontrado", "error")
    
    print_status(f"AWS_DEFAULT_REGION: {aws_region}", "success" if aws_region else "warning")
    
    # Intentar cargar desde config.json como fallback
    if not aws_access_key or not aws_secret_key:
        print_status("Intentando cargar desde config.json...", "info")
        if os.path.exists('config.json'):
            try:
                with open('config.json', 'r') as f:
                    config = json.load(f)
                    aws_access_key = aws_access_key or config.get('AWS_ACCESS_KEY_ID')
                    aws_secret_key = aws_secret_key or config.get('AWS_SECRET_ACCESS_KEY')
                    aws_region = aws_region or config.get('AWS_DEFAULT_REGION')
                print_status("Credenciales cargadas desde config.json", "success")
            except Exception as e:
                print_status(f"Error cargando config.json: {str(e)}", "error")
    
    return aws_access_key, aws_secret_key, aws_region

def test_boto3_import():
    """Verifica que boto3 esté disponible"""
    print_status("PASO 2: Verificando boto3...", "step")
    
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError
        print_status("boto3 importado correctamente", "success")
        print_status(f"Versión de boto3: {boto3.__version__}", "info")
        return True, boto3, ClientError, NoCredentialsError
    except ImportError as e:
        print_status(f"Error importando boto3: {str(e)}", "error")
        print_status("Instalar con: pip install boto3", "warning")
        return False, None, None, None

def test_aws_connection(aws_access_key, aws_secret_key, aws_region, boto3, ClientError, NoCredentialsError):
    """Prueba la conexión básica con AWS"""
    print_status("PASO 3: Probando conexión con AWS...", "step")
    
    if not aws_access_key or not aws_secret_key:
        print_status("Credenciales faltantes - no se puede conectar", "error")
        return False, None, None
    
    try:
        # Crear cliente DynamoDB
        print_status("Creando cliente DynamoDB...", "info")
        dynamodb_client = boto3.client(
            'dynamodb',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        # Crear resource DynamoDB
        dynamodb_resource = boto3.resource(
            'dynamodb',
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )
        
        print_status("Cliente DynamoDB creado exitosamente", "success")
        return True, dynamodb_client, dynamodb_resource
        
    except NoCredentialsError:
        print_status("Error: Credenciales de AWS no válidas", "error")
        return False, None, None
    except Exception as e:
        print_status(f"Error creando cliente DynamoDB: {str(e)}", "error")
        return False, None, None

def test_aws_permissions(dynamodb_client, ClientError):
    """Prueba los permisos básicos de AWS"""
    print_status("PASO 4: Verificando permisos de AWS...", "step")
    
    try:
        # Intentar listar tablas (operación básica)
        print_status("Probando permiso ListTables...", "info")
        response = dynamodb_client.list_tables()
        
        table_count = len(response.get('TableNames', []))
        print_status(f"ListTables exitoso - {table_count} tablas encontradas", "success")
        
        # Mostrar todas las tablas
        if table_count > 0:
            print_status("Tablas existentes en tu cuenta:", "info")
            for table_name in response['TableNames']:
                print(f"    📋 {table_name}")
        else:
            print_status("No hay tablas en esta región/cuenta", "warning")
        
        return True, response['TableNames']
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        
        if error_code == 'AccessDenied':
            print_status("Error: Sin permisos para DynamoDB", "error")
            print_status("Verifica que el usuario IAM tenga la política correcta", "warning")
        elif error_code == 'UnauthorizedOperation':
            print_status("Error: Usuario no autorizado para esta operación", "error")
        else:
            print_status(f"Error AWS ({error_code}): {error_message}", "error")
        
        return False, []
    except Exception as e:
        print_status(f"Error inesperado probando permisos: {str(e)}", "error")
        return False, []

def check_required_tables(existing_tables, dynamodb_resource, ClientError):
    """Verifica si existen las tablas requeridas"""
    print_status("PASO 5: Verificando tablas requeridas...", "step")
    
    required_tables = ['tech-trends-analyses', 'tech-trends-categories']
    table_status = {}
    
    for table_name in required_tables:
        print_status(f"Verificando tabla: {table_name}", "info")
        
        if table_name in existing_tables:
            try:
                # Obtener información detallada de la tabla
                table = dynamodb_resource.Table(table_name)
                status = table.table_status
                item_count = table.item_count
                creation_date = table.creation_date_time
                
                print_status(f"✓ {table_name} - Estado: {status}", "success")
                print_status(f"  Items: {item_count}, Creada: {creation_date.strftime('%Y-%m-%d %H:%M')}", "info")
                
                table_status[table_name] = {
                    'exists': True,
                    'status': status,
                    'item_count': item_count,
                    'creation_date': creation_date
                }
                
            except ClientError as e:
                print_status(f"✗ Error accediendo a {table_name}: {e.response['Error']['Code']}", "error")
                table_status[table_name] = {'exists': False, 'error': str(e)}
        else:
            print_status(f"✗ {table_name} NO existe", "error")
            table_status[table_name] = {'exists': False, 'error': 'Table not found'}
    
    return table_status

def test_table_operations(dynamodb_resource, table_status, ClientError):
    """Prueba operaciones básicas en las tablas"""
    print_status("PASO 6: Probando operaciones básicas...", "step")
    
    for table_name, status in table_status.items():
        if not status.get('exists', False):
            print_status(f"Saltando {table_name} (no existe)", "warning")
            continue
        
        if status.get('status') != 'ACTIVE':
            print_status(f"Saltando {table_name} (no está ACTIVE)", "warning")
            continue
        
        print_status(f"Probando operaciones en {table_name}...", "info")
        
        try:
            table = dynamodb_resource.Table(table_name)
            
            if table_name == 'tech-trends-categories':
                # Probar operación simple en categorías
                test_item = {
                    'category_id': 'test-connection-001',
                    'name': 'Test Connection',
                    'description': 'Test de conectividad',
                    'created_at': datetime.now().isoformat()
                }
                
                # Escribir item de prueba
                table.put_item(Item=test_item)
                print_status(f"  ✓ PutItem exitoso en {table_name}", "success")
                
                # Leer item de prueba
                response = table.get_item(Key={'category_id': 'test-connection-001'})
                if 'Item' in response:
                    print_status(f"  ✓ GetItem exitoso en {table_name}", "success")
                
                # Limpiar item de prueba
                table.delete_item(Key={'category_id': 'test-connection-001'})
                print_status(f"  ✓ DeleteItem exitoso en {table_name}", "success")
            
            elif table_name == 'tech-trends-analyses':
                # Probar operación simple en análisis
                test_item = {
                    'analysis_id': 'test-connection-001',
                    'timestamp': datetime.now().isoformat(),
                    'name': 'Test Connection',
                    'query': 'test'
                }
                
                # Escribir item de prueba
                table.put_item(Item=test_item)
                print_status(f"  ✓ PutItem exitoso en {table_name}", "success")
                
                # Leer item de prueba
                response = table.get_item(
                    Key={
                        'analysis_id': 'test-connection-001',
                        'timestamp': test_item['timestamp']
                    }
                )
                if 'Item' in response:
                    print_status(f"  ✓ GetItem exitoso en {table_name}", "success")
                
                # Limpiar item de prueba
                table.delete_item(
                    Key={
                        'analysis_id': 'test-connection-001',
                        'timestamp': test_item['timestamp']
                    }
                )
                print_status(f"  ✓ DeleteItem exitoso en {table_name}", "success")
        
        except ClientError as e:
            error_code = e.response['Error']['Code']
            print_status(f"  ✗ Error en {table_name} ({error_code}): {e.response['Error']['Message']}", "error")
        except Exception as e:
            print_status(f"  ✗ Error inesperado en {table_name}: {str(e)}", "error")

def provide_solutions(table_status, existing_tables):
    """Proporciona soluciones para los problemas encontrados"""
    print_status("PASO 7: Soluciones y recomendaciones...", "step")
    
    required_tables = ['tech-trends-analyses', 'tech-trends-categories']
    missing_tables = [t for t in required_tables if not table_status.get(t, {}).get('exists', False)]
    
    if missing_tables:
        print_status("🛠️ CREAR TABLAS FALTANTES:", "warning")
        
        for table_name in missing_tables:
            print(f"\n{Colors.YELLOW}📋 Para crear {table_name}:{Colors.ENDC}")
            print("1. Ve a AWS Console → DynamoDB → Crear tabla")
            
            if table_name == 'tech-trends-analyses':
                print(f"   Nombre: {table_name}")
                print("   Clave de partición: analysis_id (String)")
                print("   Clave de ordenación: timestamp (String)")
            elif table_name == 'tech-trends-categories':
                print(f"   Nombre: {table_name}")
                print("   Clave de partición: category_id (String)")
                print("   (Sin clave de ordenación)")
            
            print("   Configuración: Predeterminada (On-demand)")
    
    if existing_tables:
        print_status("💡 OTRAS RECOMENDACIONES:", "info")
        print("• Si las tablas tienen nombres diferentes, puedes modificar el código")
        print("• Verifica que estés en la región correcta (us-east-2)")
        print("• Asegúrate de que el usuario IAM tenga todos los permisos")

def main():
    """Función principal de diagnóstico"""
    print(f"{Colors.BOLD}{Colors.BLUE}")
    print("=" * 70)
    print("🔍 DIAGNÓSTICO ESPECÍFICO DE AWS DYNAMODB")
    print("Tech Trends Explorer v2.0")
    print("=" * 70)
    print(f"{Colors.ENDC}")
    
    try:
        # Paso 1: Cargar credenciales
        aws_access_key, aws_secret_key, aws_region = load_aws_credentials()
        
        # Paso 2: Verificar boto3
        boto3_ok, boto3, ClientError, NoCredentialsError = test_boto3_import()
        if not boto3_ok:
            return
        
        # Paso 3: Probar conexión
        connection_ok, dynamodb_client, dynamodb_resource = test_aws_connection(
            aws_access_key, aws_secret_key, aws_region, boto3, ClientError, NoCredentialsError
        )
        if not connection_ok:
            return
        
        # Paso 4: Verificar permisos
        permissions_ok, existing_tables = test_aws_permissions(dynamodb_client, ClientError)
        if not permissions_ok:
            return
        
        # Paso 5: Verificar tablas requeridas
        table_status = check_required_tables(existing_tables, dynamodb_resource, ClientError)
        
        # Paso 6: Probar operaciones
        test_table_operations(dynamodb_resource, table_status, ClientError)
        
        # Paso 7: Proporcionar soluciones
        provide_solutions(table_status, existing_tables)
        
        # Resumen final
        print(f"\n{Colors.BOLD}📊 RESUMEN DEL DIAGNÓSTICO{Colors.ENDC}")
        print("=" * 50)
        
        required_tables = ['tech-trends-analyses', 'tech-trends-categories']
        working_tables = [t for t in required_tables if table_status.get(t, {}).get('exists', False)]
        
        print(f"Credenciales AWS: {'✅ OK' if aws_access_key and aws_secret_key else '❌ FALTA'}")
        print(f"Conexión DynamoDB: {'✅ OK' if connection_ok else '❌ FALLA'}")
        print(f"Permisos básicos: {'✅ OK' if permissions_ok else '❌ FALLA'}")
        print(f"Tablas funcionando: {len(working_tables)}/{len(required_tables)}")
        
        if len(working_tables) == len(required_tables):
            print_status("🎉 ¡DynamoDB completamente funcional!", "success")
        elif len(working_tables) > 0:
            print_status("⚠️ DynamoDB parcialmente funcional - faltan tablas", "warning")
        else:
            print_status("❌ DynamoDB no funcional - crear tablas requeridas", "error")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ Diagnóstico interrumpido por el usuario{Colors.ENDC}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error inesperado: {str(e)}{Colors.ENDC}")
        import traceback
        print(f"{Colors.RED}{traceback.format_exc()}{Colors.ENDC}")

if __name__ == "__main__":
    main()