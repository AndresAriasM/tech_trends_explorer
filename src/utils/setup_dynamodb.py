# scripts/setup_dynamodb.py
"""
Script para configurar las tablas de DynamoDB necesarias para Tech Trends Explorer

Uso:
    python scripts/setup_dynamodb.py --region us-east-2 --profile default
    python scripts/setup_dynamodb.py --access-key YOUR_KEY --secret-key YOUR_SECRET --region us-east-2
"""

import boto3
import argparse
import json
import sys
import time
from botocore.exceptions import ClientError, NoCredentialsError

def create_table_if_not_exists(dynamodb, table_name, key_schema, attribute_definitions, 
                              billing_mode='PAY_PER_REQUEST', global_secondary_indexes=None):
    """
    Crea una tabla de DynamoDB si no existe
    
    Args:
        dynamodb: Cliente de DynamoDB
        table_name: Nombre de la tabla
        key_schema: Esquema de claves
        attribute_definitions: Definiciones de atributos
        billing_mode: Modo de facturación
        global_secondary_indexes: Índices secundarios globales
    """
    try:
        # Verificar si la tabla ya existe
        existing_tables = dynamodb.list_tables()['TableNames']
        
        if table_name in existing_tables:
            print(f"✅ La tabla '{table_name}' ya existe")
            
            # Verificar estado de la tabla
            table = dynamodb.describe_table(TableName=table_name)
            status = table['Table']['TableStatus']
            
            if status == 'ACTIVE':
                print(f"✅ La tabla '{table_name}' está activa y lista para usar")
            else:
                print(f"⏳ La tabla '{table_name}' está en estado: {status}")
            
            return True
        
        # Crear la tabla
        print(f"🔨 Creando tabla '{table_name}'...")
        
        create_table_params = {
            'TableName': table_name,
            'KeySchema': key_schema,
            'AttributeDefinitions': attribute_definitions,
            'BillingMode': billing_mode
        }
        
        if global_secondary_indexes:
            create_table_params['GlobalSecondaryIndexes'] = global_secondary_indexes
        
        response = dynamodb.create_table(**create_table_params)
        
        print(f"⏳ Esperando que la tabla '{table_name}' esté activa...")
        
        # Esperar hasta que la tabla esté activa
        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(
            TableName=table_name,
            WaiterConfig={
                'Delay': 5,
                'MaxAttempts': 20
            }
        )
        
        print(f"✅ Tabla '{table_name}' creada exitosamente")
        return True
        
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ResourceInUseException':
            print(f"✅ La tabla '{table_name}' ya existe")
            return True
        else:
            print(f"❌ Error creando tabla '{table_name}': {e}")
            return False
    except Exception as e:
        print(f"❌ Error inesperado creando tabla '{table_name}': {e}")
        return False

def setup_tech_trends_tables(aws_access_key_id=None, aws_secret_access_key=None, region_name='us-east-2', profile_name=None):
    """
    Configura todas las tablas necesarias para Tech Trends Explorer
    
    Args:
        aws_access_key_id: Access Key ID de AWS
        aws_secret_access_key: Secret Access Key de AWS
        region_name: Región de AWS
        profile_name: Nombre del perfil de AWS CLI
    """
    try:
        # Configurar cliente de DynamoDB
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
            dynamodb = session.client('dynamodb', region_name=region_name)
        elif aws_access_key_id and aws_secret_access_key:
            dynamodb = boto3.client(
                'dynamodb',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            # Usar credenciales por defecto
            dynamodb = boto3.client('dynamodb', region_name=region_name)
        
        print(f"🔗 Conectando a DynamoDB en región: {region_name}")
        
        # Verificar conexión
        try:
            dynamodb.list_tables()
            print("✅ Conexión exitosa con DynamoDB")
        except NoCredentialsError:
            print("❌ Credenciales de AWS no encontradas")
            print("💡 Asegúrate de configurar AWS CLI o proporcionar credenciales")
            return False
        except Exception as e:
            print(f"❌ Error de conexión: {e}")
            return False
        
        success = True
        
        # 1. Tabla principal de análisis
        print("\n📊 Configurando tabla principal de análisis...")
        
        analyses_table_success = create_table_if_not_exists(
            dynamodb=dynamodb,
            table_name='tech-trends-analyses',
            key_schema=[
                {
                    'AttributeName': 'analysis_id',
                    'KeyType': 'HASH'  # Partition key
                },
                {
                    'AttributeName': 'timestamp',
                    'KeyType': 'RANGE'  # Sort key
                }
            ],
            attribute_definitions=[
                {
                    'AttributeName': 'analysis_id',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'timestamp',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'category_id',
                    'AttributeType': 'S'
                },
                {
                    'AttributeName': 'analysis_type',
                    'AttributeType': 'S'
                }
            ],
            global_secondary_indexes=[
                {
                    'IndexName': 'CategoryIndex',
                    'KeySchema': [
                        {
                            'AttributeName': 'category_id',
                            'KeyType': 'HASH'
                        },
                        {
                            'AttributeName': 'timestamp',
                            'KeyType': 'RANGE'
                        }
                    ],
                    'Projection': {
                        'ProjectionType': 'ALL'
                    }
                },
                {
                    'IndexName': 'TypeIndex',
                    'KeySchema': [
                        {
                            'AttributeName': 'analysis_type',
                            'KeyType': 'HASH'
                        },
                        {
                            'AttributeName': 'timestamp',
                            'KeyType': 'RANGE'
                        }
                    ],
                    'Projection': {
                        'ProjectionType': 'ALL'
                    }
                }
            ]
        )
        
        if not analyses_table_success:
            success = False
        
        # 2. Tabla de categorías
        print("\n📂 Configurando tabla de categorías...")
        
        categories_table_success = create_table_if_not_exists(
            dynamodb=dynamodb,
            table_name='tech-trends-categories',
            key_schema=[
                {
                    'AttributeName': 'category_id',
                    'KeyType': 'HASH'  # Partition key
                }
            ],
            attribute_definitions=[
                {
                    'AttributeName': 'category_id',
                    'AttributeType': 'S'
                }
            ]
        )
        
        if not categories_table_success:
            success = False
        
        # 3. Crear categorías por defecto
        if categories_table_success:
            print("\n📋 Configurando categorías por defecto...")
            try:
                # Usar el recurso de DynamoDB para operaciones de items
                if profile_name:
                    session = boto3.Session(profile_name=profile_name)
                    dynamodb_resource = session.resource('dynamodb', region_name=region_name)
                elif aws_access_key_id and aws_secret_access_key:
                    dynamodb_resource = boto3.resource(
                        'dynamodb',
                        aws_access_key_id=aws_access_key_id,
                        aws_secret_access_key=aws_secret_access_key,
                        region_name=region_name
                    )
                else:
                    dynamodb_resource = boto3.resource('dynamodb', region_name=region_name)
                
                categories_table = dynamodb_resource.Table('tech-trends-categories')
                
                # Esperar que la tabla esté disponible
                categories_table.wait_until_exists()
                
                default_categories = [
                    {
                        'category_id': 'default',
                        'name': 'Sin categoría',
                        'description': 'Categoría por defecto para análisis sin categorizar',
                        'created_at': '2025-01-01T00:00:00Z'
                    },
                    {
                        'category_id': 'technology',
                        'name': 'Tecnología',
                        'description': 'Análisis relacionados con tecnologías emergentes',
                        'created_at': '2025-01-01T00:00:00Z'
                    },
                    {
                        'category_id': 'ai',
                        'name': 'Inteligencia Artificial',
                        'description': 'Análisis específicos de IA y Machine Learning',
                        'created_at': '2025-01-01T00:00:00Z'
                    },
                    {
                        'category_id': 'blockchain',
                        'name': 'Blockchain',
                        'description': 'Análisis de tecnologías blockchain y criptomonedas',
                        'created_at': '2025-01-01T00:00:00Z'
                    },
                    {
                        'category_id': 'biotech',
                        'name': 'Biotecnología',
                        'description': 'Análisis de biotecnología y ciencias de la vida',
                        'created_at': '2025-01-01T00:00:00Z'
                    }
                ]
                
                for category in default_categories:
                    try:
                        # Verificar si la categoría ya existe
                        response = categories_table.get_item(
                            Key={'category_id': category['category_id']}
                        )
                        
                        if 'Item' not in response:
                            # La categoría no existe, crearla
                            categories_table.put_item(Item=category)
                            print(f"✅ Categoría '{category['name']}' creada")
                        else:
                            print(f"✅ Categoría '{category['name']}' ya existe")
                            
                    except Exception as e:
                        print(f"⚠️ Error creando categoría '{category['name']}': {e}")
                
            except Exception as e:
                print(f"⚠️ Error configurando categorías por defecto: {e}")
        
        if success:
            print("\n🎉 ¡Configuración de DynamoDB completada exitosamente!")
            print("\n📋 Resumen de tablas creadas:")
            print("  • tech-trends-analyses (análisis principales)")
            print("  • tech-trends-categories (categorías)")
            print("\n💡 Próximos pasos:")
            print("  1. Configura las credenciales de AWS en Tech Trends Explorer")
            print("  2. Selecciona 'DynamoDB' como modo de almacenamiento")
            print("  3. ¡Comienza a analizar tendencias tecnológicas!")
        else:
            print("\n❌ Algunos errores ocurrieron durante la configuración")
            print("💡 Revisa los mensajes de error y las credenciales de AWS")
        
        return success
        
    except Exception as e:
        print(f"❌ Error general en la configuración: {e}")
        return False

def delete_tables(aws_access_key_id=None, aws_secret_access_key=None, region_name='us-east-2', profile_name=None):
    """
    Elimina todas las tablas de Tech Trends Explorer (¡USAR CON CUIDADO!)
    
    Args:
        aws_access_key_id: Access Key ID de AWS
        aws_secret_access_key: Secret Access Key de AWS
        region_name: Región de AWS
        profile_name: Nombre del perfil de AWS CLI
    """
    print("⚠️  ADVERTENCIA: Esta operación eliminará TODAS las tablas y datos de Tech Trends Explorer")
    confirmation = input("¿Estás seguro? Escribe 'DELETE' para confirmar: ")
    
    if confirmation != 'DELETE':
        print("❌ Operación cancelada")
        return False
    
    try:
        # Configurar cliente de DynamoDB
        if profile_name:
            session = boto3.Session(profile_name=profile_name)
            dynamodb = session.client('dynamodb', region_name=region_name)
        elif aws_access_key_id and aws_secret_access_key:
            dynamodb = boto3.client(
                'dynamodb',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )
        else:
            dynamodb = boto3.client('dynamodb', region_name=region_name)
        
        tables_to_delete = ['tech-trends-analyses', 'tech-trends-categories']
        
        for table_name in tables_to_delete:
            try:
                print(f"🗑️ Eliminando tabla '{table_name}'...")
                dynamodb.delete_table(TableName=table_name)
                
                # Esperar que la tabla sea eliminada
                waiter = dynamodb.get_waiter('table_not_exists')
                waiter.wait(TableName=table_name)
                
                print(f"✅ Tabla '{table_name}' eliminada")
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ResourceNotFoundException':
                    print(f"⚠️ La tabla '{table_name}' no existe")
                else:
                    print(f"❌ Error eliminando tabla '{table_name}': {e}")
            except Exception as e:
                print(f"❌ Error inesperado eliminando tabla '{table_name}': {e}")
        
        print("\n✅ Operación de eliminación completada")
        return True
        
    except Exception as e:
        print(f"❌ Error general en la eliminación: {e}")
        return False

def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(description='Configurar tablas de DynamoDB para Tech Trends Explorer')
    
    parser.add_argument('--access-key', help='AWS Access Key ID')
    parser.add_argument('--secret-key', help='AWS Secret Access Key')
    parser.add_argument('--region', default='us-east-2', help='Región de AWS (default: us-east-2)')
    parser.add_argument('--profile', help='Perfil de AWS CLI a usar')
    parser.add_argument('--delete', action='store_true', help='Eliminar todas las tablas (¡CUIDADO!)')
    
    args = parser.parse_args()
    
    print("🚀 Tech Trends Explorer - Configuración de DynamoDB")
    print("=" * 50)
    
    if args.delete:
        success = delete_tables(
            aws_access_key_id=args.access_key,
            aws_secret_access_key=args.secret_key,
            region_name=args.region,
            profile_name=args.profile
        )
    else:
        success = setup_tech_trends_tables(
            aws_access_key_id=args.access_key,
            aws_secret_access_key=args.secret_key,
            region_name=args.region,
            profile_name=args.profile
        )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()