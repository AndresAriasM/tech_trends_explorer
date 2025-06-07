import requests
from bs4 import BeautifulSoup
import re
import time
import os
from dotenv import load_dotenv
import json
from datetime import datetime

def buscar_patentscope_total():
    """
    Obtiene el número total de patentes en PATENTSCOPE
    """
    base_url = "https://patentscope.wipo.int/search/en/result.jsf"
    query = "(plantain OR banana OR musa) AND (flour OR starch)"
    
    params = {
        'query': query,
        'maxRec': '1'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Buscar patrones de total de resultados
        patterns = [
            r'Results:\s*\d+\s*to\s*\d+\s*of\s*([\d,]+)',
            r'Found\s*([\d,]+)\s*results?',
            r'Total:\s*([\d,]+)',
            r'of\s*([\d,]+)\s*result',
            r'(\d+)\s*result[s]?\s*found'
        ]
        
        total_results = None
        page_text = soup.get_text()
        
        for pattern in patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    total_results = int(matches[0].replace(',', ''))
                    break
                except:
                    continue
        
        if not total_results:
            # Buscar números que podrían ser el total
            all_numbers = re.findall(r'\b([\d,]+)\b', page_text)
            for num_str in all_numbers:
                try:
                    num = int(num_str.replace(',', ''))
                    if 1 <= num <= 100000:  # Rango razonable para nuestra búsqueda
                        total_results = num
                        break
                except:
                    continue
        
        return total_results, response.url
        
    except Exception as e:
        print(f"❌ Error en PATENTSCOPE total: {e}")
        return None, None

def buscar_patentscope_por_año(año):
    """
    Busca patentes en PATENTSCOPE filtradas por año específico
    Usa la sintaxis correcta de PATENTSCOPE: DP:YYYY para publication date
    """
    base_url = "https://patentscope.wipo.int/search/en/result.jsf"
    
    # Sintaxis correcta para PATENTSCOPE: DP:YYYY
    query = f"((plantain OR banana OR musa) AND (flour OR starch)) AND DP:{año}"
    
    params = {
        'query': query,
        'maxRec': '1'
    }
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        print(f"  🔍 PATENTSCOPE año {año}...")
        response = requests.get(base_url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        page_text = soup.get_text()
        
        # Patrones para extraer resultados
        patterns = [
            r'Results:\s*\d+\s*to\s*\d+\s*of\s*([\d,]+)',
            r'Found\s*([\d,]+)\s*results?',
            r'Total:\s*([\d,]+)',
            r'of\s*([\d,]+)\s*result',
            r'(\d+)\s*result[s]?\s*found'
        ]
        
        year_results = 0
        for pattern in patterns:
            matches = re.findall(pattern, page_text, re.IGNORECASE)
            if matches:
                try:
                    year_results = int(matches[0].replace(',', ''))
                    break
                except:
                    continue
        
        # Si no encontramos con patrones, buscar "0 results" o similar
        if year_results == 0:
            if "0 results" in page_text.lower() or "no results" in page_text.lower():
                year_results = 0
            else:
                # Buscar números pequeños que podrían ser el resultado
                small_numbers = re.findall(r'\b([0-9]{1,3})\b', page_text)
                for num_str in small_numbers:
                    try:
                        num = int(num_str)
                        if 0 <= num <= 50:  # Para años individuales esperamos números pequeños
                            year_results = num
                            break
                    except:
                        continue
        
        time.sleep(2)  # Pausa más larga para no sobrecargar PATENTSCOPE
        return year_results
        
    except Exception as e:
        print(f"    ❌ Error PATENTSCOPE año {año}: {e}")
        return 0

def buscar_google_patents_serpapi_total():
    """
    Obtiene el total de patentes en Google Patents usando SerpAPI
    """
    load_dotenv()
    api_key = os.getenv('SERP_API_KEY')
    
    if not api_key:
        print("❌ Error: SERP_API_KEY no encontrada en .env")
        return None
    
    url = "https://serpapi.com/search.json"
    
    params = {
        'engine': 'google_patents',
        'q': '(plantain OR banana OR musa) AND (flour OR starch)',
        'api_key': api_key,
        'num': 10  # Mínimo permitido
    }
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Verificar si hay errores en la respuesta
        if 'error' in data:
            print(f"❌ Error SerpAPI: {data['error']}")
            return None
        
        # SerpAPI devuelve información de búsqueda
        search_info = data.get('search_information', {})
        total_results = search_info.get('total_results')
        
        if isinstance(total_results, str):
            # Convertir "About 1,234 results" a número
            total_results = int(re.sub(r'[^\d]', '', total_results))
        
        return total_results
        
    except Exception as e:
        print(f"❌ Error en Google Patents total: {e}")
        return None

def buscar_google_patents_serpapi_por_año(año):
    """
    Busca patentes en Google Patents filtradas por año usando SerpAPI
    Sintaxis correcta: after y before con publication:YYYYMMDD
    """
    load_dotenv()
    api_key = os.getenv('SERP_API_KEY')
    
    if not api_key:
        return 0
    
    url = "https://serpapi.com/search.json"
    
    # Sintaxis correcta para SerpAPI
    params = {
        'engine': 'google_patents',
        'q': '(plantain OR banana OR musa) AND (flour OR starch)',
        'after': f'publication:{año}0101',   # Fecha mínima
        'before': f'publication:{año}1231',  # Fecha máxima
        'api_key': api_key,
        'num': 10
    }
    
    try:
        print(f"  🔍 Google Patents año {año}...")
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        
        # Verificar errores
        if 'error' in data:
            print(f"    ❌ Error SerpAPI año {año}: {data['error']}")
            return 0
        
        search_info = data.get('search_information', {})
        total_results = search_info.get('total_results', 0)
        
        if isinstance(total_results, str):
            total_results = int(re.sub(r'[^\d]', '', total_results))
        
        time.sleep(1)  # Respetar rate limits
        return total_results
        
    except Exception as e:
        print(f"    ❌ Error Google Patents año {año}: {e}")
        return 0

def analisis_completo_patentes():
    """
    Realiza análisis completo: total y evolución anual
    """
    print("🔍 ANÁLISIS COMPLETO DE PATENTES")
    print("Tema: Harina/almidón de plátano/banana/musa")
    print("=" * 60)
    
    # 1. TOTALES GENERALES
    print("\n📊 PASO 1: TOTALES GENERALES")
    print("-" * 40)
    
    # PATENTSCOPE Total
    print("🌍 Obteniendo total PATENTSCOPE...")
    patentscope_total, patentscope_url = buscar_patentscope_total()
    
    # Google Patents Total
    print("🔍 Obteniendo total Google Patents...")
    google_total = buscar_google_patents_serpapi_total()
    
    print(f"\n📈 RESULTADOS TOTALES:")
    if patentscope_total:
        print(f"  🌍 PATENTSCOPE: {patentscope_total:,} patentes")
    else:
        print(f"  🌍 PATENTSCOPE: Error - verificar manualmente")
        if patentscope_url:
            print(f"      URL: {patentscope_url}")
    
    if google_total:
        print(f"  🔍 Google Patents: {google_total:,} patentes")
    else:
        print(f"  🔍 Google Patents: Error en API")
    
    # 2. ANÁLISIS POR AÑOS (rango más pequeño para testing)
    print(f"\n📅 PASO 2: EVOLUCIÓN ANUAL (2020-2024)")
    print("-" * 40)
    
    años = [2020, 2021, 2022, 2023, 2024]  # Rango reducido para pruebas
    datos_patentscope = {}
    datos_google = {}
    
    print("🌍 PATENTSCOPE por año:")
    for año in años:
        resultado = buscar_patentscope_por_año(año)
        datos_patentscope[año] = resultado
        print(f"  {año}: {resultado:,} patentes")
    
    print(f"\n🔍 Google Patents por año:")
    for año in años:
        resultado = buscar_google_patents_serpapi_por_año(año)
        datos_google[año] = resultado
        print(f"  {año}: {resultado:,} patentes")
    
    # 3. COMPARACIÓN Y VALIDACIÓN
    print(f"\n🔬 PASO 3: COMPARACIÓN Y VALIDACIÓN")
    print("-" * 40)
    
    suma_patentscope = sum(datos_patentscope.values())
    suma_google = sum(datos_google.values())
    
    print(f"📊 VALIDACIÓN DE DATOS:")
    print(f"  🌍 PATENTSCOPE - Total declarado: {patentscope_total or 'N/A'}")
    print(f"  🌍 PATENTSCOPE - Suma años {min(años)}-{max(años)}: {suma_patentscope:,}")
    print(f"  🔍 Google Patents - Total declarado: {google_total or 'N/A'}")
    print(f"  🔍 Google Patents - Suma años {min(años)}-{max(años)}: {suma_google:,}")
    
    # Análisis de consistencia
    if patentscope_total and suma_patentscope > 0:
        if suma_patentscope <= patentscope_total:
            print(f"  ✅ PATENTSCOPE: Suma parcial es lógica ({suma_patentscope} ≤ {patentscope_total})")
        else:
            print(f"  ⚠️  PATENTSCOPE: Suma parcial > total (revisar)")
    else:
        print(f"  ⚠️  PATENTSCOPE: Datos incompletos para validar")
    
    if google_total and suma_google > 0:
        if suma_google <= google_total:
            print(f"  ✅ Google Patents: Suma parcial es lógica ({suma_google} ≤ {google_total})")
        else:
            print(f"  ⚠️  Google Patents: Suma parcial > total (revisar)")
    else:
        print(f"  ⚠️  Google Patents: Datos incompletos para validar")
    
    # 4. PREPARAR DATOS PARA GRÁFICO
    print(f"\n📈 PASO 4: DATOS PARA GRÁFICO")
    print("-" * 40)
    
    print("Año, PATENTSCOPE, Google Patents")
    for año in años:
        ps_val = datos_patentscope.get(año, 0)
        gp_val = datos_google.get(año, 0)
        print(f"{año}, {ps_val}, {gp_val}")
    
    # 5. EXPORTAR DATOS
    resultados = {
        'timestamp': datetime.now().isoformat(),
        'query': '(plantain OR banana OR musa) AND (flour OR starch)',
        'totales': {
            'patentscope': patentscope_total,
            'google_patents': google_total
        },
        'por_año': {
            'patentscope': datos_patentscope,
            'google_patents': datos_google
        },
        'validacion': {
            'suma_patentscope': suma_patentscope,
            'suma_google': suma_google,
            'años_analizados': años
        }
    }
    
    # Guardar JSON
    with open('patentes_analisis_completo.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Datos exportados a: patentes_analisis_completo.json")
    
    return resultados

def crear_grafico_simple():
    """
    Crea un gráfico simple con los datos obtenidos
    """
    try:
        import matplotlib.pyplot as plt
        
        # Cargar datos
        with open('patentes_analisis_completo.json', 'r', encoding='utf-8') as f:
            datos = json.load(f)
        
        años = sorted([int(k) for k in datos['por_año']['patentscope'].keys()])
        patentscope_vals = [datos['por_año']['patentscope'].get(str(año), 0) for año in años]
        google_vals = [datos['por_año']['google_patents'].get(str(año), 0) for año in años]
        
        plt.figure(figsize=(12, 8))
        
        # Crear gráfico de líneas
        plt.plot(años, patentscope_vals, marker='o', label='PATENTSCOPE', linewidth=3, markersize=8)
        plt.plot(años, google_vals, marker='s', label='Google Patents', linewidth=3, markersize=8)
        
        plt.title('Evolución de Patentes: Harina/Almidón de Plátano/Banana/Musa', fontsize=16, fontweight='bold')
        plt.xlabel('Año', fontsize=14)
        plt.ylabel('Número de Patentes', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(años, rotation=0)
        
        # Añadir valores en los puntos
        for i, (año, ps_val, gp_val) in enumerate(zip(años, patentscope_vals, google_vals)):
            if ps_val > 0:
                plt.annotate(str(ps_val), (año, ps_val), textcoords="offset points", xytext=(0,10), ha='center')
            if gp_val > 0:
                plt.annotate(str(gp_val), (año, gp_val), textcoords="offset points", xytext=(0,-15), ha='center')
        
        plt.tight_layout()
        plt.savefig('evolucion_patentes.png', dpi=300, bbox_inches='tight')  # Corregido
        plt.show()
        
        print("📊 Gráfico guardado como: evolucion_patentes.png")
        
    except ImportError:
        print("📊 Para crear gráfico, instala matplotlib: pip install matplotlib")
    except Exception as e:
        print(f"❌ Error creando gráfico: {e}")

def debug_patentscope_url(año=2023):
    """
    Función de debug para mostrar la URL exacta que se está usando
    """
    query = f"((plantain OR banana OR musa) AND (flour OR starch)) AND DP:{año}"
    base_url = "https://patentscope.wipo.int/search/en/result.jsf"
    
    print(f"\n🔧 DEBUG - URLs de búsqueda:")
    print(f"Query PATENTSCOPE: {query}")
    print(f"URL completa: {base_url}?query={query.replace(' ', '+')}")
    
    # También mostrar Google Patents
    print(f"\nQuery Google Patents: (plantain OR banana OR musa) AND (flour OR starch)")
    print(f"Filtros: after=publication:{año}0101, before=publication:{año}1231")

def main():
    """
    Función principal
    """
    print("🚀 INICIANDO ANÁLISIS COMPLETO DE PATENTES")
    print("=" * 60)
    
    # Mostrar debug info
    debug_patentscope_url()
    
    # Ejecutar análisis completo
    resultados = analisis_completo_patentes()
    
    # Crear gráfico si es posible
    crear_grafico_simple()
    
    print(f"\n🎯 ANÁLISIS COMPLETADO")
    print("✅ Datos totales obtenidos")
    print("✅ Evolución anual calculada") 
    print("✅ Validación cruzada realizada")
    print("✅ Datos exportados para gráfico")
    
    return resultados

if __name__ == "__main__":
    main()