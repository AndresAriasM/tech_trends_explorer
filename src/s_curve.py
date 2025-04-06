# src/s_curve.py
import streamlit as st
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from datetime import datetime
import re
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from utils.scopus_builder import scopus_equation_interface, parse_scopus_query
from scipy.optimize import curve_fit
import json
import io
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.utils import get_column_letter

# Definir la función sigmoidal al nivel global para que esté disponible en toda la aplicación
def sigmoid(x, L, x0, k):
    """
    Función sigmoidal de tres parámetros.
    """
    return L / (1 + np.exp(-k * (x - x0)))

class ScopusPublicationsAnalyzer:
    """
    Clase para conectarse a la API de Scopus, buscar publicaciones académicas y analizarlas por año
    para crear una curva en S.
    """
    
    def __init__(self, api_key=None):
        """
        Inicializa el analizador.
        
        Args:
            api_key: La clave API de Scopus.
        """
        self.api_key = api_key
        self.base_url = "https://api.elsevier.com/content/search/scopus"
        self.headers = {
            "X-ELS-APIKey": api_key,
            "Accept": "application/json"
        }
        self.publications_by_year = {}
        
    def search_publications(self, query, max_results=1000):
        """
        Busca publicaciones en Scopus utilizando una consulta, sin filtros de año.
        
        Args:
            query: Consulta de búsqueda en formato de sintaxis de Scopus
            max_results: Número máximo de resultados a recuperar
            
        Returns:
            Resultados de la búsqueda o None si hay error
        """
        # Añadir filtro específico para excluir patentes
        query_papers = f"({query}) AND NOT (DOCTYPE(pt))"
        
        # Mostrar ecuación de búsqueda
        st.info(f"📝 Ecuación de búsqueda (Papers): {query_papers}")
        
        # Configurar parámetros de búsqueda
        base_url = "https://api.elsevier.com/content/search/scopus"
        headers = {
            "X-ELS-APIKey": self.api_key,
            "Accept": "application/json"
        }
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Intentar la búsqueda
        try:
            # Verificar si la búsqueda funciona
            status_text.text("🔍 Verificando si la búsqueda funciona...")
            
            response = requests.get(
                base_url,
                headers=headers,
                params={"query": query_papers, "count": 1}
            )
            
            if response.status_code != 200:
                st.error(f"❌ Error en la búsqueda de papers: {response.status_code}")
                st.code(response.text)
                
                # Probar con una versión simplificada de la consulta para diagnosticar
                status_text.text("🔍 Probando con búsqueda simplificada...")
                simple_query = 'TITLE("banana") AND TITLE("flour") AND NOT (DOCTYPE(pt))'
                
                st.info(f"📝 Intentando con: {simple_query}")
                
                response = requests.get(
                    base_url,
                    headers=headers,
                    params={"query": simple_query, "count": 1}
                )
                
                if response.status_code == 200:
                    st.success("✅ La búsqueda simplificada funciona")
                    st.info("❗ El problema está en la complejidad de la ecuación original")
                    
                    # Usar la ecuación simplificada
                    query_papers = simple_query
                    st.info(f"📝 Nueva ecuación de búsqueda: {query_papers}")
                else:
                    st.error(f"❌ Error también con búsqueda simplificada: {response.status_code}")
                    st.code(response.text)
                    return None
            
            # Realizar la búsqueda con la ecuación (original o simplificada)
            status_text.text("🔍 Obteniendo resultados completos...")
            
            response = requests.get(
                base_url,
                headers=headers,
                params={"query": query_papers, "count": 25, "sort": "coverDate"}
            )
            
            if response.status_code == 200:
                data = response.json()
                total_results = int(data["search-results"]["opensearch:totalResults"])
                st.success(f"✅ Búsqueda exitosa: {total_results} papers encontrados")
                
                if total_results == 0:
                    status_text.text("❌ No se encontraron papers con esta ecuación")
                    return None
                
                # Recopilar todos los resultados
                all_results = []
                params = {"query": query_papers, "count": 25, "start": 0, "sort": "coverDate"}
                
                while params["start"] < min(total_results, max_results):
                    response = requests.get(base_url, headers=headers, params=params)
                    
                    if response.status_code == 200:
                        data = response.json()
                        entries = data["search-results"].get("entry", [])
                        all_results.extend(entries)
                        
                        # Actualizar progreso
                        progress = min(len(all_results) / min(total_results, max_results), 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"⏳ Recuperados {len(all_results)} de {min(total_results, max_results)} papers ({progress*100:.1f}%)...")
                        
                        params["start"] += params["count"]
                        
                        if params["start"] < min(total_results, max_results):
                            time.sleep(0.5)  # pequeña pausa entre solicitudes
                    else:
                        st.error(f"❌ Error al recuperar página: {response.status_code}")
                        st.code(response.text)
                        break
                
                # Limpieza y mensaje final
                progress_bar.empty()
                
                if all_results:
                    status_text.text(f"✅ Recuperados {len(all_results)} papers en total")
                    return all_results
                else:
                    status_text.text("❌ No se pudieron recuperar papers")
                    return None
            else:
                st.error(f"❌ Error en la búsqueda: {response.status_code}")
                st.code(response.text)
                return None
        except Exception as e:
            st.error(f"❌ Error inesperado en búsqueda de papers: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None
        
    def categorize_by_year(self, results):
        """
        Categoriza los resultados de publicaciones por año y completa los años faltantes con ceros.
        
        Args:
            results: Lista de resultados de la búsqueda
            
        Returns:
            Diccionario con el recuento por año
        """
        publications_by_year = {}
        
        st.text("📅 Categorizando publicaciones por año...")
        
        for entry in results:
            # Intentar extraer el año de diferentes campos
            year = None
            
            # Método 1: Del campo prism:coverDate
            if "prism:coverDate" in entry:
                try:
                    date_str = entry["prism:coverDate"]
                    year = datetime.strptime(date_str, "%Y-%m-%d").year
                except (ValueError, TypeError):
                    pass
            
            # Método 2: Del campo prism:publicationDate
            if not year and "prism:publicationDate" in entry:
                try:
                    date_str = entry["prism:publicationDate"]
                    year = datetime.strptime(date_str, "%Y-%m-%d").year
                except (ValueError, TypeError):
                    pass
                    
            # Método 3: Del texto de la fecha de portada
            if not year and "prism:coverDisplayDate" in entry:
                try:
                    year_str = entry["prism:coverDisplayDate"]
                    # Buscar un año en el texto con regex
                    match = re.search(r'(19|20)\d{2}', year_str)
                    if match:
                        year = int(match.group(0))
                except (ValueError, TypeError, AttributeError):
                    pass
            
            # Si no se pudo determinar el año, continuar con la siguiente entrada
            if not year:
                continue
            
            # Incrementar el contador para este año
            publications_by_year[year] = publications_by_year.get(year, 0) + 1
        
        # Completar años faltantes con ceros
        if publications_by_year:
            min_year = min(publications_by_year.keys())
            max_year = max(publications_by_year.keys())
            
            # Crear rango completo de años
            all_years = list(range(min_year, max_year + 1))
            
            # Completar con ceros los años faltantes
            complete_publications = {year: publications_by_year.get(year, 0) for year in all_years}
            
            # Ordenar por año
            self.publications_by_year = dict(sorted(complete_publications.items()))
        else:
            self.publications_by_year = {}
            
        return self.publications_by_year


class PatentDataManager:
    """
    Clase para gestionar datos de patentes cargados manualmente mediante archivos Excel.
    """
    
    @staticmethod
    def create_template(start_year=1990, end_year=None):
        """
        Crea una plantilla Excel para la carga de datos de patentes por año.
        
        Args:
            start_year: Año inicial para la plantilla
            end_year: Año final para la plantilla (por defecto es el año actual)
        
        Returns:
            BytesIO: Objeto de bytes con el contenido del archivo Excel
        """
        if end_year is None:
            end_year = datetime.now().year
        
        # Crear workbook y hoja
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Datos de Patentes"
        
        # Añadir encabezados con estilo
        ws['A1'] = "Año"
        ws['B1'] = "Número de Patentes"
        ws['C1'] = "Notas/Observaciones"
        
        # Aplicar estilos a los encabezados
        for col in ['A', 'B', 'C']:
            cell = ws[f'{col}1']
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color="E0E0E0", end_color="E0E0E0", fill_type="solid")
            cell.alignment = Alignment(horizontal='center')
        
        # Ajustar anchos de columna
        ws.column_dimensions['A'].width = 10
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 30
        
        # Añadir filas para cada año
        for i, year in enumerate(range(start_year, end_year + 1), start=2):
            ws[f'A{i}'] = year
            ws[f'B{i}'] = 0  # Valor predeterminado
            
            # Aplicar formato numérico a la columna de patentes
            ws[f'B{i}'].number_format = '0'
        
        # Añadir instrucciones en una nueva hoja
        ws_instructions = wb.create_sheet(title="Instrucciones")
        
        instructions = [
            "INSTRUCCIONES PARA COMPLETAR LA PLANTILLA DE PATENTES",
            "",
            "1. Completa la columna 'Número de Patentes' con la cantidad de patentes por año.",
            "2. Los datos deben ser valores numéricos enteros.",
            "3. No elimines ningún año de la lista.",
            "4. Puedes dejar años con valor 0 si no hay datos disponibles.",
            "5. La columna 'Notas/Observaciones' es opcional y puedes usarla para información adicional.",
            "6. Guarda el archivo cuando hayas completado los datos.",
            "7. Sube el archivo en la aplicación para analizar los datos de patentes.",
            "",
            "CÓMO OBTENER LOS DATOS DE PATENTES:",
            "",
            "- Opción 1: Busca en la pestaña 'Patents' de Scopus y anota el número de resultados por año.",
            "- Opción 2: Utiliza otras bases de datos de patentes como PatentScope, Google Patents, o Espacenet.",
            "- Opción 3: Consulta informes de oficinas de patentes nacionales o internacionales.",
            "",
            "Nota: Para una análisis preciso, asegúrate de utilizar la misma ecuación de búsqueda o criterios consistentes para todos los años."
        ]
        
        for i, line in enumerate(instructions, start=1):
            ws_instructions[f'A{i}'] = line
        
        # Ajustar ancho de columna en la hoja de instrucciones
        ws_instructions.column_dimensions['A'].width = 100
        
        # Guardar en un objeto de bytes
        output = io.BytesIO()
        wb.save(output)
        output.seek(0)
        
        return output
    
    @staticmethod
    def load_data(uploaded_file):
        """
        Carga y procesa datos de patentes desde un archivo Excel subido.
        
        Args:
            uploaded_file: Archivo Excel subido mediante st.file_uploader
            
        Returns:
            dict: Diccionario con datos de patentes por año {año: conteo}
        """
        try:
            # Cargar el archivo Excel
            df = pd.read_excel(uploaded_file, sheet_name="Datos de Patentes")
            
            # Verificar que las columnas necesarias existen
            if "Año" not in df.columns or "Número de Patentes" not in df.columns:
                st.error("❌ El archivo no tiene el formato esperado. Asegúrate de usar la plantilla proporcionada.")
                return None
            
            # Convertir a diccionario {año: conteo}
            patents_by_year = dict(zip(df["Año"], df["Número de Patentes"]))
            
            # Filtrar años con valores nulos o no numéricos
            patents_by_year = {year: int(count) for year, count in patents_by_year.items() 
                              if pd.notna(count) and str(count).replace('.', '', 1).isdigit()}
            
            # Ordenar por año
            patents_by_year = dict(sorted(patents_by_year.items()))
            
            return patents_by_year
            
        except Exception as e:
            st.error(f"❌ Error al cargar el archivo: {str(e)}")
            return None


class TechnologyAnalyzer:
    """
    Clase para analizar datos de publicaciones o patentes y generar análisis de curva en S.
    """
    
    @staticmethod
    def analyze_s_curve(yearly_data):
        """
        Realiza un análisis completo de la curva en S con cálculos matemáticos.
        
        Args:
            yearly_data: Diccionario con conteo por año {año: conteo}
            
        Returns:
            DataFrame con los datos analizados, figura, info de ajuste y parámetros
        """
        if not yearly_data:
            st.error("No hay datos para analizar")
            return None, None, None, None
        
        # Crear DataFrame base
        df = pd.DataFrame({
            'Año': list(yearly_data.keys()),
            'Cantidad': list(yearly_data.values())
        })
        
        # Calcular acumulado
        df['Acumulado'] = df['Cantidad'].cumsum()
        
        # Establecer año como índice para cálculos
        df_analysis = df.copy()
        df_analysis.set_index('Año', inplace=True)
        
        # Calcular la tasa de crecimiento anual
        df_analysis['TasaCrecimiento'] = df_analysis['Acumulado'].pct_change() * 100
        
        # Añadir la segunda derivada al DataFrame
        df_analysis['SegundaDerivada'] = np.gradient(np.gradient(df_analysis['Acumulado']))
        
        # Encontrar los años de puntos de inflexión exactos
        try:
            puntos_inflexion_exacto = df_analysis[df_analysis['SegundaDerivada'] == 0].index.tolist()
            if not puntos_inflexion_exacto and len(df_analysis) > 0:
                # Si no hay puntos exactos con segunda derivada = 0, encontrar el más cercano
                punto_cercano_idx = np.abs(df_analysis['SegundaDerivada']).idxmin()
                puntos_inflexion_exacto = [punto_cercano_idx]
        except:
            puntos_inflexion_exacto = []
        
        # Preparar datos para el ajuste
        x_data = np.array(df_analysis.index)
        y_data = np.array(df_analysis['Acumulado'])
        
        # Ajustar la curva sigmoidal a los datos
        try:
            # Usar los mismos valores iniciales y parámetros 
            popt, pcov = curve_fit(
                sigmoid, 
                x_data, 
                y_data, 
                p0=[max(df_analysis['Acumulado']), np.median(df_analysis.index), 0.1], 
                maxfev=5000
            )
            
            # Calcular los puntos de la curva ajustada
            curva_ajustada = sigmoid(x_data, *popt)
            df_analysis['Ajustada'] = curva_ajustada
            
            # Crear una tabla con datos relevantes
            df_parametros = pd.DataFrame({
                'Parámetro': ['L', 'x0', 'k'],
                'Valor ajustado': popt,
                'Error estándar': np.sqrt(np.diag(pcov)),
                'Valor T': popt/np.sqrt(np.diag(pcov)),
                'Validación': ['Válido' if abs(valor_t) > 2 else 'No válido' 
                              for valor_t in popt/np.sqrt(np.diag(pcov))]
            })
            
            # Cálculo de R²
            r_squared = 1 - np.sum((y_data - curva_ajustada)**2) / np.sum((y_data - np.mean(y_data))**2)
            
            # Determinar la fase actual
            ultima_derivada = df_analysis['SegundaDerivada'].iloc[-3:].mean() if len(df_analysis) >= 3 else 0
            
            if ultima_derivada > 0:
                fase = "Fase inicial (crecimiento acelerado)"
                descripcion = "La tecnología está en su fase temprana con crecimiento acelerado."
            elif ultima_derivada < 0:
                fase = "Fase de madurez (crecimiento desacelerado)"
                descripcion = "La tecnología está madurando, el crecimiento se está desacelerando."
            else:
                fase = "Punto de inflexión"
                descripcion = "La tecnología está en el punto de inflexión entre el crecimiento acelerado y desacelerado."
            
            # Crear figura de Plotly con la curva acumulada y el ajuste
            fig = px.line(
                df_analysis, x=df_analysis.index, y=['Acumulado', 'Ajustada'],
                labels={'value': 'Cantidad Acumulada', 'variable': 'Tipo de Curva', 'Año': 'Año'},
                title='Curva S - Acumulado por Año',
                color_discrete_map={'Acumulado': 'blue', 'Ajustada': 'red'},
                markers=True
            )
            
            # Añadir puntos de inflexión si existen
            for punto in puntos_inflexion_exacto:
                fig.add_trace(
                    go.Scatter(
                        x=[punto],
                        y=[df_analysis.loc[punto, 'Acumulado']],
                        mode='markers',
                        name=f'Punto de inflexión ({punto})',
                        marker=dict(color='red', size=15, symbol='star')
                    )
                )
            
            # Mejorar aspecto visual
            fig.update_layout(
                width=600,
                height=600,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                annotations=[
                    dict(
                        x=0.5,
                        y=-0.15,
                        xref='paper',
                        yref='paper',
                        text=f'R² = {r_squared:.4f} | Fase: {fase}',
                        showarrow=False
                    )
                ]
            )
            
            # Información de ajuste
            ajuste_info = {
                'R2': r_squared,
                'L': popt[0],
                'x0': popt[1],
                'k': popt[2],
                'Fase': fase,
                'Descripción': descripcion,
                'Puntos_inflexion': puntos_inflexion_exacto
            }
            
            # Restaurar el índice para el DataFrame original
            df_analysis = df_analysis.reset_index()
            
            return df_analysis, fig, ajuste_info, df_parametros
            
        except Exception as e:
            st.error(f"Error en el ajuste de curva: {str(e)}")
            # Restaurar el índice
            df_analysis = df_analysis.reset_index()
            return df_analysis, None, None, None
    
    @staticmethod
    def display_data_table(yearly_data, title="Datos por Año"):
        """
        Muestra una tabla con los datos por año.
        
        Args:
            yearly_data: Diccionario {año: conteo}
            title: Título de la tabla
            
        Returns:
            DataFrame con los datos
        """
        if not yearly_data:
            st.error("No hay datos para mostrar")
            return None
        
        df = pd.DataFrame({
            'Año': list(yearly_data.keys()),
            'Cantidad': list(yearly_data.values())
        })
        
        # Calcular acumulado
        df['Acumulado'] = df['Cantidad'].cumsum()
        
        # Formatear años como enteros sin comas
        df['Año'] = df['Año'].astype(int)
        
        # Mostrar tabla
        st.write(f"### 📋 {title}")
        
        # Usar el componente nativo de Streamlit para tablas con estilo
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Año": st.column_config.NumberColumn(format="%d"),
                "Cantidad": st.column_config.NumberColumn(format="%d"),
                "Acumulado": st.column_config.NumberColumn(format="%d")
            }
        )
        
        return df
    
    @staticmethod
    def display_s_curve_analysis(analysis_df, analysis_fig, ajuste_info, parametros, title="Análisis de Curva en S"):
        """
        Muestra el análisis completo de la curva en S.
        
        Args:
            analysis_df: DataFrame con los datos analizados
            analysis_fig: Figura de Plotly
            ajuste_info: Información del ajuste
            parametros: DataFrame con los parámetros del modelo
            title: Título del análisis
        """
        st.write(f"### 📊 {title}")
        
        # Mostrar gráfico de análisis
        if analysis_fig:
            st.plotly_chart(analysis_fig, use_container_width=True)
            
            # Mostrar información del ajuste
            if ajuste_info:
                st.write("#### Parámetros del Modelo Ajustado")
                
                # Crear columnas para mostrar métricas
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("R² del ajuste", f"{ajuste_info['R2']:.4f}")
                with col2:
                    st.metric("Punto de inflexión (x0)", f"{ajuste_info['x0']:.1f}")
                with col3:
                    st.metric("Máximo teórico (L)", f"{ajuste_info['L']:.1f}")
                
                # Mostrar parámetros del modelo
                if parametros is not None:
                    st.write("#### Parámetros del modelo sigmoidal")
                    
                    # Usar st.dataframe con estilo
                    st.dataframe(
                        parametros,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Valor ajustado": st.column_config.NumberColumn("Valor ajustado", format="%.4f"),
                            "Error estándar": st.column_config.NumberColumn("Error estándar", format="%.4f"),
                            "Valor T": st.column_config.NumberColumn("Valor T", format="%.4f")
                        }
                    )
                
                # Mostrar fase actual
                st.info(f"**Fase actual de la tecnología**: {ajuste_info['Fase']}")
                st.write(ajuste_info['Descripción'])
    
    @staticmethod
    def export_data(df, filename_prefix, query):
        """
        Genera un botón para exportar los datos a CSV.
        
        Args:
            df: DataFrame con los datos
            filename_prefix: Prefijo para el nombre del archivo
            query: Consulta utilizada
        """
        if df is not None:
            st.write("### 📥 Exportar Datos")
            
            # Botón para descargar CSV
            csv = df.to_csv(index=False)
            query_term_clean = re.sub(r'[^\w\s]', '', query[:30])
            filename = f"{filename_prefix}_{query_term_clean}_{datetime.now().strftime('%Y%m%d')}.csv"
            
            st.download_button(
                label=f"📥 Descargar datos {filename_prefix} (CSV)",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )


def run_s_curve_analysis():
    """
    Ejecuta el análisis de curva en S para publicaciones académicas y patentes.
    """
    st.title("📈 Análisis de Curvas en S: Papers y Patentes")
    
    st.write("""
    Esta herramienta te permite analizar tendencias tecnológicas a lo largo del tiempo 
    utilizando datos de publicaciones académicas (papers) y patentes. El análisis de curvas en S 
    te ayuda a entender la fase de madurez en la que se encuentra una tecnología.
    """)
    
    # Configuración de la API de Scopus
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        api_key_input = st.text_input(
            "API Key de Scopus",
            value=st.session_state.get('scopus_api_key', ''),
            type="password",
            help="Ingresa tu API key de Scopus/Elsevier"
        )
        
        # Opción para usar una API key por defecto (la que funciona en Colab)
        use_default_key = st.checkbox(
            "Usar API key de ejemplo", 
            value=True, 
            help="Usa una API key de ejemplo para probar la aplicación"
        )
        
        if use_default_key:
            api_key = "113f57bcfb9e922c5a33ec02233ee24d"  # API key que funciona en Colab
        else:
            api_key = api_key_input
        
        st.session_state.scopus_api_key = api_key
        
        # Opciones para tipos de datos
        st.subheader("Tipos de datos")
        analyze_papers = st.checkbox("Analizar papers", value=True)
        analyze_patents = st.checkbox("Analizar patentes", value=True)
    
    # Pestañas para diferentes métodos de construcción de consulta
    tab1, tab2 = st.tabs(["Generador de Ecuaciones", "Ecuación Manual"])
    
    with tab1:
        # Utilizar el constructor de ecuaciones Scopus
        scopus_query = scopus_equation_interface()
    
    with tab2:
        # Entrada manual de ecuación
        manual_query = st.text_area(
            "Ecuación de búsqueda",
            placeholder='Ej: TITLE("Plantain" OR "banana" OR "musa") AND TITLE("flour" OR "starch")',
            height=100,
            key="manual_query_input"
        )
        scopus_query = manual_query if manual_query else ""
    
    # Número de resultados a recuperar para papers
    max_results = st.slider(
        "Número máximo de resultados a recuperar (para papers)",
        min_value=10,
        max_value=5000,
        value=1000,
        step=10,
        help="Mayor número = análisis más completo, pero toma más tiempo"
    )
    
    # Sección para patentes - Gestión de plantilla y carga de archivos
    if analyze_patents:
        st.write("## 📑 Datos de Patentes")
        st.write("""
        Para analizar datos de patentes, debes completar una plantilla Excel con los datos por año.
        Descarga la plantilla, complétala con tus datos y súbela a continuación.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Opciones para la plantilla
            st.write("### 📋 Opciones de Plantilla")
            
            start_year = st.number_input(
                "Año inicial", 
                min_value=1900,
                max_value=datetime.now().year - 5,
                value=1990,
                help="El primer año para incluir en la plantilla"
            )
            
            end_year = st.number_input(
                "Año final", 
                min_value=start_year + 5,
                max_value=datetime.now().year,
                value=datetime.now().year,
                help="El último año para incluir en la plantilla"
            )
            
            # Generar y descargar plantilla
            template_bytes = PatentDataManager.create_template(start_year, end_year)
            
            st.download_button(
                label="📥 Descargar Plantilla Excel",
                data=template_bytes,
                file_name=f"plantilla_patentes_{start_year}-{end_year}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Descarga una plantilla Excel para completar con datos de patentes"
            )
        
        with col2:
            # Subir archivo con datos de patentes
            st.write("### 📤 Cargar Datos de Patentes")
            
            uploaded_file = st.file_uploader(
                "Selecciona archivo Excel con datos de patentes",
                type=["xlsx", "xls"],
                help="Sube el archivo Excel completado con datos de patentes por año"
            )
            
            # Si hay un archivo cargado, guardarlo en session_state
            if uploaded_file is not None:
                # Guardar los datos en session_state para mantenerlos entre recargas
                st.session_state.patent_data_file = uploaded_file
                st.success("✅ Archivo cargado correctamente")
            
            # Opción para usar datos de ejemplo cuando no hay archivo
            use_sample_data = st.checkbox(
                "Usar datos de ejemplo", 
                value=not bool(uploaded_file),
                help="Usa datos de ejemplo para probar la aplicación"
            )
    
    # Botón de búsqueda y análisis
    search_button = st.button(
        "🔍 Analizar",
        type="primary", 
        use_container_width=True,
        disabled=not api_key or not scopus_query
    )
    
    # Ejecutar análisis cuando se presiona el botón
    if search_button:
        # Contenedor para los resultados
        results_container = st.container()
        
        with results_container:
            # Determinar qué tipo de análisis realizar
            if not analyze_papers and not analyze_patents:
                st.warning("Selecciona al menos un tipo de datos para analizar (papers o patentes).")
                return
            
            if analyze_papers:
                st.write("## 📚 Análisis de Publicaciones Académicas (Papers)")
                
                with st.spinner("Analizando publicaciones académicas..."):
                    # Instanciar el analizador de papers
                    papers_analyzer = ScopusPublicationsAnalyzer(api_key)
                    
                    # Buscar papers
                    papers_results = papers_analyzer.search_publications(
                        scopus_query,
                        max_results=max_results
                    )
                    
                    # Verificar si se obtuvieron resultados
                    if papers_results:
                        # Categorizar por año
                        papers_by_year = papers_analyzer.categorize_by_year(papers_results)
                        
                        # Verificar si hay datos disponibles
                        if papers_by_year:
                            # Mostrar tabla de papers por año
                            df_papers = TechnologyAnalyzer.display_data_table(
                                papers_by_year, 
                                title="Tabla de Papers por Año"
                            )
                            
                            # Realizar análisis de curva en S
                            analysis_df, analysis_fig, ajuste_info, parametros = TechnologyAnalyzer.analyze_s_curve(papers_by_year)
                            
                            # Mostrar análisis de curva en S
                            TechnologyAnalyzer.display_s_curve_analysis(
                                analysis_df, 
                                analysis_fig, 
                                ajuste_info, 
                                parametros,
                                title="Análisis de Curva en S - Papers"
                            )
                            
                            # Exportar datos
                            TechnologyAnalyzer.export_data(
                                analysis_df if analysis_df is not None else df_papers,
                                "papers",
                                scopus_query
                            )
                        else:
                            st.warning("No se pudieron categorizar los papers por año.")
                    else:
                        st.error("No se pudieron obtener resultados de papers. Por favor, verifica tu API key y la ecuación de búsqueda.")
            
            if analyze_patents:
                st.write("## 📑 Análisis de Patentes")
                
                with st.spinner("Analizando datos de patentes..."):
                    # Obtener datos de patentes (cargados o de ejemplo)
                    if uploaded_file is not None:
                        # Cargar datos desde el archivo subido
                        patents_by_year = PatentDataManager.load_data(uploaded_file)
                    elif use_sample_data:
                        # Generar datos de ejemplo basados en la consulta
                        patents_by_year = generate_sample_patent_data(scopus_query)
                    else:
                        st.warning("⚠️ No se han cargado datos de patentes y no se están usando datos de ejemplo.")
                        patents_by_year = None
                    
                    # Verificar si hay datos disponibles
                    if patents_by_year:
                        # Mostrar tabla de patentes por año
                        df_patents = TechnologyAnalyzer.display_data_table(
                            patents_by_year, 
                            title="Tabla de Patentes por Año"
                        )
                        
                        # Realizar análisis de curva en S
                        analysis_df, analysis_fig, ajuste_info, parametros = TechnologyAnalyzer.analyze_s_curve(patents_by_year)
                        
                        # Mostrar análisis de curva en S
                        TechnologyAnalyzer.display_s_curve_analysis(
                            analysis_df, 
                            analysis_fig, 
                            ajuste_info, 
                            parametros,
                            title="Análisis de Curva en S - Patentes"
                        )
                        
                        # Exportar datos
                        TechnologyAnalyzer.export_data(
                            analysis_df if analysis_df is not None else df_patents,
                            "patentes",
                            scopus_query
                        )
                    else:
                        st.warning("No hay datos de patentes para analizar. Por favor, carga un archivo o activa los datos de ejemplo.")
            
            # Si se analizaron ambos tipos de datos, mostrar comparación
            if analyze_papers and analyze_patents and papers_by_year and patents_by_year:
                st.write("## 🔄 Comparación Papers vs Patentes")
                
                with st.spinner("Generando comparación..."):
                    # Crear DataFrame de comparación
                    compare_years = sorted(set(list(papers_by_year.keys()) + list(patents_by_year.keys())))
                    
                    compare_data = {
                        'Año': compare_years,
                        'Papers': [papers_by_year.get(year, 0) for year in compare_years],
                        'Patentes': [patents_by_year.get(year, 0) for year in compare_years]
                    }
                    
                    df_compare = pd.DataFrame(compare_data)
                    
                    # Calcular acumulados
                    df_compare['Papers Acumulados'] = df_compare['Papers'].cumsum()
                    df_compare['Patentes Acumuladas'] = df_compare['Patentes'].cumsum()
                    
                    # Mostrar tabla comparativa
                    st.write("### 📊 Tabla Comparativa")
                    st.dataframe(
                        df_compare,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Crear gráfico comparativo
                    fig_compare = go.Figure()
                    
                    # Añadir líneas para papers y patentes (acumulados)
                    fig_compare.add_trace(go.Scatter(
                        x=df_compare['Año'],
                        y=df_compare['Papers Acumulados'],
                        mode='lines+markers',
                        name='Papers (Acumulados)',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig_compare.add_trace(go.Scatter(
                        x=df_compare['Año'],
                        y=df_compare['Patentes Acumuladas'],
                        mode='lines+markers',
                        name='Patentes (Acumuladas)',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Mejorar aspecto visual
                    fig_compare.update_layout(
                        title="Comparación de Curvas S: Papers vs Patentes",
                        xaxis_title="Año",
                        yaxis_title="Cantidad Acumulada",
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                        plot_bgcolor='white',
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )
                    
                    # Mostrar gráfico
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    # Calcular y mostrar brecha temporal
                    st.write("### 🕰️ Análisis de Brecha Temporal (Time Lag)")
                    
                    # Encontrar puntos de inflexión para ambas curvas (si existen)
                    papers_df, _, papers_info, _ = TechnologyAnalyzer.analyze_s_curve(papers_by_year)
                    patents_df, _, patents_info, _ = TechnologyAnalyzer.analyze_s_curve(patents_by_year)
                    
                    if papers_info and patents_info and 'x0' in papers_info and 'x0' in patents_info:
                        # Calcular brecha entre puntos de inflexión
                        time_lag = patents_info['x0'] - papers_info['x0']
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Punto de inflexión Papers", 
                                f"{papers_info['x0']:.1f}",
                                help="Año estimado del punto de inflexión para publicaciones académicas"
                            )
                        
                        with col2:
                            st.metric(
                                "Punto de inflexión Patentes", 
                                f"{patents_info['x0']:.1f}",
                                help="Año estimado del punto de inflexión para patentes"
                            )
                        
                        with col3:
                            st.metric(
                                "Time Lag (años)", 
                                f"{time_lag:.1f}",
                                delta=None,
                                help="Diferencia temporal entre los puntos de inflexión de patentes y papers"
                            )
                        
                        # Interpretación del time lag
                        if time_lag > 0:
                            st.info(f"Las patentes muestran un retraso de aproximadamente {time_lag:.1f} años respecto a las publicaciones académicas. Esto sugiere que la investigación académica ha precedido a la comercialización de esta tecnología.")
                        elif time_lag < 0:
                            st.info(f"Las patentes muestran un adelanto de aproximadamente {abs(time_lag):.1f} años respecto a las publicaciones académicas. Esto sugiere que la comercialización ha precedido a la investigación académica profunda en esta tecnología.")
                        else:
                            st.info("No hay brecha temporal significativa entre publicaciones académicas y patentes, sugiriendo un desarrollo paralelo entre investigación y comercialización.")
            
            # Mostrar explicación metodológica
            with st.expander("Metodología del análisis de curva en S"):
                st.markdown("""
                ### Metodología del análisis de curva en S
                
                Este análisis utiliza la siguiente metodología:
                
                1. **Recopilación de datos**: 
                   - Papers: Se obtienen datos de publicaciones académicas desde Scopus.
                   - Patentes: Se cargan datos manualmente desde un archivo Excel.
                
                2. **Cálculo de acumulado**: Se calcula el número acumulado por cada año.
                
                3. **Análisis matemático**:
                   - Se calcula la tasa de crecimiento anual 
                   - Se calcula la segunda derivada para identificar puntos de inflexión
                   - Se ajusta un modelo sigmoidal de tres parámetros a los datos acumulados
                
                4. **Modelo sigmoidal**: Se utiliza la función:
                   ```
                   f(x) = L / (1 + exp(-k * (x - x0)))
                   ```
                   Donde:
                   - L = valor máximo teórico (asíntota)
                   - x0 = punto medio (punto de inflexión)
                   - k = tasa de crecimiento
                
                5. **Identificación de la fase**: Se determina la fase actual de la tecnología basándose en la segunda derivada.
                
                6. **Cálculo de brecha temporal (time lag)**: Se compara la diferencia entre los puntos de inflexión de papers y patentes.
                """)
            
            # Opción para guardar los resultados en la base de datos
            if analyze_papers or analyze_patents:
                st.write("## 💾 Guardar Resultados")
                st.write("Puedes guardar este análisis para futuras comparaciones.")
                
                # Opciones para guardar
                save_expander = st.expander("Guardar este análisis", expanded=False)
                
                with save_expander:
                    # Inicializar sistema de base de datos
                    from data_storage import initialize_github_db
                    
                    # Usar almacenamiento local en lugar de GitHub
                    db = initialize_github_db(use_local=True)
                    
                    if db is None:
                        st.error("❌ No se pudo inicializar el sistema de almacenamiento.")
                    else:
                        # Formulario para guardar
                        with st.form("save_analysis_form"):
                            # Nombre del análisis
                            analysis_name = st.text_input(
                                "Nombre para este análisis",
                                value=f"Análisis de {scopus_query[:30]}..." if len(scopus_query) > 30 else f"Análisis de {scopus_query}"
                            )
                            
                            # Seleccionar categoría
                            categories = db.get_all_categories()
                            category_options = {cat["name"]: cat["id"] for cat in categories}
                            
                            selected_category = st.selectbox(
                                "Categoría",
                                options=list(category_options.keys()),
                                index=0
                            )
                            
                            selected_category_id = category_options[selected_category]
                            
                            # Opción para crear nueva categoría
                            new_category = st.checkbox("Crear nueva categoría")
                            
                            if new_category:
                                new_cat_name = st.text_input("Nombre de la nueva categoría")
                                new_cat_desc = st.text_input("Descripción (opcional)")
                            
                            # Botón para guardar
                            submit = st.form_submit_button("Guardar Análisis")
                            
                            if submit:
                                # Crear nueva categoría si es necesario
                                if new_category and new_cat_name:
                                    category_id = db.create_category(new_cat_name, new_cat_desc)
                                    if not category_id:
                                        st.error("❌ Error al crear la categoría.")
                                        st.stop()
                                else:
                                    category_id = selected_category_id
                                
                                # Preparar datos para guardar
                                paper_data = papers_by_year if analyze_papers and 'papers_by_year' in locals() else None
                                patent_data = patents_by_year if analyze_patents and 'patents_by_year' in locals() else None
                                
                                paper_metrics = ajuste_info if analyze_papers and 'ajuste_info' in locals() else None
                                patent_metrics = ajuste_info if analyze_patents and 'ajuste_info' in locals() else None
                                
                                # Guardar en la base de datos
                                analysis_id = db.save_s_curve_analysis(
                                    query=scopus_query,
                                    paper_data=paper_data,
                                    patent_data=patent_data,
                                    paper_metrics=paper_metrics,
                                    patent_metrics=patent_metrics,
                                    category_id=category_id,
                                    analysis_name=analysis_name
                                )
                                
                                if analysis_id:
                                    st.success(f"✅ Análisis guardado correctamente con ID: {analysis_id}")
                                    st.info("Puedes ver y comparar todos los análisis guardados en la pestaña 'Datos Guardados'.")
                                else:
                                    st.error("❌ Error al guardar el análisis.")
    else:
        # Mostrar instrucciones cuando no se ha realizado búsqueda
        st.info("""
        ### 🚀 Cómo comenzar:
        
        1. Selecciona los tipos de datos que deseas analizar (papers y/o patentes) en el panel lateral
        2. La API key de ejemplo está activada por defecto para papers
        3. Usa el generador de ecuaciones para construir tu consulta de búsqueda
        4. Si deseas analizar patentes, descarga la plantilla Excel, complétala y súbela
        5. Haz clic en "Analizar" para iniciar el análisis
        
        ### 📝 Consulta de ejemplo:
        
        TITLE("banana") AND TITLE("flour")
        """)


def generate_sample_patent_data(query, start_year=1990, end_year=None):
    """
    Genera datos de muestra para patentes basados en la consulta de búsqueda.
    Útil para demostración cuando no hay datos reales disponibles.
    
    Args:
        query (str): Consulta de búsqueda para extraer términos relevantes
        start_year (int): Año inicial para los datos
        end_year (int): Año final para los datos (por defecto es el año actual)
        
    Returns:
        dict: Diccionario con datos de patentes por año {año: conteo}
    """
    if end_year is None:
        end_year = datetime.now().year
    
    # Extraer términos de búsqueda para hacer la demostración más realista
    search_terms = []
    for term in re.findall(r'"([^"]+)"', query):
        search_terms.append(term)
    
    if not search_terms and query:
        # Si no hay términos entre comillas, tomar palabras individuales
        search_terms = re.findall(r'\b\w+\b', query)
    
    # Crear semilla para reproducibilidad basada en los términos de búsqueda
    seed = sum(ord(c) for c in "".join(search_terms)) if search_terms else 42
    np.random.seed(seed)
    
    # Generar años
    years = list(range(start_year, end_year + 1))
    
    # Generar conteos con tendencia creciente para patentes (curva típica)
    # Patrones comunes en datos de patentes: crecimiento lento inicial, aceleración, desaceleración
    x = np.linspace(0, 1, len(years))
    
    # Crear curva sigmoidal modificada con inicio más lento
    base_values = 1000 * (1 / (1 + np.exp(-12 * (x - 0.6))))
    
    # Añadir variación aleatoria
    noise = np.random.normal(0, 0.1, len(years))
    values_with_noise = base_values * (1 + noise)
    
    # Convertir a enteros y asegurar no negativos
    patent_counts = np.maximum(0, np.round(values_with_noise)).astype(int)
    
    # Crear diccionario {año: conteo}
    patents_by_year = dict(zip(years, patent_counts))
    
    # Añadir tendencia específica según términos de búsqueda
    # (Por ejemplo, términos más recientes tendrán más patentes en años recientes)
    recency_factor = 0.5  # Peso para la preferencia por términos recientes
    if search_terms and len(search_terms) > 1:
        for i, term in enumerate(search_terms):
            term_weight = (i / len(search_terms)) * recency_factor
            for year in years:
                year_factor = (year - start_year) / (end_year - start_year)
                boost = int(10 * term_weight * year_factor * np.random.random())
                patents_by_year[year] += boost
    
    return patents_by_year


if __name__ == "__main__":
    run_s_curve_analysis()