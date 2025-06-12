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
    def create_template(start_year=1920, end_year=None):
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
        Versión mejorada con validación de datos suficientes.
        
        Args:
            yearly_data: Diccionario con conteo por año {año: conteo}
            
        Returns:
            DataFrame con los datos analizados, figura, info de ajuste y parámetros
        """
        if not yearly_data:
            st.error("No hay datos para analizar")
            return None, None, None, None
        
        # Verificar que hay suficientes datos
        if len(yearly_data) < 3:
            st.warning(f"⚠️ Solo hay {len(yearly_data)} años de datos. Se necesitan al menos 3 años para un análisis completo.")
            # Crear DataFrame básico sin análisis avanzado
            df = pd.DataFrame({
                'Año': list(yearly_data.keys()),
                'Cantidad': list(yearly_data.values())
            })
            df['Acumulado'] = df['Cantidad'].cumsum()
            
            # Crear gráfico simple
            fig = px.line(
                df, x='Año', y='Acumulado',
                title='Datos Acumulados por Año (Análisis Básico)',
                markers=True
            )
            fig.update_layout(
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
            # Info básica
            basic_info = {
                'total_publications': sum(yearly_data.values()),
                'years_span': len(yearly_data),
                'avg_per_year': sum(yearly_data.values()) / len(yearly_data),
                'note': 'Datos insuficientes para análisis completo de curva S'
            }
            
            return df, fig, basic_info, None
        
        # Crear DataFrame base (solo si hay suficientes datos)
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
        
        # Calcular segunda derivada SOLO si hay suficientes datos
        if len(df_analysis) >= 3:
            try:
                df_analysis['SegundaDerivada'] = np.gradient(np.gradient(df_analysis['Acumulado']))
            except ValueError as e:
                st.warning(f"⚠️ No se pudo calcular la segunda derivada: {str(e)}")
                df_analysis['SegundaDerivada'] = 0
        else:
            df_analysis['SegundaDerivada'] = 0
        
        # Encontrar los años de puntos de inflexión exactos
        try:
            if len(df_analysis) >= 3 and 'SegundaDerivada' in df_analysis.columns:
                puntos_inflexion_exacto = df_analysis[df_analysis['SegundaDerivada'] == 0].index.tolist()
                if not puntos_inflexion_exacto and len(df_analysis) > 0:
                    # Si no hay puntos exactos con segunda derivada = 0, encontrar el más cercano
                    punto_cercano_idx = np.abs(df_analysis['SegundaDerivada']).idxmin()
                    puntos_inflexion_exacto = [punto_cercano_idx]
            else:
                puntos_inflexion_exacto = []
        except Exception as e:
            st.warning(f"⚠️ Error calculando puntos de inflexión: {str(e)}")
            puntos_inflexion_exacto = []
        
        # Preparar datos para el ajuste
        x_data = np.array(df_analysis.index)
        y_data = np.array(df_analysis['Acumulado'])
        
        # Ajustar la curva sigmoidal a los datos SOLO si hay suficientes puntos
        if len(x_data) >= 4:  # Mínimo 4 puntos para ajuste sigmoidal
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
                if len(df_analysis) >= 3:
                    ultima_derivada = df_analysis['SegundaDerivada'].iloc[-3:].mean() if len(df_analysis) >= 3 else 0
                else:
                    ultima_derivada = 0
                
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
                
            except Exception as e:
                st.warning(f"⚠️ Error en el ajuste de curva sigmoidal: {str(e)}")
                # Crear gráfico simple sin ajuste
                fig = px.line(
                    df_analysis, x=df_analysis.index, y='Acumulado',
                    title='Datos Acumulados por Año (Sin Ajuste)',
                    markers=True
                )
                
                # Info básica sin ajuste
                ajuste_info = {
                    'total_publications': sum(yearly_data.values()),
                    'years_span': len(yearly_data),
                    'avg_per_year': sum(yearly_data.values()) / len(yearly_data),
                    'note': 'No se pudo realizar ajuste sigmoidal'
                }
                df_parametros = None
        else:
            # No hay suficientes datos para ajuste sigmoidal
            st.warning(f"⚠️ Solo hay {len(x_data)} puntos de datos. Se necesitan al menos 4 para ajuste sigmoidal.")
            
            # Crear gráfico simple
            fig = px.line(
                df_analysis, x=df_analysis.index, y='Acumulado',
                title='Datos Acumulados por Año (Análisis Básico)',
                markers=True
            )
            
            # Info básica
            ajuste_info = {
                'total_publications': sum(yearly_data.values()),
                'years_span': len(yearly_data),
                'avg_per_year': sum(yearly_data.values()) / len(yearly_data),
                'note': 'Datos insuficientes para ajuste sigmoidal'
            }
            df_parametros = None
        
        # Mejorar aspecto visual del gráfico
        fig.update_layout(
            width=600,
            height=600,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            plot_bgcolor='white',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        # Restaurar el índice para el DataFrame original
        df_analysis = df_analysis.reset_index()
        
        return df_analysis, fig, ajuste_info, df_parametros
    
    
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
                mime="text/csv",
                key=f"download_csv_{filename_prefix}_{int(time.time())}"  # ← KEY ÚNICO AGREGADO
            )


class GooglePatentsAnalyzer:
    """
    Clase para conectarse a Google Patents vía SerpAPI y analizar patentes por año
    para crear una curva en S.
    """
    
    def __init__(self, api_key=None):
        """
        Inicializa el analizador de patentes.
        
        Args:
            api_key: La clave API de SerpAPI.
        """
        self.api_key = api_key
        self.base_url = "https://serpapi.com/search"
        self.patents_by_year = {}
        
    def search_patents(self, query, max_results=1000, start_year=None, end_year=None):
        """
        Busca patentes en Google Patents utilizando búsquedas por año individual.
        Versión mejorada con mejor manejo de errores para evitar error 400.
        """
        # Construir query específica para patentes
        patents_query = self._build_patents_query(query)
        
        # Mostrar ecuación de búsqueda con debugging
        st.info(f"📝 Ecuación de búsqueda (Patentes): `{patents_query}`")
        st.caption(f"🔧 Query original Scopus: {query[:100]}...")
        
        # Determinar rango de años
        current_year = datetime.now().year
        if not start_year:
            start_year = current_year - 15
        if not end_year:
            end_year = current_year
        
        # Verificar que el rango sea válido
        if start_year > end_year:
            st.error("❌ El año inicial no puede ser mayor que el año final")
            return None
        
        total_years = end_year - start_year + 1
        st.info(f"🗓️ Analizando {total_years} años: {start_year} - {end_year}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        patents_by_year = {}
        total_patents_found = 0
        successful_years = 0
        failed_years = []
        
        try:
            # Hacer una búsqueda de prueba inicial MUY SIMPLE
            status_text.text("🔍 Verificando conectividad con Google Patents...")
            
            # Test con query ultra-simple para evitar error 400
            test_params = {
                "engine": "google_patents",
                "q": "patent",  # Query de prueba ultra-simple
                "api_key": self.api_key,
                "num": 10
            }
            
            response = requests.get(self.base_url, params=test_params, timeout=30)
            
            st.write(f"🔧 Debug - Test response status: {response.status_code}")
            
            if response.status_code != 200:
                st.error(f"❌ Error de conectividad: {response.status_code}")
                try:
                    error_data = response.json()
                    st.code(f"Error details: {error_data}")
                except:
                    st.code(f"Response content: {response.text}")
                return None
            
            test_data = response.json()
            if "error" in test_data:
                st.error(f"❌ Error de SerpAPI: {test_data['error']}")
                return None
            
            st.success("✅ Conectividad verificada. Iniciando búsqueda por años...")
            
            # Ahora probar con la query real
            status_text.text(f"🧪 Probando query real: {patents_query}")
            
            real_test_params = {
                "engine": "google_patents",
                "q": patents_query,
                "api_key": self.api_key,
                "num": 10
            }
            
            response = requests.get(self.base_url, params=real_test_params, timeout=30)
            
            if response.status_code != 200:
                st.warning(f"⚠️ Query original falló ({response.status_code}), usando query simple")
                patents_query = "technology patent"  # Fallback
                st.info(f"📝 Usando query simplificada: {patents_query}")
            else:
                test_data = response.json()
                if "error" in test_data:
                    st.warning(f"⚠️ Query original con error: {test_data['error']}")
                    patents_query = "technology patent"  # Fallback
                    st.info(f"📝 Usando query simplificada: {patents_query}")
            
            # Iterar por cada año
            for i, year in enumerate(range(start_year, end_year + 1)):
                # Actualizar progreso
                progress = (i + 1) / total_years
                progress_bar.progress(progress)
                status_text.text(f"📅 Analizando año {year}... ({i+1}/{total_years})")
                
                try:
                    # Construir query específica para este año
                    year_query = f"{patents_query} after:{year-1}-12-31 before:{year+1}-01-01"
                    
                    # Parámetros para búsqueda anual
                    year_params = {
                        "engine": "google_patents",
                        "q": year_query,
                        "api_key": self.api_key,
                        "num": 10  # Solo necesitamos el total_results, no los datos
                    }
                    
                    # Hacer request para este año
                    response = requests.get(self.base_url, params=year_params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if "error" in data:
                            st.warning(f"⚠️ Error en año {year}: {data['error']}")
                            failed_years.append(year)
                            patents_by_year[year] = 0
                            continue
                        
                        # Extraer número total de resultados
                        search_info = data.get("search_information", {})
                        year_total = search_info.get("total_results", 0)
                        
                        # Convertir a entero si viene como string
                        try:
                            year_total = int(str(year_total).replace(",", ""))
                        except (ValueError, AttributeError):
                            year_total = 0
                        
                        patents_by_year[year] = year_total
                        total_patents_found += year_total
                        successful_years += 1
                        
                        # Mostrar progreso detallado cada 5 años o si hay muchos resultados
                        if year % 5 == 0 or year_total > 1000:
                            status_text.text(f"📅 Año {year}: {year_total:,} patentes encontradas")
                    
                    else:
                        st.warning(f"⚠️ Error HTTP {response.status_code} en año {year}")
                        failed_years.append(year)
                        patents_by_year[year] = 0
                    
                    # Pausa entre requests para respetar rate limits
                    time.sleep(0.3)  # 300ms entre años
                    
                except Exception as year_error:
                    st.warning(f"⚠️ Error procesando año {year}: {str(year_error)}")
                    failed_years.append(year)
                    patents_by_year[year] = 0
                    continue
            
            # Limpiar indicadores de progreso
            progress_bar.empty()
            
            # Mostrar resumen final
            if successful_years > 0:
                status_text.text(f"✅ Búsqueda completada: {total_patents_found:,} patentes en {successful_years} años")
                
                # Mostrar estadísticas detalladas
                with st.expander("📊 Estadísticas Detalladas", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total de Patentes", f"{total_patents_found:,}")
                        st.metric("Años Analizados", f"{successful_years}/{total_years}")
                    
                    with col2:
                        if patents_by_year:
                            max_year = max(patents_by_year, key=patents_by_year.get)
                            max_count = patents_by_year[max_year]
                            st.metric("Año Pico", f"{max_year}")
                            st.metric("Patentes en Pico", f"{max_count:,}")
                    
                    with col3:
                        avg_per_year = total_patents_found / successful_years if successful_years > 0 else 0
                        st.metric("Promedio Anual", f"{avg_per_year:.0f}")
                        if failed_years:
                            st.metric("Años con Error", len(failed_years))
                    
                    # Mostrar años con errores si los hay
                    if failed_years:
                        st.warning(f"⚠️ Años con errores: {', '.join(map(str, failed_years))}")
                    
                    # Mostrar top 5 años
                    if patents_by_year:
                        top_years = sorted(patents_by_year.items(), key=lambda x: x[1], reverse=True)[:5]
                        st.write("**Top 5 años con más patentes:**")
                        for year, count in top_years:
                            st.write(f"- {year}: {count:,} patentes")
                
                return patents_by_year
            else:
                status_text.text("❌ No se pudieron obtener datos de ningún año")
                return None
                
        except Exception as e:
            st.error(f"❌ Error general en búsqueda por años: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return None
        
        finally:
            # Limpiar elementos de UI
            try:
                progress_bar.empty()
                status_text.empty()
            except:
                pass

    def get_patents_summary_stats(self, query):
        """
        Obtiene estadísticas rápidas sin análisis completo.
        Útil para verificar si vale la pena hacer el análisis completo.
        
        Args:
            query: Consulta de búsqueda
            
        Returns:
            Dict con estadísticas básicas
        """
        try:
            patents_query = self._build_patents_query(query)
            
            # Búsqueda general para obtener total
            params = {
                "engine": "google_patents",
                "q": patents_query,
                "api_key": self.api_key,
                "num": 10
            }
            
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if "error" not in data:
                    search_info = data.get("search_information", {})
                    total_results = search_info.get("total_results", 0)
                    
                    # Convertir a entero
                    try:
                        total_results = int(str(total_results).replace(",", ""))
                    except:
                        total_results = 0
                    
                    return {
                        "total_patents": total_results,
                        "query_used": patents_query,
                        "worth_analyzing": total_results > 10,
                        "estimated_time": f"{total_results // 1000 * 2} segundos" if total_results > 1000 else "< 30 segundos"
                    }
            
            return None
            
        except Exception as e:
            st.error(f"Error obteniendo estadísticas: {str(e)}")
            return None
    
    def _build_patents_query(self, scopus_query):
        """
        Convierte una ecuación de Scopus a formato SIMPLE compatible con Google Patents.
        Implementación mejorada para evitar error 400.
        
        Args:
            scopus_query: Query en formato Scopus
            
        Returns:
            str: Query simplificada para Google Patents
        """
        try:
            # Paso 1: Extraer términos principales entre comillas
            quoted_terms = re.findall(r'"([^"]+)"', scopus_query)
            
            # Paso 2: Si no hay términos entre comillas, extraer de campos Scopus
            if not quoted_terms:
                # Extraer de campos TITLE, ABS, KEY
                title_terms = re.findall(r'TITLE\(([^)]+)\)', scopus_query)
                abs_terms = re.findall(r'ABS\(([^)]+)\)', scopus_query)
                key_terms = re.findall(r'KEY\(([^)]+)\)', scopus_query)
                titleabs_terms = re.findall(r'TITLE-ABS-KEY\(([^)]+)\)', scopus_query)
                
                # Combinar todos los términos encontrados
                all_terms = title_terms + abs_terms + key_terms + titleabs_terms
                
                # Limpiar términos (quitar comillas si las tienen)
                for term in all_terms:
                    clean_term = term.strip().strip('"').strip("'")
                    if clean_term and len(clean_term) > 2:
                        quoted_terms.append(clean_term)
            
            # Paso 3: Si aún no hay términos, extraer palabras importantes
            if not quoted_terms:
                # Limpiar la query de campos Scopus
                clean_query = re.sub(r'(TITLE|ABS|KEY|TITLE-ABS-KEY|AUTH|AFFIL|PUBYEAR|DOCTYPE|SUBJAREA)\([^)]*\)', '', scopus_query)
                clean_query = re.sub(r'[()><=]', ' ', clean_query)
                clean_query = re.sub(r'\b(AND|OR|NOT)\b', ' ', clean_query)
                
                # Extraer palabras significativas (3+ caracteres, solo letras)
                words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_query)
                
                # Filtrar palabras muy comunes
                stop_words = {
                    'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'way', 'will', 'with', 'been', 'call', 'come', 'each', 'find', 'have', 'here', 'into', 'just', 'like', 'long', 'look', 'made', 'make', 'many', 'more', 'over', 'part', 'said', 'than', 'that', 'them', 'they', 'this', 'time', 'very', 'well', 'were', 'what', 'when', 'word', 'work', 'your'
                }
                
                # Mantener solo palabras relevantes
                quoted_terms = [word for word in words if word.lower() not in stop_words and len(word) >= 4]
            
            # Paso 4: Limitar a máximo 2 términos principales para evitar complejidad
            if len(quoted_terms) > 2:
                # Priorizar términos más largos (más específicos)
                quoted_terms = sorted(quoted_terms, key=len, reverse=True)[:2]
            
            # Paso 5: Construir query ultra-simple para Google Patents
            if len(quoted_terms) == 0:
                # Fallback absoluto
                simple_query = "technology"
            elif len(quoted_terms) == 1:
                # Un solo término entre comillas
                simple_query = f'"{quoted_terms[0]}"'
            else:
                # Máximo dos términos con AND simple
                simple_query = f'"{quoted_terms[0]}" AND "{quoted_terms[1]}"'
            
            # Paso 6: NO agregar filtros temporales aquí (se manejan por año individualmente)
            # Esto evita queries complejas que causan error 400
            
            # Paso 7: Validar longitud final (Google Patents tiene límite ~80 caracteres)
            if len(simple_query) > 80:
                # Usar solo el primer término si es muy larga
                if quoted_terms:
                    simple_query = f'"{quoted_terms[0]}"'
                else:
                    simple_query = "patent"
            
            return simple_query
            
        except Exception as e:
            st.warning(f"Error simplificando query: {str(e)}")
            # Fallback de emergencia ultra-simple
            return "technology"

    def _extract_core_concept(self, scopus_query):
        """
        Extrae el concepto central de una query compleja de Scopus.
        
        Args:
            scopus_query: Query original de Scopus
            
        Returns:
            str: Concepto central identificado
        """
        # Patrones para identificar conceptos tecnológicos centrales
        tech_patterns = [
            r'"([^"]*(?:AI|artificial intelligence|machine learning)[^"]*)"',
            r'"([^"]*(?:blockchain|crypto|bitcoin)[^"]*)"',
            r'"([^"]*(?:quantum|qubit)[^"]*)"',
            r'"([^"]*(?:CRISPR|gene editing)[^"]*)"',
            r'"([^"]*(?:graphene|nanotube)[^"]*)"',
            r'"([^"]*(?:solar|photovoltaic|battery)[^"]*)"',
            r'"([^"]*(?:robot|automation)[^"]*)"',
            r'"([^"]*(?:biotech|pharmaceutical)[^"]*)"'
        ]
        
        for pattern in tech_patterns:
            match = re.search(pattern, scopus_query, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # Si no encuentra patrones específicos, usar el primer término entre comillas
        quoted = re.findall(r'"([^"]+)"', scopus_query)
        if quoted:
            return quoted[0]
        
        return None
    
    def categorize_by_year(self, results):
        """
        Categoriza las patentes por año y completa los años faltantes con ceros.
        
        Args:
            results: Lista de resultados de la búsqueda
            
        Returns:
            Diccionario con el recuento por año
        """
        patents_by_year = {}
        
        st.text("📅 Categorizando patentes por año...")
        
        for entry in results:
            year = None
            
            # Método 1: publication_date (fecha de publicación)
            if "publication_date" in entry:
                try:
                    date_str = entry["publication_date"]
                    # Puede venir como "2023-05-15" o "May 15, 2023"
                    year = self._extract_year_from_date_string(date_str)
                except (ValueError, TypeError):
                    pass
            
            # Método 2: patent_date (fecha de la patente)
            if not year and "patent_date" in entry:
                try:
                    date_str = entry["patent_date"]
                    year = self._extract_year_from_date_string(date_str)
                except (ValueError, TypeError):
                    pass
            
            # Método 3: Buscar año en el título o snippet
            if not year:
                text_to_search = f"{entry.get('title', '')} {entry.get('snippet', '')}"
                year = self._extract_year_from_text(text_to_search)
            
            # Método 4: filing_date como último recurso
            if not year and "filing_date" in entry:
                try:
                    date_str = entry["filing_date"]
                    year = self._extract_year_from_date_string(date_str)
                except (ValueError, TypeError):
                    pass
            
            # Si se encontró un año válido, contabilizarlo
            if year and 1900 <= year <= datetime.now().year:
                patents_by_year[year] = patents_by_year.get(year, 0) + 1
        
        # Completar años faltantes con ceros
        if patents_by_year:
            min_year = min(patents_by_year.keys())
            max_year = max(patents_by_year.keys())
            
            # Crear rango completo de años
            all_years = list(range(min_year, max_year + 1))
            
            # Completar con ceros los años faltantes
            complete_patents = {year: patents_by_year.get(year, 0) for year in all_years}
            
            # Ordenar por año
            self.patents_by_year = dict(sorted(complete_patents.items()))
        else:
            self.patents_by_year = {}
            
        return self.patents_by_year
    
    def _extract_year_from_date_string(self, date_str):
        """
        Extrae el año de una cadena de fecha en varios formatos.
        
        Args:
            date_str: Cadena de fecha
            
        Returns:
            int: Año extraído o None si no se puede extraer
        """
        if not date_str:
            return None
        
        # Formatos comunes de fecha
        date_formats = [
            '%Y-%m-%d',           # 2023-05-15
            '%Y/%m/%d',           # 2023/05/15
            '%m/%d/%Y',           # 05/15/2023
            '%B %d, %Y',          # May 15, 2023
            '%b %d, %Y',          # May 15, 2023
            '%Y',                 # 2023
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt).year
            except ValueError:
                continue
        
        # Si no funciona ningún formato, usar regex
        return self._extract_year_from_text(date_str)
    
    def _extract_year_from_text(self, text):
        """
        Extrae año de texto usando regex.
        
        Args:
            text: Texto donde buscar el año
            
        Returns:
            int: Año encontrado o None
        """
        if not text:
            return None
        
        # Buscar años entre 1900 y año actual
        current_year = datetime.now().year
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', text)
        
        if years:
            valid_years = [int(y) for y in years if 1900 <= int(y) <= current_year]
            if valid_years:
                # Retornar el año más reciente si hay múltiples
                return max(valid_years)
        
        return None
    

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
    
    # Configuración de APIs en el sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        api_key_input = st.text_input(
            "API Key de Scopus",
            value=st.session_state.get('scopus_api_key', ''),
            type="password",
            help="Ingresa tu API key de Scopus/Elsevier"
        )
        
        use_default_key = st.checkbox(
            "Usar API key de ejemplo", 
            value=True, 
            help="Usa una API key de ejemplo para probar la aplicación"
        )
        
        if use_default_key:
            api_key = "113f57bcfb9e922c5a33ec02233ee24d"
        else:
            api_key = api_key_input
        
        st.session_state.scopus_api_key = api_key
        
        # Verificar SerpAPI para patentes
        serp_api_available = bool(st.session_state.get('serp_api_key'))
        if serp_api_available:
            st.success("✅ SerpAPI configurado para patentes")
        else:
            st.warning("⚠️ Configura SerpAPI para buscar patentes")
        
        # Opciones para tipos de datos
        st.subheader("Tipos de datos")
        analyze_papers = st.checkbox("Analizar papers", value=True)
        analyze_patents = st.checkbox("Analizar patentes", value=serp_api_available)
    
    # PESTAÑAS PRINCIPALES
    tab1, tab2 = st.tabs(["🔍 Análisis Automático", "📤 Carga Manual (Excel)"])
    
    # =================================================================
    # PESTAÑA 1: ANÁLISIS AUTOMÁTICO
    # =================================================================
    with tab1:
        st.header("🔍 Análisis Automático")
        st.write("Busca automáticamente papers en Scopus y patentes en Google Patents")
        
        # Pestañas para diferentes métodos de construcción de consulta
        query_tab1, query_tab2 = st.tabs(["Generador de Ecuaciones", "Ecuación Manual"])
        
        with query_tab1:
            # Utilizar el constructor de ecuaciones Scopus
            scopus_query = scopus_equation_interface()
        
        with query_tab2:
            # Entrada manual de ecuación
            manual_query = st.text_area(
                "Ecuación de búsqueda",
                placeholder='Ej: TITLE("Plantain" OR "banana" OR "musa") AND TITLE("flour" OR "starch")',
                height=100,
                key="manual_query_input_auto"
            )
            scopus_query = manual_query if manual_query else ""
        
        # Configuración de búsqueda
        with st.expander("⚙️ Configuración de Búsqueda", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Papers (Scopus):**")
                max_results = st.slider(
                    "Número máximo de resultados",
                    min_value=500,
                    max_value=10000,
                    value=2000,
                    step=100,
                    help="Mayor número = análisis más completo, pero toma más tiempo"
                )
            
            with col2:
                st.write("**Patentes (Google Patents):**")
                col2a, col2b = st.columns(2)
                
                with col2a:
                    patents_start_year = st.number_input(
                        "Año inicial", 
                        min_value=1950, 
                        max_value=datetime.now().year-1,
                        value=datetime.now().year - 20,
                        key="patents_start_year_main"
                    )
                    # Guardar en session_state
                    st.session_state['patents_start_year_config'] = patents_start_year
                
                with col2b:
                    patents_end_year = st.number_input(
                        "Año final", 
                        min_value=patents_start_year,
                        max_value=datetime.now().year,
                        value=datetime.now().year,
                        key="patents_end_year_main"
                    )
                    # Guardar en session_state
                    st.session_state['patents_end_year_config'] = patents_end_year
                
                years_span = patents_end_year - patents_start_year + 1
                st.caption(f"📡 {years_span} requests a SerpAPI (uno por año)")
        
        # BOTÓN PRINCIPAL DE ANÁLISIS
        if st.button("🔍 Analizar Tecnología", type="primary", use_container_width=True):
            if not scopus_query:
                st.error("Por favor, ingresa una ecuación de búsqueda válida")
                return
            
            if not analyze_papers and not analyze_patents:
                st.warning("Selecciona al menos un tipo de datos para analizar (papers o patentes).")
                return
            
            # Variables para almacenar resultados
            papers_by_year = None
            patents_by_year = None
            papers_analysis = None
            patents_analysis = None
            
            # =============================================================
            # 1. ANÁLISIS DE PAPERS
            # =============================================================
            if analyze_papers:
                st.write("## 📚 Análisis de Publicaciones Académicas (Papers)")
                
                if not api_key:
                    st.error("⚠️ Se requiere API Key de Scopus para analizar papers")
                else:
                    with st.spinner("🔄 Analizando publicaciones académicas..."):
                        try:
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
                                    
                                    # Guardar para comparación posterior
                                    papers_analysis = {
                                        'df': analysis_df if analysis_df is not None else df_papers,
                                        'metrics': ajuste_info
                                    }
                                    
                                    # Exportar datos
                                    TechnologyAnalyzer.export_data(
                                        analysis_df if analysis_df is not None else df_papers,
                                        "papers",
                                        scopus_query
                                    )
                                else:
                                    st.warning("No se pudieron categorizar los papers por año.")
                            else:
                                st.error("No se pudieron obtener resultados de papers. Verifica tu API key y la ecuación de búsqueda.")
                        
                        except Exception as e:
                            st.error(f"Error en el análisis de papers: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
            
            # =============================================================
            # 2. ANÁLISIS DE PATENTES - DIRECTO Y AUTOMÁTICO
            # =============================================================
            if analyze_patents:
                st.write("## 📑 Análisis de Patentes")
                
                if not st.session_state.get('serp_api_key'):
                    st.error("⚠️ Se requiere SerpAPI Key para analizar patentes. Configúrala en el panel lateral.")
                    patents_by_year = None
                    patents_analysis = None
                else:
                    with st.spinner("🔄 Analizando patentes con Google Patents..."):
                        try:
                            # Instanciar el analizador de patentes
                            patents_analyzer = GooglePatentsAnalyzer(st.session_state.serp_api_key)
                            
                            # Obtener rango de años desde la configuración (agregada abajo)
                            patents_start_year = st.session_state.get('patents_start_year_config', datetime.now().year - 15)
                            patents_end_year = st.session_state.get('patents_end_year_config', datetime.now().year)
                            
                            # Ejecutar búsqueda por años
                            patents_by_year = patents_analyzer.search_patents(
                                scopus_query,
                                max_results=max_results,
                                start_year=patents_start_year,
                                end_year=patents_end_year
                            )
                            
                            # Verificar si se obtuvieron resultados
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
                                
                                # Guardar para comparación posterior
                                patents_analysis = {
                                    'df': analysis_df if analysis_df is not None else df_patents,
                                    'metrics': ajuste_info
                                }
                                
                                # Exportar datos
                                TechnologyAnalyzer.export_data(
                                    analysis_df if analysis_df is not None else df_patents,
                                    "patentes",
                                    scopus_query
                                )
                            else:
                                st.warning("No se pudieron obtener resultados de patentes.")
                                patents_by_year = None
                                patents_analysis = None
                        
                        except Exception as e:
                            st.error(f"Error en el análisis de patentes: {str(e)}")
                            import traceback
                            st.code(traceback.format_exc())
                            patents_by_year = None
                            patents_analysis = None
                                
            # =============================================================
            # 3. COMPARACIÓN PAPERS vs PATENTES
            # =============================================================
            if papers_by_year and patents_by_year:
                st.write("## 🔄 Comparación Papers vs Patentes")
                
                with st.spinner("📊 Generando comparación..."):
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
                    st.dataframe(df_compare, use_container_width=True, hide_index=True)
                    
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
                    
                    # Configurar aspecto visual
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
                    
                    # =============================================================
                    # 4. ANÁLISIS DE BRECHA TEMPORAL (TIME LAG)
                    # =============================================================
                    st.write("### 🕰️ Análisis de Brecha Temporal (Time Lag)")
                    
                    if (papers_analysis and patents_analysis and 
                        papers_analysis['metrics'] and patents_analysis['metrics'] and
                        'x0' in papers_analysis['metrics'] and 'x0' in patents_analysis['metrics']):
                        
                        # Calcular brecha entre puntos de inflexión
                        papers_inflection = papers_analysis['metrics']['x0']
                        patents_inflection = patents_analysis['metrics']['x0']
                        time_lag = patents_inflection - papers_inflection
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "Punto de inflexión Papers", 
                                f"{papers_inflection:.1f}",
                                help="Año estimado del punto de inflexión para publicaciones académicas"
                            )
                        
                        with col2:
                            st.metric(
                                "Punto de inflexión Patentes", 
                                f"{patents_inflection:.1f}",
                                help="Año estimado del punto de inflexión para patentes"
                            )
                        
                        with col3:
                            delta_color = "normal" if abs(time_lag) < 1 else ("inverse" if time_lag > 0 else "off")
                            st.metric(
                                "Time Lag (años)", 
                                f"{time_lag:.1f}",
                                delta=f"{'+' if time_lag > 0 else ''}{time_lag:.1f}",
                                help="Diferencia temporal entre los puntos de inflexión"
                            )
                        
                        # Interpretación del time lag
                        if time_lag > 2:
                            st.info(f"🔍 **Interpretación**: Las patentes muestran un retraso significativo de {time_lag:.1f} años respecto a las publicaciones académicas. Esto sugiere que la investigación académica ha precedido considerablemente a la comercialización de esta tecnología.")
                        elif time_lag < -2:
                            st.info(f"🔍 **Interpretación**: Las patentes muestran un adelanto significativo de {abs(time_lag):.1f} años respecto a las publicaciones académicas. Esto sugiere que la comercialización ha precedido a la investigación académica profunda en esta tecnología.")
                        else:
                            st.info("🔍 **Interpretación**: No hay brecha temporal significativa entre publicaciones académicas y patentes, sugiriendo un desarrollo relativamente paralelo entre investigación y comercialización.")
                    else:
                        st.warning("No se pudieron calcular los puntos de inflexión para ambas curvas.")
                
                # Exportar comparación completa
                st.write("### 📥 Exportar Comparación Completa")
                combined_csv = df_compare.to_csv(index=False)
                st.download_button(
                    label="📊 Descargar Comparación Completa (CSV)",
                    data=combined_csv,
                    file_name=f"comparacion_papers_patents_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="download_comparison_complete_csv"
                )
            
            # Opción para guardar análisis completo
            _show_save_analysis_option(scopus_query, papers_by_year, patents_by_year, papers_analysis, patents_analysis)
    
    # =================================================================
    # PESTAÑA 2: CARGA MANUAL (EXCEL)
    # =================================================================
    with tab2:
        st.header("📤 Carga Manual de Datos")
        st.write("""
        **Método auxiliar:** Si necesitas datos más específicos, de fuentes adicionales, 
        o quieres complementar los resultados automáticos, puedes usar plantillas Excel.
        """)
        
        # Subtabs para papers y patentes
        excel_tab1, excel_tab2 = st.tabs(["📄 Papers (Excel)", "📑 Patentes (Excel)"])
        
        with excel_tab1:
            st.write("### 📄 Carga Manual de Papers")
            st.info("Esta funcionalidad permite cargar datos de papers desde Excel si tienes fuentes adicionales.")
            st.write("*Próximamente: Plantilla para carga manual de papers*")
        
        with excel_tab2:
            st.write("### 📑 Carga Manual de Patentes")
            _show_excel_patents_interface()

# =================================================================
# FUNCIONES AUXILIARES
# =================================================================

def _show_save_analysis_option(query, papers_data, patents_data, papers_analysis, patents_analysis):
    """Muestra opciones para guardar el análisis completo"""
    if papers_data or patents_data:
        with st.expander("💾 Guardar Análisis Completo", expanded=False):
            st.write("Guarda este análisis para futuras comparaciones y referencias.")
            
            # Mostrar resumen de datos disponibles
            available_data = []
            if papers_data:
                available_data.append(f"✅ Papers: {sum(papers_data.values())} publicaciones")
            if patents_data:
                available_data.append(f"✅ Patentes: {sum(patents_data.values())} patentes")
            
            st.info("**Datos disponibles para guardar:**\n" + "\n".join(available_data))
            
            # Formulario para guardar
            with st.form("save_complete_analysis_form"):
                analysis_name = st.text_input(
                    "Nombre para este análisis",
                    value=f"Análisis completo - {query[:30]}..." if len(query) > 30 else f"Análisis completo - {query}"
                )
                
                save_method = st.radio(
                    "Método de guardado",
                    options=["Sistema de Base de Datos", "Archivo directo"],
                    index=0
                )
                
                submit = st.form_submit_button("💾 Guardar Análisis Completo", type="primary")
                
                if submit and analysis_name:
                    _save_complete_analysis(
                        analysis_name, query, papers_data, patents_data, 
                        papers_analysis, patents_analysis, save_method
                    )

def _show_excel_patents_interface():
    """Interfaz para carga manual de patentes via Excel"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### 📋 Descargar Plantilla")
        
        start_year = st.number_input(
            "Año inicial", 
            min_value=1900,
            max_value=datetime.now().year - 5,
            value=1920,
            key="excel_start_year"
        )
        
        end_year = st.number_input(
            "Año final", 
            min_value=start_year + 5,
            max_value=datetime.now().year,
            value=datetime.now().year,
            key="excel_end_year"
        )
        
        template_bytes = PatentDataManager.create_template(start_year, end_year)
        
        st.download_button(
            label="📥 Descargar Plantilla Excel",
            data=template_bytes,
            file_name=f"plantilla_patentes_{start_year}-{end_year}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_excel_template_manual"
        )
    
    with col2:
        st.write("#### 📤 Subir Archivo")
        
        uploaded_file = st.file_uploader(
            "Selecciona archivo Excel",
            type=["xlsx", "xls"],
            key="excel_file_uploader_manual"
        )
        
        if uploaded_file is not None:
            patents_data = PatentDataManager.load_data(uploaded_file)
            
            if patents_data:
                st.success("✅ Archivo cargado correctamente")
                
                # Mostrar análisis inmediato
                df_patents = TechnologyAnalyzer.display_data_table(
                    patents_data, 
                    title="Datos de Patentes Cargados"
                )
                
                analysis_df, analysis_fig, ajuste_info, parametros = TechnologyAnalyzer.analyze_s_curve(patents_data)
                
                TechnologyAnalyzer.display_s_curve_analysis(
                    analysis_df, analysis_fig, ajuste_info, parametros,
                    title="Análisis de Curva en S - Patentes (Excel)"
                )
                
                TechnologyAnalyzer.export_data(
                    analysis_df if analysis_df is not None else df_patents,
                    "patentes_excel",
                    "datos_manuales"
                )

def _save_complete_analysis(name, query, papers_data, patents_data, papers_analysis, patents_analysis, method):
    """Guarda el análisis completo"""
    try:
        if method == "Sistema de Base de Datos":
            from data_storage import initialize_github_db
            
            db = initialize_github_db(use_local=True)
            if db:
                paper_metrics = papers_analysis['metrics'] if papers_analysis else None
                patent_metrics = patents_analysis['metrics'] if patents_analysis else None
                
                analysis_id = db.save_s_curve_analysis(
                    query=query,
                    paper_data=papers_data,
                    patent_data=patents_data,
                    paper_metrics=paper_metrics,
                    patent_metrics=patent_metrics,
                    analysis_name=name
                )
                
                if analysis_id:
                    st.success(f"✅ Análisis completo guardado con ID: {analysis_id}")
                else:
                    st.error("❌ Error al guardar en base de datos")
            else:
                st.error("❌ No se pudo inicializar el sistema de base de datos")
        
        else:  # Archivo directo
            from data_storage import save_analysis_direct
            
            paper_metrics = papers_analysis['metrics'] if papers_analysis else None
            patent_metrics = patents_analysis['metrics'] if patents_analysis else None
            
            result = save_analysis_direct(
                analysis_name=name,
                query=query,
                paper_data=papers_data,
                patent_data=patents_data,
                paper_metrics=paper_metrics,
                patent_metrics=patent_metrics
            )
            
            if result:
                st.success(f"✅ Análisis guardado en: {result}")
            else:
                st.error("❌ Error al guardar archivo directo")
                
    except Exception as e:
        st.error(f"❌ Error al guardar: {str(e)}")

def add_direct_save_option(analysis_name, query, paper_data=None, patent_data=None, paper_metrics=None, patent_metrics=None):
    """
    Añade una opción para guardar directamente los datos del análisis.
    """
    from data_storage import save_analysis_direct
    
    if st.button("💾 Guardar datos directamente", type="primary", key="save_direct_analysis"):
        result = save_analysis_direct(
            analysis_name=analysis_name,
            query=query,
            paper_data=paper_data,
            patent_data=patent_data,
            paper_metrics=paper_metrics,
            patent_metrics=patent_metrics
        )
        
        if result:
            st.success(f"Datos guardados correctamente en: {result}")
            # Mostrar opción para descargar el archivo
            try:
                with open(result, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    
                st.download_button(
                    label="📥 Descargar archivo JSON",
                    data=file_content,
                    file_name=os.path.basename(result),
                    mime="application/json",
                    key=f"download_json_direct_{int(time.time())}"  # ← KEY ÚNICO AGREGADO
                )
            except Exception as e:
                st.warning(f"No se pudo preparar descarga: {str(e)}")

if __name__ == "__main__":
    run_s_curve_analysis()