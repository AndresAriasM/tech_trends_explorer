import pandas as pd
import re
import streamlit as st
import requests
import time
from collections import Counter
from dotenv import load_dotenv
import nltk
import sqlite3
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from datetime import datetime
from serpapi import GoogleSearch
import numpy as np
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build
import json
from datetime import datetime
import folium
from folium.plugins import HeatMap
from streamlit_folium import folium_static
from config import CONFIG




class QueryBuilder:
    @staticmethod
    def build_google_query(topics, min_year=None, include_patents=True):
        base_query = ' AND '.join([f'"{topic.strip()}"' for topic in topics if topic.strip()])
        filters = []
        if min_year:
            filters.append(f'after:{min_year}')
        if not include_patents:
            filters.append('-patent')
        
        full_query = base_query
        if filters:
            full_query += ' ' + ' '.join(filters)
        return full_query

    @staticmethod
    def build_scopus_query(topics, min_year=None):
        terms = [f'TITLE-ABS-KEY("{topic.strip()}")' for topic in topics if topic.strip()]
        base_query = ' AND '.join(terms)
        if min_year:
            base_query += f' AND PUBYEAR > {min_year}'
        return base_query

class ResultAnalyzer:
    def __init__(self):
        # Palabras comunes en inglés para filtrar
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there',
            'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
            'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no',
            'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your',
            'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
            'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also',
            'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these',
            'give', 'day', 'most', 'us', 'was', 'is', 'are', 'were', 'been',
            'based', 'using', 'since', 'more', 'has', 'been', 'such', 'may',
            'very', 'both', 'each', 'between', 'under', 'same', 'through',
            'until'
        }

    def extract_year(self, text):
        """
        Extrae el año del texto con validación estricta
        """
        current_year = datetime.now().year
        MAX_VALID_YEAR = min(current_year, 2025)  
        MIN_VALID_YEAR = 1970  
        
        def is_valid_year(year):
            try:
                year_num = int(year)
                return MIN_VALID_YEAR <= year_num <= MAX_VALID_YEAR
            except ValueError:
                return False
        
        # Limpiar el texto de números que no son años
        cleaned_text = text.lower()
        cleaned_text = re.sub(r'\d+\s*(?:kb|mb|gb|kib|mib|gib|bytes?)', '', cleaned_text)
        cleaned_text = re.sub(r'(?:page|p\.)\s*\d+|\d+\s*(?:pages?|p\.)', '', cleaned_text)
        
        # Buscar fechas explícitas primero
        date_patterns = [
            r'published.*?in\s*(20\d{2})',
            r'publication\s*date:?\s*(20\d{2})',
            r'©\s*(20\d{2})',
            r'\d{1,2}\s*(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(20\d{2})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, cleaned_text)
            for match in matches:
                if is_valid_year(match):
                    return int(match)
        
        # Buscar en el título cualquier año entre 1970 y el año actual
        year_pattern = r'\b(19[7-9]\d|20[0-2]\d)\b'
        if years := re.findall(year_pattern, cleaned_text):
            valid_years = [int(y) for y in years if is_valid_year(y)]
            if valid_years:
                return max(valid_years)  # Retornar el año más reciente
        
        # Si no encontramos un año válido, retornar el año actual
        return min(current_year, MAX_VALID_YEAR)

    def extract_country(self, text):
        countries = ['USA', 'United States', 'UK', 'China', 'Japan', 'Germany', 'France']
        for country in countries:
            if country.lower() in text.lower():
                return country
        return None

    def extract_keywords(self, text):
        # Limpieza básica del texto
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        
        # Filtrar palabras comunes, números y años
        keywords = [
            word for word in words 
            if (
                word not in self.common_words 
                and len(word) > 3 
                and word.isalpha()  # Solo palabras que contengan letras
                and not word.isdigit()  # Excluir números
                and not re.match(r'.*\d+.*', word)  # Excluir palabras con números
                and not re.match(r'20\d{2}', word)  # Excluir años
            )
        ]
        
        return keywords

    def classify_result_type(self, url, title):
        url_lower = url.lower()
        title_lower = title.lower()
        
        if any(term in url_lower for term in ['.pdf', '/pdf/']):
            return 'PDF'
        elif any(term in url_lower for term in ['patent', 'uspto.gov', 'espacenet']):
            return 'Patent'
        elif any(term in url_lower for term in ['scholar.google', 'sciencedirect', 'springer', 'ieee']):
            return 'Academic Article'
        elif any(term in url_lower for term in ['news.google', 'reuters', 'bloomberg']):
            return 'News'
        elif 'cite' in url_lower or 'citation' in title_lower:
            return 'Citation'
        else:
            return 'Web Page'

    def analyze_results(self, results, search_topics):
        """Analiza los resultados y genera estadísticas"""
        # Obtener palabras de búsqueda para filtrarlas
        search_words = set()
        for topic in search_topics:
            words = re.sub(r'[^\w\s]', ' ', topic.lower()).split()
            search_words.update(words)

        processed_results = []
        all_keywords = []
        
        for item in results:
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            full_text = f"{title}. {snippet}"
            
            # Extraer palabras clave, excluyendo las palabras de búsqueda
            text_words = re.sub(r'[^\w\s]', ' ', full_text.lower()).split()
            keywords = [
                word for word in text_words 
                if (word not in self.common_words and 
                    word not in search_words and 
                    len(word) > 3 and 
                    word.isalpha() and  # Solo palabras que contengan letras
                    not word.isdigit() and  # Excluir números
                    not re.match(r'.*\d+.*', word))  # Excluir palabras con números
            ]
            all_keywords.extend(keywords)
            
            processed = {
                'title': title,
                'link': item.get('link', ''),
                'snippet': snippet,
                'year': self.extract_year(full_text),
                'country': self.extract_country(full_text),
                'type': self.classify_result_type(item.get('link', ''), title)
            }
            processed_results.append(processed)
        
        # Convertir a DataFrame para análisis
        df = pd.DataFrame(processed_results)
        
        # Contar palabras clave más comunes
        keyword_counts = Counter(all_keywords)
        
        # Filtrar palabras no deseadas adicionales
        filtered_keywords = [
            (word, count) for word, count in keyword_counts.most_common(20)
            if word not in search_words
        ]
        
        # Generar estadísticas
        stats = {
            'total_results': len(df),
            'by_type': df['type'].value_counts().to_dict(),
            'by_year': df['year'].value_counts().sort_index().to_dict(),
            'by_country': df['country'].value_counts().to_dict(),
            'common_keywords': filtered_keywords
        }
        
        return processed_results, stats

class NewsAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self.SERP_API_BASE_URL = "https://serpapi.com/search"

    def plot_gartner_analysis(self, yearly_stats, inflection_points):
        """Crea visualización de menciones por año con puntos de inflexión"""
        fig = go.Figure()
        
        # Línea principal de menciones
        df = pd.DataFrame(yearly_stats)
        fig.add_trace(go.Scatter(
            x=df['year'],
            y=df['mention_count'],
            mode='lines+markers',
            name='Menciones',
            line=dict(color='blue', width=2)
        ))
        
        # Agregar puntos de inflexión
        colors = {
            'innovation_trigger': 'green',
            'peak': 'red',
            'trough': 'orange',
            'slope': 'purple',
            'plateau': 'blue'
        }
        
        descriptions = {
            'innovation_trigger': 'Inicio de la Innovación',
            'peak': 'Pico de Expectativas',
            'trough': 'Valle de la Desilusión',
            'slope': 'Pendiente de la Iluminación',
            'plateau': 'Meseta de la Productividad'
        }
        
        for point_type, point_data in inflection_points.items():
            if point_data:
                fig.add_trace(go.Scatter(
                    x=[point_data['year']],
                    y=[point_data['mentions']],
                    mode='markers+text',
                    name=descriptions[point_type],
                    marker=dict(
                        size=15,
                        color=colors[point_type],
                        symbol='star'
                    ),
                    text=[descriptions[point_type]],
                    textposition="top center",
                    showlegend=True
                ))
        
        fig.update_layout(
            title="Análisis de Puntos de Inflexión de Gartner",
            xaxis_title="Año",
            yaxis_title="Número de Menciones",
            height=500,
            showlegend=True,
            hovermode='x'
        )
        
        return fig
    
    def extract_source_name(self, url):
        """Extrae el nombre de la fuente de la URL"""
        try:
            domain = re.search(r'https?://(?:www\.)?([^/]+)', url).group(1)
            source_name = domain.split('.')[0].title()
            return source_name
        except:
            return "Fuente desconocida"

    def perform_news_search(self, serp_api_key, query):
        """
        Realiza búsqueda optimizada usando SerpAPI PRO, maximizando los resultados por consulta
        """
        try:
            all_results = []
            current_year = datetime.now().year
            start_year = current_year - 11
            
            # Definir rangos de búsqueda más amplios
            date_ranges = [
                (start_year, start_year + 3),
                (start_year + 4, start_year + 7),
                (start_year + 8, current_year)
            ]
            
            print(f"Iniciando búsqueda optimizada desde {start_year} hasta {current_year}")
            
            for start_date, end_date in date_ranges:
                start = 0
                has_more = True
                found_in_range = 0
                
                print(f"\nBuscando en rango: {start_date}-{end_date}")
                
                while has_more:
                    try:
                        # Construir query con rango de fecha
                        date_query = f"{query} after:{start_date}-01-01 before:{end_date}-12-31"
                        
                        params = {
                            "q": date_query,
                            "tbm": "nws",
                            "api_key": serp_api_key,
                            "start": start,
                            "num": 100
                        }
                        
                        print(f"Consultando resultados {start+1}-{start+100}")
                        response = requests.get(self.SERP_API_BASE_URL, params=params)
                        
                        # Imprimir detalles de la respuesta para debugging
                        print(f"Status Code: {response.status_code}")
                        
                        response.raise_for_status()
                        data = response.json()
                        
                        # Verificar la estructura de la respuesta
                        print(f"Keys en respuesta: {data.keys()}")
                        
                        if "news_results" in data and data["news_results"]:
                            results = data["news_results"]
                            print(f"Encontrados {len(results)} resultados en esta página")
                            found_in_range += len(results)
                            
                            # Procesar resultados con manejo de errores individual
                            for item in results:
                                try:
                                    processed = self._process_news_item(item)
                                    if processed:
                                        all_results.append(processed)
                                        print(".", end="", flush=True)  # Indicador de progreso
                                except Exception as e:
                                    print(f"\nError procesando item: {str(e)}")
                                    continue
                            
                            print(f"\nProcesados {len(results)} resultados exitosamente")
                            
                            if len(results) < 100:
                                has_more = False
                                print(f"No hay más resultados en este rango. Total encontrado: {found_in_range}")
                            else:
                                start += len(results)
                                time.sleep(0.5)
                        else:
                            print("No se encontraron news_results en la respuesta")
                            if data.get("error"):
                                print(f"Error reportado: {data['error']}")
                            has_more = False
                    
                    except requests.exceptions.RequestException as e:
                        print(f"Error en la solicitud HTTP: {str(e)}")
                        has_more = False
                    except Exception as e:
                        print(f"Error inesperado: {str(e)}")
                        has_more = False
                        time.sleep(1)
            
            # Verificar resultados finales
            if not all_results:
                print("No se encontraron resultados en ningún rango")
                return False, "No se encontraron resultados"
            
            # Remover duplicados
            unique_results = {result['link']: result for result in all_results}.values()
            final_results = list(unique_results)
            
            print(f"\nBúsqueda completada exitosamente:")
            print(f"- Total de resultados encontrados: {len(all_results)}")
            print(f"- Resultados únicos después de filtrar: {len(final_results)}")
            
            return True, final_results
            
        except Exception as e:
            print(f"Error general en la búsqueda: {str(e)}")
            return False, str(e)

    def _process_news_item(self, item):
        """Procesa un resultado de noticia con mejor manejo de errores"""
        try:
            # Verificar campos requeridos
            if not item.get('title') or not item.get('link'):
                print(f"Item descartado por falta de campos requeridos")
                return None

            # Extraer campos con manejo de errores
            processed = {
                'title': str(item.get('title', '')),
                'link': str(item.get('link', '')),
                'snippet': str(item.get('snippet', '')),
                'source': str(item.get('source', '')),
                'date': str(item.get('date', '')),
                'year': self._extract_year_from_date(item.get('date', '')),
                'thumbnail': str(item.get('thumbnail', '')),
                'keywords': [],
                'sentiment': 0.0,
                'country': None
            }
            
            # Análisis de sentimiento
            text = f"{processed['title']} {processed['snippet']}"
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(text)
                processed['sentiment'] = sentiment_scores['compound']
            except Exception as e:
                print(f"Error en análisis de sentimiento: {str(e)}")
                processed['sentiment'] = 0.0

            # Extraer keywords y país
            try:
                processed['keywords'] = self._extract_keywords(text)
            except Exception as e:
                print(f"Error extrayendo keywords: {str(e)}")

            try:
                processed['country'] = self._extract_country(text)
            except Exception as e:
                print(f"Error extrayendo país: {str(e)}")

            return processed
            
        except Exception as e:
            print(f"Error procesando noticia: {str(e)}")
            return None

    def _extract_year_from_date(self, date_str):
        """Extrae el año de una fecha con manejo de errores mejorado"""
        try:
            if not date_str:
                return datetime.now().year
                
            # Intentar diferentes formatos de fecha
            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%b %d, %Y', '%Y']:
                try:
                    return datetime.strptime(date_str.split('T')[0], fmt).year
                except ValueError:
                    continue
                    
            # Si no se puede extraer el año, buscar un año en el string
            match = re.search(r'20\d{2}|19\d{2}', date_str)
            if match:
                year = int(match.group())
                if 1970 <= year <= datetime.now().year:
                    return year
                    
            return datetime.now().year
            
        except Exception as e:
            print(f"Error extrayendo año de {date_str}: {str(e)}")
            return datetime.now().year

    def _extract_year_from_text(self, text):
        """Extrae el año del texto con validación"""
        current_year = datetime.now().year
        years = re.findall(r'\b(19[7-9]\d|20[0-2]\d)\b', text)
        if years:
            valid_years = [int(y) for y in years if CONFIG['MIN_YEAR'] <= int(y) <= current_year]
            if valid_years:
                return max(valid_years)
        return current_year

    def _extract_month(self, date_str):
        """Extrae el mes de una fecha con manejo de errores"""
        try:
            if not date_str:
                return None
                
            # Intentar diferentes formatos de fecha
            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%S', '%b %d, %Y']:
                try:
                    return datetime.strptime(date_str.split('T')[0], fmt).month
                except ValueError:
                    continue
                    
            return None
            
        except Exception:
            return None

    def analyze_hype_cycle(self, news_results):
        """Análisis del Hype Cycle basado en puntos de inflexión"""
        try:
            # Crear DataFrame con los resultados limpios
            df_data = []
            for result in news_results:
                try:
                    df_data.append({
                        'year': int(result['year']),
                        'sentiment': float(result['sentiment']),
                        'month': self._extract_month(result.get('date', ''))
                    })
                except (ValueError, TypeError, KeyError):
                    continue

            if not df_data:
                return None

            df = pd.DataFrame(df_data)
            
            # Agrupar por año y calcular estadísticas
            yearly_stats = df.groupby('year').agg({
                'sentiment': ['mean', 'count', 'std'],
            }).reset_index()
            
            yearly_stats.columns = ['year', 'sentiment_mean', 'mention_count', 'sentiment_std']
            yearly_stats = yearly_stats.sort_values('year')
            
            # Obtener puntos de inflexión
            inflection_points = self.analyze_gartner_points(yearly_stats)
            
            # Determinar fase actual basada en puntos de inflexión
            current_year = datetime.now().year
            
            # Identificar la fase actual basada en los puntos de inflexión
            if inflection_points['innovation_trigger']:
                innovation_year = inflection_points['innovation_trigger']['year']
                if current_year - innovation_year <= 2:
                    phase = "Innovation Trigger"
                    confidence = 0.85
                
            if inflection_points['peak']:
                peak_year = inflection_points['peak']['year']
                if current_year - peak_year <= 1:
                    phase = "Peak of Inflated Expectations"
                    confidence = 0.9
                elif inflection_points['trough']:
                    trough_year = inflection_points['trough']['year']
                    if current_year - trough_year <= 1:
                        phase = "Trough of Disillusionment"
                        confidence = 0.85
                    elif current_year - trough_year <= 3:
                        phase = "Slope of Enlightenment"
                        confidence = 0.8
                    else:
                        phase = "Plateau of Productivity"
                        confidence = 0.75
                else:
                    # Si hay pico pero no valle, y ha pasado tiempo
                    if current_year - peak_year > 1:
                        phase = "Trough of Disillusionment"
                        confidence = 0.7
            else:
                # Si no hay pico identificado, usar métricas auxiliares
                latest_stats = yearly_stats.iloc[-1]
                mention_trend = yearly_stats['mention_count'].pct_change().mean()
                
                if mention_trend > 0.3:
                    phase = "Innovation Trigger"
                    confidence = 0.6
                else:
                    phase = "Pre-Innovation Trigger"
                    confidence = 0.5

            # Imprimir información de diagnóstico
            print("\nAnálisis de Puntos de Inflexión:")
            for point_type, point_data in inflection_points.items():
                if point_data:
                    print(f"{point_type}: Año {point_data['year']}, Menciones {point_data['mentions']}")
            print(f"\nFase determinada: {phase} (Confianza: {confidence:.2f})")

            return {
                'phase': phase,
                'confidence': confidence,
                'yearly_stats': yearly_stats,
                'inflection_points': inflection_points,
                'metrics': {
                    'latest_year': int(yearly_stats.iloc[-1]['year']),
                    'total_mentions': int(yearly_stats['mention_count'].sum()),
                    'peak_mentions': int(yearly_stats['mention_count'].max())
                }
            }
                
        except Exception as e:
            print(f"Error en análisis del Hype Cycle: {str(e)}")
            return None

    def analyze_gartner_points(self, yearly_stats):
        """Analiza los puntos de inflexión según el modelo de Gartner"""
        try:
            df = pd.DataFrame(yearly_stats)
            df = df.sort_values('year')
            
            # Calcular cambios porcentuales y tendencias
            df['mention_pct_change'] = df['mention_count'].pct_change()
            df['sentiment_change'] = df['sentiment_mean'].diff()
            
            inflection_points = {
                'innovation_trigger': None,
                'peak': None,
                'trough': None
            }
            
            if not df.empty:
                # Punto de innovación: primer año con menciones significativas
                threshold = df['mention_count'].mean() * 0.1
                innovation_data = df[df['mention_count'] >= threshold].iloc[0]
                inflection_points['innovation_trigger'] = {
                    'year': int(innovation_data['year']),
                    'mentions': int(innovation_data['mention_count']),
                    'sentiment': float(innovation_data['sentiment_mean'])
                }
                
                # Pico: máximo de menciones
                peak_data = df.loc[df['mention_count'].idxmax()]
                inflection_points['peak'] = {
                    'year': int(peak_data['year']),
                    'mentions': int(peak_data['mention_count']),
                    'sentiment': float(peak_data['sentiment_mean'])
                }
                
                # Valle: mínimo después del pico
                post_peak = df[df['year'] > peak_data['year']]
                if not post_peak.empty:
                    trough_data = post_peak.loc[post_peak['mention_count'].idxmin()]
                    inflection_points['trough'] = {
                        'year': int(trough_data['year']),
                        'mentions': int(trough_data['mention_count']),
                        'sentiment': float(trough_data['sentiment_mean'])
                    }
            
            return inflection_points
            
        except Exception as e:
            print(f"Error analizando puntos de inflexión: {str(e)}")
            return {
                'innovation_trigger': None,
                'peak': None,
                'trough': None
            }

    def plot_hype_cycle(self, hype_data, topics):
        """Visualización optimizada para modo oscuro del Hype Cycle con puntos de inflexión"""
        try:
            fig = go.Figure()

            # Crear curva base del Hype Cycle
            x = np.linspace(0, 100, 1000)
            y = self._hype_cycle_curve(x)

            # Añadir la curva principal
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name='Hype Cycle',
                line=dict(
                    color='#4A90E2',  # Azul claro
                    width=3,
                    shape='spline'
                )
            ))

            # Agregar línea base en y=0
            fig.add_hline(
                y=0,
                line=dict(color='gray', width=1, dash='dash'),
                opacity=0.5
            )

            # Definir puntos de inflexión
            inflection_points = {
                'Innovation Trigger': 15,
                'Peak of Inflated Expectations': 30,
                'Trough of Disillusionment': 60,
                'Slope of Enlightenment': 75,
                'Plateau of Productivity': 90
            }

            # Posicionar la tecnología basado en los datos
            if isinstance(topics, list) and topics and hype_data and 'inflection_points' in hype_data:
                # Calcular posición basada en puntos de inflexión reales
                current_year = datetime.now().year
                x_pos_grouped = self._calculate_position_from_points(
                    hype_data['inflection_points'],
                    current_year
                )
                y_pos_grouped = self._hype_cycle_curve(x_pos_grouped)

                # Ajustar posición si hay superposición con puntos de inflexión
                for phase, x_pos in inflection_points.items():
                    y_pos = self._hype_cycle_curve(x_pos)
                    if abs(y_pos_grouped - y_pos) < 0.05:  # Umbral de superposición
                        y_pos_grouped += 5
                        break

                # Añadir punto agrupado de tecnología
                fig.add_trace(go.Scatter(
                    x=[x_pos_grouped],
                    y=[y_pos_grouped],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color='#000080',  # Azul marino
                        symbol='circle',
                        line=dict(color='white', width=1)
                    ),
                    text=['Tecnología'],
                    textposition='top center',
                    textfont=dict(color='white'),
                    name='Tecnología',
                    hovertemplate=(
                        f"<b>Tecnología</b><br>" +
                        "Posición: %{x:.0f}<br>" +
                        "Expectativa: %{y:.1f}<br>" +
                        "<extra></extra>"
                    )
                ))

            # Añadir puntos de inflexión
            for phase, x_pos in inflection_points.items():
                y_pos = self._hype_cycle_curve(x_pos)
                fig.add_trace(go.Scatter(
                    x=[x_pos],
                    y=[y_pos],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color='yellow',
                        symbol='star'
                    ),
                    text=[phase],
                    textposition="top center",
                    textfont=dict(size=12, color='white'),
                    showlegend=False
                ))

            # Estilo para modo oscuro
            fig.update_layout(
                title={
                    'text': "Análisis del Hype Cycle",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top',
                    'font': dict(size=24, color='white')
                },
                height=700,
                showlegend=True,
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                xaxis=dict(
                    showticklabels=False,
                    title="Madurez de la Tecnología",
                    titlefont=dict(color='white'),
                    showgrid=False,
                    zeroline=False
                ),
                yaxis=dict(
                    showticklabels=False,
                    title="Expectativas",
                    titlefont=dict(color='white'),
                    showgrid=False,
                    zeroline=False
                ),
                legend=dict(
                    yanchor="top",
                    y=1.10,
                    xanchor="left",
                    x=0.01,
                    font=dict(color='white'),
                    bgcolor='rgba(50, 50, 50, 0.8)'
                ),
                margin=dict(t=100, l=50, r=50, b=100),
                hovermode='closest'
            )

            # Líneas de cuadrícula sutiles
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')

            return fig

        except Exception as e:
            print(f"Error en la visualización: {str(e)}")
            return None

    def _calculate_position_from_points(self, inflection_points, current_year):
        """Calcula la posición en el Hype Cycle basada en puntos de inflexión"""
        try:
            if inflection_points['innovation_trigger']:
                innovation_year = inflection_points['innovation_trigger']['year']
                if current_year - innovation_year <= 2:
                    return 15  # Innovation Trigger
            
            if inflection_points['peak']:
                peak_year = inflection_points['peak']['year']
                if current_year - peak_year <= 1:
                    return 30  # Peak
                
                if inflection_points['trough']:
                    trough_year = inflection_points['trough']['year']
                    if current_year - trough_year <= 1:
                        return 60  # Trough
                    elif current_year - trough_year <= 3:
                        return 75  # Slope
                    else:
                        return 90  # Plateau
                else:
                    years_since_peak = current_year - peak_year
                    if years_since_peak <= 2:
                        return 45  # Entre Peak y Trough
                    else:
                        return 60  # Trough
            
            return 15  # Default a Innovation Trigger
            
        except Exception as e:
            print(f"Error calculando posición: {str(e)}")
            return 15

    def _hype_cycle_curve(self, x):
        """Genera la curva del Hype Cycle"""
        return (60 * np.exp(-((x-20)/15)**2) -    # Primer pico
                20 * np.exp(-((x-60)/40)**2) +    # Valle
                40 * np.exp(-((x-90)/20)**2))     # Meseta


    def display_results(self, results, st_object):
        """
        Muestra los resultados usando un objeto streamlit proporcionado
        
        Args:
            results: Los resultados a mostrar
            st_object: El objeto streamlit para mostrar los resultados
        """
        if not results:
            st_object.warning("No hay resultados para mostrar")
            return

        st_object.write("### 📰 Resultados del Análisis")
        
        # Mostrar estadísticas generales
        df = pd.DataFrame(results)
        
        # Métricas principales
        col1, col2, col3 = st_object.columns(3)
        with col1:
            st_object.metric(
                "Total de Menciones",
                len(df),
                help="Número total de artículos encontrados"
            )
        with col2:
            avg_sentiment = df['sentiment'].mean()
            st_object.metric(
                "Sentimiento Promedio",
                f"{avg_sentiment:.2f}",
                help="Promedio del análisis de sentimiento (-1 a 1)"
            )
        with col3:
            recent_count = len(df[df['year'] >= datetime.now().year - 1])
            st_object.metric(
                "Menciones Recientes",
                recent_count,
                help="Menciones en el último año"
            )
        
        # Mostrar distribución temporal
        st_object.write("#### 📊 Distribución Temporal")
        yearly_counts = df['year'].value_counts().sort_index()
        fig = px.bar(
            x=yearly_counts.index,
            y=yearly_counts.values,
            labels={'x': 'Año', 'y': 'Número de Menciones'},
            title="Menciones por Año"
        )
        fig.update_layout(showlegend=False)
        st_object.plotly_chart(fig, use_container_width=True)
        
        # Mostrar artículos detallados
        st_object.write("#### 📑 Artículos Detallados")
        for idx, article in enumerate(results, 1):
            with st_object.expander(f"{idx}. {article['title']}", expanded=False):
                st_object.write("**Fuente:**", article['source'])
                st_object.write("**Fecha:**", article['date'])
                
                # Mostrar sentimiento con color
                sentiment = float(article['sentiment'])
                sentiment_color = "green" if sentiment > 0 else "red"
                st_object.markdown(
                    f"**Sentimiento:** <span style='color:{sentiment_color}'>{sentiment:.2f}</span>",
                    unsafe_allow_html=True
                )
                
                st_object.write("**Resumen:**", article['snippet'])
                st_object.write("**Keywords:**", ", ".join([k[0] for k in article['keywords']]))
                if article.get('country'):
                    st_object.write("**País:**", article['country'])
                st_object.markdown(f"[🔗 Leer artículo completo]({article['link']})")

    def _calculate_position(self, growth, sentiment):
        """Calcula la posición en el Hype Cycle basada en métricas disponibles"""
        if growth > 0.3 and sentiment > 0:
            return 15  # Innovation Trigger
        elif sentiment > 0.3:
            return 30  # Peak of Expectations
        elif sentiment < 0:
            return 60  # Trough of Disillusionment
        elif sentiment > 0 and growth > 0.1:
            return 75  # Slope of Enlightenment
        else:
            return 90  # Plateau of Productivity

    def _hype_cycle_curve(self, x):
        """Genera la curva del Hype Cycle"""
        return 60 * np.exp(-((x-20)/10)**2) - 20 * np.exp(-((x-60)/40)**2) + 40 * np.exp(-((x-90)/15)**2)

    def _extract_keywords(self, text):
        """Extrae palabras clave relevantes del texto"""
        # Implementar extracción de keywords
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to'])
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        return Counter(keywords).most_common(5)

    def _extract_country(self, text):
        """Extrae menciones de países del texto"""
        countries = {
            'USA': ['united states', 'usa', 'u.s.'],
            'UK': ['united kingdom', 'uk', 'britain'],
            'China': ['china', 'chinese'],
            'Japan': ['japan', 'japanese'],
            'Germany': ['germany', 'german']
        }
        
        text_lower = text.lower()
        for country, patterns in countries.items():
            if any(pattern in text_lower for pattern in patterns):
                return country
        return None

    def _extract_tech_mentions(self, text):
        """Extrae menciones de tecnologías específicas"""
        tech_terms = [
            'ai', 'artificial intelligence', 'machine learning', 'blockchain',
            'iot', 'internet of things', 'cloud', 'big data', '5g', 
            'automation', 'robotics', 'virtual reality', 'vr', 'ar',
            'augmented reality', 'quantum', 'cybersecurity'
        ]
        
        text_lower = text.lower()
        mentions = {}
        for term in tech_terms:
            count = text_lower.count(term)
            if count > 0:
                mentions[term] = count
        return mentions

    def display_advanced_analysis(self, results, query_info, st_object):
        """
        Muestra análisis avanzado con múltiples visualizaciones
        """
        if not results:
            st_object.warning("No hay resultados para mostrar")
            return

        # 1. Mostrar ecuación de búsqueda
        st_object.write("### 📝 Ecuación de Búsqueda")
        search_query = query_info.get('google_query') or query_info.get('search_query', 'No disponible')
        st_object.code(search_query)
        
        if 'time_range' in query_info:
            st_object.caption(f"Rango de tiempo: {query_info['time_range']}")

        # 2. Métricas generales
        col1, col2, col3 = st_object.columns(3)
        df = pd.DataFrame(results)
        
        with col1:
            st_object.metric("Total de Noticias", len(df))
        with col2:
            avg_sentiment = df['sentiment'].mean()
            st_object.metric("Sentimiento Promedio", f"{avg_sentiment:.2f}")
        with col3:
            recent_count = len(df[df['year'] >= datetime.now().year - 1])
            st_object.metric("Noticias Recientes", recent_count)

        # 3. Mapa de calor de noticias
        st_object.write("### 🌎 Distribución Geográfica")
        self._plot_news_map(df, st_object)

        # 4. Análisis temporal y sentimientos
        col1, col2 = st_object.columns(2)
        
        with col1:
            st_object.write("### 📈 Evolución Temporal")
            yearly_counts = df['year'].value_counts().sort_index()
            fig_temporal = px.line(
                x=yearly_counts.index, 
                y=yearly_counts.values,
                markers=True,
                labels={'x': 'Año', 'y': 'Número de Noticias'}
            )
            st_object.plotly_chart(fig_temporal, use_container_width=True)
        
        with col2:
            st_object.write("### 💭 Evolución del Sentimiento")
            sentiment_by_year = df.groupby('year')['sentiment'].mean()
            fig_sentiment = px.line(
                x=sentiment_by_year.index,
                y=sentiment_by_year.values,
                markers=True,
                labels={'x': 'Año', 'y': 'Sentimiento Promedio'}
            )
            st_object.plotly_chart(fig_sentiment, use_container_width=True)

        # 5. Nube de palabras y keywords
        st_object.write("### 🔤 Análisis de Palabras Clave")
        self._plot_keyword_analysis(df, st_object)

    def _plot_news_map(self, df, st_object):
        """Genera mapa de calor de noticias"""
        import folium
        from folium.plugins import HeatMap
        from streamlit_folium import folium_static
        
        # Coordenadas predefinidas para países principales
        country_coords = {
            'USA': [37.0902, -95.7129],
            'UK': [55.3781, -3.4360],
            'China': [35.8617, 104.1954],
            'Spain': [40.4637, -3.7492],
            'Germany': [51.1657, 10.4515],
            'France': [46.2276, 2.2137],
            # Añadir más países según necesidad
        }

        # Crear mapa base
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # Procesar datos para el mapa de calor
        heat_data = []
        for country in df['country'].dropna().unique():
            if country in country_coords:
                count = len(df[df['country'] == country])
                heat_data.append(
                    country_coords[country] + [count]
                )

        # Añadir capa de calor
        HeatMap(heat_data).add_to(m)
        
        # Mostrar mapa
        folium_static(m)

    def _plot_keyword_analysis(self, df, st_object):
        """Analiza y visualiza palabras clave con filtrado personalizado"""
        # Definir stopwords y palabras bloqueadas
        default_stopwords = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            # ... resto de stopwords ...
        ])

        # Palabras específicas del dominio para bloquear
        domain_blocklist = set([
            'http', 'https', 'com', 'www', 'html', 'htm', 'pdf', 'org',
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ])

        # Combinar todas las palabras bloqueadas
        all_blocked_words = default_stopwords | domain_blocklist

        # Unir y filtrar keywords
        all_keywords = []
        for kw_list in df['keywords']:
            if isinstance(kw_list, list):
                all_keywords.extend([
                    k[0].lower() for k in kw_list 
                    if isinstance(k, tuple) and 
                    len(k[0]) > 2 and  # Filtrar palabras muy cortas
                    k[0].lower() not in all_blocked_words and
                    not k[0].isdigit()  # Filtrar números
                ])

        if not all_keywords:
            st_object.warning("No se encontraron palabras clave después del filtrado")
            return

        # Contar frecuencias
        keyword_freq = Counter(all_keywords).most_common(20)
        
        # Crear DataFrame para mejor manipulación
        keywords_df = pd.DataFrame(keyword_freq, columns=['Palabra', 'Frecuencia'])
        
        # Crear gráfico mejorado
        fig = px.bar(
            keywords_df,
            x='Palabra',
            y='Frecuencia',
            title="Palabras Clave más Frecuentes",
            color='Frecuencia',
            color_continuous_scale='Viridis'
        )
        
        # Mejorar el diseño del gráfico
        fig.update_layout(
            xaxis_title="Palabra Clave",
            yaxis_title="Frecuencia de Aparición",
            xaxis_tickangle=45,
            showlegend=False,
            height=500
        )
        
        # Mostrar gráfico
        st_object.plotly_chart(fig, use_container_width=True)
        

    def show_hype_cycle_news_table(self, st, news_results):
        """Muestra una tabla interactiva con el mapa y los detalles de las noticias"""
        # Calcular estadísticas por país
        country_stats = {}
        for result in news_results:
            country = result.get('country')
            if country:
                if country not in country_stats:
                    country_stats[country] = {'count': 0, 'sentiment': 0}
                country_stats[country]['count'] += 1
                country_stats[country]['sentiment'] += result.get('sentiment', 0)

        # Calcular promedios de sentimiento
        for country in country_stats:
            country_stats[country]['avg_sentiment'] = country_stats[country]['sentiment'] / country_stats[country]['count']

        # Crear mapa base
        m = folium.Map(location=[20, 0], zoom_start=2, tiles='CartoDB positron')
        
        # Coordenadas de países principales
        country_coords = {
            'USA': [37.0902, -95.7129],
            'UK': [55.3781, -3.4360],
            'China': [35.8617, 104.1954],
            'Japan': [36.2048, 138.2529],
            'Germany': [51.1657, 10.4515],
            'India': [20.5937, 78.9629],
            'Brazil': [-14.2350, -51.9253],
            'Spain': [40.4637, -3.7492]
        }

        # Añadir marcadores al mapa
        for country, stats in country_stats.items():
            if country in country_coords:
                color = 'green' if stats['avg_sentiment'] > 0 else 'red'
                radius = stats['count'] * 5  # Tamaño basado en número de menciones
                
                folium.CircleMarker(
                    location=country_coords[country],
                    radius=radius,
                    color=color,
                    fill=True,
                    popup=f"{country}: {stats['count']} menciones"
                ).add_to(m)

        # Mostrar mapa y ranking
        st.write("### 🌍 Distribución Global de la Tecnología")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            folium_static(m)
        
        with col2:
            st.write("#### 🏆 Ranking de Países")
            # Crear DataFrame para ranking
            data = []
            for country, stats in country_stats.items():
                data.append({
                    "País": country,
                    "Menciones": int(stats['count']),
                    "Sentimiento": round(stats['avg_sentiment'], 2)
                })
            
            ranking = pd.DataFrame(data)
            if not ranking.empty:
                ranking = ranking.sort_values(by="Menciones", ascending=False)
                st.dataframe(
                    ranking,
                    hide_index=True,
                    column_config={
                        "País": "País",
                        "Menciones": st.column_config.NumberColumn("Menciones", format="%d"),
                        "Sentimiento": st.column_config.NumberColumn("Sentimiento", format="%.2f")
                    }
                )

        # Mostrar lista de noticias
        st.write("### 📰 Artículos Analizados")
        for i, result in enumerate(news_results, 1):
            with st.expander(f"📄 {i}. {result['title']}", expanded=False):
                st.write(result.get('summary', 'No hay resumen disponible'))
                st.write(f"🔗 [Ver noticia completa]({result['link']})")
                st.write(f"📅 Año: {result['year']}")
                st.write(f"🌍 País: {result.get('country', 'No especificado')}")
                sentiment = result.get('sentiment', 0)
                st.write(f"💭 Sentimiento: {'Positivo' if sentiment > 0 else 'Negativo'} ({sentiment:.2f})")