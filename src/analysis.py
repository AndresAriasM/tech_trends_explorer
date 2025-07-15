#src/analysis.py
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
from serpapi.google_search import GoogleSearch
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
        """
        Construye la consulta para Google Custom Search respetando la estructura de operadores
        """
        if not topics:
            return ""
            
        query_parts = []
        
        for i, topic in enumerate(topics):
            if not topic['value'].strip():
                continue
                
            term = topic['value'].strip()
            
            # Aplicar coincidencia exacta si está seleccionada y el término no está ya entre comillas
            if topic['exact_match'] and not (term.startswith('"') and term.endswith('"')):
                term = f'"{term}"'
            
            # Solo añadimos el operador si no es el último término
            if i == len(topics) - 1:
                query_parts.append(term)
            else:
                query_parts.append(f"{term} {topic['operator']}")
        
        # Unir todas las partes
        base_query = " ".join(query_parts)
        
        # Añadir filtros adicionales
        filters = []
        if min_year:
            filters.append(f"after:{min_year}")
        if not include_patents:
            filters.append("-patent")
        
        # Combinar query base con filtros
        final_query = base_query
        if filters:
            final_query += " " + " ".join(filters)
        
        return final_query

    @staticmethod
    def build_scopus_query(topics, min_year=None):
        """
        Construye la consulta para Scopus respetando estructura de operadores
        
        Args:
            topics: Lista de diccionarios con la estructura:
                {
                    'value': str,          # término de búsqueda
                    'operator': str,       # 'AND', 'OR', o 'NOT'
                    'exact_match': bool    # si debe ser coincidencia exacta
                }
            min_year: Año mínimo para la búsqueda
        
        Returns:
            str: Query formateada para Scopus
        """
        if not topics:
            return ""
            
        query_parts = []
        
        for i, topic in enumerate(topics):
            if not topic['value'].strip():
                continue
                
            term = topic['value'].strip()
            
            # Aplicar coincidencia exacta si está seleccionada y el término no está ya entre comillas
            if topic['exact_match'] and not (term.startswith('"') and term.endswith('"')):
                term = f'"{term}"'
            
            # En Scopus, los términos van envueltos en TITLE-ABS-KEY()
            scopus_term = f'TITLE-ABS-KEY({term})'
            
            # Solo añadimos el operador si no es el último término
            if i == len(topics) - 1:
                query_parts.append(scopus_term)
            else:
                # Convertir el operador NOT a AND NOT para Scopus
                operator = 'AND NOT' if topic['operator'] == 'NOT' else topic['operator']
                query_parts.append(f"{scopus_term} {operator}")
        
        # Unir todas las partes
        base_query = " ".join(query_parts)
        
        # Añadir filtro de año si existe
        if min_year:
            base_query += f" AND PUBYEAR > {min_year}"
            
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
        countries = ['USA', 'United States', 'UK', 'China', 'Japan', 'Germany', 'France', 'India', 'Italy', 'Canada',
                     'South Korea', 'Russia', 'Brazil', 'Australia', 'Spain', 'Mexico', 'Indonesia', 'Netherlands',
                     'Turkey', 'Saudi Arabia', 'Switzerland', 'Sweden', 'Poland', 'Belgium', 'Norway', 'Austria',
                     'UAE', 'Iran', 'Ireland, Denmark', 'Colombia', 'South Africa', 'Egypt', 'Chile', 'Argentina',
                     'Finland', 'Czech Republic', 'Portugal', 'Greece', 'Vietnam', 'New Zealand', 'Thailand', 'Algeria',
                     'Qatar', 'Peru', 'Romania', 'Hungary', 'Kazakhstan', 'Ukraine', 'Iraq', 'Morocco', 'Bangladesh',
                     'Puerto Rico', 'Philippines', 'Pakistan', 'Venezuela', 'Croatia']
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
        """
        Analiza los resultados y genera estadísticas.
        
        Args:
            results: Lista de resultados a analizar
            search_topics: Lista de diccionarios con la información de los topics
        """
        # Obtener palabras de búsqueda para filtrarlas
        search_words = set()
        for topic in search_topics:
            if isinstance(topic, dict):
                # Si el topic viene en formato diccionario
                words = re.sub(r'[^\w\s]', ' ', topic['value'].lower()).split()
            else:
                # Si el topic viene en formato string
                words = re.sub(r'[^\w\s]', ' ', str(topic).lower()).split()
                
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
        VERSIÓN HÍBRIDA: Selecciona estrategia según complejidad de la consulta
        
        - Consultas SIMPLES (1 término) → Lógica EXPLORATORIA (múltiples llamadas)
        - Consultas COMPLEJAS (múltiples términos/operadores) → Lógica DIRECTA (pocas llamadas)
        """
        try:
            all_results = []
            current_year = datetime.now().year
            start_year = current_year - 12
            
            # Limpiar la query de filtros de fecha existentes
            clean_query = re.sub(r'\s*(?:after|before):\d{4}(?:-\d{2}-\d{2})?\s*', '', query).strip()
            
            # PASO 1: Analizar complejidad de la consulta
            query_complexity = self._analyze_query_complexity(clean_query)
            
            print(f"🔍 Iniciando búsqueda HÍBRIDA desde {start_year} hasta {current_year}")
            print(f"📝 Query: {clean_query}")
            print(f"🧠 Complejidad detectada: {query_complexity['level']} ({query_complexity['reason']})")
            
            # PASO 2: Seleccionar estrategia según complejidad
            if query_complexity['level'] == 'SIMPLE':
                print(f"🚀 Usando ESTRATEGIA EXPLORATORIA (términos simples)")
                return self._search_simple_query(serp_api_key, clean_query, start_year, current_year)
            else:
                print(f"🎯 Usando ESTRATEGIA DIRECTA (consulta compleja)")
                return self._search_complex_query(serp_api_key, clean_query, start_year, current_year)
                
        except Exception as e:
            print(f"❌ Error general en búsqueda: {str(e)}")
            return False, str(e)

    def _analyze_query_complexity(self, query):
        """
        Analiza la complejidad de una consulta para determinar estrategia de búsqueda
        
        Returns:
            dict: {'level': 'SIMPLE'|'COMPLEX', 'reason': str, 'score': int}
        """
        complexity_score = 0
        reasons = []
        
        # Detectar operadores booleanos
        boolean_operators = ['AND', 'OR', 'NOT', '&', '|', '-']
        for operator in boolean_operators:
            if operator in query.upper():
                complexity_score += 2
                reasons.append(f"operador {operator}")
        
        # Detectar múltiples términos entre comillas
        quoted_terms = re.findall(r'"[^"]+"', query)
        if len(quoted_terms) > 1:
            complexity_score += len(quoted_terms)
            reasons.append(f"{len(quoted_terms)} términos exactos")
        
        # Detectar múltiples palabras (sin comillas)
        words_outside_quotes = re.sub(r'"[^"]+"', '', query).strip()
        word_count = len([w for w in words_outside_quotes.split() if len(w) > 2])
        if word_count > 2:
            complexity_score += word_count - 2
            reasons.append(f"{word_count} palabras separadas")
        
        # Detectar filtros específicos
        filters = ['site:', 'filetype:', 'intitle:', 'inurl:', 'after:', 'before:']
        for filter_term in filters:
            if filter_term in query.lower():
                complexity_score += 1
                reasons.append(f"filtro {filter_term}")
        
        # Detectar frases largas (más de 4 palabras juntas)
        phrases = [phrase for phrase in query.split() if len(phrase.split()) > 4]
        if phrases:
            complexity_score += len(phrases)
            reasons.append("frases largas")
        
        # Determinar nivel de complejidad
        if complexity_score == 0:
            level = 'SIMPLE'
            reason = "término único sin operadores"
        elif complexity_score <= 2:
            level = 'SIMPLE'
            reason = "complejidad baja: " + ", ".join(reasons[:2])
        else:
            level = 'COMPLEX'
            reason = "complejidad alta: " + ", ".join(reasons[:3])
        
        return {
            'level': level,
            'reason': reason,
            'score': complexity_score,
            'operators_found': len([r for r in reasons if 'operador' in r]),
            'terms_count': len(quoted_terms) + word_count
        }

    def _search_simple_query(self, serp_api_key, query, start_year, current_year):
        """
        ESTRATEGIA EXPLORATORIA: Para consultas simples (1-2 términos)
        Usa lógica de exploración inicial + rangos dinámicos
        """
        base_params = {
            "api_key": serp_api_key,
            "tbm": "nws",
            "num": 100,
            "safe": "off",
            "gl": "us",
            "hl": "en",
            "filter": "0"
        }
        
        all_results = []
        total_api_calls = 0
        
        print(f"\n🧭 PASO 1: Consulta exploratoria...")
        
        # Consulta exploratoria inicial
        exploratory_query = f"{query} after:{start_year}-01-01 before:{current_year}-12-31"
        params = {**base_params, "q": exploratory_query, "start": 0}
        
        try:
            response = requests.get(self.SERP_API_BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()
            total_api_calls += 1
            
            if "news_results" in data and data["news_results"]:
                exploratory_results = data["news_results"]
                exploratory_count = len(exploratory_results)
                
                print(f"  📊 Resultados exploratorios: {exploratory_count}")
                
                # Procesar resultados exploratorios
                for item in exploratory_results:
                    if self._is_valid_result(item):
                        processed = self._process_news_item(item)
                        if processed:
                            all_results.append(processed)
                
                # Decisión basada en volumen
                if exploratory_count < 100:
                    print(f"  ✅ OPTIMIZACIÓN: Solo {exploratory_count} resultados totales")
                    print(f"  💰 Estrategia simple suficiente")
                    
                    # Verificar página 2 solo si hay exactamente 100
                    if exploratory_count == 100:
                        params_page2 = {**base_params, "q": exploratory_query, "start": 100}
                        try:
                            response2 = requests.get(self.SERP_API_BASE_URL, params=params_page2)
                            response2.raise_for_status()
                            data2 = response2.json()
                            total_api_calls += 1
                            
                            if "news_results" in data2 and data2["news_results"]:
                                for item in data2["news_results"]:
                                    if self._is_valid_result(item):
                                        processed = self._process_news_item(item)
                                        if processed:
                                            all_results.append(processed)
                        except Exception:
                            pass
                else:
                    print(f"  📈 MUCHOS DATOS: Iniciando búsqueda por rangos...")
                    
                    # Determinar tamaño de rangos según densidad
                    if exploratory_count >= 500:
                        range_size = 1
                        max_calls_per_range = 8
                    elif exploratory_count >= 200:
                        range_size = 2
                        max_calls_per_range = 6
                    else:
                        range_size = 3
                        max_calls_per_range = 4
                    
                    print(f"  🎯 Configuración: Rangos de {range_size} año(s), máx {max_calls_per_range} llamadas/rango")
                    
                    # Generar rangos dinámicos
                    date_ranges = []
                    current_start = start_year
                    while current_start <= current_year:
                        range_end = min(current_start + range_size - 1, current_year)
                        date_ranges.append((current_start, range_end))
                        current_start = range_end + 1
                    
                    # Buscar por rangos
                    for range_idx, (start_date, end_date) in enumerate(date_ranges):
                        if total_api_calls >= 50:  # Límite de seguridad
                            print(f"  🛑 Límite de seguridad alcanzado")
                            break
                        
                        start = 0
                        calls_in_range = 0
                        
                        while start < 800 and calls_in_range < max_calls_per_range:
                            try:
                                date_query = f"{query} after:{start_date}-01-01 before:{end_date}-12-31"
                                params = {**base_params, "q": date_query, "start": start}
                                
                                response = requests.get(self.SERP_API_BASE_URL, params=params)
                                response.raise_for_status()
                                data = response.json()
                                
                                total_api_calls += 1
                                calls_in_range += 1
                                
                                if "news_results" in data and data["news_results"]:
                                    results = data["news_results"]
                                    batch_size = len(results)
                                    
                                    for item in results:
                                        if self._is_valid_result(item):
                                            processed = self._process_news_item(item)
                                            if processed:
                                                all_results.append(processed)
                                    
                                    if batch_size < 100:
                                        break
                                    else:
                                        start += batch_size
                                        time.sleep(0.2)
                                else:
                                    break
                                    
                            except Exception as e:
                                print(f"    ⚠️ Error en rango: {str(e)}")
                                break
            else:
                return False, "No se encontraron resultados en consulta exploratoria"
                
        except Exception as e:
            return False, f"Error en consulta exploratoria: {str(e)}"
        
        # Procesar y retornar resultados
        unique_results = self._remove_duplicates(all_results)
        
        print(f"\n✅ ESTRATEGIA EXPLORATORIA COMPLETADA:")
        print(f"   💰 Total llamadas API: {total_api_calls}")
        print(f"   📄 Resultados únicos: {len(unique_results)}")
        
        return True, unique_results

    def _search_complex_query(self, serp_api_key, query, start_year, current_year):
        """
        ESTRATEGIA DIRECTA: Para consultas complejas (múltiples términos/operadores)
        Usa rangos predefinidos con pocas llamadas por rango
        """
        base_params = {
            "api_key": serp_api_key,
            "tbm": "nws",
            "num": 100,
            "safe": "off",
            "gl": "us",
            "hl": "en",
            "filter": "0"
        }
        
        all_results = []
        total_api_calls = 0
        
        print(f"\n🎯 Usando ESTRATEGIA DIRECTA para consulta compleja...")
        
        # Rangos predefinidos más amplios (menos llamadas)
        date_ranges = [
            (start_year, start_year + 2),      # Primeros 3 años
            (start_year + 3, start_year + 5),  # Siguientes 3 años
            (start_year + 6, start_year + 8),  # Siguientes 3 años
            (start_year + 9, current_year)     # Últimos años
        ]
        
        print(f"  📅 Configurados {len(date_ranges)} rangos amplios")
        
        for range_idx, (start_date, end_date) in enumerate(date_ranges):
            start = 0
            calls_in_range = 0
            
            print(f"\n  📅 Rango {range_idx + 1}/{len(date_ranges)}: {start_date}-{end_date}")
            
            # Máximo 3 llamadas por rango para consultas complejas
            while start < 300 and calls_in_range < 3:
                try:
                    date_query = f"{query} after:{start_date}-01-01 before:{end_date}-12-31"
                    params = {**base_params, "q": date_query, "start": start}
                    
                    response = requests.get(self.SERP_API_BASE_URL, params=params)
                    response.raise_for_status()
                    data = response.json()
                    
                    total_api_calls += 1
                    calls_in_range += 1
                    
                    if "news_results" in data and data["news_results"]:
                        results = data["news_results"]
                        batch_size = len(results)
                        
                        print(f"    🔗 Llamada {calls_in_range}: {batch_size} resultados")
                        
                        for item in results:
                            if self._is_valid_result(item):
                                processed = self._process_news_item(item)
                                if processed:
                                    all_results.append(processed)
                        
                        if batch_size < 100:
                            print(f"    ✅ Fin de rango (última página)")
                            break
                        else:
                            start += batch_size
                            time.sleep(0.2)
                            
                    else:
                        print(f"    ❌ Sin resultados")
                        break
                        
                except Exception as e:
                    print(f"    ⚠️ Error: {str(e)}")
                    break
        
        # Procesar y retornar resultados
        unique_results = self._remove_duplicates(all_results)
        
        print(f"\n✅ ESTRATEGIA DIRECTA COMPLETADA:")
        print(f"   💰 Total llamadas API: {total_api_calls}")
        print(f"   📄 Resultados únicos: {len(unique_results)}")
        
        return True, unique_results

    def _remove_duplicates(self, results):
        """Elimina duplicados basándose en URL y título similar"""
        unique_results = []
        seen_urls = set()
        seen_titles = set()
        
        for result in results:
            url = result.get('link', '')
            title = result.get('title', '').lower().strip()
            
            if url not in seen_urls and title not in seen_titles:
                seen_urls.add(url)
                seen_titles.add(title)
                unique_results.append(result)
        
        return unique_results

    def _is_valid_result(self, item):
        """
        Valida si un resultado debe ser incluido basado en criterios de calidad
        """
        try:
            # Verificar campos requeridos
            if not all(key in item for key in ['title', 'link', 'snippet']):
                return False
                
            # Verificar longitud mínima del contenido
            if len(item['snippet']) < 50:  # Contenido muy corto puede ser de baja calidad
                return False
                
            # Verificar si es una fuente bloqueada
            blocked_domains = [
                'pinterest',
                'facebook.com',
                'twitter.com',
                'instagram.com'
            ]
            if any(domain in item['link'].lower() for domain in blocked_domains):
                return False
                
            return True
                
        except Exception:
            return False
    

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
        """VERSIÓN CORREGIDA: Análisis del Hype Cycle con lógica mejorada"""
        try:
            # Crear DataFrame con validación mejorada
            df_data = []
            for result in news_results:
                try:
                    year = int(result['year'])
                    sentiment = float(result['sentiment'])
                    
                    # Validar rangos
                    if 2000 <= year <= datetime.now().year and -1 <= sentiment <= 1:
                        df_data.append({
                            'year': year,
                            'sentiment': sentiment,
                            'month': self._extract_month(result.get('date', ''))
                        })
                except (ValueError, TypeError, KeyError):
                    continue

            if len(df_data) < 3:  # Necesitamos datos mínimos
                return None

            df = pd.DataFrame(df_data)
            
            # Agrupar por año con estadísticas completas
            yearly_stats = df.groupby('year').agg({
                'sentiment': ['mean', 'count', 'std', 'min', 'max'],
            }).reset_index()
            
            yearly_stats.columns = ['year', 'sentiment_mean', 'mention_count', 'sentiment_std', 'sentiment_min', 'sentiment_max']
            yearly_stats = yearly_stats.sort_values('year')
            
            # Calcular métricas adicionales
            yearly_stats['mention_change'] = yearly_stats['mention_count'].pct_change()
            yearly_stats['sentiment_change'] = yearly_stats['sentiment_mean'].diff()
            yearly_stats['momentum'] = yearly_stats['mention_change'] * yearly_stats['sentiment_mean']
            
            # Obtener puntos de inflexión mejorados
            inflection_points = self.analyze_gartner_points(yearly_stats)
            
            # LÓGICA CORREGIDA: Determinar fase actual con elif
            current_year = datetime.now().year
            phase = "Pre-Innovation Trigger"
            confidence = 0.5
            
            # Verificar datos del último año
            latest_data = yearly_stats.iloc[-1]
            
            # 1. INNOVATION TRIGGER
            if inflection_points['innovation_trigger']:
                innovation_year = inflection_points['innovation_trigger']['year']
                years_since_innovation = current_year - innovation_year
                
                if years_since_innovation <= 1:
                    phase = "Innovation Trigger"
                    confidence = 0.85
                
                # 2. PEAK OF INFLATED EXPECTATIONS  
                elif inflection_points['peak']:
                    peak_year = inflection_points['peak']['year']
                    years_since_peak = current_year - peak_year
                    peak_sentiment = inflection_points['peak']['sentiment']
                    
                    # Detectar si estamos en el pico (alta actividad + alto sentimiento)
                    if years_since_peak <= 1 and peak_sentiment > 0.2:
                        phase = "Peak of Inflated Expectations"
                        confidence = 0.9
                    
                    # 3. TROUGH OF DISILLUSIONMENT
                    elif inflection_points['trough']:
                        trough_year = inflection_points['trough']['year']
                        years_since_trough = current_year - trough_year
                        
                        if years_since_trough <= 1:
                            phase = "Trough of Disillusionment"
                            confidence = 0.85
                        
                        # 4. SLOPE OF ENLIGHTENMENT
                        elif 1 < years_since_trough <= 4:
                            # Verificar recuperación gradual
                            recent_trend = yearly_stats['mention_change'].tail(2).mean()
                            if recent_trend > 0:
                                phase = "Slope of Enlightenment"
                                confidence = 0.8
                            else:
                                phase = "Trough of Disillusionment"
                                confidence = 0.75
                        
                        # 5. PLATEAU OF PRODUCTIVITY
                        elif years_since_trough > 4:
                            # Verificar estabilidad
                            recent_std = yearly_stats['mention_count'].tail(3).std()
                            recent_mean = yearly_stats['mention_count'].tail(3).mean()
                            
                            if recent_std / recent_mean < 0.3:  # Coeficiente de variación bajo
                                phase = "Plateau of Productivity"
                                confidence = 0.85
                            else:
                                phase = "Slope of Enlightenment"
                                confidence = 0.7
                    
                    # Si hay pico pero no valle detectado
                    else:
                        if years_since_peak > 2:
                            phase = "Trough of Disillusionment"
                            confidence = 0.7
                        else:
                            # Analizar tendencia después del pico
                            post_peak_data = yearly_stats[yearly_stats['year'] > peak_year]
                            if not post_peak_data.empty:
                                trend = post_peak_data['mention_change'].mean()
                                if trend < -0.2:
                                    phase = "Trough of Disillusionment"
                                    confidence = 0.75
                                else:
                                    phase = "Peak of Inflated Expectations"
                                    confidence = 0.65
                
                # Si solo hay innovation trigger
                elif years_since_innovation > 1:
                    # Buscar indicios de pico
                    recent_growth = yearly_stats['mention_change'].tail(2).mean()
                    recent_sentiment = yearly_stats['sentiment_mean'].tail(2).mean()
                    
                    if recent_growth > 0.5 and recent_sentiment > 0.3:
                        phase = "Peak of Inflated Expectations"
                        confidence = 0.7
                    else:
                        phase = "Innovation Trigger"
                        confidence = 0.6
            
            # Si no hay puntos de inflexión claros, usar métricas de respaldo
            else:
                latest_mentions = latest_data['mention_count']
                latest_sentiment = latest_data['sentiment_mean']
                growth_trend = yearly_stats['mention_change'].tail(3).mean()
                
                if growth_trend > 0.3 and latest_sentiment > 0.1:
                    phase = "Innovation Trigger"
                    confidence = 0.6
                elif growth_trend < -0.3:
                    phase = "Trough of Disillusionment"
                    confidence = 0.55
                else:
                    phase = "Pre-Innovation Trigger"
                    confidence = 0.5

            return {
                'phase': phase,
                'confidence': confidence,
                'yearly_stats': yearly_stats,
                'inflection_points': inflection_points,
                'metrics': {
                    'latest_year': int(yearly_stats.iloc[-1]['year']),
                    'total_mentions': int(yearly_stats['mention_count'].sum()),
                    'peak_mentions': int(yearly_stats['mention_count'].max()),
                    'avg_sentiment': float(yearly_stats['sentiment_mean'].mean()),
                    'sentiment_volatility': float(yearly_stats['sentiment_std'].mean())
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
                'Innovation Trigger': 5,
                'Peak of Inflated Expectations': 20,
                'Trough of Disillusionment': 50,
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
                        y_pos_grouped += 6
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
                    return 5  # Innovation Trigger
            
            if inflection_points['peak']:
                peak_year = inflection_points['peak']['year']
                if current_year - peak_year <= 1:
                    return 20  # Peak
                
                if inflection_points['trough']:
                    trough_year = inflection_points['trough']['year']
                    if current_year - trough_year <= 1:
                        return 50  # Trough
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
            
            return 5  # Default a Innovation Trigger
            
        except Exception as e:
            print(f"Error calculando posición: {str(e)}")
            return 5

    def _hype_cycle_curve(self, x):
        """Genera la curva del Hype Cycle"""
        return (60 * np.exp(-((x-20)/15)**2) -    # Primer pico
                20 * np.exp(-((x-60)/40)**2) +    # Valle
                40 * np.exp(-((x-90)/20)**2))     # Meseta


    def display_results(self, results, st_object):
        """Muestra los resultados con información geográfica mejorada"""
        if not results:
            st_object.warning("No hay resultados para mostrar")
            return

        st_object.write("### 📰 Artículos Encontrados")
        st_object.write(f"Mostrando {len(results)} resultados")

        # Diccionario de banderas de países
        country_emojis = {
            'Afghanistan': '🇦🇫', 'Albania': '🇦🇱', 'Algeria': '🇩🇿', 'Andorra': '🇦🇩', 'Angola': '🇦🇴', 'Antigua and Barbuda': '🇦🇬',
            'Argentina': '🇦🇷', 'Armenia': '🇦🇲', 'Australia': '🇦🇺', 'Austria': '🇦🇹', 'Azerbaijan': '🇦🇿', 'Bahamas': '🇧🇸',
            'Bahrain': '🇧🇭', 'Bangladesh': '🇧🇩', 'Barbados': '🇧🇧', 'Belarus': '🇧🇾', 'Belgium': '🇧🇪', 'Belize': '🇧🇿',
            'Benin': '🇧🇯', 'Bhutan': '🇧🇹', 'Bolivia': '🇧🇴', 'Bosnia and Herzegovina': '🇧🇦', 'Botswana': '🇧🇼', 'Brazil': '🇧🇷',
            'Brunei': '🇧🇳', 'Bulgaria': '🇧🇬', 'Burkina Faso': '🇧🇫', 'Burundi': '🇧🇮', 'Cabo Verde': '🇨🇻', 'Cambodia': '🇰🇭',
            'Cameroon': '🇨🇲', 'Canada': '🇨🇦', 'Central African Republic': '🇨🇫', 'Chad': '🇹🇩', 'Chile': '🇨🇱', 'China': '🇨🇳',
            'Colombia': '🇨🇴', 'Comoros': '🇰🇲', 'Congo (Congo-Brazzaville)': '🇨🇬', 'Congo (DRC)': '🇨🇩', 'Costa Rica': '🇨🇷',
            'Croatia': '🇭🇷', 'Cuba': '🇨🇺', 'Cyprus': '🇨🇾', 'Czech Republic': '🇨🇿', 'Denmark': '🇩🇰', 'Djibouti': '🇩🇯',
            'Dominica': '🇩🇲', 'Dominican Republic': '🇩🇴', 'Ecuador': '🇪🇨', 'Egypt': '🇪🇬', 'El Salvador': '🇸🇻', 'Equatorial Guinea': '🇬🇶',
            'Eritrea': '🇪🇷', 'Estonia': '🇪🇪', 'Eswatini': '🇸🇿', 'Ethiopia': '🇪🇹', 'Fiji': '🇫🇯', 'Finland': '🇫🇮', 'France': '🇫🇷',
            'Gabon': '🇬🇦', 'Gambia': '🇬🇲', 'Georgia': '🇬🇪', 'Germany': '🇩🇪', 'Ghana': '🇬🇭', 'Greece': '🇬🇷', 'Grenada': '🇬🇩',
            'Guatemala': '🇬🇹', 'Guinea': '🇬🇳', 'Guinea-Bissau': '🇬🇼', 'Guyana': '🇬🇾', 'Haiti': '🇭🇹', 'Honduras': '🇭🇳', 'Hungary': '🇭🇺',
            'Iceland': '🇮🇸', 'India': '🇮🇳', 'Indonesia': '🇮🇩', 'Iran': '🇮🇷', 'Iraq': '🇮🇶', 'Ireland': '🇮🇪', 'Israel': '🇮🇱',
            'Italy': '🇮🇹', 'Ivory Coast': '🇨🇮', 'Jamaica': '🇯🇲', 'Japan': '🇯🇵', 'Jordan': '🇯🇴', 'Kazakhstan': '🇰🇿', 'Kenya': '🇰🇪',
            'Kiribati': '🇰🇮', 'Kuwait': '🇰🇼', 'Kyrgyzstan': '🇰🇬', 'Laos': '🇱🇦', 'Latvia': '🇱🇻', 'Lebanon': '🇱🇧', 'Lesotho': '🇱🇸',
            'Liberia': '🇱🇷', 'Libya': '🇱🇾', 'Liechtenstein': '🇱🇮', 'Lithuania': '🇱🇹', 'Luxembourg': '🇱🇺', 'Madagascar': '🇲🇬',
            'Malawi': '🇲🇼', 'Malaysia': '🇲🇾', 'Maldives': '🇲🇻', 'Mali': '🇲🇱', 'Malta': '🇲🇹', 'Marshall Islands': '🇲🇭',
            'Mauritania': '🇲🇷', 'Mauritius': '🇲🇺', 'Mexico': '🇲🇽', 'Micronesia': '🇫🇲', 'Moldova': '🇲🇩', 'Monaco': '🇲🇨',
            'Mongolia': '🇲🇳', 'Montenegro': '🇲🇪', 'Morocco': '🇲🇦', 'Mozambique': '🇲🇿', 'Myanmar': '🇲🇲', 'Namibia': '🇳🇦',
            'Nauru': '🇳🇷', 'Nepal': '🇳🇵', 'Netherlands': '🇳🇱', 'New Zealand': '🇳🇿', 'Nicaragua': '🇳🇮', 'Niger': '🇳🇪',
            'Nigeria': '🇳🇬', 'North Korea': '🇰🇵', 'North Macedonia': '🇲🇰', 'Norway': '🇳🇴', 'Oman': '🇴🇲', 'Pakistan': '🇵🇰',
            'Palau': '🇵🇼', 'Panama': '🇵🇦', 'Papua New Guinea': '🇵🇬', 'Paraguay': '🇵🇾', 'Peru': '🇵🇪', 'Philippines': '🇵🇭',
            'Poland': '🇵🇱', 'Portugal': '🇵🇹', 'Qatar': '🇶🇦', 'Romania': '🇷🇴', 'Russia': '🇷🇺', 'Rwanda': '🇷🇼', 'Saint Kitts and Nevis': '🇰🇳',
            'Saint Lucia': '🇱🇨', 'Saint Vincent and the Grenadines': '🇻🇨', 'Samoa': '🇼🇸', 'San Marino': '🇸🇲', 'Saudi Arabia': '🇸🇦',
            'Senegal': '🇸🇳', 'Serbia': '🇷🇸', 'Seychelles': '🇸🇨', 'Sierra Leone': '🇸🇱', 'Singapore': '🇸🇬', 'Slovakia': '🇸🇰',
            'Slovenia': '🇸🇮', 'Solomon Islands': '🇸🇧', 'Somalia': '🇸🇴', 'South Africa': '🇿🇦', 'South Korea': '🇰🇷', 'South Sudan': '🇸🇸',
            'Spain': '🇪🇸', 'Sri Lanka': '🇱🇰', 'Sudan': '🇸🇩', 'Suriname': '🇸🇷', 'Sweden': '🇸🇪', 'Switzerland': '🇨🇭', 'Syria': '🇸🇾',
            'Taiwan': '🇹🇼', 'Tajikistan': '🇹🇯', 'Tanzania': '🇹🇿', 'Thailand': '🇹🇭', 'Timor-Leste': '🇹🇱', 'Togo': '🇹🇬',
            'Tonga': '🇹🇴', 'Trinidad and Tobago': '🇹🇹', 'Tunisia': '🇹🇳', 'Turkey': '🇹🇷', 'Turkmenistan': '🇹🇲', 'Tuvalu': '🇹🇻',
            'Uganda': '🇺🇬', 'Ukraine': '🇺🇦', 'United Arab Emirates': '🇦🇪', 'United Kingdom': '🇬🇧', 'United States': '🇺🇸',
            'Uruguay': '🇺🇾', 'Uzbekistan': '🇺🇿', 'Vanuatu': '🇻🇺', 'Vatican City': '🇻🇦', 'Venezuela': '🇻🇪', 'Vietnam': '🇻🇳',
            'Yemen': '🇾🇪', 'Zambia': '🇿🇲', 'Zimbabwe': '🇿🇼'
        }

        
        # Mostrar resultados
        for idx, result in enumerate(results, 1):
            with st_object.expander(f"📄 {idx}. {result['title']}", expanded=False):
                col1, col2 = st_object.columns([2,1])
                
                with col1:
                    st_object.markdown("**Descripción:**")
                    st_object.write(result['snippet'])
                    st_object.markdown(f"🔗 [Ver artículo completo]({result['link']})")
                
                with col2:
                    st_object.markdown("**Detalles:**")
                    st_object.markdown(f"📅 **Fecha:** {result.get('date', 'No especificada')}")
                    
                    # Mostrar país con bandera emoji
                    country = result.get('country', 'No especificado')
                    flag = country_emojis.get(country, '🌐')
                    st_object.markdown(f"🌍 **País:** {flag} {country}")
                    
                    # Mostrar sentimiento con color
                    sentiment = result.get('sentiment', 0)
                    sentiment_color = "green" if sentiment > 0 else "red"
                    st_object.markdown(
                        f"💭 **Sentimiento:** <span style='color:{sentiment_color}'>{sentiment:.2f}</span>",
                        unsafe_allow_html=True
                    )
                    
                    # Mostrar fuente
                    st_object.markdown(f"📰 **Fuente:** {result.get('source', 'No especificada')}")
                    
                    # Mostrar palabras clave si existen
                    if result.get('keywords'):
                        keywords = [k[0] for k in result['keywords'][:3]]  # Top 3 keywords
                        st_object.markdown(f"🏷️ **Keywords:** {', '.join(keywords)}")

                    
    def _calculate_position(self, growth, sentiment):
        """Calcula la posición en el Hype Cycle basada en métricas disponibles"""
        if growth > 0.3 and sentiment > 0:
            return 5  # Innovation Trigger
        elif sentiment > 0.3:
            return 20  # Peak of Expectations
        elif sentiment < 0:
            return 50  # Trough of Disillusionment
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
            'USA': ['united states', 'usa', 'u.s.'], 'UK': ['united kingdom', 'uk', 'britain'],
            'China': ['china', 'chinese'], 'Japan': ['japan', 'japanese'], 'Germany': ['germany', 'german'],
            'France': ['france', 'french'], 'Spain': ['spain', 'spanish'], 'Italy': ['italy', 'italian'],
            'India': ['india', 'indian'], 'Brazil': ['brazil', 'brazilian'], 'Canada': ['canada', 'canadian'],
            'Australia': ['australia', 'australian'], 'Mexico': ['mexico', 'mexican'], 'Russia': ['russia', 'russian'],
            'South Korea': ['south korea', 'korea'], 'North Korea': ['north korea', 'dprk'], 'Netherlands': ['netherlands', 'dutch'],
            'Sweden': ['sweden', 'swedish'], 'Switzerland': ['switzerland', 'swiss'], 'Singapore': ['singapore', 'singaporean'],
            'Hong Kong': ['hong kong'], 'Taiwan': ['taiwan', 'taiwanese'], 'New Zealand': ['new zealand', 'kiwi'],
            'Norway': ['norway', 'norwegian'], 'Denmark': ['denmark', 'danish'], 'Finland': ['finland', 'finnish'],
            'Belgium': ['belgium', 'belgian'], 'Austria': ['austria', 'austrian'], 'Ireland': ['ireland', 'irish'],
            'Portugal': ['portugal', 'portuguese'], 'Greece': ['greece', 'greek'], 'Poland': ['poland', 'polish'],
            'Czech Republic': ['czech republic', 'czech'], 'Slovakia': ['slovakia', 'slovak'], 'Hungary': ['hungary', 'hungarian'],
            'Turkey': ['turkey', 'turkish'], 'South Africa': ['south africa', 'south african'], 'Argentina': ['argentina', 'argentinian'],
            'Chile': ['chile', 'chilean'], 'Colombia': ['colombia', 'colombian'], 'Peru': ['peru', 'peruvian'],
            'Venezuela': ['venezuela', 'venezuelan'], 'Ecuador': ['ecuador', 'ecuadorian'], 'Paraguay': ['paraguay', 'paraguayan'],
            'Uruguay': ['uruguay', 'uruguayan'], 'Bolivia': ['bolivia', 'bolivian'], 'Panama': ['panama', 'panamanian'],
            'Costa Rica': ['costa rica', 'costa rican'], 'El Salvador': ['el salvador', 'salvadoran'],
            'Guatemala': ['guatemala', 'guatemalan'], 'Honduras': ['honduras', 'honduran'], 'Nicaragua': ['nicaragua', 'nicaraguan'],
            'Cuba': ['cuba', 'cuban'], 'Dominican Republic': ['dominican republic', 'dominican'],
            'Jamaica': ['jamaica', 'jamaican'], 'Trinidad and Tobago': ['trinidad and tobago', 'trinidadian'],
            'Egypt': ['egypt', 'egyptian'], 'Nigeria': ['nigeria', 'nigerian'], 'Kenya': ['kenya', 'kenyan'],
            'Ethiopia': ['ethiopia', 'ethiopian'], 'Uganda': ['uganda', 'ugandan'], 'Ghana': ['ghana', 'ghanaian'],
            'Algeria': ['algeria', 'algerian'], 'Morocco': ['morocco', 'moroccan'], 'Tunisia': ['tunisia', 'tunisian'],
            'Libya': ['libya', 'libyan'], 'Sudan': ['sudan', 'sudanese'], 'South Sudan': ['south sudan', 'south sudanese'],
            'Angola': ['angola', 'angolan'], 'Zambia': ['zambia', 'zambian'], 'Zimbabwe': ['zimbabwe', 'zimbabwean'],
            'Mozambique': ['mozambique', 'mozambican'], 'Namibia': ['namibia', 'namibian'], 'Botswana': ['botswana', 'botswanan'],
            'Madagascar': ['madagascar', 'malagasy'], 'Democratic Republic of the Congo': ['dr congo', 'democratic republic of congo'],
            'Congo': ['congo', 'republic of congo'], 'Rwanda': ['rwanda', 'rwandan'], 'Burundi': ['burundi', 'burundian'],
            'Tanzania': ['tanzania', 'tanzanian'], 'Saudi Arabia': ['saudi arabia', 'saudi'], 'UAE': ['uae', 'united arab emirates'],
            'Qatar': ['qatar', 'qatari'], 'Bahrain': ['bahrain', 'bahraini'], 'Kuwait': ['kuwait', 'kuwaiti'],
            'Oman': ['oman', 'omani'], 'Lebanon': ['lebanon', 'lebanese'], 'Jordan': ['jordan', 'jordanian'],
            'Syria': ['syria', 'syrian'], 'Iraq': ['iraq', 'iraqi'], 'Iran': ['iran', 'iranian'], 'Pakistan': ['pakistan', 'pakistani'],
            'Afghanistan': ['afghanistan', 'afghan'], 'Bangladesh': ['bangladesh', 'bangladeshi'], 'Sri Lanka': ['sri lanka', 'sri lankan'],
            'Nepal': ['nepal', 'nepali'], 'Bhutan': ['bhutan', 'bhutanese'], 'Maldives': ['maldives', 'maldivian'],
            'Indonesia': ['indonesia', 'indonesian'], 'Malaysia': ['malaysia', 'malaysian'], 'Philippines': ['philippines', 'filipino'],
            'Thailand': ['thailand', 'thai'], 'Vietnam': ['vietnam', 'vietnamese'], 'Cambodia': ['cambodia', 'cambodian'],
            'Laos': ['laos', 'laotian'], 'Myanmar': ['myanmar', 'burmese'], 'Mongolia': ['mongolia', 'mongolian'],
            'Kazakhstan': ['kazakhstan', 'kazakh'], 'Uzbekistan': ['uzbekistan', 'uzbek'], 'Turkmenistan': ['turkmenistan', 'turkmen'],
            'Kyrgyzstan': ['kyrgyzstan', 'kyrgyz'], 'Tajikistan': ['tajikistan', 'tajik'], 'Georgia': ['georgia', 'georgian'],
            'Armenia': ['armenia', 'armenian'], 'Azerbaijan': ['azerbaijan', 'azerbaijani'], 'Belarus': ['belarus', 'belarusian'],
            'Ukraine': ['ukraine', 'ukrainian'], 'Moldova': ['moldova', 'moldovan'], 'Lithuania': ['lithuania', 'lithuanian'],
            'Latvia': ['latvia', 'latvian'], 'Estonia': ['estonia', 'estonian'], 'Serbia': ['serbia', 'serbian'],
            'Montenegro': ['montenegro', 'montenegrin'], 'Bosnia and Herzegovina': ['bosnia and herzegovina', 'bosnian'],
            'North Macedonia': ['north macedonia', 'macedonian'], 'Albania': ['albania', 'albanian'], 'Slovenia': ['slovenia', 'slovenian'],
            'Croatia': ['croatia', 'croatian']
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
        """
        Crea un mapa mundial de noticias - VERSIÓN CORREGIDA
        Mantiene las coordenadas de países existentes y corrige el error de columnas
        """
        try:
            # Verificar si hay datos de ubicación
            if df.empty or 'country' not in df.columns:
                st_object.info("No hay datos de ubicación disponibles para crear el mapa")
                return
            
            # Contar menciones por país
            country_counts = df['country'].value_counts()
            
            if country_counts.empty:
                st_object.info("No hay datos de países válidos para crear el mapa")
                return
            
            # MANTENER coordenadas de países existentes (como solicitó el usuario)
            country_coords = {
            'USA': {'coords': [37.0902, -95.7129], 'full_name': 'Estados Unidos'},
            'UK': {'coords': [55.3781, -3.4360], 'full_name': 'Reino Unido'},
            'China': {'coords': [35.8617, 104.1954], 'full_name': 'China'},
            'Japan': {'coords': [36.2048, 138.2529], 'full_name': 'Japón'},
            'Germany': {'coords': [51.1657, 10.4515], 'full_name': 'Alemania'},
            'France': {'coords': [46.2276, 2.2137], 'full_name': 'Francia'},
            'Spain': {'coords': [40.4637, -3.7492], 'full_name': 'España'},
            'Italy': {'coords': [41.8719, 12.5674], 'full_name': 'Italia'},
            'India': {'coords': [20.5937, 78.9629], 'full_name': 'India'},
            'Brazil': {'coords': [-14.2350, -51.9253], 'full_name': 'Brasil'},
            'Canada': {'coords': [56.1304, -106.3468], 'full_name': 'Canadá'},
            'Australia': {'coords': [-25.2744, 133.7751], 'full_name': 'Australia'},
            'South Korea': {'coords': [35.9078, 127.7669], 'full_name': 'Corea del Sur'},
            'Russia': {'coords': [61.5240, 105.3188], 'full_name': 'Rusia'},
            'Netherlands': {'coords': [52.1326, 5.2913], 'full_name': 'Países Bajos'},
            'Sweden': {'coords': [60.1282, 18.6435], 'full_name': 'Suecia'},
            'Israel': {'coords': [31.0461, 34.8516], 'full_name': 'Israel'},
            'Switzerland': {'coords': [46.8182, 8.2275], 'full_name': 'Suiza'},
            'Norway': {'coords': [60.4720, 8.4689], 'full_name': 'Noruega'},
            'Denmark': {'coords': [56.2639, 9.5018], 'full_name': 'Dinamarca'},
            'Finland': {'coords': [61.9241, 25.7482], 'full_name': 'Finlandia'},
            'Belgium': {'coords': [50.5039, 4.4699], 'full_name': 'Bélgica'},
            'Austria': {'coords': [47.5162, 14.5501], 'full_name': 'Austria'},
            'Ireland': {'coords': [53.1424, -7.6921], 'full_name': 'Irlanda'},
            'Portugal': {'coords': [39.3999, -8.2245], 'full_name': 'Portugal'},
            'Greece': {'coords': [39.0742, 21.8243], 'full_name': 'Grecia'},
            'Poland': {'coords': [51.9194, 19.1451], 'full_name': 'Polonia'},
            'Czech Republic': {'coords': [49.8175, 15.4729], 'full_name': 'República Checa'},
            'Turkey': {'coords': [38.9637, 35.2433], 'full_name': 'Turquía'},
            'South Africa': {'coords': [-30.5595, 22.9375], 'full_name': 'Sudáfrica'},
            'Argentina': {'coords': [-38.4161, -63.6167], 'full_name': 'Argentina'},
            'Chile': {'coords': [-35.6751, -71.5430], 'full_name': 'Chile'},
            'Colombia': {'coords': [4.5709, -74.2973], 'full_name': 'Colombia'},
            'Peru': {'coords': [-9.1900, -75.0152], 'full_name': 'Perú'},
            'Egypt': {'coords': [26.8206, 30.8025], 'full_name': 'Egipto'},
            'Nigeria': {'coords': [9.0820, 8.6753], 'full_name': 'Nigeria'},
            'Kenya': {'coords': [-1.2921, 36.8219], 'full_name': 'Kenia'},
            'Croatia': {'coords': [45.1000, 15.2000], 'full_name': 'Croacia'},
            'UAE': {'coords': [23.4241, 53.8478], 'full_name': 'Emiratos Árabes Unidos'},
            'Saudi Arabia': {'coords': [23.8859, 45.0792], 'full_name': 'Arabia Saudita'},
            'Qatar': {'coords': [25.3548, 51.1839], 'full_name': 'Catar'},
            'Singapore': {'coords': [1.3521, 103.8198], 'full_name': 'Singapur'},
            'Hong Kong': {'coords': [22.3193, 114.1694], 'full_name': 'Hong Kong'},
            'Taiwan': {'coords': [23.6978, 120.9605], 'full_name': 'Taiwán'},
            'Afghanistan': {'coords': [33.9391, 67.7100], 'full_name': 'Afganistán'},
            'Bangladesh': {'coords': [23.6850, 90.3563], 'full_name': 'Bangladesh'},
            'Bhutan': {'coords': [27.5142, 90.4336], 'full_name': 'Bután'},
            'Pakistan': {'coords': [30.3753, 69.3451], 'full_name': 'Pakistán'},
            'Sri Lanka': {'coords': [7.8731, 80.7718], 'full_name': 'Sri Lanka'},
            'Maldives': {'coords': [3.2028, 73.2207], 'full_name': 'Maldivas'},
            'Nepal': {'coords': [28.3949, 84.1240], 'full_name': 'Nepal'},
            'Myanmar': {'coords': [21.9162, 95.9560], 'full_name': 'Myanmar'},
            'Cambodia': {'coords': [12.5657, 104.9910], 'full_name': 'Camboya'},
            'Laos': {'coords': [19.8563, 102.4955], 'full_name': 'Laos'},
            'Vietnam': {'coords': [14.0583, 108.2772], 'full_name': 'Vietnam'},
            'Thailand': {'coords': [15.8700, 100.9925], 'full_name': 'Tailandia'},
            'Malaysia': {'coords': [4.2105, 101.9758], 'full_name': 'Malasia'},
            'Singapore': {'coords': [1.3521, 103.8198], 'full_name': 'Singapur'},
            'Indonesia': {'coords': [-0.7893, 113.9213], 'full_name': 'Indonesia'},
            'Philippines': {'coords': [12.8797, 121.7740], 'full_name': 'Filipinas'},
            'Brunei': {'coords': [4.5353, 114.7277], 'full_name': 'Brunéi'},
            'Timor-Leste': {'coords': [-8.8742, 125.7275], 'full_name': 'Timor Oriental'},
            'New Zealand': {'coords': [-40.9006, 174.8860], 'full_name': 'Nueva Zelanda'},
            'Fiji': {'coords': [-17.7134, 178.0650], 'full_name': 'Fiyi'},
            'Solomon Islands': {'coords': [-9.6457, 160.1562], 'full_name': 'Islas Salomón'},
            'Vanuatu': {'coords': [-15.3767, 166.9592], 'full_name': 'Vanuatu'},
            'Papua New Guinea': {'coords': [-6.3149, 143.9556], 'full_name': 'Papúa Nueva Guinea'},
            'Samoa': {'coords': [-13.7590, -172.1046], 'full_name': 'Samoa'},
            'Tonga': {'coords': [-21.1789, -175.1982], 'full_name': 'Tonga'},
            'Kiribati': {'coords': [1.8709, -157.3623], 'full_name': 'Kiribati'},
            'Tuvalu': {'coords': [-7.1095, 177.6493], 'full_name': 'Tuvalu'},
            'Marshall Islands': {'coords': [7.1315, 171.1845], 'full_name': 'Islas Marshall'},
            'Micronesia': {'coords': [7.4256, 150.5508], 'full_name': 'Micronesia'},
            'Palau': {'coords': [7.5150, 134.5825], 'full_name': 'Palaos'},
            'Nauru': {'coords': [-0.5228, 166.9315], 'full_name': 'Nauru'},
            'Kiribati': {'coords': [1.8709, -157.3623], 'full_name': 'Kiribati'},
            'Ecuador': {'coords': [-1.8312, -78.1834], 'full_name': 'Ecuador'},
            'Bolivia': {'coords': [-16.2902, -63.5887], 'full_name': 'Bolivia'},
            'Paraguay': {'coords': [-23.4425, -58.4438], 'full_name': 'Paraguay'},
            'Uruguay': {'coords': [-32.5228, -55.7658], 'full_name': 'Uruguay'},
            'Panama': {'coords': [8.5379, -80.7821], 'full_name': 'Panamá'},
            'Costa Rica': {'coords': [9.7489, -83.7534], 'full_name': 'Costa Rica'},
            'Honduras': {'coords': [15.2000, -86.2419], 'full_name': 'Honduras'},
            'El Salvador': {'coords': [13.7942, -88.8965], 'full_name': 'El Salvador'},
            'Guatemala': {'coords': [15.7835, -90.2308], 'full_name': 'Guatemala'},
            'Nicaragua': {'coords': [12.8654, -85.2072], 'full_name': 'Nicaragua'},
            'Cuba': {'coords': [21.5218, -77.7812], 'full_name': 'Cuba'},
            'Jamaica': {'coords': [18.1096, -77.2975], 'full_name': 'Jamaica'},
            'Trinidad and Tobago': {'coords': [10.6918, -61.2225], 'full_name': 'Trinidad y Tobago'},
            'Haiti': {'coords': [18.9712, -72.2852], 'full_name': 'Haití'},
            'Dominican Republic': {'coords': [18.7357, -70.1627], 'full_name': 'República Dominicana'},
            'Bahamas': {'coords': [25.0343, -77.3963], 'full_name': 'Bahamas'},
            'Barbados': {'coords': [13.1939, -59.5432], 'full_name': 'Barbados'},
            'Saint Lucia': {'coords': [13.9094, -60.9789], 'full_name': 'Santa Lucía'},
            'Saint Vincent and the Grenadines': {'coords': [12.9843, -61.2872], 'full_name': 'San Vicente y las Granadinas'},
            'Grenada': {'coords': [12.1165, -61.6790], 'full_name': 'Granada'},
            'Saint Kitts and Nevis': {'coords': [17.3578, -62.782998], 'full_name': 'San Cristóbal y Nieves'},
            'Antigua and Barbuda': {'coords': [17.0608, -61.7964], 'full_name': 'Antigua y Barbuda'},
            'Dominica': {'coords': [15.4150, -61.3710], 'full_name': 'Dominica'},
            'Saint Kitts and Nevis': {'coords': [17.3578, -62.782998], 'full_name': 'San Cristóbal y Nieves'},
            'Monaco': {'coords': [43.7384, 7.4246], 'full_name': 'Mónaco'},
            'Andorra': {'coords': [42.5063, 1.5218], 'full_name': 'Andorra'},
            'San Marino': {'coords': [43.9424, 12.4578], 'full_name': 'San Marino'},
            'Liechtenstein': {'coords': [47.1660, 9.5554], 'full_name': 'Liechtenstein'},
            'Vatican City': {'coords': [41.9029, 12.4534], 'full_name': 'Ciudad del Vaticano'},
            'Moldova': {'coords': [47.4116, 28.3699], 'full_name': 'Moldavia'},
            'Belarus': {'coords': [53.7098, 27.9534], 'full_name': 'Bielorrusia'},
            'Ukraine': {'coords': [48.3794, 31.1656], 'full_name': 'Ucrania'},
            'Armenia': {'coords': [40.0691, 45.0382], 'full_name': 'Armenia'},
            'Azerbaijan': {'coords': [40.1431, 47.5769], 'full_name': 'Azerbaiyán'},
            'Georgia': {'coords': [42.3154, 43.3569], 'full_name': 'Georgia'},
            'Kazakhstan': {'coords': [48.0196, 66.9237], 'full_name': 'Kazajistán'},
            'Uzbekistan': {'coords': [41.3775, 64.5853], 'full_name': 'Uzbekistán'},
            'Turkmenistan': {'coords': [38.9697, 59.5563], 'full_name': 'Turkmenistán'},
            'Kyrgyzstan': {'coords': [41.2044, 74.7661], 'full_name': 'Kirguistán'},
            'Tajikistan': {'coords': [38.8610, 71.2761], 'full_name': 'Tayikistán'},
            'Montenegro': {'coords': [42.7087, 19.3744], 'full_name': 'Montenegro'},
            'Serbia': {'coords': [44.0165, 21.0059], 'full_name': 'Serbia'},
            'Bosnia and Herzegovina': {'coords': [43.9159, 17.6791], 'full_name': 'Bosnia y Herzegovina'},
            'Mongolia': {'coords': [46.8625, 103.8467], 'full_name': 'Mongolia'},
            'North Korea': {'coords': [40.3399, 127.5101], 'full_name': 'Corea del Norte'},
            'South Korea': {'coords': [35.9078, 127.7669], 'full_name': 'Corea del Sur'},
            'North Macedonia': {'coords': [41.6086, 21.7453], 'full_name': 'Macedonia del Norte'},
            'Morocco': {'coords': [31.7917, -7.0926], 'full_name': 'Marruecos'},
            'Algeria': {'coords': [28.0339, 1.6596], 'full_name': 'Argelia'},
            'Tunisia': {'coords': [33.8869, 9.5375], 'full_name': 'Túnez'},
            'Libya': {'coords': [26.3351, 17.2283], 'full_name': 'Libia'},
            'Sudan': {'coords': [12.8628, 30.2176], 'full_name': 'Sudán'},
            'South Sudan': {'coords': [6.8770, 31.3070], 'full_name': 'Sudán del Sur'},
            'Ethiopia': {'coords': [9.1450, 40.4897], 'full_name': 'Etiopía'},
            'Eritrea': {'coords': [15.1794, 39.7823], 'full_name': 'Eritrea'},
            'Djibouti': {'coords': [11.8251, 42.5903], 'full_name': 'Yibuti'},
            'Somalia': {'coords': [5.1521, 46.1996], 'full_name': 'Somalia'},
            'Ghana': {'coords': [7.9465, -1.0232], 'full_name': 'Ghana'},
            'Ivory Coast': {'coords': [7.5400, -5.5471], 'full_name': 'Costa de Marfil'},
            'Senegal': {'coords': [14.4974, -14.4524], 'full_name': 'Senegal'},
            'Mali': {'coords': [17.5707, -3.9962], 'full_name': 'Mali'},
            'Burkina Faso': {'coords': [12.2383, -1.5616], 'full_name': 'Burkina Faso'},
            'Niger': {'coords': [17.6078, 8.0817], 'full_name': 'Níger'},
            'Chad': {'coords': [15.4542, 18.7322], 'full_name': 'Chad'},
            'Cameroon': {'coords': [7.3697, 12.3547], 'full_name': 'Camerún'},
            'Central African Republic': {'coords': [6.6111, 20.9394], 'full_name': 'República Centroafricana'},
            'Uganda': {'coords': [1.3733, 32.2903], 'full_name': 'Uganda'},
            'Rwanda': {'coords': [-1.9403, 29.8739], 'full_name': 'Ruanda'},
            'Burundi': {'coords': [-3.3731, 29.9189], 'full_name': 'Burundi'},
            'Tanzania': {'coords': [-6.369028, 34.888822], 'full_name': 'Tanzania'},
            'Madagascar': {'coords': [-18.766947, 46.869107], 'full_name': 'Madagascar'},
            'Mozambique': {'coords': [-18.665695, 35.529562], 'full_name': 'Mozambique'},
            'Zambia': {'coords': [-13.133897, 27.849332], 'full_name': 'Zambia'},
            'Zimbabwe': {'coords': [-19.015438, 29.154857], 'full_name': 'Zimbabue'},
            'Namibia': {'coords': [-22.957640, 18.490410], 'full_name': 'Namibia'},
            'Botswana': {'coords': [-22.328474, 24.684866], 'full_name': 'Botsuana'},
            'Lesotho': {'coords': [-29.609988, 28.233608], 'full_name': 'Lesoto'},
            'Eswatini': {'coords': [-26.522503, 31.465866], 'full_name': 'Esuatini'},
            'Angola': {'coords': [-11.202692, 17.873887], 'full_name': 'Angola'},
            'DR Congo': {'coords': [-4.038333, 21.758664], 'full_name': 'República Democrática del Congo'},
            'Congo': {'coords': [-0.228021, 15.827659], 'full_name': 'República del Congo'},
            'Gabon': {'coords': [-0.803689, 11.609444], 'full_name': 'Gabón'},
            'Luxembourg': {'coords': [49.8153, 6.1296], 'full_name': 'Luxemburgo'},
            'Hungary': {'coords': [47.1625, 19.5033], 'full_name': 'Hungría'},
            'Slovakia': {'coords': [48.6690, 19.6990], 'full_name': 'Eslovaquia'},
            'Slovenia': {'coords': [46.1512, 14.9955], 'full_name': 'Eslovenia'},
            'Romania': {'coords': [45.9432, 24.9668], 'full_name': 'Rumanía'},
            'Bulgaria': {'coords': [42.7339, 25.4858], 'full_name': 'Bulgaria'},
            'Albania': {'coords': [41.1533, 20.1683], 'full_name': 'Albania'},
            'Kosovo': {'coords': [42.6026, 20.9030], 'full_name': 'Kosovo'},
            'Estonia': {'coords': [58.5953, 25.0136], 'full_name': 'Estonia'},
            'Latvia': {'coords': [56.8796, 24.6032], 'full_name': 'Letonia'},
            'Lithuania': {'coords': [55.1694, 23.8813], 'full_name': 'Lituania'},
            'Iceland': {'coords': [64.9631, -19.0208], 'full_name': 'Islandia'},
            'Malta': {'coords': [35.9375, 14.3754], 'full_name': 'Malta'},
            'Cyprus': {'coords': [35.1264, 33.4299], 'full_name': 'Chipre'},
            'Iraq': {'coords': [33.2232, 43.6793], 'full_name': 'Irak'},
            'Iran': {'coords': [32.4279, 53.6880], 'full_name': 'Irán'},
            'Syria': {'coords': [34.8021, 38.9968], 'full_name': 'Siria'},
            'Lebanon': {'coords': [33.8547, 35.8623], 'full_name': 'Líbano'},
            'Jordan': {'coords': [30.5852, 36.2384], 'full_name': 'Jordania'},
            'Yemen': {'coords': [15.5527, 48.5164], 'full_name': 'Yemen'},
            'Oman': {'coords': [21.4735, 55.9754], 'full_name': 'Omán'},
            'Kuwait': {'coords': [29.3117, 47.4818], 'full_name': 'Kuwait'},
            'Bahrain': {'coords': [26.0667, 50.5577], 'full_name': 'Baréin'},
            'Madagascar': {'coords': [-18.7669, 46.8691], 'full_name': 'Madagascar'},
            'New Caledonia': {'coords': [-20.9043, 165.6180], 'full_name': 'Nueva Caledonia'},
            'French Polynesia': {'coords': [-17.6797, -149.4068], 'full_name': 'Polinesia Francesa'},
            'Greenland': {'coords': [71.7069, -42.6043], 'full_name': 'Groenlandia'},
            'Guyana': {'coords': [4.8604, -58.9302], 'full_name': 'Guyana'},
            'Suriname': {'coords': [3.9193, -56.0278], 'full_name': 'Surinam'},
            'French Guiana': {'coords': [3.9339, -53.1258], 'full_name': 'Guayana Francesa'},
            'Belize': {'coords': [17.1899, -88.4976], 'full_name': 'Belice'},
        }
            
            # Preparar datos para el mapa
            map_data = []
            stats_data = []  # Para la tabla de estadísticas
            
            for country, count in country_counts.items():
                # Limpiar nombre del país
                country_clean = str(country).strip()
                
                if country_clean in country_coords:
                    coords = country_coords[country_clean]
                    map_data.append({
                        'country': country_clean,
                        'display_name': coords['name'],
                        'lat': coords['lat'],
                        'lon': coords['lon'],
                        'count': int(count),  # Asegurar que sea entero
                        'size': min(max(count * 3, 8), 50)  # Tamaño para el mapa
                    })
                    
                    # CORREGIR: Usar nombres de columnas consistentes
                    stats_data.append({
                        'País': coords['name'],
                        'Codigo': country_clean,
                        'Total': int(count),  # CAMBIAR 'Menciones' por 'Total'
                        'Porcentaje': round((count / len(df)) * 100, 1)
                    })
            
            if not map_data:
                st_object.warning("No se encontraron coordenadas para los países en los datos")
                return
            
            # Crear DataFrame para el mapa
            map_df = pd.DataFrame(map_data)
            
            # Crear mapa usando plotly
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Crear figura del mapa
            fig = px.scatter_geo(
                map_df,
                lat='lat',
                lon='lon',
                size='count',
                hover_name='display_name',
                hover_data={
                    'lat': False,
                    'lon': False,
                    'count': ':d',
                    'display_name': False
                },
                size_max=50,
                projection='natural earth',
                title="Distribución Geográfica de Noticias",
                color='count',
                color_continuous_scale='viridis'
            )
            
            # Personalizar el mapa
            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" +
                            "Noticias: %{marker.size}<br>" +
                            "<extra></extra>"
            )
            
            fig.update_layout(
                title={
                    'text': "Distribución Geográfica de Noticias",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                geo=dict(
                    showframe=False,
                    showcoastlines=True,
                    projection_type='natural earth'
                ),
                height=500
            )
            
            # Mostrar el mapa
            st_object.plotly_chart(fig, use_container_width=True)
            
            # CORREGIR: Crear tabla de estadísticas con nombres de columnas correctos
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                
                # CORREGIR: Usar 'Total' en lugar de 'Menciones'
                try:
                    stats_df = stats_df.sort_values('Total', ascending=False)
                except KeyError:
                    # Si 'Total' no existe, intentar con otros nombres posibles
                    possible_columns = ['Menciones', 'count', 'Count', 'total']
                    sort_column = 'Total'  # default
                    
                    for col in possible_columns:
                        if col in stats_df.columns:
                            sort_column = col
                            break
                    
                    stats_df = stats_df.sort_values(sort_column, ascending=False)
                
                # Mostrar tabla de estadísticas
                st_object.subheader("📊 Estadísticas por País")
                
                # Configurar columnas para mejor visualización
                column_config = {
                    'País': st_object.column_config.TextColumn("País", width="medium"),
                    'Codigo': st_object.column_config.TextColumn("Código", width="small"),
                    'Total': st_object.column_config.NumberColumn("Noticias", width="small"),
                    'Porcentaje': st_object.column_config.NumberColumn("% Total", format="%.1f%%", width="small")
                }
                
                st_object.dataframe(
                    stats_df, 
                    use_container_width=True,
                    hide_index=True,
                    column_config=column_config
                )
                
                # Métricas adicionales
                col1, col2, col3 = st_object.columns(3)
                
                with col1:
                    total_countries = len(stats_df)
                    st_object.metric("Países con Noticias", total_countries)
                
                with col2:
                    top_country = stats_df.iloc[0] if not stats_df.empty else None
                    if top_country is not None:
                        st_object.metric("País Líder", top_country['País'])
                
                with col3:
                    total_news = stats_df['Total'].sum() if 'Total' in stats_df.columns else 0
                    st_object.metric("Total Noticias Mapeadas", total_news)
        
        except Exception as e:
            st_object.error(f"Error creando mapa de noticias: {str(e)}")
            
            # Debug información
            with st_object.expander("🔍 Información de Debug"):
                st_object.write(f"**Error:** {str(e)}")
                if not df.empty:
                    st_object.write(f"**Columnas disponibles:** {list(df.columns)}")
                    st_object.write(f"**Filas en DataFrame:** {len(df)}")
                    if 'country' in df.columns:
                        unique_countries = df['country'].unique()
                        st_object.write(f"**Países únicos:** {len(unique_countries)}")
                        st_object.write(f"**Primeros países:** {list(unique_countries)[:10]}")
                else:
                    st_object.write("**DataFrame está vacío**")

    def _plot_keyword_analysis(self, df, st_object):
        """Analiza y visualiza palabras clave con filtrado personalizado"""
        # Definir stopwords y palabras bloqueadas
        default_stopwords = set([
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
            'give', 'day', 'most', 'us', 'using', 'great', 'must', 'go', 'may'
            
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