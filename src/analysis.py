import pandas as pd
import re
from collections import Counter
from dotenv import load_dotenv
import nltk
import sqlite3
import json
import os
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from datetime import datetime
import numpy as np
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googleapiclient.discovery import build

import sqlite3
import json
from datetime import datetime
import os

class PersistentCache:
    def __init__(self, db_path='data/cache.db'):
        os.makedirs('data', exist_ok=True)
        self.db_path = db_path
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS searches (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    query TEXT NOT NULL,
                    results TEXT NOT NULL,
                    query_info TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def serialize_data(self, data):
        """Convierte los datos a formato JSON manteniendo la estructura"""
        if isinstance(data, (list, dict)):
            return json.dumps(data, default=str)
        return json.dumps([])

    def deserialize_data(self, data_str, default=None):
        """Convierte JSON a diccionarios/listas de Python"""
        try:
            return json.loads(data_str)
        except:
            return default if default is not None else []

    def add_search(self, query, results, query_info):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Serializar datos asegurando estructura correcta
            results_json = self.serialize_data(results)
            query_info_json = self.serialize_data(query_info)
            
            cursor.execute('SELECT id FROM searches WHERE query = ?', (query,))
            existing = cursor.fetchone()
            
            if existing:
                cursor.execute('''
                    UPDATE searches 
                    SET results = ?, query_info = ?, timestamp = CURRENT_TIMESTAMP
                    WHERE query = ?
                ''', (results_json, query_info_json, query))
            else:
                cursor.execute('SELECT COUNT(*) FROM searches')
                count = cursor.fetchone()[0]
                
                if count >= 3:
                    cursor.execute('DELETE FROM searches WHERE id IN (SELECT id FROM searches ORDER BY timestamp ASC LIMIT 1)')
                
                cursor.execute('''
                    INSERT INTO searches (query, results, query_info)
                    VALUES (?, ?, ?)
                ''', (query, results_json, query_info_json))
            
            conn.commit()

    def get_search(self, query):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT results, query_info, timestamp 
                FROM searches 
                WHERE query = ?
            ''', (query,))
            
            result = cursor.fetchone()
            if result:
                return {
                    'results': self.deserialize_data(result[0], []),
                    'query_info': self.deserialize_data(result[1], {}),
                    'timestamp': result[2]
                }
            return None

    def get_recent_searches(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT query, results, timestamp 
                FROM searches 
                ORDER BY timestamp DESC
            ''')
            
            searches = []
            for row in cursor.fetchall():
                results = self.deserialize_data(row[1], [])
                searches.append({
                    'query': row[0],
                    'result_count': len(results),
                    'timestamp': row[2]
                })
            
            return searches

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
            'give', 'day', 'most', 'us', 'was', 'is', 'are', 'were', 'been'
        }

    def extract_year(self, text):
        years = re.findall(r'\b20\d{2}\b', text)
        return int(min(years)) if years else None

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

    def analyze_gartner_points(self, yearly_stats):
        """Analiza los puntos de inflexión según el modelo de Gartner"""
        # Convertir a DataFrame si no lo es
        df = pd.DataFrame(yearly_stats)
        df = df.sort_values('year')
        
        # Calcular cambios porcentuales y tendencias
        df['mention_pct_change'] = df['mention_count'].pct_change()
        df['sentiment_change'] = df['sentiment_mean'].diff()
        
        # Identificar puntos de inflexión
        inflection_points = {
            'innovation_trigger': None,
            'peak': None,
            'trough': None,
            'slope': None,
            'plateau': None
        }
        
        # Buscar punto de innovación (primer año con menciones significativas)
        if not df.empty:
            first_significant = df[df['mention_count'] >= df['mention_count'].mean() * 0.1].iloc[0]
            inflection_points['innovation_trigger'] = {
                'year': first_significant['year'],
                'mentions': first_significant['mention_count'],
                'sentiment': first_significant['sentiment_mean']
            }
            
            # Buscar pico (máximo de menciones)
            peak_row = df.loc[df['mention_count'].idxmax()]
            inflection_points['peak'] = {
                'year': peak_row['year'],
                'mentions': peak_row['mention_count'],
                'sentiment': peak_row['sentiment_mean']
            }
            
            # Buscar valle (mínimo después del pico)
            post_peak = df[df['year'] > peak_row['year']]
            if not post_peak.empty:
                trough_row = post_peak.loc[post_peak['mention_count'].idxmin()]
                inflection_points['trough'] = {
                    'year': trough_row['year'],
                    'mentions': trough_row['mention_count'],
                    'sentiment': trough_row['sentiment_mean']
                }
        
        return inflection_points

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

    def perform_news_search(self, api_key, search_engine_id, query):
        """Realiza búsqueda específica en noticias"""
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            all_results = []
            
            # Modificar query para noticias
            news_query = f"{query} site:news.google.com"
            
            for start_index in range(1, 91, 10):
                result = service.cse().list(
                    q=news_query,
                    cx=search_engine_id,
                    num=10,
                    start=start_index,
                    sort='date'  # Ordenar por fecha
                ).execute()
                
                items = result.get('items', [])
                if not items:
                    break
                    
                all_results.extend(items)
            
            return True, all_results
        except Exception as e:
            return False, str(e)

    

    def analyze_hype_cycle(self, news_results):
        """Analiza resultados para determinar posición en Hype Cycle"""
        analyzed_results = []
        
        for item in news_results:
            try:
                # Extraer fecha del snippet o título
                text = f"{item.get('title', '')} {item.get('snippet', '')}"
                # Buscar patrones de fecha en el texto
                date_pattern = r'(\d{4})'
                year_match = re.search(date_pattern, text)
                
                if year_match:
                    year = year_match.group(1)
                else:
                    continue  # Saltar si no hay año
                
                # Analizar sentimiento
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                
                analyzed_results.append({
                    'year': year,
                    'year_month': f"{year}-01",  # Usar enero como mes por defecto
                    'sentiment': sentiment['compound'],
                    'text': text,
                    'title': item.get('title', 'Sin título'),
                    'link': item.get('link', '#')
                })
                
            except Exception as e:
                print(f"Error procesando noticia: {str(e)}")
                continue
        
        if not analyzed_results:
            return {
                'phase': "No hay suficientes datos",
                'monthly_stats': pd.DataFrame(),
                'sentiment_trend': pd.Series(),
                'results': []
            }
        
        # Convertir a DataFrame
        df = pd.DataFrame(analyzed_results)
        
        # Agrupar por año
        yearly_stats = df.groupby('year').agg({
            'sentiment': ['mean', 'count']
        }).reset_index()
        
        yearly_stats.columns = ['year', 'sentiment_mean', 'mention_count']
        
        # Determinar fase del Hype Cycle
        latest_stats = yearly_stats.iloc[-1]
        avg_sentiment = latest_stats['sentiment_mean']
        mention_trend = yearly_stats['mention_count'].pct_change().mean()
        
        # Lógica para determinar la fase
        if mention_trend > 0.5 and avg_sentiment > 0:
            phase = "Innovation Trigger"
        elif avg_sentiment > 0.3 and mention_trend > 0:
            phase = "Peak of Inflated Expectations"
        elif avg_sentiment < 0 or mention_trend < -0.2:
            phase = "Trough of Disillusionment"
        elif avg_sentiment > 0 and mention_trend > 0:
            phase = "Slope of Enlightenment"
        else:
            phase = "Plateau of Productivity"
        
        return {
            'phase': phase,
            'yearly_stats': yearly_stats,
            'sentiment_trend': yearly_stats['sentiment_mean'],
            'results': analyzed_results
        }

    def plot_hype_cycle(self, hype_data):
        """Genera visualización del Hype Cycle con mejor diseño"""
        fig = go.Figure()
        
        # Crear curva del Hype Cycle con forma más pronunciada
        x = np.linspace(0, 100, 1000)
        # Modificar la ecuación para una curva más similar a la imagen
        y = 60 * np.exp(-((x-20)/10)**2) - 20 * np.exp(-((x-60)/40)**2) + 40 * np.exp(-((x-90)/15)**2)
        
        # Añadir la curva principal con un azul más suave
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Curva del Hype Cycle',
            line=dict(color='rgb(65, 105, 225)', width=3)
        ))
        
        # Definir posiciones de las fases
        phases = {
            "Innovation Trigger": 15,
            "Peak of Inflated Expectations": 30,
            "Trough of Disillusionment": 60,
            "Slope of Enlightenment": 75,
            "Plateau of Productivity": 90
        }
        
        # Marcar todas las fases
        for phase, x_pos in phases.items():
            y_pos = 60 * np.exp(-((x_pos-20)/10)**2) - 20 * np.exp(-((x_pos-60)/40)**2) + 40 * np.exp(-((x_pos-90)/15)**2)
            
            # Resaltar la fase actual
            if phase == hype_data['phase']:
                marker_color = 'red'
                marker_size = 15
                text_pos = 'top center'
            else:
                marker_color = 'black'  # Puntos en negro
                marker_size = 8
                text_pos = 'middle right'
            
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode='markers+text',
                marker=dict(size=marker_size, color=marker_color),
                text=[phase],
                textposition=text_pos,
                textfont=dict(color='black', size=12),  # Texto en negro
                showlegend=False
            ))
        
        # Personalizar diseño del gráfico
        fig.update_layout(
            title={
                'text': "Posición en el Hype Cycle de Gartner",
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=20)
            },
            xaxis_title="Madurez de la Tecnología",
            yaxis_title="Expectativas",
            height=700,  # Hacer el gráfico más alto
            width=1000,  # Y más ancho
            showlegend=False,
            plot_bgcolor='white',
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                zeroline=False
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                zeroline=False
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Eliminar los números de los ejes
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        
        return fig

# Agregar al inicio de analysis.py

class SearchCache:
    def __init__(self, max_size=3):
        self.max_size = max_size
        self.cache = {}
        self.search_history = []  # Para mantener el orden de las búsquedas

    def add_search(self, query, results, stats):
        """Agrega una búsqueda al caché"""
        # Si la búsqueda ya existe, actualizar su posición
        if query in self.cache:
            self.search_history.remove(query)
        
        # Agregar la nueva búsqueda
        self.search_history.append(query)
        self.cache[query] = {
            'results': results,
            'stats': stats,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Mantener solo las últimas max_size búsquedas
        if len(self.search_history) > self.max_size:
            oldest_query = self.search_history.pop(0)
            del self.cache[oldest_query]

    def get_search(self, query):
        """Obtiene una búsqueda del caché"""
        return self.cache.get(query)

    def get_recent_searches(self):
        """Retorna las búsquedas recientes en orden"""
        recent_searches = []
        for query in reversed(self.search_history):
            search_data = self.cache[query]
            recent_searches.append({
                'query': query,
                'timestamp': search_data['timestamp'],
                'result_count': len(search_data['results'])
            })
        return recent_searches