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
        """Realiza búsqueda específica en noticias usando sitios de noticias relevantes"""
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            all_results = []
            
            # En lugar de restringir a sitios específicos, usaremos términos que indiquen contenido noticioso
            news_terms = [
                'news', 
                'press release',
                'announced',
                'launches',
                'reveals'
            ]
            
            # Modificar query para encontrar contenido tipo noticia
            news_query = f'{query} ({" OR ".join(news_terms)})'
            
            # También podemos agregar palabras clave relacionadas con noticias
            news_keywords = ['news', 'article', 'press release', 'announcement']
            news_query += f' AND ({" OR ".join(news_keywords)})'
            
            for start_index in range(1, 91, 10):  # Hasta 90 resultados
                result = service.cse().list(
                    q=news_query,
                    cx=search_engine_id,
                    num=10,
                    start=start_index,
                    sort='date',  # Ordenar por fecha
                    dateRestrict='y5'  # Restringir a los últimos 5 años
                ).execute()
                
                items = result.get('items', [])
                if not items:
                    break
                    
                # Filtrar y procesar resultados
                for item in items:
                    # Verificar si es realmente una noticia (puedes agregar más criterios)
                    if self._is_news_content(item):
                        all_results.append(item)
            
            return True, all_results
        except Exception as e:
            return False, str(e)

    def _is_news_content(self, item):
        """Helper method para verificar si un resultado es realmente una noticia"""
        # Obtener texto combinado para análisis
        text = f"{item.get('title', '')} {item.get('snippet', '')}"
        text = text.lower()
        
        # Palabras clave que indican contenido de noticias
        news_indicators = [
            'announced', 'reported', 'launched', 'released',
            'unveiled', 'introduced', 'published', 'news',
            'article', 'press release', 'coverage'
        ]
        
        # Verificar si contiene indicadores de noticias
        has_news_indicators = any(indicator in text for indicator in news_indicators)
        
        # Verificar la URL
        url = item.get('link', '').lower()
        is_news_site = any(
            site in url for site in [
                'news', 'article', 'blog', 'press', 
                'techcrunch', 'wired', 'verge', 'zdnet',
                'reuters', 'bloomberg'
            ]
        )
        
        return has_news_indicators or is_news_site

    

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

    def plot_hype_cycle(self, hype_data, topics):
        """
        Genera visualización del Hype Cycle con temas posicionados
        """
        fig = go.Figure()
        
        # Crear curva del Hype Cycle
        x = np.linspace(0, 100, 1000)
        y = 60 * np.exp(-((x-20)/10)**2) - 20 * np.exp(-((x-60)/40)**2) + 40 * np.exp(-((x-90)/15)**2)
        
        # Añadir la curva principal
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name='Curva del Hype Cycle',
            line=dict(color='rgb(65, 105, 225)', width=3)
        ))

        # Posicionar los temas en la curva
        if isinstance(topics, list) and topics:
            sentiment = hype_data['sentiment_trend'].mean()
            mentions_trend = hype_data['yearly_stats']['mention_count'].pct_change().mean()
            
            # Determinar posición en x basada en métricas
            if mentions_trend > 0.5 and sentiment > 0:
                x_pos = 15  # Innovation Trigger
            elif sentiment > 0.3:
                x_pos = 30  # Peak of Expectations
            elif sentiment < 0:
                x_pos = 60  # Trough of Disillusionment
            elif sentiment > 0 and mentions_trend > 0:
                x_pos = 75  # Slope of Enlightenment
            else:
                x_pos = 90  # Plateau of Productivity

            # Añadir cada tema a la curva
            for topic in topics:
                if topic.strip():  # Solo procesar temas no vacíos
                    y_pos = 60 * np.exp(-((x_pos-20)/10)**2) - 20 * np.exp(-((x_pos-60)/40)**2) + 40 * np.exp(-((x_pos-90)/15)**2)
                    fig.add_trace(go.Scatter(
                        x=[x_pos],
                        y=[y_pos],
                        mode='markers+text',
                        marker=dict(size=10, color='red'),
                        text=[topic],
                        textposition='top center',
                        name=topic
                    ))

        # Marcar las fases
        phases = {
            "Innovation Trigger": 15,
            "Peak of Inflated Expectations": 30,
            "Trough of Disillusionment": 60,
            "Slope of Enlightenment": 75,
            "Plateau of Productivity": 90
        }
        
        for phase, x_pos in phases.items():
            y_pos = 60 * np.exp(-((x_pos-20)/10)**2) - 20 * np.exp(-((x_pos-60)/40)**2) + 40 * np.exp(-((x_pos-90)/15)**2)
            fig.add_trace(go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode='text',
                text=[phase],
                textposition='bottom center',
                textfont=dict(color='black', size=10),
                showlegend=False
            ))
        
        # Personalizar diseño
        fig.update_layout(
            title="Posición de Temas en el Hype Cycle",
            height=700,
            width=1000,
            showlegend=True,
            plot_bgcolor='white',
            xaxis=dict(showticklabels=False, title="Madurez de la Tecnología"),
            yaxis=dict(showticklabels=False, title="Expectativas")
        )
        
        return fig
    
    def extract_news_details(self, text):
        """Extrae detalles adicionales del texto de la noticia"""
        # Eliminada la detección de países
        
        # Extraer autores
        author_patterns = [
            r'by\s+([\w\s]+)(?=\s+for|\.|\n)',
            r'author[s]?:?\s+([\w\s]+)(?=\s+for|\.|\n)',
            r'written\s+by\s+([\w\s]+)(?=\s+for|\.|\n)'
        ]
        
        authors = []
        for pattern in author_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                authors.extend([author.strip() for author in matches])
        
        # Extraer palabras clave
        # Eliminar palabras comunes y símbolos
        words = re.findall(r'\b\w+\b', text.lower())
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        keywords = [word for word in words if word not in stopwords and len(word) > 3]
        # Obtener las palabras más frecuentes
        keyword_freq = Counter(keywords).most_common(5)
        
        return {
            'authors': list(set(authors)),  # Eliminar duplicados
            'keywords': [kw for kw, _ in keyword_freq]
        }

    def analyze_hype_cycle(self, news_results):
        """Versión mejorada del análisis del Hype Cycle con más detalles"""
        analyzed_results = []
        
        for item in news_results:
            try:
                text = f"{item.get('title', '')} {item.get('snippet', '')}"
                year_match = re.search(r'(\d{4})', text)
                
                if not year_match:
                    continue
                    
                year = year_match.group(1)
                sentiment = self.sentiment_analyzer.polarity_scores(text)
                
                # Extraer detalles adicionales
                details = self.extract_news_details(text)
                
                # Crear resumen más corto si el snippet es muy largo
                snippet = item.get('snippet', '')
                summary = snippet[:200] + '...' if len(snippet) > 200 else snippet
                
                analyzed_results.append({
                    'year': year,
                    'sentiment': sentiment['compound'],
                    'title': item.get('title', 'Sin título'),
                    'summary': summary,
                    'link': item.get('link', '#'),
                    'country': details['country'],
                    'authors': details['authors'],
                    'keywords': details['keywords'],
                    'source': self.extract_source_name(item.get('link', '')),
                    'date_analyzed': datetime.now().strftime('%Y-%m-%d')
                })
                
            except Exception as e:
                print(f"Error procesando noticia: {str(e)}")
                continue
        
        # Ordenar por año y sentimiento
        analyzed_results.sort(key=lambda x: (x['year'], x['sentiment']), reverse=True)
        
        # Análisis del Hype Cycle
        if not analyzed_results:
            return {
                'phase': "No hay suficientes datos",
                'yearly_stats': pd.DataFrame(),
                'sentiment_trend': pd.Series(),
                'results': []
            }

        # Convertir a DataFrame para análisis
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
            'results': analyzed_results  # Ahora contiene más detalles
        }

    def extract_source_name(self, url):
        """Extrae el nombre de la fuente de la URL"""
        try:
            domain = re.search(r'https?://(?:www\.)?([^/]+)', url).group(1)
            # Limpiar el dominio para obtener un nombre más legible
            source_name = domain.split('.')[0].title()
            return source_name
        except:
            return "Fuente desconocida"

    def show_hype_cycle_news_table(st, news_results):
        """Muestra una tabla interactiva con los detalles de las noticias"""
        st.write("### 📰 Noticias Analizadas para el Hype Cycle")
        
        # Agregar filtros
        col1, col2 = st.columns(2)
        with col1:
            years = sorted(set(r.get('year') for r in news_results if r.get('year')))
            year_filter = st.multiselect(
                "Filtrar por año",
                options=years,
                default=[]
            )
        with col2:
            sentiment_filter = st.select_slider(
                "Filtrar por sentimiento",
                options=['Todos', 'Solo positivos', 'Solo negativos']
            )
        
        # Aplicar filtros
        filtered_results = news_results
        if year_filter:
            filtered_results = [r for r in filtered_results if r.get('year') in year_filter]
        if country_filter:
            filtered_results = [r for r in filtered_results if r.get('country') in country_filter]
        if sentiment_filter != 'Todos':
            filtered_results = [r for r in filtered_results if 
                (sentiment_filter == 'Solo positivos' and r.get('sentiment', 0) > 0) or
                (sentiment_filter == 'Solo negativos' and r.get('sentiment', 0) < 0)]
        
        # Mostrar resultados
        for idx, result in enumerate(filtered_results, 1):
            with st.expander(f"📄 {idx}. {result['title']}", expanded=False):
                col1, col2 = st.columns([2,1])
                
                with col1:
                    st.markdown("**Resumen:**")
                    st.write(result['summary'])
                    st.markdown(f"**Palabras clave:** {', '.join(result['keywords'])}")
                    st.markdown(f"🔗 [Ver noticia completa]({result['link']})")
                
                with col2:
                    st.markdown("**Detalles:**")
                    st.markdown(f"📅 **Año:** {result['year']}")
                    st.markdown(f"🌍 **País:** {result.get('country', 'No especificado')}")
                    st.markdown(f"📰 **Fuente:** {result['source']}")
                    if result['authors']:
                        st.markdown(f"✍️ **Autores:** {', '.join(result['authors'])}")
                    
                    # Mostrar sentimiento con color
                    sentiment = result['sentiment']
                    sentiment_color = 'green' if sentiment > 0 else 'red'
                    st.markdown(f"💭 **Sentimiento:** <span style='color:{sentiment_color}'>{sentiment:.2f}</span>", 
                            unsafe_allow_html=True)
        