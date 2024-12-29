# utils/search_utils.py
from googleapiclient.discovery import build
from datetime import datetime
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
import pandas as pd

# Descargar recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

class SearchProcessor:
    def __init__(self, api_key, search_engine_id):
        self.service = build("customsearch", "v1", developerKey=api_key)
        self.search_engine_id = search_engine_id
        self.stop_words = set(stopwords.words('english'))

    def build_query(self, topics, min_year):
        """Construye la ecuación de búsqueda optimizada"""
        base_query = ' AND '.join([f'"{topic.strip()}"' for topic in topics if topic.strip()])
        time_filter = f' after:{min_year}'
        return base_query + time_filter

    def perform_search(self, query, **kwargs):
        """Realiza la búsqueda en Google"""
        try:
            results = self.service.cse().list(
                q=query,
                cx=self.search_engine_id,
                num=100,  # Máximo permitido por página
                **kwargs
            ).execute()
            return results.get('items', [])
        except Exception as e:
            raise Exception(f"Error en la búsqueda: {str(e)}")

    def extract_keywords(self, text):
        """Extrae palabras clave relevantes del texto"""
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        
        # Filtrar solo sustantivos y adjetivos
        keywords = [word for word, tag in tagged 
                   if tag.startswith(('NN', 'JJ')) 
                   and word not in self.stop_words
                   and len(word) > 2]
        
        return list(set(keywords))

    def process_results(self, results):
        """Procesa y estructura los resultados de búsqueda"""
        processed = []
        for item in results:
            # Extraer año del snippet o título
            year = self.extract_year(item.get('snippet', '') + item.get('title', ''))
            if year:
                processed.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', ''),
                    'year': year,
                    'keywords': self.extract_keywords(item.get('snippet', '')),
                    'domain': self.extract_domain(item.get('link', '')),
                    'type': self.determine_content_type(item)
                })
        return processed

    @staticmethod
    def extract_year(text):
        """Extrae el año de un texto"""
        import re
        current_year = datetime.now().year
        years = re.findall(r'\b20\d{2}\b', text)
        valid_years = [int(y) for y in years if 2014 <= int(y) <= current_year]
        return min(valid_years) if valid_years else None

    @staticmethod
    def extract_domain(url):
        """Extrae el dominio base de una URL"""
        from urllib.parse import urlparse
        try:
            return urlparse(url).netloc
        except:
            return None

    @staticmethod
    def determine_content_type(item):
        """Determina el tipo de contenido basado en la URL y metatags"""
        url = item.get('link', '').lower()
        if 'pdf' in url:
            return 'PDF'
        elif 'patent' in url:
            return 'Patent'
        elif any(domain in url for domain in ['scholar.google', 'sciencedirect', 'springer']):
            return 'Academic'
        else:
            return 'Web'

    def analyze_trends(self, processed_results):
        """Analiza tendencias en los resultados"""
        df = pd.DataFrame(processed_results)
        
        analysis = {
            'total_results': len(df),
            'year_distribution': df['year'].value_counts().sort_index().to_dict(),
            'content_types': df['type'].value_counts().to_dict(),
            'top_domains': df['domain'].value_counts().head(10).to_dict(),
            'top_keywords': self.get_top_keywords(df),
            'trend_direction': self.calculate_trend_direction(df)
        }
        
        return analysis

    def get_top_keywords(self, df):
        """Obtiene las palabras clave más frecuentes"""
        all_keywords = [kw for keywords in df['keywords'] for kw in keywords]
        from collections import Counter
        return dict(Counter(all_keywords).most_common(20))

    def calculate_trend_direction(self, df):
        """Calcula la dirección de la tendencia basada en la distribución temporal"""
        yearly_counts = df['year'].value_counts().sort_index()
        if len(yearly_counts) < 2:
            return 0
        
        # Calculamos la pendiente de la tendencia
        from scipy import stats
        slope, _ = stats.linregress(range(len(yearly_counts)), yearly_counts.values)[:2]
        return slope