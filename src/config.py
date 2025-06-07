#src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY'),
    'SEARCH_ENGINE_ID': os.getenv('SEARCH_ENGINE_ID'),
    'SERP_API_KEY': os.getenv('SERP_API_KEY'),
    'MIN_YEAR': 2014,
    'MAX_RESULTS': 100
}
