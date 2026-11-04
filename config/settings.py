"""
Scout-AI : Centralized Configuration
======================================
All API keys, database connections, and project constants.

IMPORTANT — SECURITY:
  Never hardcode API keys or passwords here.
  Use environment variables via a `.env` file (see `.env.example`).

Required environment variables (copy .env.example → .env and fill in values):
  FOOTBALL_DATA_API_KEY  — Football-Data.org API key
  NEWS_API_KEY           — NewsAPI key
  MONGO_URI              — MongoDB connection URI
  NEO4J_URI              — Neo4J bolt URI
  NEO4J_USER             — Neo4J username
  NEO4J_PASSWORD         — Neo4J password
  ELASTIC_HOST           — ElasticSearch host URL
"""

import os
import sys
from dotenv import load_dotenv

# Load .env file if present (does nothing in production if .env doesn't exist)
load_dotenv()

# ============================================================
# HELPER — ENVIRONMENT VARIABLE VALIDATION
# ============================================================


def _get_env(key: str, default: str = None, required: bool = False) -> str:
    """
    Retrieves an environment variable.

    Args:
        key: Environment variable name.
        default: Optional default value for non-sensitive settings.
        required: If True and variable is missing/empty, prints a warning.

    Returns:
        Variable value or default.
    """
    value = os.getenv(key, default)
    if required and not value:
        print(
            f"[Scout-AI] WARNING: Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in the value.",
            file=sys.stderr,
        )
    return value


def validate_api_keys():
    """
    Validates that API keys are set before making API calls.
    Call this at the start of any collector that needs live API access.

    Returns:
        dict with 'football_data' and 'news_api' boolean status.
    """
    status = {
        "football_data": bool(FOOTBALL_DATA_API_KEY),
        "news_api": bool(NEWS_API_KEY),
    }
    if not status["football_data"]:
        print(
            "[Scout-AI] WARNING: FOOTBALL_DATA_API_KEY not set — "
            "Football-Data API calls will fail. Use --demo mode or set the key.",
            file=sys.stderr,
        )
    if not status["news_api"]:
        print(
            "[Scout-AI] WARNING: NEWS_API_KEY not set — "
            "NewsAPI calls will fail. Use --demo mode or set the key.",
            file=sys.stderr,
        )
    return status


# ============================================================
# EXTERNAL APIS
# ============================================================

# Football-Data.org (free tier: 10 requests/minute)
# Register at: https://www.football-data.org/client/register
FOOTBALL_DATA_API_KEY = _get_env("FOOTBALL_DATA_API_KEY", required=False)
FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

# NewsAPI (free tier: 100 requests/day)
# Register at: https://newsapi.org/register
NEWS_API_KEY = _get_env("NEWS_API_KEY", required=False)
NEWS_API_BASE_URL = "https://newsapi.org/v2"

# ============================================================
# NoSQL DATABASES
# ============================================================

# MongoDB (primary storage — Content & Usage data)
MONGO_URI = _get_env("MONGO_URI", default="mongodb://localhost:27017")
MONGO_DB_NAME = "scout_ai"

# Neo4J (graph relationships — Structure Mining)
NEO4J_URI = _get_env("NEO4J_URI", default="bolt://localhost:7687")
NEO4J_USER = _get_env("NEO4J_USER", default="neo4j")
NEO4J_PASSWORD = _get_env("NEO4J_PASSWORD", default="")

# ElasticSearch (full-text search — Articles & Sentiment)
# NOTE: Python client elasticsearch==7.17.9 vs Docker image 8.12.0
# If you upgrade the Docker image, also run: pip install elasticsearch>=8.0
ELASTIC_HOST = _get_env("ELASTIC_HOST", default="http://localhost:9200")
ELASTIC_INDEX_ARTICLES = "scout_articles"
ELASTIC_INDEX_SENTIMENT = "scout_sentiment"

# ============================================================
# WEB SCRAPING — ETHICAL PARAMETERS
# ============================================================

# Delay between requests (rate limiting compliance)
REQUEST_DELAY_MIN = 1.5   # seconds minimum between 2 requests
REQUEST_DELAY_MAX = 3.0   # seconds maximum (randomised)

# Rotating User-Agent pool
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
]

# Default request headers
DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

# Request timeout
REQUEST_TIMEOUT = 15  # seconds

# ============================================================
# TARGET PLAYERS & CLUBS (project sample)
# ============================================================

TARGET_PLAYERS = [
    {"name": "Kylian Mbappé",    "id_tm": "342229", "team": "Real Madrid"},
    {"name": "Erling Haaland",   "id_tm": "418560", "team": "Manchester City"},
    {"name": "Jude Bellingham",  "id_tm": "581678", "team": "Real Madrid"},
    {"name": "Vinicius Junior",  "id_tm": "371998", "team": "Real Madrid"},
    {"name": "Bukayo Saka",      "id_tm": "433177", "team": "Arsenal"},
    {"name": "Jamal Musiala",    "id_tm": "580195", "team": "Bayern Munich"},
    {"name": "Lamine Yamal",     "id_tm": "937958", "team": "FC Barcelona"},
    {"name": "Florian Wirtz",    "id_tm": "598577", "team": "Bayer Leverkusen"},
    {"name": "Victor Osimhen",   "id_tm": "401923", "team": "Galatasaray"},
    {"name": "Mohamed Salah",    "id_tm": "148455", "team": "Liverpool"},
]

TARGET_LEAGUES = [
    {"name": "Premier League", "code": "PL",  "id": 2021},
    {"name": "La Liga",        "code": "PD",  "id": 2014},
    {"name": "Bundesliga",     "code": "BL1", "id": 2002},
    {"name": "Serie A",        "code": "SA",  "id": 2019},
    {"name": "Ligue 1",        "code": "FL1", "id": 2015},
]

# ============================================================
# NLP & AI MODELS
# ============================================================

# BERT models for NER and Sentiment Analysis
BERT_MODEL_NER = "Jean-Baptiste/camembert-ner"           # French NER
BERT_MODEL_SENTIMENT = "nlptown/bert-base-multilingual-uncased-sentiment"
SPACY_MODEL = "fr_core_news_md"                          # French spaCy model

# ============================================================
# EXPORT & LOCAL STORAGE
# ============================================================

# Resolved absolute path to the data/ directory at project root
DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data")
)

# Path to models/ directory at project root
MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)

# Path to figures/ directory at project root
FIGURES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "figures")
)

EXPORT_FORMAT = "json"  # json or csv
