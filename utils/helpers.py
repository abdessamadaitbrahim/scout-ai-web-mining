"""
Scout-AI : Fonctions Utilitaires
==================================
Helpers pour le scraping éthique, nettoyage, logging et export des données.
"""

import os
import re
import json
import csv
import time
import random
import logging
import hashlib
from datetime import datetime
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    REQUEST_DELAY_MIN, REQUEST_DELAY_MAX,
    USER_AGENTS, DEFAULT_HEADERS, REQUEST_TIMEOUT, DATA_DIR
)

# LOGGING


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Configure un logger avec format standardisé."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger("scout-ai")

# SCRAPING ÉTHIQUE


def check_robots_txt(url: str) -> bool:
    """
    Vérifie si le scraping est autorisé par robots.txt.
    Conformité éthique requise par le cahier des charges.
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        allowed = rp.can_fetch("*", url)
        if not allowed:
            logger.warning(f"robots.txt INTERDIT le scraping de: {url}")
        return allowed
    except Exception as e:
        logger.warning(f"Impossible de lire robots.txt pour {url}: {e}")
        return True  # En cas d'erreur, on continue prudemment


def polite_delay():
    """Délai aléatoire entre requêtes pour respecter les serveurs."""
    delay = random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX)
    time.sleep(delay)


def get_random_headers() -> dict:
    """Retourne des headers avec User-Agent aléatoire."""
    headers = DEFAULT_HEADERS.copy()
    headers["User-Agent"] = random.choice(USER_AGENTS)
    return headers


def safe_request(url: str, method="GET", params=None, headers=None,
                 check_robots=True, max_retries=3, **kwargs) -> requests.Response | None:
    """
    Requête HTTP sécurisée avec :
    - Vérification robots.txt
    - Délai poli entre requêtes
    - Retry automatique avec backoff exponentiel
    - Gestion d'erreurs robuste
    """
    if check_robots and not check_robots_txt(url):
        return None

    if headers is None:
        headers = get_random_headers()

    for attempt in range(max_retries):
        try:
            polite_delay()
            response = requests.request(
                method, url, params=params, headers=headers,
                timeout=REQUEST_TIMEOUT, **kwargs
            )
            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else "?"
            if status == 429:  # Rate limited
                wait = (2 ** attempt) * 5
                logger.warning(f"Rate limited (429). Attente de {wait}s...")
                time.sleep(wait)
            elif status in (403, 404):
                logger.error(f"HTTP {status} pour {url}")
                return None
            else:
                logger.error(f"HTTP {status} pour {url}: {e}")
        except requests.exceptions.ConnectionError:
            logger.error(f"Erreur de connexion pour {url} (tentative {attempt+1})")
            time.sleep(2 ** attempt)
        except requests.exceptions.Timeout:
            logger.error(f"Timeout pour {url} (tentative {attempt+1})")
        except Exception as e:
            logger.error(f"Erreur inattendue pour {url}: {e}")
            return None

    logger.error(f"Échec après {max_retries} tentatives pour {url}")
    return None

# NETTOYAGE DE DONNÉES


def clean_text(text: str) -> str:
    """Nettoie un texte : supprime HTML, espaces multiples, caractères spéciaux."""
    if not text:
        return ""
    # Supprime les balises HTML
    text = re.sub(r'<[^>]+>', '', text)
    # Supprime les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    # Supprime les caractères de contrôle
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    return text.strip()


def anonymize_data(text: str) -> str:
    """
    Anonymise les données personnelles (emails, téléphones).
    Conformité RGPD requise par le cahier des charges.
    """
    # Anonymise les emails
    text = re.sub(r'[\w.+-]+@[\w-]+\.[\w.-]+', '[EMAIL_ANONYMISÉ]', text)
    # Anonymise les numéros de téléphone
    text = re.sub(r'\+?\d[\d\s\-]{8,}\d', '[TEL_ANONYMISÉ]', text)
    return text


def generate_doc_id(content: str) -> str:
    """Génère un ID unique pour un document basé sur son contenu."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]

# EXPORT DES DONNÉES


def ensure_data_dir():
    """Crée le répertoire data/ s'il n'existe pas."""
    os.makedirs(DATA_DIR, exist_ok=True)


def export_to_json(data: list | dict, filename: str) -> str:
    """Exporte des données en JSON dans data/."""
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    logger.info(f"Exporté {len(data) if isinstance(data, list) else 1} entrée(s) → {filepath}")
    return filepath


def export_to_csv(data: list[dict], filename: str) -> str:
    """Exporte une liste de dictionnaires en CSV dans data/."""
    ensure_data_dir()
    filepath = os.path.join(DATA_DIR, filename)
    if not data:
        logger.warning(f"Aucune donnée à exporter pour {filename}")
        return filepath
    keys = data[0].keys()
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    logger.info(f"Exporté {len(data)} lignes → {filepath}")
    return filepath


def load_json(filename: str) -> list | dict:
    """Charge un fichier JSON depuis data/."""
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

# STATISTIQUES & MONITORING


class CollectionStats:
    """Suivi des statistiques de collecte en temps réel."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.start_time = datetime.now()
        self.success_count = 0
        self.error_count = 0
        self.total_items = 0

    def record_success(self, items=1):
        self.success_count += 1
        self.total_items += items

    def record_error(self):
        self.error_count += 1

    def summary(self) -> dict:
        elapsed = (datetime.now() - self.start_time).total_seconds()
        return {
            "source": self.source_name,
            "durée_secondes": round(elapsed, 1),
            "requêtes_réussies": self.success_count,
            "requêtes_échouées": self.error_count,
            "total_items_collectés": self.total_items,
            "taux_succès": f"{self.success_count/(self.success_count+self.error_count)*100:.1f}%"
            if (self.success_count + self.error_count) > 0 else "N/A"
        }

    def print_summary(self):
        s = self.summary()
        logger.info(f"=== Bilan {s['source']} ===")
        for k, v in s.items():
            logger.info(f"  {k}: {v}")
