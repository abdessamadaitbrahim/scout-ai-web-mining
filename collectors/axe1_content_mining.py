"""
Scout-AI : Axe 1 - Web Content Mining
=======================================
Analyse de la Réputation, Extraction Sémantique & Intelligence Mercato

Ce module implémente :
1. Collecte d'articles de presse via NewsAPI
2. Scraping de statistiques de joueurs (Football-Data API + Transfermarkt)
3. Pipeline NLP : NER (reconnaissance d'entités) + Analyse de Sentiment
4. Calcul de la "Cote de Popularité" joueur-club

Modèle utilisé : BERT fine-tuné (CamemBERT pour NER FR, BERT multilingual pour sentiment)
"""

import os
import sys
import json
import re
from datetime import datetime, timedelta
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    FOOTBALL_DATA_API_KEY, FOOTBALL_DATA_BASE_URL,
    NEWS_API_KEY, NEWS_API_BASE_URL,
    TARGET_PLAYERS, TARGET_LEAGUES,
    BERT_MODEL_NER, BERT_MODEL_SENTIMENT
)
from utils.helpers import (
    safe_request, clean_text, anonymize_data, generate_doc_id,
    export_to_json, export_to_csv, setup_logger, CollectionStats
)

logger = setup_logger("axe1-content-mining")

# 1. COLLECTE D'ARTICLES DE PRESSE (NewsAPI)


class NewsCollector:
    """
    Collecte d'articles de presse via NewsAPI.
    Sources : médias sportifs internationaux et francophones.
    """

    def __init__(self, api_key: str = NEWS_API_KEY):
        self.api_key = api_key
        self.base_url = NEWS_API_BASE_URL
        self.stats = CollectionStats("NewsAPI")

    def search_articles(self, query: str, language: str = "fr",
                        from_date: str = None, page_size: int = 20) -> list[dict]:
        """
        Recherche d'articles via NewsAPI /everything endpoint.

        Args:
            query: Mots-clés de recherche (ex: "Mbappé transfert")
            language: Langue des articles (fr, en, es)
            from_date: Date de début (format YYYY-MM-DD)
            page_size: Nombre d'articles par page (max 100)

        Returns:
            Liste de dictionnaires avec les articles normalisés
        """
        if not from_date:
            from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        params = {
            "q": query,
            "language": language,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": min(page_size, 100),
            "apiKey": self.api_key
        }

        response = safe_request(
            f"{self.base_url}/everything",
            params=params,
            check_robots=False  # API officielle, pas de scraping
        )

        if not response:
            self.stats.record_error()
            return []

        data = response.json()
        articles = []

        for article in data.get("articles", []):
            cleaned = {
                "doc_id": generate_doc_id(article.get("url", "")),
                "titre": clean_text(article.get("title", "")),
                "description": clean_text(article.get("description", "")),
                "contenu": clean_text(article.get("content", "")),
                "source": article.get("source", {}).get("name", "Inconnu"),
                "auteur": anonymize_data(article.get("author", "") or ""),
                "url": article.get("url", ""),
                "date_publication": article.get("publishedAt", ""),
                "image_url": article.get("urlToImage", ""),
                "query": query,
                "langue": language,
                "date_collecte": datetime.now().isoformat(),
            }
            articles.append(cleaned)

        self.stats.record_success(len(articles))
        logger.info(f"Collecté {len(articles)} articles pour '{query}'")
        return articles

    def collect_player_articles(self, players: list[dict] = None) -> list[dict]:
        """Collecte des articles pour tous les joueurs cibles."""
        if players is None:
            players = TARGET_PLAYERS

        all_articles = []
        for player in players:
            # Recherche en français et en anglais
            for lang in ["fr", "en"]:
                keywords = [
                    f"{player['name']} transfert",
                    f"{player['name']} {player['team']}",
                ]
                for kw in keywords:
                    articles = self.search_articles(kw, language=lang)
                    all_articles.extend(articles)

        # Dédoublonnage par doc_id
        seen = set()
        unique = []
        for a in all_articles:
            if a["doc_id"] not in seen:
                seen.add(a["doc_id"])
                unique.append(a)

        logger.info(f"Total articles uniques collectés: {len(unique)}")
        return unique

# 2. COLLECTE DE STATISTIQUES (Football-Data API)


class FootballStatsCollector:
    """
    Collecte de statistiques de matchs et classements via Football-Data.org API.
    API gratuite avec 10 requêtes/minute.
    """

    def __init__(self, api_key: str = FOOTBALL_DATA_API_KEY):
        self.api_key = api_key
        self.base_url = FOOTBALL_DATA_BASE_URL
        self.headers = {"X-Auth-Token": api_key}
        self.stats = CollectionStats("Football-Data API")

    def _api_request(self, endpoint: str, params: dict = None) -> dict | None:
        """Requête à l'API Football-Data."""
        url = f"{self.base_url}/{endpoint}"
        response = safe_request(url, headers=self.headers, params=params, check_robots=False)
        if response:
            self.stats.record_success()
            return response.json()
        self.stats.record_error()
        return None

    def get_league_standings(self, league_id: int) -> list[dict]:
        """Récupère le classement d'une ligue."""
        data = self._api_request(f"competitions/{league_id}/standings")
        if not data:
            return []

        standings = []
        for table in data.get("standings", []):
            if table.get("type") == "TOTAL":
                for team in table.get("table", []):
                    standings.append({
                        "position": team["position"],
                        "equipe": team["team"]["name"],
                        "equipe_id": team["team"]["id"],
                        "matchs_joues": team["playedGames"],
                        "victoires": team["won"],
                        "nuls": team["draw"],
                        "defaites": team["lost"],
                        "buts_pour": team["goalsFor"],
                        "buts_contre": team["goalsAgainst"],
                        "difference_buts": team["goalDifference"],
                        "points": team["points"],
                        "competition_id": league_id,
                        "date_collecte": datetime.now().isoformat(),
                    })
        return standings

    def get_league_scorers(self, league_id: int) -> list[dict]:
        """Récupère les meilleurs buteurs d'une ligue."""
        data = self._api_request(f"competitions/{league_id}/scorers", {"limit": 20})
        if not data:
            return []

        scorers = []
        for scorer in data.get("scorers", []):
            player = scorer.get("player", {})
            team = scorer.get("team", {})
            scorers.append({
                "joueur": player.get("name", ""),
                "joueur_id": player.get("id", ""),
                "nationalite": player.get("nationality", ""),
                "poste": player.get("position", ""),
                "date_naissance": player.get("dateOfBirth", ""),
                "equipe": team.get("name", ""),
                "equipe_id": team.get("id", ""),
                "buts": scorer.get("goals", 0),
                "passes_decisives": scorer.get("assists", 0),
                "penaltys": scorer.get("penalties", 0),
                "matchs_joues": scorer.get("playedMatches", 0),
                "competition_id": league_id,
                "date_collecte": datetime.now().isoformat(),
            })
        return scorers

    def get_recent_matches(self, league_id: int, matchday: int = None) -> list[dict]:
        """Récupère les matchs récents d'une compétition."""
        params = {}
        if matchday:
            params["matchday"] = matchday

        data = self._api_request(f"competitions/{league_id}/matches", params)
        if not data:
            return []

        matches = []
        for match in data.get("matches", [])[-20:]:  # 20 derniers matchs
            matches.append({
                "match_id": match.get("id"),
                "competition_id": league_id,
                "journee": match.get("matchday"),
                "date": match.get("utcDate", ""),
                "statut": match.get("status", ""),
                "equipe_domicile": match.get("homeTeam", {}).get("name", ""),
                "equipe_exterieur": match.get("awayTeam", {}).get("name", ""),
                "score_domicile": match.get("score", {}).get("fullTime", {}).get("home"),
                "score_exterieur": match.get("score", {}).get("fullTime", {}).get("away"),
                "date_collecte": datetime.now().isoformat(),
            })
        return matches

    def collect_all_leagues(self, leagues: list[dict] = None) -> dict:
        """Collecte complète pour toutes les ligues cibles."""
        if leagues is None:
            leagues = TARGET_LEAGUES

        all_data = {
            "classements": [],
            "buteurs": [],
            "matchs": [],
        }

        for league in leagues:
            logger.info(f"Collecte {league['name']}...")
            lid = league["id"]

            standings = self.get_league_standings(lid)
            all_data["classements"].extend(standings)

            scorers = self.get_league_scorers(lid)
            all_data["buteurs"].extend(scorers)

            matches = self.get_recent_matches(lid)
            all_data["matchs"].extend(matches)

        logger.info(
            f"Stats collectées: {len(all_data['classements'])} classements, "
            f"{len(all_data['buteurs'])} buteurs, {len(all_data['matchs'])} matchs"
        )
        return all_data

# 3. SCRAPING TRANSFERMARKT (valeurs marchandes)


class TransfermarktScraper:
    """
    Scraping de Transfermarkt pour les valeurs marchandes et profils.
    Respecte robots.txt et rate limiting.
    """

    BASE_URL = "https://www.transfermarkt.com"

    def __init__(self):
        self.stats = CollectionStats("Transfermarkt")
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "fr-FR,fr;q=0.9",
        }

    def get_player_profile(self, player_id: str, player_name: str) -> dict | None:
        """
        Récupère le profil d'un joueur depuis Transfermarkt.

        Args:
            player_id: ID Transfermarkt du joueur
            player_name: Nom du joueur (pour l'URL)
        """
        # Normalisation du nom pour l'URL
        slug = player_name.lower().replace(" ", "-").replace("é", "e").replace("ü", "u")
        url = f"{self.BASE_URL}/{slug}/profil/spieler/{player_id}"

        response = safe_request(url, headers=self.headers)
        if not response:
            self.stats.record_error()
            return None

        soup = BeautifulSoup(response.text, "lxml")

        profile = {
            "player_id_tm": player_id,
            "nom": player_name,
            "url_source": url,
        }

        # Extraction de la valeur marchande
        market_value_elem = soup.select_one(".tm-market-value-development-current-value")
        if market_value_elem:
            profile["valeur_marchande"] = clean_text(market_value_elem.text)

        # Extraction des infos de base
        info_table = soup.select(".info-table__content")
        for i in range(0, len(info_table) - 1, 2):
            label = clean_text(info_table[i].text).lower()
            value = clean_text(info_table[i + 1].text)
            if "âge" in label or "age" in label:
                profile["age"] = value
            elif "national" in label:
                profile["nationalite"] = value
            elif "position" in label:
                profile["poste"] = value
            elif "pied" in label or "foot" in label:
                profile["pied_fort"] = value
            elif "agent" in label:
                profile["agent"] = anonymize_data(value)
            elif "club" in label:
                profile["club_actuel"] = value

        # Extraction de l'historique de transferts
        profile["historique_transferts"] = self._extract_transfer_history(soup)
        profile["date_collecte"] = datetime.now().isoformat()

        self.stats.record_success()
        logger.info(f"Profil collecté: {player_name}")
        return profile

    def _extract_transfer_history(self, soup: BeautifulSoup) -> list[dict]:
        """Extrait l'historique de transferts depuis la page profil."""
        transfers = []
        transfer_rows = soup.select(".tm-transfer-history__list-item")

        for row in transfer_rows[:10]:  # 10 derniers transferts max
            transfer = {}
            date_elem = row.select_one(".tm-transfer-history__date")
            if date_elem:
                transfer["date"] = clean_text(date_elem.text)

            from_elem = row.select_one(".tm-transfer-history__old-club")
            if from_elem:
                transfer["de_club"] = clean_text(from_elem.text)

            to_elem = row.select_one(".tm-transfer-history__new-club")
            if to_elem:
                transfer["vers_club"] = clean_text(to_elem.text)

            fee_elem = row.select_one(".tm-transfer-history__market-value")
            if fee_elem:
                transfer["montant"] = clean_text(fee_elem.text)

            if transfer:
                transfers.append(transfer)

        return transfers

    def collect_all_players(self, players: list[dict] = None) -> list[dict]:
        """Collecte les profils de tous les joueurs cibles."""
        if players is None:
            players = TARGET_PLAYERS

        profiles = []
        for player in players:
            profile = self.get_player_profile(player["id_tm"], player["name"])
            if profile:
                profiles.append(profile)

        return profiles

# 4. PIPELINE NLP (NER + Sentiment Analysis)


class NLPPipeline:
    """
    Pipeline NLP basé sur BERT fine-tuné :
    - NER : Reconnaissance d'entités nommées (joueurs, clubs, lieux)
    - Sentiment Analysis : Analyse du sentiment des articles
    - Cote de Popularité : Score composite

    Note: Les modèles sont chargés à la demande (lazy loading).
    Pour exécuter sans GPU, les inférences sont optimisées CPU.
    """

    def __init__(self):
        self.ner_pipeline = None
        self.sentiment_pipeline = None
        self.stats = CollectionStats("NLP Pipeline")
        self._models_loaded = False

    def _load_models(self):
        """Charge les modèles BERT (lazy loading)."""
        if self._models_loaded:
            return

        try:
            from transformers import pipeline as hf_pipeline

            logger.info("Chargement du modèle NER (CamemBERT)...")
            self.ner_pipeline = hf_pipeline(
                "ner",
                model=BERT_MODEL_NER,
                aggregation_strategy="simple"
            )

            logger.info("Chargement du modèle Sentiment (BERT multilingual)...")
            self.sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=BERT_MODEL_SENTIMENT
            )

            self._models_loaded = True
            logger.info("Modèles NLP chargés avec succès")

        except ImportError:
            logger.warning(
                "transformers/torch non installés. "
                "Utilisation du mode fallback (TextBlob)."
            )
            self._models_loaded = False

    def extract_entities(self, text: str) -> list[dict]:
        """
        Extraction d'entités nommées (NER) depuis un texte.

        Returns:
            Liste d'entités : {text, label, score}
            Labels : PER (personne), ORG (organisation), LOC (lieu)
        """
        self._load_models()

        if self.ner_pipeline:
            # Mode BERT
            entities = self.ner_pipeline(text[:512])  # Limite BERT = 512 tokens
            return [
                {
                    "texte": ent["word"],
                    "type": ent["entity_group"],
                    "score": round(ent["score"], 4),
                }
                for ent in entities
                if ent["score"] > 0.7  # Seuil de confiance
            ]
        else:
            # Mode fallback simple avec regex
            return self._fallback_ner(text)

    def _fallback_ner(self, text: str) -> list[dict]:
        """NER de secours sans modèle BERT (basé sur des patterns)."""
        entities = []
        # Détection basique de noms propres (mots capitalisés)
        proper_nouns = re.findall(r'\b[A-Z][a-zéèêëàâôùûïîç]+(?:\s[A-Z][a-zéèêëàâôùûïîç]+)*\b', text)
        for noun in set(proper_nouns):
            # Heuristique : les clubs connus sont des ORG
            known_clubs = ["Real Madrid", "Barcelona", "Manchester", "Liverpool",
                           "Bayern", "PSG", "Arsenal", "Chelsea", "Juventus"]
            entity_type = "ORG" if any(c in noun for c in known_clubs) else "PER"
            entities.append({"texte": noun, "type": entity_type, "score": 0.5})
        return entities

    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyse du sentiment d'un texte.

        Returns:
            {label: "POSITIVE"|"NEGATIVE"|"NEUTRAL", score: 0.0-1.0, stars: 1-5}
        """
        self._load_models()

        if self.sentiment_pipeline:
            # Mode BERT - retourne 1-5 étoiles
            result = self.sentiment_pipeline(text[:512])[0]
            stars = int(result["label"].split()[0])
            if stars >= 4:
                label = "POSITIVE"
            elif stars <= 2:
                label = "NEGATIVE"
            else:
                label = "NEUTRAL"
            return {"label": label, "score": round(result["score"], 4), "stars": stars}
        else:
            # Mode fallback avec TextBlob
            return self._fallback_sentiment(text)

    def _fallback_sentiment(self, text: str) -> dict:
        """Analyse de sentiment de secours (lexique simple)."""
        positive_words = {"excellent", "formidable", "victoire", "gagné", "record",
                          "brille", "exceptionnel", "talent", "réussi", "succès",
                          "goal", "champion", "superbe", "magnifique", "incroyable"}
        negative_words = {"blessé", "défaite", "perdu", "échec", "crise",
                          "suspendu", "scandale", "problème", "mauvais", "déception",
                          "éliminé", "relégation", "flop", "catastrophe"}

        words = set(text.lower().split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count:
            return {"label": "POSITIVE", "score": 0.7, "stars": 4}
        elif neg_count > pos_count:
            return {"label": "NEGATIVE", "score": 0.7, "stars": 2}
        else:
            return {"label": "NEUTRAL", "score": 0.5, "stars": 3}

    def process_articles(self, articles: list[dict]) -> list[dict]:
        """
        Traite une liste d'articles avec NER + Sentiment.
        Enrichit chaque article avec les entités et le sentiment.
        """
        enriched = []
        for article in articles:
            text = f"{article.get('titre', '')} {article.get('description', '')} {article.get('contenu', '')}"

            if not text.strip():
                continue

            # NER
            entities = self.extract_entities(text)

            # Sentiment
            sentiment = self.analyze_sentiment(text)

            article_enriched = {
                **article,
                "entites_nommees": entities,
                "sentiment": sentiment,
                "nb_entites": len(entities),
                "joueurs_mentions": [e["texte"] for e in entities if e["type"] == "PER"],
                "clubs_mentions": [e["texte"] for e in entities if e["type"] == "ORG"],
            }
            enriched.append(article_enriched)
            self.stats.record_success()

        return enriched

    def compute_popularity_score(self, player_name: str,
                                  articles: list[dict]) -> dict:
        """
        Calcule la "Cote de Popularité" d'un joueur.

        Formule : CP = α × Mentions + β × SentimentMoyen + γ × DiversitéSources
        où α=0.4, β=0.35, γ=0.25

        Returns:
            {player, score, mentions, sentiment_avg, source_diversity, details}
        """
        player_articles = [
            a for a in articles
            if player_name.lower() in
            (a.get("titre", "") + " " + a.get("contenu", "")).lower()
        ]

        if not player_articles:
            return {
                "joueur": player_name,
                "cote_popularite": 0,
                "mentions": 0,
                "sentiment_moyen": 0,
                "diversite_sources": 0,
            }

        # Nombre de mentions (normalisé)
        mentions = len(player_articles)
        mentions_norm = min(mentions / 50, 1.0)  # Normaliser sur 50 articles max

        # Sentiment moyen
        sentiments = [a["sentiment"]["stars"] for a in player_articles if "sentiment" in a]
        sentiment_avg = sum(sentiments) / len(sentiments) if sentiments else 3
        sentiment_norm = (sentiment_avg - 1) / 4  # Normaliser 1-5 → 0-1

        # Diversité des sources
        sources = set(a.get("source", "") for a in player_articles)
        diversity_norm = min(len(sources) / 10, 1.0)  # Normaliser sur 10 sources max

        # Score composite
        alpha, beta, gamma = 0.4, 0.35, 0.25
        score = alpha * mentions_norm + beta * sentiment_norm + gamma * diversity_norm

        return {
            "joueur": player_name,
            "cote_popularite": round(score * 100, 2),
            "mentions": mentions,
            "sentiment_moyen": round(sentiment_avg, 2),
            "diversite_sources": len(sources),
            "sources": list(sources),
            "distribution_sentiment": {
                "positif": len([a for a in player_articles if a.get("sentiment", {}).get("label") == "POSITIVE"]),
                "neutre": len([a for a in player_articles if a.get("sentiment", {}).get("label") == "NEUTRAL"]),
                "negatif": len([a for a in player_articles if a.get("sentiment", {}).get("label") == "NEGATIVE"]),
            },
            "date_calcul": datetime.now().isoformat(),
        }

# 5. FONCTION PRINCIPALE AXE 1


def run_axe1(use_api=True, use_scraping=True, use_nlp=True) -> dict:
    """
    Exécute la pipeline complète de l'Axe 1 - Web Content Mining.

    Args:
        use_api: Activer la collecte via APIs
        use_scraping: Activer le scraping Transfermarkt
        use_nlp: Activer le pipeline NLP

    Returns:
        Dictionnaire avec toutes les données collectées et enrichies
    """
    logger.info("=" * 60)
    logger.info("AXE 1 - WEB CONTENT MINING")
    logger.info("=" * 60)

    results = {}

    # --- 1. Collecte d'articles ---
    if use_api:
        logger.info("\n Collecte d'articles de presse...")
        news = NewsCollector()
        articles = news.collect_player_articles()
        export_to_json(articles, "axe1_articles_bruts.json")
        results["articles"] = articles
        news.stats.print_summary()
    else:
        # Charger depuis le cache
        try:
            articles = json.load(open(os.path.join("data", "axe1_articles_bruts.json")))
            results["articles"] = articles
        except FileNotFoundError:
            articles = []
            results["articles"] = []

    # --- 2. Statistiques Football ---
    if use_api:
        logger.info("\n Collecte des statistiques...")
        stats_collector = FootballStatsCollector()
        stats_data = stats_collector.collect_all_leagues()
        export_to_json(stats_data["classements"], "axe1_classements.json")
        export_to_json(stats_data["buteurs"], "axe1_buteurs.json")
        export_to_json(stats_data["matchs"], "axe1_matchs.json")
        results["stats"] = stats_data
        stats_collector.stats.print_summary()

    # --- 3. Profils Transfermarkt ---
    if use_scraping:
        logger.info("\n Scraping Transfermarkt...")
        tm = TransfermarktScraper()
        profiles = tm.collect_all_players()
        export_to_json(profiles, "axe1_profils_transfermarkt.json")
        results["profils"] = profiles
        tm.stats.print_summary()

    # --- 4. Pipeline NLP ---
    if use_nlp and articles:
        logger.info("\n Pipeline NLP (NER + Sentiment)...")
        nlp = NLPPipeline()
        enriched_articles = nlp.process_articles(articles)
        export_to_json(enriched_articles, "axe1_articles_enrichis.json")
        results["articles_enrichis"] = enriched_articles

        # Calcul des cotes de popularité
        logger.info("\n Calcul des cotes de popularité...")
        popularity = []
        for player in TARGET_PLAYERS:
            score = nlp.compute_popularity_score(player["name"], enriched_articles)
            popularity.append(score)

        # Tri par cote décroissante
        popularity.sort(key=lambda x: x["cote_popularite"], reverse=True)
        export_to_json(popularity, "axe1_cotes_popularite.json")
        results["cotes_popularite"] = popularity

        nlp.stats.print_summary()

        # Affichage résumé
        logger.info("\n CLASSEMENT COTES DE POPULARITÉ:")
        for i, p in enumerate(popularity, 1):
            logger.info(f"  {i}. {p['joueur']}: {p['cote_popularite']}/100 "
                        f"({p['mentions']} mentions, sentiment={p['sentiment_moyen']}/5)")

    logger.info("\n Axe 1 terminé avec succès!")
    return results


if __name__ == "__main__":
    # Mode test sans API (pour développement)
    run_axe1(use_api=False, use_scraping=False, use_nlp=True)