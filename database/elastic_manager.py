"""
Scout-AI : Module ElasticSearch
=================================
Moteur de recherche full-text pour les articles et l'analyse sémantique.

ElasticSearch stocke :
- Articles enrichis (avec recherche full-text sur titre, contenu, description)
- Index de sentiments (agrégations rapides par joueur, source, date)

Fonctionnalités :
- Recherche full-text multilingue (français + anglais)
- Filtrage par sentiment, source, joueur, date
- Agrégations pour le dashboard (top sources, distribution sentiment, timeline)
- Suggestions de recherche (autocomplete)

Mapping optimisé pour les requêtes du dashboard interactif.
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    ELASTIC_HOST, ELASTIC_INDEX_ARTICLES, ELASTIC_INDEX_SENTIMENT, DATA_DIR
)
from utils.helpers import setup_logger

logger = setup_logger("elasticsearch-manager")


class ElasticSearchManager:
    """
    Gestionnaire ElasticSearch pour Scout-AI.
    Gère l'indexation full-text et les requêtes avancées.
    """

    def __init__(self, host: str = ELASTIC_HOST):
        self.host = host
        self.es = None

    # CONNEXION


    def connect(self) -> bool:
        """Établit la connexion avec ElasticSearch."""
        try:
            from elasticsearch import Elasticsearch
            import elasticsearch

            es_version = int(elasticsearch.__version__[0])

            if es_version >= 8:
                # Client v8+ : nécessite basic_auth ou api_key, ou désactiver la sécurité
                self.es = Elasticsearch(
                    self.host,
                    verify_certs=False,
                    ssl_show_warn=False,
                    request_timeout=30,
                    # Pour ES 8 Docker sans sécurité
                    headers={"Accept": "application/json"},
                )
            else:
                # Client v7
                self.es = Elasticsearch(
                    self.host,
                    verify_certs=False,
                    timeout=30,
                )

            if not self.es.ping():
                raise ConnectionError("ElasticSearch ne répond pas au ping")

            info = self.es.info()
            if isinstance(info, dict):
                version = info.get("version", {}).get("number", "?")
            else:
                version = "?"
            logger.info(f" Connecté à ElasticSearch: {self.host} (version {version})")
            return True

        except ImportError:
            logger.error("elasticsearch non installé. Exécutez: pip install elasticsearch")
            return False
        except Exception as e:
            # Fallback : tester avec requests directement
            try:
                import requests
                resp = requests.get(f"{self.host}", timeout=5)
                if resp.status_code == 200:
                    logger.warning(f"ES répond via HTTP mais le client Python échoue: {e}")
                    logger.info(" Essayez: pip install elasticsearch==7.17.9 --break-system-packages")
                else:
                    logger.error(f" ElasticSearch HTTP {resp.status_code}")
            except Exception:
                pass
            logger.error(f" Erreur de connexion ElasticSearch: {e}")
            logger.info(" Assurez-vous qu'ElasticSearch est lancé")
            return False

    def close(self):
        """Ferme la connexion."""
        if self.es:
            self.es.close()
            logger.info("Connexion ElasticSearch fermée")

    # MAPPING & INDEX


    def create_articles_index(self):
        """
        Crée l'index 'scout_articles' avec un mapping optimisé.

        Le mapping définit les types de champs :
        - text : pour la recherche full-text (analysé, tokenisé)
        - keyword : pour les filtres exacts et agrégations
        - date : pour les requêtes temporelles
        - integer/float : pour les métriques numériques
        """
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "football_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase", "asciifolding", "stop_fr"]
                        }
                    },
                    "filter": {
                        "stop_fr": {
                            "type": "stop",
                            "stopwords": "_french_"
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "titre": {
                        "type": "text",
                        "analyzer": "football_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "description": {
                        "type": "text",
                        "analyzer": "football_analyzer"
                    },
                    "contenu": {
                        "type": "text",
                        "analyzer": "football_analyzer"
                    },
                    "source": {"type": "keyword"},
                    "auteur": {"type": "keyword"},
                    "url": {"type": "keyword", "index": False},
                    "image_url": {"type": "keyword", "index": False},
                    "date_publication": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "date_collecte": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "query": {"type": "keyword"},
                    "langue": {"type": "keyword"},
                    # Champs NLP enrichis
                    "sentiment": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "keyword"},
                            "score": {"type": "float"},
                            "stars": {"type": "integer"}
                        }
                    },
                    "nb_entites": {"type": "integer"},
                    "joueurs_mentions": {"type": "keyword"},
                    "clubs_mentions": {"type": "keyword"},
                    "entites_nommees": {
                        "type": "nested",
                        "properties": {
                            "texte": {"type": "keyword"},
                            "type": {"type": "keyword"},
                            "score": {"type": "float"}
                        }
                    }
                }
            }
        }

        index_name = ELASTIC_INDEX_ARTICLES

        # Supprimer l'index existant
        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)
            logger.info(f"Index {index_name} supprimé")

        self.es.indices.create(index=index_name, body=mapping)
        logger.info(f" Index {index_name} créé avec mapping optimisé")

    def create_sentiment_index(self):
        """
        Crée l'index dédié aux sentiments pour les agrégations rapides.
        Structure aplatie optimisée pour les agrégations du dashboard.
        """
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "doc_id": {"type": "keyword"},
                    "joueur": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "sentiment_label": {"type": "keyword"},
                    "sentiment_stars": {"type": "integer"},
                    "sentiment_score": {"type": "float"},
                    "date_publication": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                    "langue": {"type": "keyword"}
                }
            }
        }

        index_name = ELASTIC_INDEX_SENTIMENT

        if self.es.indices.exists(index=index_name):
            self.es.indices.delete(index=index_name)

        self.es.indices.create(index=index_name, body=mapping)
        logger.info(f" Index {index_name} créé")

    # INDEXATION DES DONNÉES


    def index_articles(self) -> int:
        """
        Indexe les articles enrichis dans ElasticSearch.
        Utilise le bulk API pour des performances optimales.
        """
        filepath = os.path.join(DATA_DIR, "axe1_articles_enrichis.json")
        if not os.path.exists(filepath):
            # Fallback sur articles bruts
            filepath = os.path.join(DATA_DIR, "axe1_articles_bruts.json")
            if not os.path.exists(filepath):
                logger.warning("Aucun fichier d'articles trouvé")
                return 0

        with open(filepath, "r", encoding="utf-8") as f:
            articles = json.load(f)

        logger.info(f"Indexation de {len(articles)} articles...")

        # Bulk indexation
        from elasticsearch.helpers import bulk

        actions = []
        for article in articles:
            # Nettoyer les dates pour ElasticSearch
            doc = article.copy()
            for date_field in ["date_publication", "date_collecte"]:
                if doc.get(date_field) and doc[date_field] == "":
                    doc[date_field] = None

            actions.append({
                "_index": ELASTIC_INDEX_ARTICLES,
                "_id": doc.get("doc_id", ""),
                "_source": doc
            })

        success, errors = bulk(self.es, actions, raise_on_error=False)
        # Rafraîchir l'index pour que les documents soient immédiatement consultables
        self.es.indices.refresh(index=ELASTIC_INDEX_ARTICLES)
        logger.info(f"  → {success} articles indexés, {len(errors) if isinstance(errors, list) else errors} erreurs")

        return success

    def index_sentiments(self) -> int:
        """
        Indexe les données de sentiment dans l'index dédié.
        Un document par (article × joueur mentionné).
        """
        filepath = os.path.join(DATA_DIR, "axe1_articles_enrichis.json")
        if not os.path.exists(filepath):
            return 0

        with open(filepath, "r", encoding="utf-8") as f:
            articles = json.load(f)

        from elasticsearch.helpers import bulk

        actions = []
        for article in articles:
            sentiment = article.get("sentiment", {})
            joueurs = article.get("joueurs_mentions", [])

            if not joueurs:
                joueurs = ["_general"]

            for joueur in joueurs:
                doc = {
                    "doc_id": article.get("doc_id"),
                    "joueur": joueur,
                    "source": article.get("source"),
                    "sentiment_label": sentiment.get("label"),
                    "sentiment_stars": sentiment.get("stars"),
                    "sentiment_score": sentiment.get("score"),
                    "date_publication": article.get("date_publication") or None,
                    "langue": article.get("langue"),
                }
                actions.append({
                    "_index": ELASTIC_INDEX_SENTIMENT,
                    "_source": doc
                })

        success, _ = bulk(self.es, actions, raise_on_error=False)
        self.es.indices.refresh(index=ELASTIC_INDEX_SENTIMENT)
        logger.info(f"  → {success} entrées sentiment indexées")

        return success

    # REQUÊTES DE RECHERCHE


    def search_articles(self, query: str, filters: dict = None,
                        size: int = 20) -> list:
        """
        Recherche full-text dans les articles.

        Args:
            query: Texte de recherche (ex: "Mbappé transfert Real Madrid")
            filters: Filtres optionnels {sentiment, source, langue, joueur}
            size: Nombre max de résultats

        Returns:
            Liste d'articles triés par pertinence
        """
        must = [{"multi_match": {
            "query": query,
            "fields": ["titre^3", "description^2", "contenu"],
            "type": "best_fields",
            "fuzziness": "AUTO"
        }}]

        filter_clauses = []
        if filters:
            if filters.get("sentiment"):
                filter_clauses.append(
                    {"term": {"sentiment.label": filters["sentiment"]}}
                )
            if filters.get("source"):
                filter_clauses.append(
                    {"term": {"source": filters["source"]}}
                )
            if filters.get("langue"):
                filter_clauses.append(
                    {"term": {"langue": filters["langue"]}}
                )
            if filters.get("joueur"):
                filter_clauses.append(
                    {"term": {"joueurs_mentions": filters["joueur"]}}
                )

        body = {
            "query": {
                "bool": {
                    "must": must,
                    "filter": filter_clauses
                }
            },
            "size": size,
            "highlight": {
                "fields": {
                    "titre": {},
                    "description": {},
                    "contenu": {"fragment_size": 200}
                }
            }
        }

        result = self.es.search(index=ELASTIC_INDEX_ARTICLES, body=body)

        articles = []
        for hit in result["hits"]["hits"]:
            article = hit["_source"]
            article["_score"] = hit["_score"]
            article["_highlights"] = hit.get("highlight", {})
            articles.append(article)

        return articles

    def get_sentiment_by_player(self) -> list:
        """Agrégation : sentiment moyen par joueur."""
        body = {
            "size": 0,
            "aggs": {
                "by_player": {
                    "terms": {"field": "joueur", "size": 20},
                    "aggs": {
                        "avg_stars": {"avg": {"field": "sentiment_stars"}},
                        "sentiment_dist": {
                            "terms": {"field": "sentiment_label"}
                        }
                    }
                }
            }
        }
        result = self.es.search(index=ELASTIC_INDEX_SENTIMENT, body=body)
        return result["aggregations"]["by_player"]["buckets"]

    def get_sentiment_by_source(self) -> list:
        """Agrégation : sentiment moyen par source média."""
        body = {
            "size": 0,
            "aggs": {
                "by_source": {
                    "terms": {"field": "source", "size": 20},
                    "aggs": {
                        "avg_stars": {"avg": {"field": "sentiment_stars"}},
                        "article_count": {"value_count": {"field": "doc_id"}}
                    }
                }
            }
        }
        result = self.es.search(index=ELASTIC_INDEX_SENTIMENT, body=body)
        return result["aggregations"]["by_source"]["buckets"]

    def get_sentiment_timeline(self, joueur: str = None) -> list:
        """Agrégation temporelle : évolution du sentiment par semaine."""
        body = {
            "size": 0,
            "query": {"match_all": {}},
            "aggs": {
                "by_week": {
                    "date_histogram": {
                        "field": "date_publication",
                        "calendar_interval": "week"
                    },
                    "aggs": {
                        "avg_stars": {"avg": {"field": "sentiment_stars"}},
                        "sentiment_dist": {
                            "terms": {"field": "sentiment_label"}
                        }
                    }
                }
            }
        }

        if joueur:
            body["query"] = {"term": {"joueur": joueur}}

        result = self.es.search(index=ELASTIC_INDEX_SENTIMENT, body=body)
        return result["aggregations"]["by_week"]["buckets"]

    def get_top_entities(self, entity_type: str = "PER", limit: int = 20) -> list:
        """Top entités nommées extraites des articles."""
        body = {
            "size": 0,
            "aggs": {
                "entities": {
                    "nested": {"path": "entites_nommees"},
                    "aggs": {
                        "by_type": {
                            "filter": {"term": {"entites_nommees.type": entity_type}},
                            "aggs": {
                                "names": {
                                    "terms": {
                                        "field": "entites_nommees.texte",
                                        "size": limit
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        result = self.es.search(index=ELASTIC_INDEX_ARTICLES, body=body)
        return result["aggregations"]["entities"]["by_type"]["names"]["buckets"]

    def get_index_stats(self) -> dict:
        """Statistiques des index."""
        stats = {}
        for index_name in [ELASTIC_INDEX_ARTICLES, ELASTIC_INDEX_SENTIMENT]:
            if self.es.indices.exists(index=index_name):
                count = self.es.count(index=index_name)["count"]
                stats[index_name] = count
            else:
                stats[index_name] = 0
        return stats

# FONCTION PRINCIPALE


def setup_elasticsearch() -> ElasticSearchManager | None:
    """
    Configure ElasticSearch : connexion, création des index, indexation.
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION ELASTICSEARCH")
    logger.info("=" * 60)

    manager = ElasticSearchManager()

    if not manager.connect():
        return None

    # Création des index avec mapping
    manager.create_articles_index()
    manager.create_sentiment_index()

    # Indexation des données
    n_articles = manager.index_articles()
    n_sentiments = manager.index_sentiments()

    # Résumé
    logger.info(f"\n RÉSUMÉ ELASTICSEARCH:")
    stats = manager.get_index_stats()
    for index_name, count in stats.items():
        logger.info(f"  → {index_name}: {count} documents")

    return manager


if __name__ == "__main__":
    manager = setup_elasticsearch()
    if manager:
        # Exemple de recherche
        print("\n Recherche 'Mbappé transfert':")
        results = manager.search_articles("Mbappé transfert")
        for r in results[:3]:
            print(f"  [{r['sentiment']['label']}] {r['titre'][:80]}...")
            print(f"    Source: {r['source']} | Score: {r['_score']:.2f}")

        print("\n Sentiment par joueur:")
        for bucket in manager.get_sentiment_by_player()[:5]:
            print(f"  {bucket['key']}: {bucket['avg_stars']['value']:.1f} "
                  f"({bucket['doc_count']} mentions)")

        manager.close()