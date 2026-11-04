"""
Scout-AI : Module MongoDB
===========================
Base de données documentaire pour le stockage principal.

MongoDB stocke :
- Articles de presse (bruts + enrichis NLP)
- Statistiques de football (classements, buteurs, matchs)
- Profils de joueurs (Transfermarkt)
- Cotes de popularité
- Logs de navigation (Axe 3)
- Résultats de clustering et patterns

Collections créées :
├── articles_bruts          → Articles NewsAPI non traités
├── articles_enrichis       → Articles + NER + Sentiment
├── classements             → Classements par ligue
├── buteurs                 → Top buteurs par ligue
├── matchs                  → Résultats de matchs
├── profils_joueurs         → Profils Transfermarkt
├── cotes_popularite        → Scores de popularité calculés
├── web_logs                → Logs de navigation simulés
├── patterns                → Résultats Apriori
├── clusters                → Segments d'audience K-Means
├── fan_values              → Fan Value par utilisateur
├── trends                  → Données Google Trends
└── metadata                → Métadonnées de collecte
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import MONGO_URI, MONGO_DB_NAME, DATA_DIR
from utils.helpers import setup_logger

logger = setup_logger("mongodb-manager")


class MongoDBManager:
    """
    Gestionnaire MongoDB pour Scout-AI.
    Gère la connexion, l'insertion, l'indexation et les requêtes.
    """

    def __init__(self, uri: str = MONGO_URI, db_name: str = MONGO_DB_NAME):
        self.uri = uri
        self.db_name = db_name
        self.client = None
        self.db = None

    # CONNEXION


    def connect(self) -> bool:
        """Établit la connexion avec MongoDB."""
        try:
            from pymongo import MongoClient
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test de connexion
            self.client.server_info()
            self.db = self.client[self.db_name]
            logger.info(f" Connecté à MongoDB: {self.uri} / {self.db_name}")
            return True
        except ImportError:
            logger.error("pymongo non installé. Exécutez: pip install pymongo")
            return False
        except Exception as e:
            logger.error(f" Erreur de connexion MongoDB: {e}")
            logger.info(" Assurez-vous que MongoDB est lancé (mongod)")
            return False

    def close(self):
        """Ferme la connexion."""
        if self.client:
            self.client.close()
            logger.info("Connexion MongoDB fermée")

    # CRÉATION DES INDEX


    def create_indexes(self):
        """
        Crée les index pour optimiser les requêtes fréquentes.
        Les index accélèrent les recherches sur les champs les plus utilisés.
        """
        if self.db is None:
            logger.error("Non connecté à MongoDB")
            return

        logger.info("Création des index MongoDB...")

        # Articles : recherche par joueur, source, date, sentiment
        self.db["articles_bruts"].create_index("doc_id", unique=True)
        self.db["articles_bruts"].create_index("query")
        self.db["articles_bruts"].create_index("source")
        self.db["articles_bruts"].create_index("date_publication")

        self.db["articles_enrichis"].create_index("doc_id", unique=True)
        self.db["articles_enrichis"].create_index("sentiment.label")
        self.db["articles_enrichis"].create_index("joueurs_mentions")
        self.db["articles_enrichis"].create_index("clubs_mentions")
        self.db["articles_enrichis"].create_index([
            ("source", 1), ("sentiment.stars", -1)
        ])

        # Stats : recherche par ligue, équipe, joueur
        self.db["classements"].create_index([
            ("competition_id", 1), ("position", 1)
        ])
        self.db["buteurs"].create_index([
            ("competition_id", 1), ("buts", -1)
        ])
        self.db["matchs"].create_index("competition_id")
        self.db["matchs"].create_index("date")

        # Profils joueurs
        self.db["profils_joueurs"].create_index("player_id_tm", unique=True)
        self.db["profils_joueurs"].create_index("nom")

        # Cotes de popularité
        self.db["cotes_popularite"].create_index("joueur", unique=True)
        self.db["cotes_popularite"].create_index([
            ("cote_popularite", -1)
        ])

        # Logs de navigation
        self.db["web_logs"].create_index("user_id")
        self.db["web_logs"].create_index("session_id")
        self.db["web_logs"].create_index("page_category")
        self.db["web_logs"].create_index("player_viewed")

        # Clusters
        self.db["clusters"].create_index("cluster_id", unique=True)

        # Metadata
        self.db["metadata"].create_index("type", unique=True)

        logger.info(" Index créés avec succès")

    # INSERTION DES DONNÉES


    def _load_json(self, filename: str) -> list | dict | None:
        """Charge un fichier JSON depuis data/."""
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Fichier non trouvé: {filepath}")
            return None
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    def _insert_many(self, collection_name: str, data: list[dict],
                     unique_field: str = None) -> int:
        """
        Insère des documents dans une collection.
        Si unique_field est spécifié, utilise upsert pour éviter les doublons.
        """
        if not data:
            logger.warning(f"Aucune donnée à insérer dans {collection_name}")
            return 0

        collection = self.db[collection_name]

        if unique_field:
            # Upsert : met à jour si existe, insère sinon
            count = 0
            for doc in data:
                key = doc.get(unique_field)
                if key is not None:
                    collection.update_one(
                        {unique_field: key},
                        {"$set": doc},
                        upsert=True
                    )
                    count += 1
            logger.info(f"  → {collection_name}: {count} documents upsertés")
            return count
        else:
            # Insertion simple (supprime l'ancien contenu)
            collection.delete_many({})
            result = collection.insert_many(data)
            count = len(result.inserted_ids)
            logger.info(f"  → {collection_name}: {count} documents insérés")
            return count

    def import_axe1_data(self) -> dict:
        """
        Importe toutes les données de l'Axe 1 (Content Mining).
        """
        logger.info("\n Import Axe 1 - Content Mining dans MongoDB...")
        stats = {}

        # Articles bruts
        articles = self._load_json("axe1_articles_bruts.json")
        if articles:
            stats["articles_bruts"] = self._insert_many(
                "articles_bruts", articles, unique_field="doc_id"
            )

        # Articles enrichis NLP
        enrichis = self._load_json("axe1_articles_enrichis.json")
        if enrichis:
            stats["articles_enrichis"] = self._insert_many(
                "articles_enrichis", enrichis, unique_field="doc_id"
            )

        # Classements
        classements = self._load_json("axe1_classements.json")
        if classements:
            stats["classements"] = self._insert_many("classements", classements)

        # Buteurs
        buteurs = self._load_json("axe1_buteurs.json")
        if buteurs:
            stats["buteurs"] = self._insert_many("buteurs", buteurs)

        # Matchs
        matchs = self._load_json("axe1_matchs.json")
        if matchs:
            stats["matchs"] = self._insert_many("matchs", matchs)

        # Profils Transfermarkt
        profils = self._load_json("axe1_profils_transfermarkt.json")
        if profils:
            stats["profils_joueurs"] = self._insert_many(
                "profils_joueurs", profils, unique_field="player_id_tm"
            )

        # Cotes de popularité
        cotes = self._load_json("axe1_cotes_popularite.json")
        if cotes:
            stats["cotes_popularite"] = self._insert_many(
                "cotes_popularite", cotes, unique_field="joueur"
            )

        return stats

    def import_axe3_data(self) -> dict:
        """
        Importe toutes les données de l'Axe 3 (Usage Mining).
        """
        logger.info("\n Import Axe 3 - Usage Mining dans MongoDB...")
        stats = {}

        # Logs de navigation (depuis CSV)
        import csv
        logs_path = os.path.join(DATA_DIR, "axe3_web_logs.csv")
        if os.path.exists(logs_path):
            with open(logs_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                logs = list(reader)
            if logs:
                stats["web_logs"] = self._insert_many("web_logs", logs)

        # Patterns Apriori
        patterns = self._load_json("axe3_patterns.json")
        if patterns:
            self.db["patterns"].delete_many({})
            self.db["patterns"].insert_one(patterns)
            stats["patterns"] = 1
            logger.info(f"  → patterns: 1 document inséré")

        # Associations joueurs
        assoc = self._load_json("axe3_player_associations.json")
        if assoc:
            stats["player_associations"] = self._insert_many(
                "player_associations", assoc
            )

        # Clusters
        clusters = self._load_json("axe3_clusters.json")
        if clusters:
            stats["clusters"] = self._insert_many(
                "clusters", clusters, unique_field="cluster_id"
            )

        # Trends
        trends = self._load_json("axe3_trends.json")
        if trends:
            self.db["trends"].delete_many({})
            self.db["trends"].insert_one(trends)
            stats["trends"] = 1
            logger.info(f"  → trends: 1 document inséré")

        return stats

    def save_metadata(self):
        """Sauvegarde les métadonnées de la collecte."""
        metadata = {
            "type": "collection_metadata",
            "date_import": datetime.now().isoformat(),
            "version": "1.0",
            "projet": "Scout-AI",
            "module": "Analyse de Web (M242)",
            "collections": self.db.list_collection_names(),
            "stats": {}
        }
        for coll_name in self.db.list_collection_names():
            metadata["stats"][coll_name] = self.db[coll_name].count_documents({})

        self.db["metadata"].update_one(
            {"type": "collection_metadata"},
            {"$set": metadata},
            upsert=True
        )
        logger.info(" Métadonnées sauvegardées")

    # REQUÊTES UTILES (pour le Dashboard)


    def get_top_players_by_popularity(self, limit: int = 10) -> list:
        """Retourne les joueurs triés par cote de popularité."""
        return list(
            self.db["cotes_popularite"]
            .find({}, {"_id": 0})
            .sort("cote_popularite", -1)
            .limit(limit)
        )

    def get_articles_by_sentiment(self, sentiment: str) -> list:
        """Retourne les articles filtrés par sentiment (POSITIVE/NEGATIVE/NEUTRAL)."""
        return list(
            self.db["articles_enrichis"]
            .find({"sentiment.label": sentiment}, {"_id": 0})
            .limit(50)
        )

    def get_sentiment_distribution(self) -> dict:
        """Distribution des sentiments dans les articles."""
        pipeline = [
            {"$group": {
                "_id": "$sentiment.label",
                "count": {"$sum": 1},
                "avg_stars": {"$avg": "$sentiment.stars"}
            }},
            {"$sort": {"count": -1}}
        ]
        results = list(self.db["articles_enrichis"].aggregate(pipeline))
        return {r["_id"]: {"count": r["count"], "avg_stars": round(r["avg_stars"], 2)}
                for r in results if r["_id"]}

    def get_top_sources(self, limit: int = 10) -> list:
        """Top sources médiatiques par nombre d'articles."""
        pipeline = [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": limit}
        ]
        return list(self.db["articles_enrichis"].aggregate(pipeline))

    def get_league_standings(self, league_id: int) -> list:
        """Classement d'une ligue spécifique."""
        return list(
            self.db["classements"]
            .find({"competition_id": league_id}, {"_id": 0})
            .sort("position", 1)
        )

    def get_player_profile(self, player_name: str) -> dict | None:
        """Profil complet d'un joueur."""
        return self.db["profils_joueurs"].find_one(
            {"nom": player_name}, {"_id": 0}
        )

    def get_cluster_summary(self) -> list:
        """Résumé des clusters d'audience."""
        return list(
            self.db["clusters"].find({}, {"_id": 0}).sort("cluster_id", 1)
        )

    def get_collection_stats(self) -> dict:
        """Statistiques de toutes les collections."""
        stats = {}
        for name in self.db.list_collection_names():
            stats[name] = self.db[name].count_documents({})
        return stats

# FONCTION PRINCIPALE


def setup_mongodb() -> MongoDBManager | None:
    """
    Configure MongoDB : connexion, import des données, création des index.
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION MONGODB")
    logger.info("=" * 60)

    manager = MongoDBManager()

    if not manager.connect():
        return None

    # Création des index
    manager.create_indexes()

    # Import des données
    stats_axe1 = manager.import_axe1_data()
    stats_axe3 = manager.import_axe3_data()

    # Metadata
    manager.save_metadata()

    # Résumé
    logger.info("\n RÉSUMÉ MONGODB:")
    all_stats = manager.get_collection_stats()
    for coll, count in sorted(all_stats.items()):
        logger.info(f"  → {coll}: {count} documents")

    total = sum(all_stats.values())
    logger.info(f"\n  Total: {total} documents dans {len(all_stats)} collections")

    return manager


if __name__ == "__main__":
    manager = setup_mongodb()
    if manager:
        # Exemple de requêtes
        print("\n Top joueurs par popularité:")
        for p in manager.get_top_players_by_popularity(5):
            print(f"  {p['joueur']}: {p['cote_popularite']}/100")

        print("\n Distribution des sentiments:")
        for label, data in manager.get_sentiment_distribution().items():
            print(f"  {label}: {data['count']} articles (avg={data['avg_stars']})")

        manager.close()