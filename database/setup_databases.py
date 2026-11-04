"""
Scout-AI : Configuration des Bases de Données NoSQL
=====================================================
Script principal pour configurer et peupler les 3 bases de données :

1. MongoDB    → Stockage principal (documents : articles, stats, profils)
2. Neo4J      → Graphe relationnel (joueurs, clubs, agents, médias)
3. ElasticSearch → Recherche full-text (articles) + agrégations (sentiments)

Usage:
    python -m database.setup_databases              # Configure les 3 bases
    python -m database.setup_databases --mongo      # MongoDB uniquement
    python -m database.setup_databases --neo4j      # Neo4J uniquement
    python -m database.setup_databases --elastic    # ElasticSearch uniquement
    python -m database.setup_databases --status     # Vérifier l'état des bases

Prérequis :
    - MongoDB lancé   : mongod (port 27017)
    - Neo4J lancé     : neo4j start (port 7687)
    - ElasticSearch   : service elasticsearch start (port 9200)
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.helpers import setup_logger
from config.settings import MONGO_URI, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ELASTIC_HOST

logger = setup_logger("database-setup")


def check_status():
    """Vérifie l'état de connexion de chaque base de données."""
    print("\n" + "=" * 60)
    print(" VÉRIFICATION DES BASES DE DONNÉES")
    print("=" * 60)

    status = {}

    # MongoDB
    print("\n1️⃣  MongoDB...")
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
        client.server_info()
        db = client["scout_ai"]
        collections = db.list_collection_names()
        total = sum(db[c].count_documents({}) for c in collections)
        print(f"    Connecté | {len(collections)} collections | {total} documents")
        status["mongodb"] = True
        client.close()
    except Exception as e:
        print(f"    Non disponible: {e}")
        status["mongodb"] = False

    # Neo4J
    print("\n2️⃣  Neo4J...")
    try:
        from neo4j import GraphDatabase
        connected = False
        for auth_option in [None, (NEO4J_USER, NEO4J_PASSWORD), ("neo4j", "neo4j")]:
            try:
                driver = GraphDatabase.driver(NEO4J_URI, auth=auth_option)
                with driver.session() as session:
                    result = session.run("MATCH (n) RETURN count(n) AS nodes")
                    nodes = result.single()["nodes"]
                    result2 = session.run("MATCH ()-[r]->() RETURN count(r) AS edges")
                    edges = result2.single()["edges"]
                print(f"    Connecté | {nodes} nœuds | {edges} arêtes")
                status["neo4j"] = True
                connected = True
                driver.close()
                break
            except Exception:
                continue
        if not connected:
            raise ConnectionError("Auth échoué avec toutes les méthodes")
    except Exception as e:
        print(f"    Non disponible: {e}")
        status["neo4j"] = False

    # ElasticSearch
    print("\n3️⃣  ElasticSearch...")
    try:
        from elasticsearch import Elasticsearch
        import elasticsearch
        es_version = int(elasticsearch.__version__[0])

        if es_version >= 8:
            es = Elasticsearch(
                ELASTIC_HOST,
                verify_certs=False, ssl_show_warn=False,
                request_timeout=30,
                headers={"Accept": "application/json"},
            )
        else:
            es = Elasticsearch(ELASTIC_HOST, verify_certs=False, timeout=30)

        if es.ping():
            info = es.info()
            version = info.get("version", {}).get("number", "?") if isinstance(info, dict) else "?"
            try:
                indices = list(es.indices.get(index="scout_*").keys())
            except Exception:
                indices = []
            total = sum(es.count(index=idx)["count"] for idx in indices) if indices else 0
            print(f"    Connecté (v{version}) | "
                  f"{len(indices)} index | {total} documents")
            status["elasticsearch"] = True
        else:
            print("    Ping échoué")
            status["elasticsearch"] = False
        es.close()
    except Exception as e:
        print(f"    Non disponible: {e}")
        status["elasticsearch"] = False

    # Résumé
    ok = sum(1 for v in status.values() if v)
    total = len(status)
    print(f"\n{'='*60}")
    print(f" Résultat: {ok}/{total} bases de données connectées")

    if ok < total:
        print("\n Guide d'installation rapide:")
        if not status.get("mongodb"):
            print("   MongoDB  : https://www.mongodb.com/try/download/community")
            print("              Puis lancez: mongod")
        if not status.get("neo4j"):
            print("   Neo4J    : https://neo4j.com/download/")
            print("              Puis lancez: neo4j console")
            print("               Changez le mot de passe par défaut à 'password'")
        if not status.get("elasticsearch"):
            print("   Elastic  : https://www.elastic.co/downloads/elasticsearch")
            print("              Puis lancez: elasticsearch")

    print("=" * 60)
    return status


def setup_all(mongo: bool = True, neo4j: bool = True, elastic: bool = True):
    """Configure et peuple les bases de données sélectionnées."""

    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║       SCOUT-AI  -  Configuration Base de Données         ║
    ║       MongoDB  +  Neo4J  +  ElasticSearch               ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

    start_time = datetime.now()
    results = {}

    # MONGODB

    if mongo:
        try:
            from database.mongo_manager import setup_mongodb
            manager = setup_mongodb()
            if manager is not None:
                results["mongodb"] = " OK"
                stats = manager.get_collection_stats()
                total = sum(stats.values())
                results["mongodb_docs"] = total
                manager.close()
            else:
                results["mongodb"] = " Échec de connexion"
        except Exception as e:
            results["mongodb"] = f" Erreur: {e}"
            logger.error(f"Erreur MongoDB: {e}")

    # NEO4J

    if neo4j:
        try:
            from database.neo4j_manager import setup_neo4j
            manager = setup_neo4j()
            if manager is not None:
                results["neo4j"] = " OK"
                stats = manager.get_graph_stats()
                results["neo4j_nodes"] = stats["total_nodes"]
                results["neo4j_edges"] = stats["total_edges"]
                manager.close()
            else:
                results["neo4j"] = " Échec de connexion"
        except Exception as e:
            results["neo4j"] = f" Erreur: {e}"
            logger.error(f"Erreur Neo4J: {e}")

    # ELASTICSEARCH

    if elastic:
        try:
            from database.elastic_manager import setup_elasticsearch
            manager = setup_elasticsearch()
            if manager is not None:
                results["elasticsearch"] = " OK"
                # Refresh pour que le count soit à jour
                import time
                time.sleep(1)
                stats = manager.get_index_stats()
                results["elastic_docs"] = sum(stats.values())
                manager.close()
            else:
                results["elasticsearch"] = " Échec de connexion"
        except Exception as e:
            results["elasticsearch"] = f" Erreur: {e}"
            logger.error(f"Erreur ElasticSearch: {e}")

    # RÉSUMÉ

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 60)
    print(" RÉSUMÉ DE LA CONFIGURATION DES BASES DE DONNÉES")
    print("=" * 60)
    print(f"  Durée totale: {elapsed:.1f} secondes\n")

    if "mongodb" in results:
        print(f"  MongoDB: {results['mongodb']}")
        if "mongodb_docs" in results:
            print(f"     → {results['mongodb_docs']} documents stockés")

    if "neo4j" in results:
        print(f"  Neo4J: {results['neo4j']}")
        if "neo4j_nodes" in results:
            print(f"     → {results['neo4j_nodes']} nœuds, {results['neo4j_edges']} arêtes")

    if "elasticsearch" in results:
        print(f"  ElasticSearch: {results['elasticsearch']}")
        if "elastic_docs" in results:
            print(f"     → {results['elastic_docs']} documents indexés")

    print("\n" + "=" * 60)

    # Vérifier si au moins une base fonctionne
    ok_count = sum(1 for k, v in results.items()
                   if isinstance(v, str) and v.startswith(" "))
    if ok_count == 0:
        print("  Aucune base de données n'est connectée.")
        print("   Les données restent dans data/ (JSON/CSV).")
        print("   Installez et lancez les bases, puis relancez ce script.")
    else:
        print(f" {ok_count} base(s) de données configurée(s) avec succès!")

    print("=" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scout-AI : Configuration des bases de données NoSQL"
    )
    parser.add_argument("--mongo", action="store_true", help="MongoDB uniquement")
    parser.add_argument("--neo4j", action="store_true", help="Neo4J uniquement")
    parser.add_argument("--elastic", action="store_true", help="ElasticSearch uniquement")
    parser.add_argument("--status", action="store_true", help="Vérifier l'état des bases")

    args = parser.parse_args()

    if args.status:
        check_status()
    elif args.mongo or args.neo4j or args.elastic:
        setup_all(mongo=args.mongo, neo4j=args.neo4j, elastic=args.elastic)
    else:
        # Par défaut : configurer les 3 bases
        setup_all()