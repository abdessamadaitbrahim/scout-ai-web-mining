"""
Scout-AI : Point d'entrée principal
=====================================
Système de Veille et d'Analyse de Marché des Transferts

Usage:
    python main.py              # Lancer les 3 axes
    python main.py --axe 1      # Axe 1 uniquement (Content Mining)
    python main.py --axe 2      # Axe 2 uniquement (Structure Mining)
    python main.py --axe 3      # Axe 3 uniquement (Usage Mining)
    python main.py --demo       # Mode démo (sans API, données simulées)

"""

import argparse
import os
import sys
import json
from datetime import datetime

# Ajouter le dossier racine au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import setup_logger, export_to_json

logger = setup_logger("scout-ai-main")


def run_pipeline(axe: int = 0, demo: bool = False):
    """
    Exécute la pipeline de collecte Scout-AI.

    Args:
        axe: 0 = tous, 1/2/3 = axe spécifique
        demo: Si True, utilise des données simulées (pas d'API)
    """
    start_time = datetime.now()
    results = {}

    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║              SCOUT-AI  -  Collecte de Données            ║
    ║  Système de Veille & Analyse du Marché des Transferts    ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

    if demo:
        logger.info(" MODE DÉMO activé (données simulées, pas d'appels API)")

    # AXE 1 - Web Content Mining

    if axe in (0, 1):
        try:
            from collectors.axe1_content_mining import run_axe1
            results["axe1"] = run_axe1(
                use_api=not demo,
                use_scraping=not demo,
                use_nlp=True
            )
        except Exception as e:
            logger.error(f"Erreur Axe 1: {e}")
            import traceback
            traceback.print_exc()

    # AXE 2 - Web Structure Mining

    if axe in (0, 2):
        try:
            from collectors.axe2_structure_mining import run_axe2

            # Passer les articles enrichis de l'Axe 1 si disponibles
            articles = results.get("axe1", {}).get("articles_enrichis", None)
            results["axe2"] = run_axe2(
                articles=articles,
                export_neo4j=False  # True si Neo4J est installé
            )
        except Exception as e:
            logger.error(f"Erreur Axe 2: {e}")
            import traceback
            traceback.print_exc()

    # AXE 3 - Web Usage Mining

    if axe in (0, 3):
        try:
            from collectors.axe3_usage_mining import run_axe3
            results["axe3"] = run_axe3(
                use_trends=not demo,
                n_users=500,
                n_sessions=2000
            )
        except Exception as e:
            logger.error(f"Erreur Axe 3: {e}")
            import traceback
            traceback.print_exc()

    # RÉSUMÉ FINAL

    elapsed = (datetime.now() - start_time).total_seconds()

    print("\n" + "=" * 60)
    print(" RÉSUMÉ DE LA COLLECTE")
    print("=" * 60)
    print(f"  Durée totale : {elapsed:.1f} secondes")

    if "axe1" in results:
        a1 = results["axe1"]
        n_articles = len(a1.get("articles", []))
        n_enrichis = len(a1.get("articles_enrichis", []))
        n_profils = len(a1.get("profils", []))
        print(f"\n   Axe 1 - Content Mining:")
        print(f"     Articles collectés : {n_articles}")
        print(f"     Articles enrichis (NLP) : {n_enrichis}")
        print(f"     Profils Transfermarkt : {n_profils}")
        if a1.get("cotes_popularite"):
            top = a1["cotes_popularite"][0]
            print(f"      Top popularité : {top['joueur']} ({top['cote_popularite']}/100)")

    if "axe2" in results:
        a2 = results["axe2"]
        g = a2.get("graphe", {})
        print(f"\n   Axe 2 - Structure Mining:")
        print(f"     Nœuds du graphe : {len(g.get('nodes', []))}")
        print(f"     Arêtes du graphe : {len(g.get('edges', []))}")
        print(f"     Communautés détectées : {a2.get('communautes', {}).get('nb_communautes', '?')}")
        if a2.get("hits", {}).get("top_authorities"):
            top_auth = a2["hits"]["top_authorities"][0]
            print(f"      Top Authority : {top_auth['noeud']}")

    if "axe3" in results:
        a3 = results["axe3"]
        print(f"\n   Axe 3 - Usage Mining:")
        ls = a3.get("logs_stats", {})
        print(f"     Sessions simulées : {ls.get('sessions_uniques', '?')}")
        print(f"     Utilisateurs uniques : {ls.get('utilisateurs_uniques', '?')}")
        p = a3.get("patterns", {})
        print(f"     Patterns fréquents : {len(p.get('frequent_itemsets', []))}")
        print(f"     Règles d'association : {len(p.get('association_rules', []))}")
        print(f"     Clusters identifiés : {len(a3.get('clusters', []))}")

    # Fichiers de sortie
    print(f"\n   Fichiers générés dans data/:")
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if os.path.exists(data_dir):
        for f in sorted(os.listdir(data_dir)):
            size = os.path.getsize(os.path.join(data_dir, f))
            print(f"     → {f} ({size:,} octets)")

    print("\n" + "=" * 60)
    print(" Collecte terminée avec succès!")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scout-AI : Collecte de données pour l'analyse du marché des transferts"
    )
    parser.add_argument(
        "--axe", type=int, default=0, choices=[0, 1, 2, 3],
        help="Axe à exécuter (0=tous, 1=Content, 2=Structure, 3=Usage)"
    )
    parser.add_argument(
        "--demo", action="store_true",
        help="Mode démo sans appels API (données simulées)"
    )
    parser.add_argument(
        "--db", action="store_true",
        help="Configurer les bases de données NoSQL après la collecte"
    )
    parser.add_argument(
        "--db-only", action="store_true",
        help="Configurer uniquement les bases (sans relancer la collecte)"
    )

    args = parser.parse_args()

    if args.db_only:
        # Mode base de données uniquement
        from database.setup_databases import setup_all
        setup_all()
    else:
        run_pipeline(axe=args.axe, demo=args.demo)
        if args.db:
            from database.setup_databases import setup_all
            setup_all()