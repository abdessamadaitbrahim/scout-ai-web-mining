"""
Scout-AI : Pipeline Automatisée End-to-End
============================================
Orchestre TOUT le projet en une seule commande :

  1. COLLECTE     → Scraping des 3 axes (APIs + Web + Simulation)
  2. PIPELINE IA  → 3 modèles comparés, sélection automatique du meilleur
  3. STOCKAGE     → MongoDB + Neo4J + ElasticSearch (via Docker)
  4. DASHBOARD    → Lance Streamlit pour la visualisation

Usage:
    python run_all.py                  # Tout lancer (avec APIs)
    python run_all.py --demo           # Mode démo (données simulées)
    python run_all.py --skip-collect   # Sauter la collecte (données déjà prêtes)
    python run_all.py --skip-db        # Sauter le stockage DB
    python run_all.py --no-dashboard   # Ne pas lancer le dashboard à la fin

Pipeline complète :
    Scraping (10 min) → NLP/BERT (5 min) → ML (30 sec) → DB (10 sec) → Dashboard
"""

import argparse
import os
import sys
import json
import time
import subprocess
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.helpers import setup_logger, export_to_json
from config.settings import DATA_DIR

logger = setup_logger("scout-ai-auto")

# ÉTAPE 1 : COLLECTE DES DONNÉES (3 Axes de Web Mining)


def step1_collect(demo: bool = False):
    """
    Collecte des données via les 3 axes de Web Mining.
    - Axe 1 : NewsAPI + Football-Data + Transfermarkt + NLP (BERT)
    - Axe 2 : Graphe relationnel + HITS + PageRank + Louvain
    - Axe 3 : Google Trends + Simulation logs + Apriori + K-Means
    """
    print("\n" + "=" * 70)
    print("   ÉTAPE 1/4 — COLLECTE DES DONNÉES")
    print("=" * 70)

    from main import run_pipeline
    results = run_pipeline(axe=0, demo=demo)

    # Vérifier les fichiers générés
    data_files = os.listdir(DATA_DIR) if os.path.exists(DATA_DIR) else []
    axe1_ok = any(f.startswith("axe1_") for f in data_files)
    axe2_ok = any(f.startswith("axe2_") for f in data_files)
    axe3_ok = any(f.startswith("axe3_") for f in data_files)

    print(f"\n   Axe 1 (Content Mining)   : {'OK' if axe1_ok else ' MANQUANT'}")
    print(f"   Axe 2 (Structure Mining) : {'OK' if axe2_ok else ' MANQUANT'}")
    print(f"   Axe 3 (Usage Mining)     : {'OK' if axe3_ok else ' MANQUANT'}")

    return results

# ÉTAPE 2 : PIPELINE IA (3 modèles → sélection du meilleur)


def step2_ml_pipeline():
    """
    Lance la pipeline IA comparative :
    - Compare Random Forest vs SVR vs Gradient Boosting 
    - Sauvegarde le meilleur modèle dans models/
    """
    print("\n" + "=" * 70)
    print("   ÉTAPE 2/4 — PIPELINE IA COMPARATIVE ")
    print("=" * 70)

    from pipeline_ia import run_pipeline as run_ia
    ia_results = run_ia()

    if ia_results is None:
        print("   Pipeline IA échouée (données manquantes)")
        return None, None

    # Trouver le meilleur modèle
    # pipeline_ia.run_pipeline() retourne le dict `results` de train_and_compare
    # Clés : 'Random Forest', 'SVR', 'Gradient Boosting'
    # Chaque valeur a : r2_test, mae_test, rmse_test, cv_r2_mean, etc.
    best_models = {}
    best_name = None
    best_r2 = -1
    for name, metrics in ia_results.items():
        if isinstance(metrics, dict) and "r2_test" in metrics:
            if metrics["r2_test"] > best_r2:
                best_name = name
                best_r2 = metrics["r2_test"]

    if best_name:
        best_models["transfer_prediction"] = {
            "task": "Transfer Potential Prediction",
            "best_model": best_name,
            "r2_test": best_r2,
            "metrics": {k: v for k, v in ia_results[best_name].items()
                        if isinstance(v, (int, float))}
        }

    # Vérifier les modèles sauvegardés
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    saved = [f for f in os.listdir(models_dir) if f.endswith(".joblib")] if os.path.exists(models_dir) else []
    print(f"\n   {len(saved)} fichiers modèles sauvegardés dans models/")
    for f in sorted(saved):
        size = os.path.getsize(os.path.join(models_dir, f))
        print(f"     → {f} ({size:,} octets)")

    # Recalculer la Fan Value de l'Axe 3 avec le meilleur modèle ML
    print("\n   Recalcul Fan Value (Axe 3) avec le meilleur modèle...")
    try:
        import pandas as pd
        from collectors.axe3_usage_mining import AudienceSegmenter
        from utils.helpers import export_to_csv

        logs_path = os.path.join(DATA_DIR, "axe3_web_logs.csv")
        if os.path.exists(logs_path):
            logs_df = pd.read_csv(logs_path)
            segmenter = AudienceSegmenter()
            features_df = segmenter.prepare_features(logs_df)
            clustered_df = segmenter.fit_predict(features_df)
            fan_df = segmenter.compute_fan_value(clustered_df)

            export_to_csv(
                fan_df[["nb_sessions", "duree_moyenne", "nb_joueurs_consultes",
                        "cluster", "fan_value"]].to_dict(orient="records"),
                "axe3_fan_values.csv"
            )
            avg_fv = fan_df["fan_value"].mean()
            print(f"      Fan Value recalculée: moyenne={avg_fv:.2f}/100 ({best_name})")
        else:
            print("      Logs non trouvés")
    except Exception as e:
        print(f"      Erreur recalcul: {e}")

    return ia_results, best_models

# ÉTAPE 3 : STOCKAGE DANS LES BASES DE DONNÉES NoSQL


def step3_database():
    """
    Stocke les données dans MongoDB + Neo4J + ElasticSearch.
    Lance les conteneurs Docker si nécessaire.
    """
    print("\n" + "=" * 70)
    print("   ÉTAPE 3/4 — STOCKAGE BASE DE DONNÉES")
    print("=" * 70)

    db_results = {"mongodb": False, "neo4j": False, "elasticsearch": False}

    # Vérifier si Docker est dispo et les conteneurs tournent
    docker_available = _check_docker_running()

    if not docker_available:
        print("\n    Docker non disponible ou conteneurs non lancés.")
        print("   Lancez d'abord: python setup_docker.py --up")
        print("   Passage au stockage des bases disponibles...\n")

    # --- MongoDB ---
    print("   MongoDB...")
    try:
        from database.mongo_manager import setup_mongodb
        manager = setup_mongodb()
        if manager is not None:
            stats = manager.get_collection_stats()
            total = sum(stats.values())
            print(f"      {total} documents stockés dans {len(stats)} collections")
            db_results["mongodb"] = True
            manager.close()
        else:
            print("      Connexion échouée")
    except Exception as e:
        print(f"      {e}")

    # --- Neo4J ---
    print("   Neo4J...")
    try:
        from database.neo4j_manager import setup_neo4j
        manager = setup_neo4j()
        if manager is not None:
            stats = manager.get_graph_stats()
            print(f"      {stats['total_nodes']} nœuds, {stats['total_edges']} arêtes")
            db_results["neo4j"] = True
            manager.close()
        else:
            print("      Connexion échouée")
    except Exception as e:
        print(f"      {e}")

    # --- ElasticSearch ---
    print("   ElasticSearch...")
    try:
        from database.elastic_manager import setup_elasticsearch
        manager = setup_elasticsearch()
        if manager is not None:
            stats = manager.get_index_stats()
            total = sum(stats.values())
            print(f"      {total} documents indexés")
            db_results["elasticsearch"] = True
            manager.close()
        else:
            print("      Connexion échouée")
    except Exception as e:
        print(f"      {e}")

    ok = sum(1 for v in db_results.values() if v)
    print(f"\n   Résultat: {ok}/3 bases de données configurées")

    return db_results


def _check_docker_running():
    """Vérifie si Docker tourne et si les conteneurs sont up."""
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        try:
            result = subprocess.run(
                ["docker-compose", "ps"],
                capture_output=True, text=True, timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False

# ÉTAPE 4 : LANCEMENT DU DASHBOARD


def step4_dashboard():
    """Lance le dashboard Streamlit."""
    print("\n" + "=" * 70)
    print("   ÉTAPE 4/4 — LANCEMENT DU DASHBOARD")
    print("=" * 70)

    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard_streamlit.py")

    if not os.path.exists(dashboard_path):
        print("   dashboard_streamlit.py non trouvé")
        return

    print("   Ouverture du dashboard dans le navigateur...")
    print("   URL : http://localhost:8501")
    print("   Ctrl+C pour arrêter\n")

    # Lancer Streamlit avec le même Python
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path])

# RÉSUMÉ GLOBAL


def print_final_summary(collect_results, ia_results, best_models, db_results, elapsed):
    """Affiche le résumé final de toute la pipeline."""

    print("\n" + "═" * 70)
    print("   RÉSUMÉ FINAL — SCOUT-AI PIPELINE COMPLÈTE")
    print("═" * 70)
    print(f"   Durée totale : {elapsed:.0f} secondes ({elapsed/60:.1f} min)")

    # Collecte
    print(f"\n   COLLECTE:")
    if collect_results:
        a1 = collect_results.get("axe1", {})
        print(f"     Axe 1 : {len(a1.get('articles', []))} articles, "
              f"{len(a1.get('profils', []))} profils")
        a2 = collect_results.get("axe2", {})
        g = a2.get("graphe", {})
        print(f"     Axe 2 : {len(g.get('nodes', []))} nœuds, "
              f"{len(g.get('edges', []))} arêtes")
        a3 = collect_results.get("axe3", {})
        print(f"     Axe 3 : {a3.get('logs_stats', {}).get('total_logs', '?')} logs, "
              f"{len(a3.get('clusters', []))} clusters")
    else:
        print("     (collecte sautée)")

    # IA
    print(f"\n   PIPELINE IA:")
    if best_models:
        for key, info in best_models.items():
            m = info["metrics"]
            score_parts = [f"{k}={v}" for k, v in m.items()
                           if k not in ("n_clusters", "noise_points", "eps", "inertia",
                                        "cv_r2_std", "cv_f1_std")][:3]
            print(f"     {info['task']}:  {info['best_model']} "
                  f"({', '.join(score_parts)})")
    else:
        print("     (pipeline IA sautée)")

    # Bases de données
    print(f"\n   BASES DE DONNÉES:")
    if db_results:
        icons = {"mongodb": "", "neo4j": "", "elasticsearch": ""}
        for db, ok in db_results.items():
            status = "" if ok else ""
            print(f"     {icons.get(db, '')} {db}: {status}")
    else:
        print("     (stockage DB sauté)")

    # Fichiers
    print(f"\n   FICHIERS GÉNÉRÉS:")
    if os.path.exists(DATA_DIR):
        files = sorted(os.listdir(DATA_DIR))
        total_size = sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in files)
        print(f"     {len(files)} fichiers ({total_size:,} octets)")
        for f in files:
            size = os.path.getsize(os.path.join(DATA_DIR, f))
            print(f"       → {f} ({size:,})")

    if os.path.exists(os.path.join(os.path.dirname(__file__), "figures")):
        figs = [f for f in os.listdir(os.path.join(os.path.dirname(__file__), "figures"))
                if f.endswith(".png")]
        print(f"\n     {len(figs)} figures ML dans figures/")

    print("\n" + "═" * 70)
    print("   PIPELINE SCOUT-AI TERMINÉE AVEC SUCCÈS!")
    print("═" * 70)

# MAIN


def main():
    parser = argparse.ArgumentParser(
        description="Scout-AI : Pipeline automatisée end-to-end"
    )
    parser.add_argument("--demo", action="store_true",
                        help="Mode démo (données simulées, pas d'appels API)")
    parser.add_argument("--skip-collect", action="store_true",
                        help="Sauter la collecte (utilise les données existantes)")
    parser.add_argument("--skip-ml", action="store_true",
                        help="Sauter la pipeline IA")
    parser.add_argument("--skip-db", action="store_true",
                        help="Sauter le stockage base de données")
    parser.add_argument("--no-dashboard", action="store_true",
                        help="Ne pas lancer le dashboard à la fin")

    args = parser.parse_args()

    banner = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║            SCOUT-AI  —  Pipeline Automatisée                     ║
    ║                                                                  ║
    ║   Collecte → Pipeline IA → Base de Données → Dashboard           ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

    start_time = datetime.now()
    collect_results = None
    ia_results = None
    best_models = None
    db_results = None

    # ── ÉTAPE 1 : COLLECTE ──────────────────────────────────
    if not args.skip_collect:
        collect_results = step1_collect(demo=args.demo)
    else:
        print("\n   Collecte sautée (--skip-collect)")

    # ── ÉTAPE 2 : PIPELINE IA ───────────────────────────────
    if not args.skip_ml:
        ia_results, best_models = step2_ml_pipeline()
    else:
        print("\n   Pipeline IA sautée (--skip-ml)")

    # ── ÉTAPE 3 : BASE DE DONNÉES ──────────────────────────
    if not args.skip_db:
        db_results = step3_database()
    else:
        print("\n   Stockage DB sauté (--skip-db)")

    # ── RÉSUMÉ FINAL ────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    print_final_summary(collect_results, ia_results, best_models, db_results, elapsed)

    # ── ÉTAPE 4 : DASHBOARD ─────────────────────────────────
    if not args.no_dashboard:
        step4_dashboard()
    else:
        print("\n   Lancez le dashboard manuellement :")
        print("     python -m streamlit run dashboard_streamlit.py")


if __name__ == "__main__":
    main()