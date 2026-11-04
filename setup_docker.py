"""
Scout-AI : Lancement Docker + Peuplement des Bases
====================================================
Ce script :
1. Vérifie que Docker est installé
2. Lance les 3 conteneurs (MongoDB, Neo4J, ElasticSearch)
3. Attend que chaque base soit prête (healthcheck)
4. Peuple les bases avec les données collectées

Usage:
    python setup_docker.py          → Lance tout
    python setup_docker.py --up     → Lance les conteneurs uniquement
    python setup_docker.py --down   → Arrête les conteneurs
    python setup_docker.py --reset  → Arrête + supprime les données + relance
"""

import subprocess
import sys
import time
import argparse
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import MONGO_URI, NEO4J_URI, NEO4J_USER, ELASTIC_HOST


def run_cmd(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Exécute une commande shell."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"   Erreur: {result.stderr.strip()}")
    return result


def check_docker() -> bool:
    """Vérifie que Docker et Docker Compose sont installés."""
    print(" Vérification de Docker...")

    result = run_cmd("docker --version", check=False)
    if result.returncode != 0:
        print("   Docker n'est pas installé.")
        print("   Installez Docker Desktop: https://www.docker.com/products/docker-desktop/")
        return False
    print(f"   {result.stdout.strip()}")

    # Vérifier docker compose (v2)
    result = run_cmd("docker compose version", check=False)
    if result.returncode != 0:
        # Essayer l'ancienne syntaxe
        result = run_cmd("docker-compose --version", check=False)
        if result.returncode != 0:
            print("   Docker Compose n'est pas installé.")
            return False
    print(f"   {result.stdout.strip()}")

    # Vérifier que Docker daemon tourne
    result = run_cmd("docker info", check=False)
    if result.returncode != 0:
        print("   Docker daemon n'est pas lancé.")
        print("   Lancez Docker Desktop ou exécutez: sudo systemctl start docker")
        return False
    print("   Docker daemon actif")

    return True


def get_compose_cmd() -> str:
    """Retourne la bonne commande compose (v1 ou v2)."""
    result = subprocess.run("docker compose version", shell=True, capture_output=True)
    if result.returncode == 0:
        return "docker compose"
    return "docker-compose"


def docker_up():
    """Lance les conteneurs."""
    print("\n Lancement des conteneurs...")
    compose = get_compose_cmd()
    result = run_cmd(f"{compose} up -d")
    if result.returncode == 0:
        print("   Conteneurs lancés")
    print(result.stdout)


def docker_down(volumes: bool = False):
    """Arrête les conteneurs."""
    compose = get_compose_cmd()
    flag = " -v" if volumes else ""
    print(f"\n Arrêt des conteneurs{' + suppression données' if volumes else ''}...")
    run_cmd(f"{compose} down{flag}")
    print("   Conteneurs arrêtés")


def wait_for_services(timeout: int = 120):
    """
    Attend que les 3 bases soient prêtes.
    Tente de se connecter toutes les 5 secondes pendant max timeout secondes.
    """
    print(f"\n Attente des bases de données (max {timeout}s)...")

    services = {
        "MongoDB": {"ready": False, "port": 27017},
        "Neo4J": {"ready": False, "port": 7687},
        "ElasticSearch": {"ready": False, "port": 9200},
    }

    start = time.time()

    while time.time() - start < timeout:
        all_ready = True

        # MongoDB
        if not services["MongoDB"]["ready"]:
            try:
                from pymongo import MongoClient
                client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
                client.server_info()
                client.close()
                services["MongoDB"]["ready"] = True
                print("   MongoDB prêt")
            except Exception:
                all_ready = False

        # Neo4J
        if not services["Neo4J"]["ready"]:
            try:
                from neo4j import GraphDatabase
                for auth_try in [None, (NEO4J_USER, ""), ("neo4j", "neo4j")]:
                    try:
                        driver = GraphDatabase.driver(
                            NEO4J_URI, auth=auth_try
                        )
                        with driver.session() as session:
                            session.run("RETURN 1")
                        driver.close()
                        services["Neo4J"]["ready"] = True
                        print("   Neo4J prêt")
                        break
                    except Exception:
                        continue
                if not services["Neo4J"]["ready"]:
                    all_ready = False
            except Exception:
                all_ready = False

        # ElasticSearch
        if not services["ElasticSearch"]["ready"]:
            try:
                from elasticsearch import Elasticsearch
                import elasticsearch
                es_version = int(elasticsearch.__version__[0])

                if es_version >= 8:
                    es = Elasticsearch(
                        "http://localhost:9200",
                        verify_certs=False, ssl_show_warn=False,
                        request_timeout=30,
                        headers={"Accept": "application/json"},
                    )
                else:
                    es = Elasticsearch(
                        "http://localhost:9200",
                        verify_certs=False, timeout=30,
                    )

                if es.ping():
                    services["ElasticSearch"]["ready"] = True
                    print("   ElasticSearch prêt")
                    es.close()
                else:
                    all_ready = False
            except Exception:
                # Fallback : tester directement avec requests
                try:
                    import requests as req
                    resp = req.get("http://localhost:9200", timeout=5)
                    if resp.status_code == 200:
                        services["ElasticSearch"]["ready"] = True
                        print("   ElasticSearch prêt (via HTTP direct)")
                        print("   Client Python incompatible. Corrigez avec:")
                        print("     pip install elasticsearch==7.17.9")
                except Exception:
                    all_ready = False

        if all_ready:
            elapsed = time.time() - start
            print(f"\n Toutes les bases sont prêtes ({elapsed:.0f}s)")
            return True

        # Afficher le statut
        waiting = [name for name, s in services.items() if not s["ready"]]
        if waiting:
            elapsed = int(time.time() - start)
            print(f"   [{elapsed}s] En attente de: {', '.join(waiting)}...")

        time.sleep(5)

    # Timeout
    failed = [name for name, s in services.items() if not s["ready"]]
    print(f"\n Timeout! Bases non prêtes: {', '.join(failed)}")
    return False


def populate_databases():
    """Peuple les bases avec les données collectées."""
    print("\n Peuplement des bases de données...")

    from database.setup_databases import setup_all
    setup_all()


def show_status():
    """Affiche l'état des conteneurs."""
    compose = get_compose_cmd()
    print("\n État des conteneurs:")
    result = run_cmd(f"{compose} ps")
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(
        description="Scout-AI : Gestion Docker des bases de données"
    )
    parser.add_argument("--up", action="store_true",
                        help="Lancer les conteneurs uniquement")
    parser.add_argument("--down", action="store_true",
                        help="Arrêter les conteneurs")
    parser.add_argument("--reset", action="store_true",
                        help="Reset complet (arrêt + suppression + relance)")
    parser.add_argument("--status", action="store_true",
                        help="Afficher l'état des conteneurs")
    parser.add_argument("--populate", action="store_true",
                        help="Peupler les bases uniquement (conteneurs déjà lancés)")

    args = parser.parse_args()

    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║      SCOUT-AI  -  Docker Database Manager               ║
    ║       MongoDB  +  Neo4J  +  ElasticSearch               ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)

    # Vérifications
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "docker-compose.yml")):
        print(" docker-compose.yml non trouvé. Lancez depuis le dossier scout-ai/")
        sys.exit(1)

    if args.status:
        show_status()
        from database.setup_databases import check_status
        check_status()
        return

    if args.down:
        if check_docker():
            docker_down()
        return

    if args.reset:
        if check_docker():
            docker_down(volumes=True)
            docker_up()
            if wait_for_services():
                populate_databases()
        return

    if args.populate:
        if wait_for_services(timeout=30):
            populate_databases()
        return

    if args.up:
        if check_docker():
            docker_up()
            wait_for_services()
        return

    # Mode par défaut : tout faire
    if not check_docker():
        sys.exit(1)

    docker_up()

    if wait_for_services():
        populate_databases()

    show_status()

    print("\n" + "=" * 60)
    print(" TOUT EST PRÊT!")
    print("=" * 60)
    print("   MongoDB       : mongodb://localhost:27017")
    print("   Neo4J Browser : http://localhost:7474")
    print("   ElasticSearch : http://localhost:9200")
    print()
    print("  Commandes utiles:")
    print("    python setup_docker.py --status    → État des bases")
    print("    python setup_docker.py --down      → Arrêter tout")
    print("    python setup_docker.py --reset     → Tout réinitialiser")
    print("=" * 60)


if __name__ == "__main__":
    main()