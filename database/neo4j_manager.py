"""
Scout-AI : Module Neo4J
========================
Base de données orientée graphe pour le réseau relationnel.

Neo4J stocke :
- Graphe relationnel Axe 2 (joueurs, clubs, agents, médias, ligues)
- Relations : JOUE_POUR, TRANSFERT, RUMEUR, MENTIONNE, COUVRE, REPRESENTE, COMPETE_DANS
- Scores HITS et métriques de centralité comme propriétés des nœuds

Schéma du graphe :
    (JOUEUR)-[:JOUE_POUR]->(CLUB)
    (CLUB)-[:TRANSFERT|RUMEUR]->(CLUB)
    (MEDIA)-[:MENTIONNE]->(JOUEUR)
    (MEDIA)-[:COUVRE]->(CLUB)
    (MEDIA)-[:COUVRE_RUMEUR]->(CLUB)
    (AGENT)-[:REPRESENTE]->(JOUEUR)
    (CLUB)-[:COMPETE_DANS]->(LIGUE)
"""

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATA_DIR
from utils.helpers import setup_logger

logger = setup_logger("neo4j-manager")


class Neo4JManager:
    """
    Gestionnaire Neo4J pour Scout-AI.
    Gère la connexion, l'import du graphe, et les requêtes Cypher.
    """

    def __init__(self, uri: str = NEO4J_URI, user: str = NEO4J_USER,
                 password: str = NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    # CONNEXION


    def connect(self) -> bool:
        """Établit la connexion avec Neo4J."""
        try:
            from neo4j import GraphDatabase

            # Essayer d'abord sans authentification (Docker NEO4J_AUTH=none)
            for auth in [None, (self.user, self.password)]:
                try:
                    self.driver = GraphDatabase.driver(self.uri, auth=auth)
                    with self.driver.session() as session:
                        session.run("RETURN 1")
                    auth_str = "sans auth" if auth is None else f"user={auth[0]}"
                    logger.info(f" Connecté à Neo4J: {self.uri} ({auth_str})")
                    return True
                except Exception:
                    if self.driver:
                        self.driver.close()
                    continue

            raise ConnectionError("Toutes les méthodes d'authentification ont échoué")
            return True
        except ImportError:
            logger.error("neo4j non installé. Exécutez: pip install neo4j")
            return False
        except Exception as e:
            logger.error(f" Erreur de connexion Neo4J: {e}")
            logger.info(" Assurez-vous que Neo4J est lancé (neo4j start)")
            return False

    def close(self):
        """Ferme la connexion."""
        if self.driver:
            self.driver.close()
            logger.info("Connexion Neo4J fermée")

    def _run_query(self, query: str, params: dict = None) -> list:
        """Exécute une requête Cypher et retourne les résultats."""
        with self.driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    # NETTOYAGE & CONTRAINTES


    def clear_database(self):
        """Supprime toutes les données de la base."""
        self._run_query("MATCH (n) DETACH DELETE n")
        logger.info("Base Neo4J nettoyée")

    def create_constraints(self):
        """
        Crée les contraintes d'unicité et les index.
        Optimise les performances des requêtes.
        """
        logger.info("Création des contraintes Neo4J...")

        constraints = [
            "CREATE CONSTRAINT joueur_name IF NOT EXISTS FOR (j:JOUEUR) REQUIRE j.name IS UNIQUE",
            "CREATE CONSTRAINT club_name IF NOT EXISTS FOR (c:CLUB) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT media_name IF NOT EXISTS FOR (m:MEDIA) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT agent_name IF NOT EXISTS FOR (a:AGENT) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT ligue_name IF NOT EXISTS FOR (l:LIGUE) REQUIRE l.name IS UNIQUE",
        ]

        for constraint in constraints:
            try:
                self._run_query(constraint)
            except Exception as e:
                # La contrainte existe peut-être déjà
                logger.debug(f"Contrainte ignorée: {e}")

        # Index supplémentaires
        indexes = [
            "CREATE INDEX joueur_team IF NOT EXISTS FOR (j:JOUEUR) ON (j.team)",
            "CREATE INDEX club_ligue IF NOT EXISTS FOR (c:CLUB) ON (c.ligue)",
        ]
        for idx in indexes:
            try:
                self._run_query(idx)
            except Exception:
                pass

        logger.info(" Contraintes et index créés")

    # IMPORT DU GRAPHE


    def import_graph_from_json(self) -> dict:
        """
        Importe le graphe depuis axe2_graphe.json.
        Utilise MERGE pour éviter les doublons.
        """
        logger.info("\n Import du graphe dans Neo4J...")

        filepath = os.path.join(DATA_DIR, "axe2_graphe.json")
        if not os.path.exists(filepath):
            logger.error(f"Fichier non trouvé: {filepath}")
            return {"nodes": 0, "edges": 0}

        with open(filepath, "r", encoding="utf-8") as f:
            graph_data = json.load(f)

        nodes = graph_data.get("nodes", [])
        edges = graph_data.get("edges", [])

        # Import des nœuds
        node_count = 0
        for node in nodes:
            node_type = node.get("type", "Entity")
            name = node.get("id", "")
            props = {k: v for k, v in node.items() if k not in ("type", "id")}
            props["name"] = name

            # Nettoyer les propriétés (Neo4J n'accepte pas les None)
            props = {k: v for k, v in props.items() if v is not None}

            query = f"MERGE (n:{node_type} {{name: $name}}) SET n += $props"
            try:
                self._run_query(query, {"name": name, "props": props})
                node_count += 1
            except Exception as e:
                logger.warning(f"Erreur nœud {name}: {e}")

        logger.info(f"  → {node_count} nœuds importés")

        # Import des arêtes
        edge_count = 0
        for edge in edges:
            source = edge.get("source", "")
            target = edge.get("target", "")
            rel_type = edge.get("relation", "CONNECTED_TO")
            props = {k: v for k, v in edge.items()
                     if k not in ("source", "target", "relation") and v is not None}

            # Convertir les float en string pour les propriétés Neo4J
            for k, v in props.items():
                if isinstance(v, float):
                    props[k] = round(v, 6)

            query = (
                f"MATCH (a {{name: $source}}), (b {{name: $target}}) "
                f"MERGE (a)-[r:{rel_type}]->(b) SET r += $props"
            )
            try:
                self._run_query(query, {
                    "source": source, "target": target, "props": props
                })
                edge_count += 1
            except Exception as e:
                logger.warning(f"Erreur arête {source}->{target}: {e}")

        logger.info(f"  → {edge_count} arêtes importées")

        return {"nodes": node_count, "edges": edge_count}

    def import_hits_scores(self):
        """
        Ajoute les scores HITS comme propriétés des nœuds.
        Permet de requêter directement les hubs/authorities dans Neo4J.
        """
        filepath = os.path.join(DATA_DIR, "axe2_hits.json")
        if not os.path.exists(filepath):
            return

        with open(filepath, "r", encoding="utf-8") as f:
            hits_data = json.load(f)

        logger.info("Import des scores HITS dans les nœuds...")

        for hub in hits_data.get("top_hubs", []):
            self._run_query(
                "MATCH (n {name: $name}) SET n.hub_score = $score",
                {"name": hub["noeud"], "score": hub["score_hub"]}
            )

        for auth in hits_data.get("top_authorities", []):
            self._run_query(
                "MATCH (n {name: $name}) SET n.authority_score = $score",
                {"name": auth["noeud"], "score": auth["score_authority"]}
            )

        logger.info("  → Scores HITS ajoutés aux nœuds")

    def import_centrality_scores(self):
        """
        Ajoute les métriques de centralité comme propriétés des nœuds.
        """
        filepath = os.path.join(DATA_DIR, "axe2_centralite.json")
        if not os.path.exists(filepath):
            return

        with open(filepath, "r", encoding="utf-8") as f:
            centrality_data = json.load(f)

        logger.info("Import des métriques de centralité...")

        for node_data in centrality_data:
            self._run_query(
                """MATCH (n {name: $name})
                   SET n.pagerank = $pagerank,
                       n.betweenness = $betweenness,
                       n.closeness = $closeness,
                       n.in_degree = $in_degree,
                       n.out_degree = $out_degree""",
                {
                    "name": node_data["noeud"],
                    "pagerank": node_data.get("pagerank", 0),
                    "betweenness": node_data.get("betweenness", 0),
                    "closeness": node_data.get("closeness", 0),
                    "in_degree": node_data.get("in_degree", 0),
                    "out_degree": node_data.get("out_degree", 0),
                }
            )

        logger.info(f"  → Centralité ajoutée à {len(centrality_data)} nœuds")

    # REQUÊTES CYPHER UTILES


    def get_graph_stats(self) -> dict:
        """Statistiques globales du graphe."""
        nodes = self._run_query("MATCH (n) RETURN count(n) AS count")[0]["count"]
        edges = self._run_query("MATCH ()-[r]->() RETURN count(r) AS count")[0]["count"]
        types = self._run_query(
            "MATCH (n) RETURN labels(n)[0] AS type, count(n) AS count ORDER BY count DESC"
        )
        return {"total_nodes": nodes, "total_edges": edges, "node_types": types}

    def get_player_network(self, player_name: str) -> list:
        """Retourne le réseau complet d'un joueur (club, agent, médias, rumeurs)."""
        return self._run_query("""
            MATCH (j:JOUEUR {name: $name})-[r]-(connected)
            RETURN j.name AS joueur, type(r) AS relation,
                   connected.name AS connected_to,
                   labels(connected)[0] AS connected_type,
                   properties(r) AS relation_props
        """, {"name": player_name})

    def get_transfer_rumors(self) -> list:
        """Retourne toutes les rumeurs de transfert avec leur fiabilité."""
        return self._run_query("""
            MATCH (from:CLUB)-[r:RUMEUR]->(to:CLUB)
            RETURN from.name AS de_club, to.name AS vers_club,
                   r.joueur AS joueur, r.fiabilite AS fiabilite,
                   r.source AS source
            ORDER BY r.fiabilite DESC
        """)

    def get_top_pagerank(self, limit: int = 10) -> list:
        """Top nœuds par PageRank."""
        return self._run_query("""
            MATCH (n)
            WHERE n.pagerank IS NOT NULL
            RETURN n.name AS noeud, labels(n)[0] AS type,
                   n.pagerank AS pagerank
            ORDER BY n.pagerank DESC
            LIMIT $limit
        """, {"limit": limit})

    def get_top_hubs(self, limit: int = 10) -> list:
        """Top hubs (médias influents)."""
        return self._run_query("""
            MATCH (n)
            WHERE n.hub_score IS NOT NULL
            RETURN n.name AS noeud, labels(n)[0] AS type,
                   n.hub_score AS hub_score
            ORDER BY n.hub_score DESC
            LIMIT $limit
        """, {"limit": limit})

    def get_club_connections(self, club_name: str) -> list:
        """Toutes les connexions d'un club (joueurs, transferts, médias)."""
        return self._run_query("""
            MATCH (c:CLUB {name: $name})-[r]-(connected)
            RETURN type(r) AS relation, connected.name AS entity,
                   labels(connected)[0] AS entity_type
            ORDER BY entity_type
        """, {"name": club_name})

    def get_shortest_path(self, from_name: str, to_name: str) -> list:
        """Plus court chemin entre deux entités."""
        return self._run_query("""
            MATCH path = shortestPath(
                (a {name: $from})-[*]-(b {name: $to})
            )
            RETURN [n IN nodes(path) | n.name] AS path_nodes,
                   [r IN relationships(path) | type(r)] AS path_relations,
                   length(path) AS distance
        """, {"from": from_name, "to": to_name})

    def get_community_members(self) -> list:
        """Récupère les communautés depuis les données importées."""
        filepath = os.path.join(DATA_DIR, "axe2_communautes.json")
        if not os.path.exists(filepath):
            return []
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f).get("communautes", {})

# FONCTION PRINCIPALE


def setup_neo4j() -> Neo4JManager | None:
    """
    Configure Neo4J : connexion, import du graphe, scores.
    """
    logger.info("=" * 60)
    logger.info("CONFIGURATION NEO4J")
    logger.info("=" * 60)

    manager = Neo4JManager()

    if not manager.connect():
        return None

    # Nettoyage et contraintes
    manager.clear_database()
    manager.create_constraints()

    # Import du graphe
    stats = manager.import_graph_from_json()

    # Import des scores
    manager.import_hits_scores()
    manager.import_centrality_scores()

    # Résumé
    graph_stats = manager.get_graph_stats()
    logger.info(f"\n RÉSUMÉ NEO4J:")
    logger.info(f"  → Nœuds: {graph_stats['total_nodes']}")
    logger.info(f"  → Arêtes: {graph_stats['total_edges']}")
    for t in graph_stats["node_types"]:
        logger.info(f"  → {t['type']}: {t['count']}")

    return manager


if __name__ == "__main__":
    manager = setup_neo4j()
    if manager:
        print("\n Rumeurs de transfert:")
        for r in manager.get_transfer_rumors():
            print(f"  {r['joueur']}: {r['de_club']} → {r['vers_club']} "
                  f"(fiabilité={r['fiabilite']}, source={r['source']})")

        print("\n Top PageRank:")
        for n in manager.get_top_pagerank(5):
            print(f"  {n['noeud']} ({n['type']}): {n['pagerank']}")

        manager.close()