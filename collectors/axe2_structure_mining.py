"""
Scout-AI : Axe 2 - Web Structure Mining
=========================================
Graph Mining & Analyse de l'Influence du Club

Ce module implémente :
1. Construction de graphes relationnels (joueurs, clubs, agents, médias)
2. Algorithme HITS (Hubs & Authorities) pour le réseau de rumeurs
3. Métriques de centralité (degree, betweenness, PageRank)
4. Détection de communautés (Louvain)
5. Export vers Neo4J pour stockage et requêtage

Bibliothèque principale : NetworkX + python-louvain
Base de données : Neo4J (graphe orienté)
"""

import os
import sys
import json
import random
from datetime import datetime
from collections import defaultdict

import networkx as nx
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import (
    TARGET_PLAYERS, TARGET_LEAGUES,
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
)
from utils.helpers import (
    safe_request, clean_text, export_to_json, export_to_csv,
    setup_logger, CollectionStats
)

logger = setup_logger("axe2-structure-mining")

# 1. CONSTRUCTION DU GRAPHE RELATIONNEL


class FootballGraphBuilder:
    """
    Construit un graphe relationnel de l'écosystème football.

    Noeuds : Joueurs, Clubs, Agents, Ligues, Médias
    Arêtes : JOUE_POUR, TRANSFERE_DE, REPRESENTE_PAR, COMPETE_DANS,
             RUMEUR_VERS, MENTIONNE, COUVRE
    """

    def __init__(self):
        self.graph = nx.DiGraph()  # Graphe orienté
        self.stats = CollectionStats("Graph Builder")

    def build_from_data(self, players: list[dict] = None,
                        articles: list[dict] = None,
                        transfers: list[dict] = None) -> nx.DiGraph:
        """
        Construit le graphe à partir des données collectées (Axe 1).
        Si pas de données, utilise les données de référence du projet.
        """
        logger.info("Construction du graphe relationnel...")

        # --- Ajout des joueurs et clubs ---
        self._add_players_and_clubs(players)

        # --- Ajout des relations de transferts ---
        self._add_transfer_edges(transfers)

        # --- Ajout des liens médiatiques (depuis articles) ---
        self._add_media_links(articles)

        # --- Ajout des relations inter-clubs (même ligue) ---
        self._add_league_connections()

        # --- Ajout d'agents (données simulées enrichissantes) ---
        self._add_agent_network()

        logger.info(
            f"Graphe construit : {self.graph.number_of_nodes()} nœuds, "
            f"{self.graph.number_of_edges()} arêtes"
        )
        return self.graph

    def _add_players_and_clubs(self, players: list[dict] = None):
        """Ajoute les nœuds joueurs et clubs avec leurs attributs."""
        if players is None:
            players = TARGET_PLAYERS

        clubs_added = set()

        for player in players:
            # Nœud joueur
            self.graph.add_node(
                player["name"],
                type="JOUEUR",
                team=player.get("team", ""),
                id_tm=player.get("id_tm", ""),
                valeur_marchande=player.get("valeur_marchande", "N/A"),
            )

            # Nœud club
            team = player.get("team", "")
            if team and team not in clubs_added:
                self.graph.add_node(
                    team,
                    type="CLUB",
                    ligue=self._guess_league(team),
                )
                clubs_added.add(team)

            # Arête JOUE_POUR
            if team:
                self.graph.add_edge(
                    player["name"], team,
                    relation="JOUE_POUR",
                    poids=1.0,
                    depuis=player.get("date_debut", "2024"),
                )

    def _guess_league(self, club: str) -> str:
        """Détermine la ligue d'un club (mapping simplifié)."""
        mapping = {
            "Real Madrid": "La Liga", "FC Barcelona": "La Liga",
            "Atletico Madrid": "La Liga",
            "Manchester City": "Premier League", "Liverpool": "Premier League",
            "Arsenal": "Premier League", "Chelsea": "Premier League",
            "Bayern Munich": "Bundesliga", "Bayer Leverkusen": "Bundesliga",
            "Borussia Dortmund": "Bundesliga",
            "PSG": "Ligue 1", "Marseille": "Ligue 1",
            "Juventus": "Serie A", "AC Milan": "Serie A", "Inter Milan": "Serie A",
            "Galatasaray": "Süper Lig",
        }
        return mapping.get(club, "Autre")

    def _add_transfer_edges(self, transfers: list[dict] = None):
        """Ajoute les arêtes de transferts (historiques et rumeurs)."""
        # Transferts confirmés (depuis données Axe 1 ou simulés)
        confirmed_transfers = transfers or [
            {"joueur": "Kylian Mbappé", "de": "PSG", "vers": "Real Madrid",
             "montant": "Libre", "date": "2024", "type": "confirme"},
            {"joueur": "Victor Osimhen", "de": "Napoli", "vers": "Galatasaray",
             "montant": "Prêt", "date": "2024", "type": "confirme"},
        ]

        # Rumeurs de transferts (enrichissement du graphe)
        rumeurs = [
            {"joueur": "Mohamed Salah", "de": "Liverpool", "vers": "Al-Ittihad",
             "source": "Sky Sports", "fiabilite": 0.6, "type": "rumeur"},
            {"joueur": "Florian Wirtz", "de": "Bayer Leverkusen", "vers": "Real Madrid",
             "source": "BILD", "fiabilite": 0.7, "type": "rumeur"},
            {"joueur": "Florian Wirtz", "de": "Bayer Leverkusen", "vers": "Bayern Munich",
             "source": "Kicker", "fiabilite": 0.5, "type": "rumeur"},
            {"joueur": "Bukayo Saka", "de": "Arsenal", "vers": "Manchester City",
             "source": "The Athletic", "fiabilite": 0.3, "type": "rumeur"},
            {"joueur": "Jamal Musiala", "de": "Bayern Munich", "vers": "Manchester City",
             "source": "Sport1", "fiabilite": 0.4, "type": "rumeur"},
            {"joueur": "Lamine Yamal", "de": "FC Barcelona", "vers": "PSG",
             "source": "L'Équipe", "fiabilite": 0.2, "type": "rumeur"},
        ]

        for t in confirmed_transfers + rumeurs:
            # Assurer que les nœuds existent
            for club in [t.get("de", ""), t.get("vers", "")]:
                if club and not self.graph.has_node(club):
                    self.graph.add_node(club, type="CLUB")

            if t.get("vers"):
                self.graph.add_edge(
                    t.get("de", "Club Inconnu"),
                    t.get("vers", "Club Inconnu"),
                    relation="TRANSFERT" if t["type"] == "confirme" else "RUMEUR",
                    joueur=t["joueur"],
                    montant=t.get("montant", ""),
                    fiabilite=t.get("fiabilite", 1.0),
                    source=t.get("source", ""),
                    poids=t.get("fiabilite", 1.0),
                )

            # Lien source médiatique → rumeur
            source = t.get("source")
            if source:
                if not self.graph.has_node(source):
                    self.graph.add_node(source, type="MEDIA")
                self.graph.add_edge(
                    source,
                    t.get("vers", ""),
                    relation="COUVRE_RUMEUR",
                    joueur=t["joueur"],
                    fiabilite=t.get("fiabilite", 0.5),
                    poids=t.get("fiabilite", 0.5),
                )

    def _add_media_links(self, articles: list[dict] = None):
        """Ajoute les liens entre médias et entités mentionnées."""
        if not articles:
            # Sources médiatiques de référence
            medias = ["L'Équipe", "Marca", "Sky Sports", "BILD",
                      "Gazzetta dello Sport", "The Athletic", "AS",
                      "RMC Sport", "Kicker", "BBC Sport"]
            for media in medias:
                if not self.graph.has_node(media):
                    self.graph.add_node(media, type="MEDIA")
                # Liens aléatoires vers des clubs (simulation)
                for player in TARGET_PLAYERS[:5]:
                    if random.random() > 0.5:
                        self.graph.add_edge(
                            media, player["name"],
                            relation="MENTIONNE",
                            poids=random.uniform(0.3, 1.0),
                        )
            return

        # À partir des articles réels
        for article in articles:
            source = article.get("source", "")
            if not source:
                continue
            if not self.graph.has_node(source):
                self.graph.add_node(source, type="MEDIA")

            # Liens vers joueurs mentionnés
            for joueur in article.get("joueurs_mentions", []):
                if self.graph.has_node(joueur):
                    if self.graph.has_edge(source, joueur):
                        self.graph[source][joueur]["poids"] += 0.1
                    else:
                        self.graph.add_edge(
                            source, joueur,
                            relation="MENTIONNE",
                            poids=0.5,
                        )

            # Liens vers clubs mentionnés
            for club in article.get("clubs_mentions", []):
                if self.graph.has_node(club):
                    if self.graph.has_edge(source, club):
                        self.graph[source][club]["poids"] += 0.1
                    else:
                        self.graph.add_edge(
                            source, club,
                            relation="COUVRE",
                            poids=0.5,
                        )

    def _add_league_connections(self):
        """Ajoute les nœuds ligues et les relations COMPETE_DANS."""
        for league in TARGET_LEAGUES:
            league_name = league["name"]
            if not self.graph.has_node(league_name):
                self.graph.add_node(league_name, type="LIGUE", code=league["code"])

            # Connecter les clubs à leur ligue
            for node, data in list(self.graph.nodes(data=True)):
                if data.get("type") == "CLUB" and data.get("ligue") == league_name:
                    self.graph.add_edge(
                        node, league_name,
                        relation="COMPETE_DANS",
                        poids=1.0,
                    )

    def _add_agent_network(self):
        """Ajoute le réseau d'agents (données simulées mais réalistes)."""
        agents = [
            {"nom": "Jorge Mendes (Gestifute)", "joueurs": ["Cristiano Ronaldo"]},
            {"nom": "Mino Raiola Legacy", "joueurs": ["Erling Haaland"]},
            {"nom": "Badou Sambague", "joueurs": ["Kylian Mbappé"]},
            {"nom": "CAA Stellar", "joueurs": ["Jude Bellingham", "Bukayo Saka"]},
            {"nom": "Rogon Sportmanagement", "joueurs": ["Jamal Musiala", "Florian Wirtz"]},
        ]

        for agent in agents:
            agent_name = agent["nom"]
            self.graph.add_node(agent_name, type="AGENT")

            for joueur in agent["joueurs"]:
                if self.graph.has_node(joueur):
                    self.graph.add_edge(
                        agent_name, joueur,
                        relation="REPRESENTE",
                        poids=1.0,
                    )

    def export_graph_data(self) -> dict:
        """Exporte le graphe en format sérialisable (JSON)."""
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({"id": node, **data})

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({"source": u, "target": v, **data})

        return {"nodes": nodes, "edges": edges}

# 2. ALGORITHME HITS (Hubs & Authorities)


class HITSAnalyzer:
    """
    Application de l'algorithme HITS au réseau de transferts/rumeurs.

    - Hubs : Sources de rumeurs (médias qui propagent l'information)
    - Authorities : Confirmations (clubs/joueurs qui sont les sujets validés)

    L'algorithme identifie les sources fiables et les cibles les plus "chaudes"
    du mercato.
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.hubs = {}
        self.authorities = {}

    def compute_hits(self, max_iter: int = 100, tol: float = 1e-8) -> dict:
        """
        Calcule les scores HITS (Hubs et Authorities).

        Le score Hub élevé = source médiatique influente (propageant des rumeurs confirmées)
        Le score Authority élevé = entité fréquemment citée et confirmée
        """
        try:
            hubs, authorities = nx.hits(
                self.graph,
                max_iter=max_iter,
                tol=tol,
                normalized=True
            )
            self.hubs = hubs
            self.authorities = authorities

            logger.info("Scores HITS calculés avec succès")
            return {"hubs": hubs, "authorities": authorities}

        except nx.PowerIterationFailedConvergence:
            logger.warning("HITS n'a pas convergé, augmentation des itérations...")
            hubs, authorities = nx.hits(self.graph, max_iter=500)
            self.hubs = hubs
            self.authorities = authorities
            return {"hubs": hubs, "authorities": authorities}

    def get_top_hubs(self, n: int = 10, node_type: str = None) -> list[dict]:
        """Retourne les top N hubs (sources de rumeurs les plus influentes)."""
        items = self.hubs.items()
        if node_type:
            items = [(k, v) for k, v in items
                     if self.graph.nodes[k].get("type") == node_type]

        sorted_hubs = sorted(items, key=lambda x: x[1], reverse=True)[:n]
        return [
            {"noeud": k, "score_hub": round(v, 6),
             "type": self.graph.nodes[k].get("type", "UNKNOWN")}
            for k, v in sorted_hubs
        ]

    def get_top_authorities(self, n: int = 10, node_type: str = None) -> list[dict]:
        """Retourne les top N authorities (entités les plus référencées)."""
        items = self.authorities.items()
        if node_type:
            items = [(k, v) for k, v in items
                     if self.graph.nodes[k].get("type") == node_type]

        sorted_auth = sorted(items, key=lambda x: x[1], reverse=True)[:n]
        return [
            {"noeud": k, "score_authority": round(v, 6),
             "type": self.graph.nodes[k].get("type", "UNKNOWN")}
            for k, v in sorted_auth
        ]

# 3. MÉTRIQUES DE CENTRALITÉ


class CentralityAnalyzer:
    """
    Calcul des métriques de centralité pour identifier les nœuds clés.

    Métriques :
    - Degree Centrality : nombre de connexions directes
    - Betweenness Centrality : rôle de pont entre communautés
    - Closeness Centrality : proximité moyenne aux autres nœuds
    - PageRank : importance basée sur la qualité des connexions entrantes
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.metrics = {}

    def compute_all_metrics(self) -> dict:
        """Calcule toutes les métriques de centralité."""
        logger.info("Calcul des métriques de centralité...")

        # Degree Centrality (in + out pour graphe orienté)
        in_degree = nx.in_degree_centrality(self.graph)
        out_degree = nx.out_degree_centrality(self.graph)

        # Betweenness Centrality
        betweenness = nx.betweenness_centrality(self.graph, weight="poids")

        # Closeness Centrality
        closeness = nx.closeness_centrality(self.graph)

        # PageRank
        try:
            pagerank = nx.pagerank(self.graph, weight="poids", alpha=0.85)
        except nx.PowerIterationFailedConvergence:
            pagerank = nx.pagerank(self.graph, weight="poids", alpha=0.85, max_iter=500)

        # Compilation des résultats par nœud
        results = []
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            results.append({
                "noeud": node,
                "type": node_data.get("type", "UNKNOWN"),
                "in_degree": round(in_degree.get(node, 0), 6),
                "out_degree": round(out_degree.get(node, 0), 6),
                "betweenness": round(betweenness.get(node, 0), 6),
                "closeness": round(closeness.get(node, 0), 6),
                "pagerank": round(pagerank.get(node, 0), 6),
            })

        # Tri par PageRank décroissant
        results.sort(key=lambda x: x["pagerank"], reverse=True)
        self.metrics = results

        logger.info(f"Métriques calculées pour {len(results)} nœuds")
        return results

    def get_top_influential(self, n: int = 10, metric: str = "pagerank") -> list[dict]:
        """Retourne les N nœuds les plus influents selon une métrique."""
        if not self.metrics:
            self.compute_all_metrics()
        return sorted(self.metrics, key=lambda x: x.get(metric, 0), reverse=True)[:n]

# 4. DÉTECTION DE COMMUNAUTÉS (Louvain)


def detect_communities(graph: nx.DiGraph) -> dict:
    """
    Détection de communautés via l'algorithme de Louvain.

    Convertit le graphe orienté en non-orienté pour l'algorithme,
    puis identifie les groupes de nœuds fortement connectés.
    """
    try:
        import community as community_louvain

        # Louvain nécessite un graphe non-orienté
        undirected = graph.to_undirected()

        # Calcul des communautés
        partition = community_louvain.best_partition(undirected, weight="poids")

        # Regroupement par communauté
        communities = defaultdict(list)
        for node, comm_id in partition.items():
            node_type = graph.nodes[node].get("type", "UNKNOWN")
            communities[comm_id].append({"noeud": node, "type": node_type})

        # Modularité
        modularity = community_louvain.modularity(partition, undirected)

        logger.info(
            f"Détecté {len(communities)} communautés (modularité={modularity:.4f})"
        )

        return {
            "nb_communautes": len(communities),
            "modularite": round(modularity, 4),
            "communautes": dict(communities),
            "partition": partition,
        }

    except ImportError:
        logger.warning("python-louvain non installé. Utilisation de composantes connexes.")
        undirected = graph.to_undirected()
        components = list(nx.connected_components(undirected))
        communities = {}
        for i, comp in enumerate(components):
            communities[i] = [
                {"noeud": n, "type": graph.nodes[n].get("type", "UNKNOWN")}
                for n in comp
            ]
        return {
            "nb_communautes": len(communities),
            "modularite": None,
            "communautes": communities,
        }

# 5. EXPORT VERS NEO4J


class Neo4JExporter:
    """
    Exporte le graphe NetworkX vers Neo4J pour stockage persistant
    et requêtage avancé (Cypher).
    """

    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None

    def connect(self):
        """Établit la connexion avec Neo4J."""
        try:
            from neo4j import GraphDatabase
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info(f"Connecté à Neo4J: {self.uri}")
        except ImportError:
            logger.warning("Driver Neo4J non installé (pip install neo4j)")
        except Exception as e:
            logger.error(f"Erreur de connexion Neo4J: {e}")

    def export_graph(self, graph: nx.DiGraph):
        """Exporte le graphe complet vers Neo4J."""
        if not self.driver:
            logger.warning("Non connecté à Neo4J. Export en Cypher textuel.")
            return self._generate_cypher_script(graph)

        with self.driver.session() as session:
            # Nettoyage
            session.run("MATCH (n) DETACH DELETE n")

            # Création des nœuds
            for node, data in graph.nodes(data=True):
                node_type = data.get("type", "Entity")
                props = {k: v for k, v in data.items() if k != "type"}
                props["name"] = node
                query = f"CREATE (n:{node_type} $props)"
                session.run(query, props=props)

            # Création des arêtes
            for u, v, data in graph.edges(data=True):
                rel_type = data.get("relation", "CONNECTED_TO")
                props = {k: v for k, v in data.items() if k != "relation"}
                query = (
                    f"MATCH (a {{name: $from_node}}), (b {{name: $to_node}}) "
                    f"CREATE (a)-[r:{rel_type} $props]->(b)"
                )
                session.run(query, from_node=u, to_node=v, props=props)

            logger.info("Graphe exporté vers Neo4J avec succès")

    def _generate_cypher_script(self, graph: nx.DiGraph) -> str:
        """Génère un script Cypher pour import manuel dans Neo4J."""
        lines = ["// === Scout-AI : Import Cypher ===\n"]
        lines.append("// Nettoyage\nMATCH (n) DETACH DELETE n;\n")

        # Nœuds
        lines.append("// --- NOEUDS ---")
        for node, data in graph.nodes(data=True):
            node_type = data.get("type", "Entity")
            props = {k: str(v) for k, v in data.items() if k != "type"}
            props["name"] = node
            props_str = ", ".join(f'{k}: "{v}"' for k, v in props.items())
            lines.append(f"CREATE (:{node_type} {{{props_str}}});")

        # Arêtes
        lines.append("\n// --- RELATIONS ---")
        for u, v, data in graph.edges(data=True):
            rel_type = data.get("relation", "CONNECTED_TO")
            props = {k: str(v) for k, v in data.items() if k != "relation"}
            props_str = ", ".join(f'{k}: "{v}"' for k, v in props.items())
            lines.append(
                f'MATCH (a {{name: "{u}"}}), (b {{name: "{v}"}}) '
                f'CREATE (a)-[:{rel_type} {{{props_str}}}]->(b);'
            )

        script = "\n".join(lines)
        return script

    def close(self):
        if self.driver:
            self.driver.close()

# 6. FONCTION PRINCIPALE AXE 2


def run_axe2(articles: list[dict] = None, profiles: list[dict] = None,
             export_neo4j: bool = False) -> dict:
    """
    Exécute la pipeline complète de l'Axe 2 - Web Structure Mining.

    Args:
        articles: Articles enrichis depuis l'Axe 1 (optionnel)
        profiles: Profils joueurs depuis l'Axe 1 (optionnel)
        export_neo4j: Tenter l'export vers Neo4J

    Returns:
        Dictionnaire avec graphe, HITS, centralité, communautés
    """
    logger.info("=" * 60)
    logger.info("AXE 2 - WEB STRUCTURE MINING")
    logger.info("=" * 60)

    results = {}

    # --- 1. Construction du graphe ---
    logger.info("\n Construction du graphe relationnel...")
    builder = FootballGraphBuilder()
    graph = builder.build_from_data(
        players=TARGET_PLAYERS,
        articles=articles,
    )
    graph_data = builder.export_graph_data()
    export_to_json(graph_data, "axe2_graphe.json")
    results["graphe"] = graph_data

    # --- 2. Algorithme HITS ---
    logger.info("\n Calcul HITS (Hubs & Authorities)...")
    hits = HITSAnalyzer(graph)
    hits_scores = hits.compute_hits()

    top_hubs = hits.get_top_hubs(10)
    top_auth = hits.get_top_authorities(10)
    top_hubs_media = hits.get_top_hubs(5, node_type="MEDIA")

    export_to_json({"top_hubs": top_hubs, "top_authorities": top_auth,
                     "top_hubs_media": top_hubs_media}, "axe2_hits.json")
    results["hits"] = {"top_hubs": top_hubs, "top_authorities": top_auth}

    logger.info("\n TOP HUBS (Sources de rumeurs les plus influentes):")
    for h in top_hubs[:5]:
        logger.info(f"  → {h['noeud']} ({h['type']}): {h['score_hub']}")

    logger.info("\n TOP AUTHORITIES (Entités les plus référencées):")
    for a in top_auth[:5]:
        logger.info(f"  → {a['noeud']} ({a['type']}): {a['score_authority']}")

    # --- 3. Métriques de centralité ---
    logger.info("\n Calcul des métriques de centralité...")
    centrality = CentralityAnalyzer(graph)
    metrics = centrality.compute_all_metrics()
    export_to_json(metrics, "axe2_centralite.json")
    export_to_csv(metrics, "axe2_centralite.csv")
    results["centralite"] = metrics

    logger.info("\n TOP NŒUDS PAR PAGERANK:")
    for m in metrics[:5]:
        logger.info(f"  → {m['noeud']} ({m['type']}): PageRank={m['pagerank']}")

    # --- 4. Détection de communautés ---
    logger.info("\n Détection de communautés (Louvain)...")
    communities = detect_communities(graph)
    export_to_json(communities, "axe2_communautes.json")
    results["communautes"] = communities

    for comm_id, members in list(communities.get("communautes", {}).items())[:5]:
        member_names = [m["noeud"] for m in members[:4]]
        logger.info(f"  Communauté {comm_id}: {', '.join(member_names)}...")

    # --- 5. Export Neo4J ---
    if export_neo4j:
        logger.info("\n Export vers Neo4J...")
        neo4j = Neo4JExporter()
        neo4j.connect()
        neo4j.export_graph(graph)
        neo4j.close()
    else:
        # Génère le script Cypher pour import manuel
        neo4j = Neo4JExporter()
        cypher = neo4j._generate_cypher_script(graph)
        cypher_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "axe2_import_neo4j.cypher"
        )
        os.makedirs(os.path.dirname(cypher_path), exist_ok=True)
        with open(cypher_path, "w") as f:
            f.write(cypher)
        logger.info(f"Script Cypher généré : {cypher_path}")

    logger.info("\n Axe 2 terminé avec succès!")
    return results


if __name__ == "__main__":
    run_axe2()