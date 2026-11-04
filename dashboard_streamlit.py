"""
Scout-AI : Dashboard Interactif (Streamlit)
=============================================
Tableau de bord pour visualiser les résultats des 3 axes de Web Mining.

6 onglets :
1. Vue d'ensemble       → KPIs + résumé du projet
2. Cotes de Popularité  → Classement NLP des joueurs (Axe 1)
3. Graphe Relationnel   → Réseau interactif joueurs-clubs-médias (Axe 2)
4. HITS & Centralité    → Hubs, Authorities, PageRank (Axe 2)
5. Segmentation         → Clusters K-Means + Fan Value (Axe 3)
6. Patterns             → Règles d'association Apriori (Axe 3)

Usage:
    streamlit run dashboard_streamlit.py
    → Ouvrir http://localhost:8501 dans le navigateur

"""

import os
import sys
import json
import math
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    DATA_DIR, TARGET_PLAYERS, TARGET_LEAGUES,
    MONGO_URI, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, ELASTIC_HOST
)

# Supprimer les warnings de deprecation (ES body param, etc.)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# CONFIG STREAMLIT


st.set_page_config(
    page_title="Scout-AI Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# STYLE CSS CUSTOM


st.markdown("""
<style>
    /* Fond sombre */
    .stApp {
        background-color: #0a0e17;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1e293b;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border-radius: 8px;
        padding: 8px 16px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b;
        color: #3b82f6;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #111827, #1e293b);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    .kpi-value {
        font-size: 32px;
        font-weight: 700;
        margin: 4px 0;
    }
    .kpi-label {
        font-size: 13px;
        color: #e2e8f0;
        font-weight: 500;
    }
    .kpi-sub {
        font-size: 11px;
        color: #64748b;
    }

    /* Metric cards override */
    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px;
    }

    /* Hide hamburger & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Plotly chart backgrounds */
    .stPlotlyChart {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# CHARGEMENT DES DONNÉES (depuis les Bases de Données NoSQL)

# Priorité : MongoDB / Neo4J / ElasticSearch → fallback fichiers JSON/CSV

def load_from_mongodb():
    """Charge les données depuis MongoDB (stockage principal)."""
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.server_info()  # Test connexion
        db = client["scout_ai"]

        # Vérifier que la base contient des données
        collections = db.list_collection_names()
        if not collections:
            client.close()
            return None, None, "Base vide (0 collections). Lancez: python setup_docker.py --populate"

        data = {
            "articles_bruts": list(db["articles_bruts"].find({}, {"_id": 0})),
            "articles_enrichis": list(db["articles_enrichis"].find({}, {"_id": 0})),
            "classements": list(db["classements"].find({}, {"_id": 0})),
            "buteurs": list(db["buteurs"].find({}, {"_id": 0})),
            "profils": list(db["profils_joueurs"].find({}, {"_id": 0})),
            "cotes_pop": list(db["cotes_popularite"].find({}, {"_id": 0}).sort("cote_popularite", -1)),
            "clusters": list(db["clusters"].find({}, {"_id": 0}).sort("cluster_id", 1)),
            "player_assoc": list(db["player_associations"].find({}, {"_id": 0})),
            "web_logs": list(db["web_logs"].find({}, {"_id": 0}).limit(500)),  # Échantillon
            # Patterns et trends sont des documents uniques
            "patterns": db["patterns"].find_one({}, {"_id": 0}),
            "trends": db["trends"].find_one({}, {"_id": 0}),
            # Agrégations
            "sentiment_distribution": _mongo_sentiment_distribution(db),
            "top_sources": _mongo_top_sources(db),
        }

        # Fan values depuis web_logs
        fan_df = _mongo_fan_values(db)

        client.close()
        return data, fan_df, "MongoDB"

    except Exception as e:
        return None, None, str(e)


def _mongo_sentiment_distribution(db):
    """Agrégation MongoDB : distribution des sentiments."""
    pipeline = [
        {"$group": {
            "_id": "$sentiment.label",
            "count": {"$sum": 1},
            "avg_stars": {"$avg": "$sentiment.stars"}
        }},
        {"$sort": {"count": -1}}
    ]
    results = list(db["articles_enrichis"].aggregate(pipeline))
    return {r["_id"]: {"count": r["count"], "avg_stars": round(r.get("avg_stars", 0) or 0, 2)}
            for r in results if r["_id"]}


def _mongo_top_sources(db, limit=10):
    """Agrégation MongoDB : top sources médiatiques."""
    pipeline = [
        {"$group": {"_id": "$source", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": limit}
    ]
    return list(db["articles_enrichis"].aggregate(pipeline))


def _mongo_fan_values(db):
    """Reconstruit le DataFrame fan_values depuis MongoDB."""
    try:
        import csv
        logs_path = os.path.join(DATA_DIR, "axe3_fan_values.csv")
        if os.path.exists(logs_path):
            return pd.read_csv(logs_path)
    except Exception:
        pass
    return None


def load_from_neo4j():
    """Charge le graphe relationnel depuis Neo4J."""
    try:
        from neo4j import GraphDatabase

        driver = None
        for auth_try in [None, (NEO4J_USER, NEO4J_PASSWORD), ("neo4j", "neo4j")]:
            try:
                driver = GraphDatabase.driver(NEO4J_URI, auth=auth_try)
                with driver.session() as session:
                    session.run("RETURN 1")
                break
            except Exception:
                if driver:
                    driver.close()
                driver = None
                continue

        if driver is None:
            return None, "Auth échoué"

        with driver.session() as session:
            # Nœuds
            result = session.run("""
                MATCH (n)
                RETURN n.name AS id, labels(n)[0] AS type,
                       n.team AS team, n.ligue AS ligue,
                       n.pagerank AS pagerank, n.betweenness AS betweenness,
                       n.closeness AS closeness, n.in_degree AS in_degree,
                       n.out_degree AS out_degree,
                       n.hub_score AS hub_score, n.authority_score AS authority_score
            """)
            nodes = [dict(record) for record in result]

            # Arêtes
            result = session.run("""
                MATCH (a)-[r]->(b)
                RETURN a.name AS source, b.name AS target, type(r) AS relation,
                       r.joueur AS joueur, r.fiabilite AS fiabilite,
                       r.poids AS poids, coalesce(r.source, '') AS media_source
            """)
            edges = [dict(record) for record in result]

            # Communautés (depuis fichier car Louvain pas natif en Neo4J Community)
            communautes = {}
            comm_path = os.path.join(DATA_DIR, "axe2_communautes.json")
            if os.path.exists(comm_path):
                with open(comm_path, "r", encoding="utf-8") as f:
                    communautes = json.load(f)

        driver.close()

        # Reconstruire les structures
        graphe = {"nodes": nodes, "edges": edges}

        # HITS depuis les propriétés des nœuds
        top_hubs = sorted(
            [{"noeud": n["id"], "score_hub": n["hub_score"], "type": n["type"]}
             for n in nodes if n.get("hub_score")],
            key=lambda x: x["score_hub"], reverse=True
        )[:10]
        top_auth = sorted(
            [{"noeud": n["id"], "score_authority": n["authority_score"], "type": n["type"]}
             for n in nodes if n.get("authority_score")],
            key=lambda x: x["score_authority"], reverse=True
        )[:10]
        hits_data = {"top_hubs": top_hubs, "top_authorities": top_auth}

        # Centralité depuis les propriétés des nœuds
        centralite = [
            {
                "noeud": n["id"], "type": n["type"],
                "pagerank": n.get("pagerank", 0) or 0,
                "betweenness": n.get("betweenness", 0) or 0,
                "closeness": n.get("closeness", 0) or 0,
                "in_degree": n.get("in_degree", 0) or 0,
                "out_degree": n.get("out_degree", 0) or 0,
            }
            for n in nodes if n.get("pagerank") is not None
        ]
        centralite.sort(key=lambda x: x["pagerank"], reverse=True)

        return {
            "graphe": graphe,
            "hits": hits_data,
            "centralite": centralite,
            "communautes": communautes,
        }, "Neo4J"

    except Exception as e:
        return None, str(e)


def load_from_elasticsearch():
    """Charge les données de recherche depuis ElasticSearch."""
    try:
        from elasticsearch import Elasticsearch
        import elasticsearch as es_lib

        es_version = int(es_lib.__version__[0])
        if es_version >= 8:
            es = Elasticsearch(ELASTIC_HOST, verify_certs=False,
                               ssl_show_warn=False, request_timeout=10,
                               headers={"Accept": "application/json"})
        else:
            es = Elasticsearch(ELASTIC_HOST, verify_certs=False, timeout=10)

        if not es.ping():
            return None, "Ping échoué"

        data = {}

        # Sentiment par joueur
        body = {
            "size": 0,
            "aggs": {
                "by_player": {
                    "terms": {"field": "joueur", "size": 20},
                    "aggs": {
                        "avg_stars": {"avg": {"field": "sentiment_stars"}},
                        "sentiment_dist": {"terms": {"field": "sentiment_label"}}
                    }
                }
            }
        }
        try:
            result = es.search(index="scout_sentiment", body=body)
            data["sentiment_by_player"] = result["aggregations"]["by_player"]["buckets"]
        except Exception:
            data["sentiment_by_player"] = []

        # Sentiment par source
        body2 = {
            "size": 0,
            "aggs": {
                "by_source": {
                    "terms": {"field": "source", "size": 20},
                    "aggs": {"avg_stars": {"avg": {"field": "sentiment_stars"}}}
                }
            }
        }
        try:
            result = es.search(index="scout_sentiment", body=body2)
            data["sentiment_by_source"] = result["aggregations"]["by_source"]["buckets"]
        except Exception:
            data["sentiment_by_source"] = []

        # Stats des index
        data["index_stats"] = {}
        for idx in ["scout_articles", "scout_sentiment"]:
            try:
                data["index_stats"][idx] = es.count(index=idx)["count"]
            except Exception:
                data["index_stats"][idx] = 0

        es.close()
        return data, "ElasticSearch"

    except Exception as e:
        return None, str(e)


def load_json_fallback(filename):
    """Fallback : charge depuis fichier JSON local."""
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

# CHARGEMENT EFFECTIF (DB → fallback fichiers)


# --- MongoDB ---
mongo_data, fan_values_df, mongo_status = load_from_mongodb()
if mongo_data is not None:
    articles_bruts = mongo_data["articles_bruts"]
    articles_enrichis = mongo_data["articles_enrichis"]
    classements = mongo_data["classements"]
    buteurs = mongo_data["buteurs"]
    profils = mongo_data["profils"]
    cotes_pop = mongo_data["cotes_pop"]
    clusters = mongo_data["clusters"]
    player_assoc = mongo_data["player_assoc"]
    patterns_data = mongo_data["patterns"] or {"frequent_itemsets": [], "association_rules": []}
    trends = mongo_data["trends"] or {}
    mongo_connected = True
    mongo_error = ""
else:
    # Fallback fichiers JSON
    articles_bruts = load_json_fallback("axe1_articles_bruts.json") or []
    articles_enrichis = load_json_fallback("axe1_articles_enrichis.json") or []
    classements = load_json_fallback("axe1_classements.json") or []
    buteurs = load_json_fallback("axe1_buteurs.json") or []
    profils = load_json_fallback("axe1_profils_transfermarkt.json") or []
    cotes_pop = load_json_fallback("axe1_cotes_popularite.json") or []
    clusters = load_json_fallback("axe3_clusters.json") or []
    player_assoc = load_json_fallback("axe3_player_associations.json") or []
    patterns_data = load_json_fallback("axe3_patterns.json") or {"frequent_itemsets": [], "association_rules": []}
    trends = load_json_fallback("axe3_trends.json") or {}
    fan_values_df = pd.read_csv(os.path.join(DATA_DIR, "axe3_fan_values.csv")) if os.path.exists(os.path.join(DATA_DIR, "axe3_fan_values.csv")) else None
    mongo_connected = False
    mongo_error = str(mongo_status)

# --- Neo4J ---
neo4j_data, neo4j_status = load_from_neo4j()
if neo4j_data is not None:
    graphe = neo4j_data["graphe"]
    hits_data = neo4j_data["hits"]
    centralite = neo4j_data["centralite"]
    communautes = neo4j_data["communautes"]
    neo4j_connected = True
    neo4j_error = ""
else:
    graphe = load_json_fallback("axe2_graphe.json") or {"nodes": [], "edges": []}
    hits_data = load_json_fallback("axe2_hits.json") or {"top_hubs": [], "top_authorities": []}
    centralite = load_json_fallback("axe2_centralite.json") or []
    communautes = load_json_fallback("axe2_communautes.json") or {}
    neo4j_connected = False
    neo4j_error = str(neo4j_status)

# --- ElasticSearch ---
es_data, es_status = load_from_elasticsearch()
es_connected = es_data is not None
es_error = "" if es_connected else str(es_status)

# --- Web logs (toujours depuis CSV, trop volumineux pour Mongo) ---
web_logs_path = os.path.join(DATA_DIR, "axe3_web_logs.csv")
web_logs_df = pd.read_csv(web_logs_path) if os.path.exists(web_logs_path) else None

# COULEURS


COLORS = {
    "primary": "#3b82f6", "secondary": "#8b5cf6", "success": "#10b981",
    "warning": "#f59e0b", "danger": "#ef4444", "accent1": "#06b6d4",
    "accent2": "#ec4899", "bg": "#0a0e17", "card": "#111827",
    "text": "#e2e8f0", "muted": "#94a3b8",
}

NODE_COLORS = {
    "JOUEUR": "#3b82f6", "CLUB": "#10b981", "MEDIA": "#f59e0b",
    "AGENT": "#8b5cf6", "LIGUE": "#ef4444",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0", margin=dict(t=40, b=40, l=50, r=20),
)

# SIDEBAR


with st.sidebar:
    st.markdown("## Scout-AI")
    st.markdown("**Système de Veille & Analyse du Marché des Transferts**")
    st.markdown("---")

    # État des connexions DB
    st.markdown("##### Sources de Données")
    if mongo_connected:
        st.success(f"MongoDB — {len(articles_enrichis)} articles, {len(cotes_pop)} joueurs")
    else:
        st.warning(f"MongoDB — Fallback JSON")
        with st.expander("Détail erreur MongoDB"):
            st.code(mongo_error, language="text")

    if neo4j_connected:
        st.success(f"Neo4J — {len(graphe.get('nodes',[]))} nœuds, {len(graphe.get('edges',[]))} arêtes")
    else:
        st.warning(f"Neo4J — Fallback JSON")
        with st.expander("Détail erreur Neo4J"):
            st.code(neo4j_error, language="text")

    if es_connected:
        idx_stats = es_data.get("index_stats", {})
        total_es = sum(idx_stats.values())
        st.success(f"ElasticSearch — {total_es} documents indexés")
    else:
        st.warning(f"ElasticSearch — Non connecté")
        with st.expander("Détail erreur ES"):
            st.code(es_error, language="text")

    # Bouton de reconnexion
    if not (mongo_connected and neo4j_connected and es_connected):
        st.markdown("---")
        if st.button("Reconnecter aux bases"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Lancez d'abord: `python setup_docker.py`")

    st.markdown("---")

    st.markdown("##### Données Chargées")
    st.markdown(f"- **{len(articles_bruts)}** articles collectés")
    st.markdown(f"- **{len(articles_enrichis)}** articles enrichis")
    st.markdown(f"- **{len(TARGET_PLAYERS)}** joueurs analysés")
    st.markdown(f"- **{len(profils)}** profils scrapés")
    st.markdown(f"- **{len(graphe.get('nodes',[]))}** nœuds dans le graphe")
    st.markdown(f"- **{len(patterns_data.get('association_rules',[]))}** règles Apriori")


# HEADER


st.markdown("""
<div style="text-align: center; padding: 10px 0 20px;">
    <h1 style="color: #e2e8f0; font-size: 36px; margin-bottom: 4px;">
        Scout-AI Dashboard
    </h1>
    <p style="color: #94a3b8; font-size: 14px; margin: 0;">
        Web Content Mining · Web Structure Mining · Web Usage Mining
    </p>
</div>
""", unsafe_allow_html=True)

# TABS


tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Vue d'ensemble",
    "Popularité",
    "Graphe",
    "HITS & Centralité",
    "Segmentation",
    "Prédictions Mercato",
])

# TAB 1 : VUE D'ENSEMBLE


with tab1:
    # KPIs Row 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Articles Collectés", len(articles_bruts), "NewsAPI")
    c2.metric("Articles Enrichis", len(articles_enrichis), "NER + Sentiment")
    c3.metric("Joueurs Analysés", len(TARGET_PLAYERS), "Transfermarkt")
    c4.metric("Profils Scrapés", len(profils), "Valeurs marchandes")

    # KPIs Row 2
    c5, c6, c7, c8 = st.columns(4)
    n_nodes = len(graphe.get("nodes", []))
    n_edges = len(graphe.get("edges", []))
    n_comm = communautes.get("nb_communautes", "?")
    n_rules = len(patterns_data.get("association_rules", []))
    c5.metric("Nœuds du Graphe", n_nodes, "Joueurs+Clubs+Médias")
    c6.metric("Arêtes", n_edges, "Relations")
    c7.metric("Communautés", n_comm, f"Modularité: {communautes.get('modularite', '?')}")
    c8.metric("Règles Apriori", n_rules, "Patterns comportement")

    st.markdown("---")

    col_sent, col_src = st.columns([4, 6])

    with col_sent:
        st.markdown("##### Distribution des Sentiments")
        sent_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
        for a in articles_enrichis:
            label = a.get("sentiment", {}).get("label", "NEUTRAL")
            if label in sent_counts:
                sent_counts[label] += 1

        fig_sent = go.Figure(data=[go.Pie(
            labels=list(sent_counts.keys()),
            values=list(sent_counts.values()),
            marker_colors=["#10b981", "#f59e0b", "#ef4444"],
            hole=0.55, textinfo="label+percent", textfont_size=12,
        )])
        fig_sent.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False)
        st.plotly_chart(fig_sent, width="stretch")

    with col_src:
        st.markdown("##### Top Sources Médiatiques")
        source_counts = {}
        for a in articles_enrichis or articles_bruts:
            src = a.get("source", "Inconnu")
            source_counts[src] = source_counts.get(src, 0) + 1
        top_src = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        fig_src = go.Figure(data=[go.Bar(
            x=[s[1] for s in top_src], y=[s[0] for s in top_src],
            orientation="h", marker_color=COLORS["primary"],
        )])
        fig_src.update_layout(**PLOTLY_LAYOUT, height=320,
                              yaxis=dict(autorange="reversed"), xaxis_title="Articles")
        st.plotly_chart(fig_src, width="stretch")
    # Classement des buteurs par ligue
    if buteurs:
        st.markdown("---")
        st.markdown("##### Meilleurs Buteurs par Ligue")
        league_names = {l["id"]: l["name"] for l in TARGET_LEAGUES}
        league_ids = sorted(set(b.get("competition_id") for b in buteurs))
 
        selected_league = st.selectbox(
            "Choisir une ligue",
            league_ids,
            format_func=lambda x: league_names.get(x, f"Ligue {x}")
        )
 
        df_bt = pd.DataFrame([b for b in buteurs if b.get("competition_id") == selected_league])
        if not df_bt.empty:
            df_bt = df_bt.sort_values("buts", ascending=False).reset_index(drop=True)
            df_bt.index = df_bt.index + 1
            df_bt.index.name = "#"
 
            # Graphique barh des top 10
            top10 = df_bt.head(10)
            fig_bt = go.Figure(data=[go.Bar(
                x=top10["buts"],
                y=top10["joueur"],
                orientation="h",
                marker_color=COLORS["primary"],
                text=top10.apply(lambda r: (f"{0 if pd.isna(r['buts']) else int(r['buts'])} buts "f"({0 if pd.isna(r.get('passes_decisives', 0)) else int(r.get('passes_decisives', 0))} passes)"),axis=1,),
                textposition="outside",
            )])
            fig_bt.update_layout(**PLOTLY_LAYOUT, height=380,
                                  yaxis=dict(autorange="reversed"),
                                  xaxis_title="Buts",
                                  title=dict(text=f"Top 10 Buteurs — {league_names.get(selected_league, '')}", x=0.5, font_size=14))
            st.plotly_chart(fig_bt, width="stretch")
 
            # Tableau complet
            cols_display = ["joueur", "equipe", "buts", "passes_decisives", "penaltys", "matchs_joues", "nationalite", "poste"]
            cols_available = [c for c in cols_display if c in df_bt.columns]
            st.dataframe(df_bt[cols_available].head(20), width="stretch", hide_index=False)

    # --- Recherche ElasticSearch (full-text) ---
    st.markdown("---")
    st.markdown("##### Recherche Full-Text (ElasticSearch)")

    if es_connected:
        search_query = st.text_input("Rechercher dans les articles", placeholder="Ex: Mbappé transfert Real Madrid")
        if search_query:
            try:
                from elasticsearch import Elasticsearch
                import elasticsearch as es_lib
                es_v = int(es_lib.__version__[0])
                if es_v >= 8:
                    es = Elasticsearch("http://localhost:9200", verify_certs=False,
                                       ssl_show_warn=False, request_timeout=10,
                                       headers={"Accept": "application/json"})
                else:
                    es = Elasticsearch("http://localhost:9200", verify_certs=False, timeout=10)

                body = {
                    "query": {"multi_match": {
                        "query": search_query,
                        "fields": ["titre^3", "description^2", "contenu"],
                        "fuzziness": "AUTO"
                    }},
                    "size": 10,
                    "highlight": {"fields": {"titre": {}, "description": {}}}
                }
                result = es.search(index="scout_articles", body=body)
                hits = result["hits"]["hits"]
                es.close()

                st.markdown(f"**{result['hits']['total']['value']}** résultats trouvés")
                for hit in hits:
                    src = hit["_source"]
                    score = hit["_score"]
                    sentiment = src.get("sentiment", {}).get("label", "?")
                    sent_emoji = {"POSITIVE": "", "NEGATIVE": "", "NEUTRAL": ""}.get(sentiment, "")
                    st.markdown(
                        f"{sent_emoji} **{src.get('titre', 'Sans titre')}** "
                        f"— *{src.get('source', '?')}* (score: {score:.1f})"
                    )
            except Exception as e:
                st.error(f"Erreur de recherche : {e}")
    else:
        st.info("ElasticSearch non connecté — lancez `python setup_docker.py` pour activer la recherche full-text")

# TAB 2 : COTES DE POPULARITÉ


with tab2:
    if cotes_pop:
        df_pop = pd.DataFrame(cotes_pop)

        st.markdown("##### Classement des Cotes de Popularité")
        st.markdown("""
        *Formule : CP = 0.4 × Mentions + 0.35 × Sentiment + 0.25 × Diversité Sources*
        """)

        # Bar chart principal
        fig_pop = go.Figure(data=[go.Bar(
            x=df_pop["joueur"], y=df_pop["cote_popularite"],
            marker_color=[px.colors.sample_colorscale("Viridis", [i/len(df_pop)])[0]
                          for i in range(len(df_pop))],
            text=df_pop["cote_popularite"].apply(lambda x: f"{x}/100"),
            textposition="outside",
        )])
        fig_pop.update_layout(**PLOTLY_LAYOUT, height=400,
                              yaxis_title="Score /100", xaxis_tickangle=-30)
        st.plotly_chart(fig_pop, width="stretch")

        col_s, col_d = st.columns(2)

        with col_s:
            st.markdown("##### Sentiment Moyen par Joueur")
            fig_sm = go.Figure(data=[go.Bar(
                x=df_pop["joueur"], y=df_pop["sentiment_moyen"],
                marker_color=df_pop["sentiment_moyen"].apply(
                    lambda x: "#10b981" if x >= 3.5 else ("#f59e0b" if x >= 2.5 else "#ef4444")
                ),
                text=df_pop["sentiment_moyen"].apply(lambda x: f"{x}"),
                textposition="outside",
            )])
            fig_sm.update_layout(**PLOTLY_LAYOUT, height=380,
                                 yaxis_title="Étoiles /5", xaxis_tickangle=-35, yaxis_range=[0, 5])
            st.plotly_chart(fig_sm, width="stretch")

        with col_d:
            st.markdown("##### Mentions vs Diversité des Sources")
            fig_sc = go.Figure(data=[go.Scatter(
                x=df_pop["mentions"], y=df_pop["diversite_sources"],
                mode="markers+text",
                text=df_pop["joueur"].apply(lambda x: x.split()[-1]),
                textposition="top center",
                marker=dict(
                    size=df_pop["cote_popularite"] / 2,
                    color=df_pop["sentiment_moyen"],
                    colorscale="RdYlGn", colorbar=dict(title="Sentiment"),
                    line=dict(width=1, color="white"),
                ),
            )])
            fig_sc.update_layout(**PLOTLY_LAYOUT, height=380,
                                 xaxis_title="Mentions", yaxis_title="Sources Différentes")
            st.plotly_chart(fig_sc, width="stretch")

        # Distribution détaillée
        st.markdown("---")
        st.markdown("##### Détail par Joueur")
        selected_player = st.selectbox("Sélectionner un joueur", df_pop["joueur"].tolist())
        player_data = df_pop[df_pop["joueur"] == selected_player].iloc[0]

        cp1, cp2, cp3, cp4 = st.columns(4)
        cp1.metric("Cote", f"{player_data['cote_popularite']}/100")
        cp2.metric("Mentions", player_data["mentions"])
        cp3.metric("Sentiment", f"{player_data['sentiment_moyen']}/5")
        cp4.metric("Sources", player_data["diversite_sources"])

        if "distribution_sentiment" in player_data and isinstance(player_data["distribution_sentiment"], dict):
            dist = player_data["distribution_sentiment"]
            fig_pd = go.Figure(data=[go.Bar(
                x=list(dist.keys()), y=list(dist.values()),
                marker_color=["#10b981", "#f59e0b", "#ef4444"],
            )])
            fig_pd.update_layout(**PLOTLY_LAYOUT, height=250, xaxis_title="Sentiment", yaxis_title="Articles")
            st.plotly_chart(fig_pd, width="stretch")

    else:
        st.info("Aucune donnée de popularité. Lancez la collecte avec `python main.py` (avec clé NewsAPI).")

# TAB 3 : GRAPHE RELATIONNEL


with tab3:
    st.markdown("##### Graphe Relationnel de l'Écosystème Football")

    nodes = graphe.get("nodes", [])
    edges = graphe.get("edges", [])

    if nodes:
        # Filtres
        col_f1, col_f2 = st.columns(2)
        all_types = sorted(set(n.get("type", "?") for n in nodes))
        with col_f1:
            show_types = st.multiselect("Types de nœuds", all_types, default=all_types)
        all_relations = sorted(set(e.get("relation", "?") for e in edges))
        with col_f2:
            show_relations = st.multiselect("Relations", all_relations, default=all_relations)

        # Filtrer
        filtered_nodes = [n for n in nodes if n.get("type") in show_types]
        filtered_ids = set(n["id"] for n in filtered_nodes)
        filtered_edges = [e for e in edges
                          if e["source"] in filtered_ids and e["target"] in filtered_ids
                          and e.get("relation") in show_relations]

        # Layout force-directed
        np.random.seed(42)
        pos = {}
        for i, node in enumerate(filtered_nodes):
            angle = 2 * math.pi * i / max(len(filtered_nodes), 1)
            r = 3 + np.random.uniform(-1, 1)
            pos[node["id"]] = [r * math.cos(angle), r * math.sin(angle)]

        for _ in range(80):
            for n1 in filtered_nodes:
                fx, fy = 0.0, 0.0
                for n2 in filtered_nodes:
                    if n1["id"] != n2["id"]:
                        dx = pos[n1["id"]][0] - pos[n2["id"]][0]
                        dy = pos[n1["id"]][1] - pos[n2["id"]][1]
                        dist = max(math.sqrt(dx*dx + dy*dy), 0.1)
                        force = 0.5 / (dist * dist)
                        fx += force * dx / dist
                        fy += force * dy / dist
                for e in filtered_edges:
                    if e["source"] == n1["id"] or e["target"] == n1["id"]:
                        other = e["target"] if e["source"] == n1["id"] else e["source"]
                        if other in pos:
                            dx = pos[n1["id"]][0] - pos[other][0]
                            dy = pos[n1["id"]][1] - pos[other][1]
                            fx -= 0.08 * dx
                            fy -= 0.08 * dy
                pos[n1["id"]][0] += fx * 0.04
                pos[n1["id"]][1] += fy * 0.04

        # Edge traces
        edge_x, edge_y = [], []
        for e in filtered_edges:
            if e["source"] in pos and e["target"] in pos:
                x0, y0 = pos[e["source"]]
                x1, y1 = pos[e["target"]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(width=0.6, color="#334155"),
            hoverinfo="none", showlegend=False,
        ))

        for ntype, color in NODE_COLORS.items():
            type_nodes = [n for n in filtered_nodes if n.get("type") == ntype]
            if not type_nodes:
                continue
            fig_g.add_trace(go.Scatter(
                x=[pos[n["id"]][0] for n in type_nodes],
                y=[pos[n["id"]][1] for n in type_nodes],
                mode="markers+text",
                marker=dict(
                    size=18 if ntype in ("JOUEUR", "CLUB") else 12,
                    color=color, line=dict(width=1.5, color="white"),
                ),
                text=[n["id"] for n in type_nodes],
                textposition="top center",
                textfont=dict(size=9, color="#e2e8f0"),
                name=ntype,
                hovertext=[
                    f"<b>{n['id']}</b><br>Type: {n.get('type','?')}<br>"
                    + (f"Équipe: {n.get('team','')}" if n.get('team') else "")
                    for n in type_nodes
                ],
                hoverinfo="text",
            ))

        fig_g.update_layout(
            **PLOTLY_LAYOUT, height=650,
            showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig_g, width="stretch")

        # Stats du graphe
        st.markdown(f"""
        **Statistiques** : {len(filtered_nodes)} nœuds affichés, {len(filtered_edges)} arêtes
        | {' | '.join(f'{t}: {sum(1 for n in filtered_nodes if n.get("type")==t)}' for t in show_types)}
        """)

    else:
        st.info("Aucune donnée de graphe.")

# TAB 4 : HITS & CENTRALITÉ


with tab4:
    st.markdown("##### Algorithme HITS — Hubs vs Authorities")
    st.markdown("""
    - **Hubs** = Sources médiatiques influentes qui propagent les rumeurs
    - **Authorities** = Entités (joueurs/clubs) les plus référencées par les sources fiables
    """)

    col_h, col_a = st.columns(2)

    top_hubs = hits_data.get("top_hubs", [])[:10]
    top_auth = hits_data.get("top_authorities", [])[:10]

    with col_h:
        st.markdown("###### Top Hubs (Sources)")
        if top_hubs:
            fig_h = go.Figure(data=[go.Bar(
                y=[h["noeud"] for h in top_hubs],
                x=[h["score_hub"] for h in top_hubs],
                orientation="h", marker_color=COLORS["primary"],
                text=[f"{h['score_hub']:.4f}" for h in top_hubs],
                textposition="outside",
            )])
            fig_h.update_layout(**PLOTLY_LAYOUT, height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_h, width="stretch")

    with col_a:
        st.markdown("###### Top Authorities (Cibles)")
        if top_auth:
            fig_a = go.Figure(data=[go.Bar(
                y=[a["noeud"] for a in top_auth],
                x=[a["score_authority"] for a in top_auth],
                orientation="h", marker_color=COLORS["success"],
                text=[f"{a['score_authority']:.4f}" for a in top_auth],
                textposition="outside",
            )])
            fig_a.update_layout(**PLOTLY_LAYOUT, height=380, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_a, width="stretch")

    st.markdown("---")

    if centralite:
        df_cent = pd.DataFrame(centralite)

        col_pr, col_bc = st.columns(2)

        with col_pr:
            st.markdown("##### PageRank — Top 15 Nœuds")
            top_pr = df_cent.nlargest(15, "pagerank")
            fig_pr = go.Figure(data=[go.Bar(
                x=top_pr["noeud"], y=top_pr["pagerank"],
                marker_color=top_pr["type"].map(NODE_COLORS).fillna("#666"),
                text=top_pr["type"], textposition="outside",
            )])
            fig_pr.update_layout(**PLOTLY_LAYOUT, height=400,
                                 xaxis_tickangle=-35, yaxis_title="PageRank")
            st.plotly_chart(fig_pr, width="stretch")

        with col_bc:
            st.markdown("##### Betweenness vs Closeness")
            fig_bc = go.Figure()
            for ntype, color in NODE_COLORS.items():
                mask = df_cent["type"] == ntype
                if mask.any():
                    fig_bc.add_trace(go.Scatter(
                        x=df_cent[mask]["betweenness"],
                        y=df_cent[mask]["closeness"],
                        mode="markers+text",
                        marker=dict(
                            size=df_cent[mask]["pagerank"] * 800 + 6,
                            color=color, line=dict(width=1, color="white"),
                        ),
                        text=df_cent[mask]["noeud"].apply(lambda x: x[:15]),
                        textposition="top center", textfont=dict(size=8),
                        name=ntype,
                    ))
            fig_bc.update_layout(**PLOTLY_LAYOUT, height=400,
                                 xaxis_title="Betweenness", yaxis_title="Closeness")
            st.plotly_chart(fig_bc, width="stretch")

       
# TAB 5 : SEGMENTATION & FAN VALUE


with tab5:
    if clusters:
        st.markdown("##### Segmentation d'Audience — K-Means Clustering")

        col_pie, col_radar = st.columns([4, 6])

        with col_pie:
            cluster_colors = [COLORS["primary"], COLORS["success"], COLORS["warning"],
                              COLORS["danger"], COLORS["secondary"]]
            fig_pie = go.Figure(data=[go.Pie(
                labels=[c["segment_label"] for c in clusters],
                values=[c["taille"] for c in clusters],
                marker_colors=cluster_colors[:len(clusters)],
                hole=0.5, textinfo="label+percent",
            )])
            fig_pie.update_layout(**PLOTLY_LAYOUT, height=380, showlegend=False,
                                  title=dict(text="Répartition des Segments", x=0.5, font_size=14))
            st.plotly_chart(fig_pie, width="stretch")

        with col_radar:
            categories = ["stats", "news", "social", "commerce", "profil"]
            fig_radar = go.Figure()
            for i, c in enumerate(clusters):
                dist = c.get("distribution_categories", {})
                values = [dist.get(cat, 0) for cat in categories] + [dist.get(categories[0], 0)]
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=categories + [categories[0]],
                    fill="toself", name=f"{c['segment_label']} ({c['taille']})",
                    opacity=0.65,
                ))
            fig_radar.update_layout(
                **PLOTLY_LAYOUT, height=400,
                title=dict(text="Profil de Navigation par Segment", x=0.5, font_size=14),
                polar=dict(bgcolor="rgba(0,0,0,0)",
                           radialaxis=dict(gridcolor="#334155")),
            )
            st.plotly_chart(fig_radar, width="stretch")

        st.markdown("---")

        col_fv, col_heat = st.columns(2)

        with col_fv:
            st.markdown("##### Distribution Fan Value")
            # Afficher le modèle ML utilisé
            best_models_info = load_json_fallback("ia_best_models.json")
            if best_models_info and "task2" in best_models_info:
                info = best_models_info["task2"]
                r2 = info.get("metrics", {}).get("r2", "?")
                st.caption(f"Modèle utilisé : **{info['best_model']}** (R²={r2})")

            if fan_values_df is not None and "fan_value" in fan_values_df.columns:
                fig_fv = go.Figure()
                valid_cluster_ids = set(c["cluster_id"] for c in clusters) if clusters else set()
                for cid in sorted(fan_values_df["cluster"].unique()):
                    cid_int = int(cid)
                    if valid_cluster_ids and cid_int not in valid_cluster_ids:
                        continue  # Ignorer les clusters qui n'existent plus
                    mask = fan_values_df["cluster"] == cid
                    label = clusters[cid_int]["segment_label"] if cid_int < len(clusters) else f"Cluster {cid}"
                    fig_fv.add_trace(go.Histogram(
                        x=fan_values_df[mask]["fan_value"],
                        name=label, opacity=0.7, nbinsx=20,
                    ))
                fig_fv.update_layout(**PLOTLY_LAYOUT, height=380, barmode="overlay",
                                     xaxis_title="Fan Value (/100)", yaxis_title="Utilisateurs")
                st.plotly_chart(fig_fv, width="stretch")

        with col_heat:
            st.markdown("##### Co-occurrence des Joueurs")
            if player_assoc:
                players_list = sorted(set(
                    [a["joueur_1"] for a in player_assoc] + [a["joueur_2"] for a in player_assoc]
                ))
                matrix = pd.DataFrame(0, index=players_list, columns=players_list)
                for a in player_assoc:
                    matrix.loc[a["joueur_1"], a["joueur_2"]] = a["co_occurrences"]
                    matrix.loc[a["joueur_2"], a["joueur_1"]] = a["co_occurrences"]

                short = [n.split()[-1] for n in players_list]
                fig_heat = go.Figure(data=[go.Heatmap(
                    z=matrix.values, x=short, y=short,
                    colorscale="Blues", showscale=True,
                )])
                fig_heat.update_layout(**PLOTLY_LAYOUT, height=380)
                st.plotly_chart(fig_heat, width="stretch")

       


# TAB 7 : PRÉDICTIONS MERCATO (meilleur modèle IA)


with tab6:
    st.markdown("##### Prédictions du Marché des Transferts")
    st.markdown("Classement prédictif basé sur la **fusion des 3 axes de Web Mining** — "
                "identifie les opportunités sous-évaluées et anticipe les tendances du mercato.")

    # Charger les prédictions
    transfer_preds = load_json_fallback("ia_transfer_predictions.json")
    model_info = load_json_fallback("ia_best_models.json")

    if model_info:
        best_name = model_info.get("best_model", "?")
        metrics = model_info.get("metrics", {})
        st.caption(f"Modèle utilisé : **{best_name}** "
                   f"(R²={metrics.get('r2', '?')}, MAE={metrics.get('mae', '?')}) — "
                   f"Entraîné le {model_info.get('date', '?')[:10] if model_info.get('date') else '?'}")

        # Comparaison des 3 modèles
        all_models = model_info.get("all_models", {})
        if all_models:
            st.markdown("###### Comparaison des 3 algorithmes")
            comp_df = pd.DataFrame(all_models).T
            comp_df.index.name = "Modèle"
            st.dataframe(comp_df.style.format("{:.4f}"), width="stretch")

    st.markdown("---")

    if transfer_preds:
        df_preds = pd.DataFrame(transfer_preds)

        # Classement des 10 joueurs analysés
        st.markdown("###### Classement Prédictif — Potentiel de Transfert")

        fig_ranking = go.Figure(data=[go.Bar(
            x=df_preds["prediction_ml"],
            y=df_preds["joueur"],
            orientation="h",
            marker_color=df_preds["prediction_ml"].apply(
                lambda x: "#ef4444" if x >= 70 else ("#f59e0b" if x >= 50 else ("#3b82f6" if x >= 30 else "#64748b"))
            ),
            text=df_preds["prediction_ml"].apply(lambda x: f"{x:.1f}/100"),
            textposition="outside",
        )])
        fig_ranking.update_layout(**PLOTLY_LAYOUT, height=450, xaxis_title="Transfer Market Score (/100)",
                                   yaxis=dict(autorange="reversed"), xaxis_range=[0, 110])
        st.plotly_chart(fig_ranking, width="stretch")

        # Catégories
        st.markdown("---")
        st.markdown("###### Catégories de Joueurs")
        col_cat1, col_cat2, col_cat3, col_cat4 = st.columns(4)

        hot = df_preds[df_preds["prediction_ml"] >= 70]
        rising = df_preds[(df_preds["prediction_ml"] >= 50) & (df_preds["prediction_ml"] < 70)]
        underval = df_preds[(df_preds["prediction_ml"] >= 30) & (df_preds["prediction_ml"] < 50)]
        stable = df_preds[df_preds["prediction_ml"] < 30]

        with col_cat1:
            st.metric("Très demandés", len(hot))
            for _, r in hot.iterrows():
                st.markdown(f"**{r['joueur']}** ({r['prediction_ml']:.0f})")
        with col_cat2:
            st.metric("En hausse", len(rising))
            for _, r in rising.iterrows():
                st.markdown(f"**{r['joueur']}** ({r['prediction_ml']:.0f})")
        with col_cat3:
            st.metric("Sous-évalués", len(underval))
            for _, r in underval.iterrows():
                st.markdown(f"**{r['joueur']}** ({r['prediction_ml']:.0f})")
        with col_cat4:
            st.metric("Stables", len(stable))
            for _, r in stable.iterrows():
                st.markdown(f"**{r['joueur']}** ({r['prediction_ml']:.0f})")

        # Détail par joueur
        st.markdown("---")
        st.markdown("###### Détail par Joueur")
        selected = st.selectbox("Sélectionner un joueur", df_preds["joueur"].tolist(), key="mercato_player")
        row = df_preds[df_preds["joueur"] == selected].iloc[0]

        mc1, mc2, mc3, mc4 = st.columns(4)
        color = "#ef4444" if row["prediction_ml"] >= 70 else ("#f59e0b" if row["prediction_ml"] >= 50 else "#3b82f6")
        mc1.markdown(f"""
        <div style="text-align:center; padding:16px; background:#111827; border-radius:12px; border:2px solid {color};">
            <h1 style="color:{color}; margin:0;">{row['prediction_ml']:.1f}</h1>
            <p style="color:#e2e8f0;">Transfer Score</p>
        </div>""", unsafe_allow_html=True)
        mc2.metric("Club", row.get("club", "?"))
        mc3.metric("Mentions", row.get("mentions", 0))
        mc4.metric("Rumeurs", row.get("nb_rumeurs", 0))

        # Facteurs du joueur
        st.markdown("")
        factors = {
            "Mentions presse": row.get("mentions", 0),
            "Sentiment": row.get("sentiment", 0),
            "Rumeurs": row.get("nb_rumeurs", 0),
            "PageRank": row.get("pagerank", 0) * 10000,
            "Authority HITS": row.get("authority_score", 0) * 1000,
            "Co-occurrences": row.get("co_occurrences", 0),
            "Vues fans": row.get("nb_vues", 0),
        }
        fig_factors = go.Figure(data=[go.Bar(
            x=list(factors.values()), y=list(factors.keys()),
            orientation="h", marker_color=color,
        )])
        fig_factors.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_title="Valeur")
        st.plotly_chart(fig_factors, width="stretch")

    else:
        st.warning("Aucune prédiction disponible. Lancez : `python pipeline_ia.py`")


# FOOTER


st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 12px; padding: 10px;">
    Scout-AI v1.0 — Système de Veille & Analyse du Marché des Transferts<br>
    Web Content Mining · Web Structure Mining · Web Usage Mining<br>
    
</div>
""", unsafe_allow_html=True)