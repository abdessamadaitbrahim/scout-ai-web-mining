"""
Scout-AI : Axe 3 - Web Usage Mining
=====================================
Analyse des Tendances de Recherche, Mining de Séquences & Predictive Fan Value

Ce module implémente :
1. Collecte des tendances Google Trends (pytrends)
2. Simulation de logs de navigation (web usage data)
3. Frequent Pattern Mining (Apriori) pour les comportements d'achat/consultation
4. Clustering K-Means pour la segmentation des audiences
5. Prédiction de la "Fan Value" (valeur médiatique prédictive)

Algorithmes : Apriori (mlxtend), K-Means (scikit-learn)
"""

import os
import sys
import json
import random
import hashlib
from datetime import datetime, timedelta
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config.settings import TARGET_PLAYERS, TARGET_LEAGUES
from utils.helpers import (
    export_to_json, export_to_csv, anonymize_data,
    setup_logger, CollectionStats
)

logger = setup_logger("axe3-usage-mining")

# 1. COLLECTE GOOGLE TRENDS


class TrendsCollector:
    """
    Collecte des tendances de recherche Google pour les joueurs.
    Utilise pytrends (API non-officielle Google Trends).
    """

    def __init__(self):
        self.pytrends = None
        self.stats = CollectionStats("Google Trends")

    def _init_pytrends(self):
        """Initialise la connexion pytrends."""
        try:
            from pytrends.request import TrendReq
            self.pytrends = TrendReq(hl='fr-FR', tz=60)
            return True
        except ImportError:
            logger.warning("pytrends non installé. Utilisation de données simulées.")
            return False

    def get_player_trends(self, player_names: list[str],
                          timeframe: str = "today 3-m") -> pd.DataFrame:
        """
        Récupère les tendances de recherche pour une liste de joueurs.

        Args:
            player_names: Noms des joueurs (max 5 par requête Google Trends)
            timeframe: Période ("today 3-m", "today 12-m", "2024-01-01 2024-12-31")

        Returns:
            DataFrame avec les tendances temporelles
        """
        if not self._init_pytrends():
            return self._simulate_trends(player_names)

        all_data = pd.DataFrame()

        # Google Trends limite à 5 mots-clés par requête
        chunks = [player_names[i:i+5] for i in range(0, len(player_names), 5)]

        for chunk in chunks:
            try:
                self.pytrends.build_payload(
                    chunk,
                    cat=294,  # Catégorie Sports
                    timeframe=timeframe,
                    geo='',  # Mondial
                )
                data = self.pytrends.interest_over_time()
                if not data.empty:
                    data = data.drop(columns=["isPartial"], errors="ignore")
                    all_data = pd.concat([all_data, data], axis=1)
                self.stats.record_success(len(chunk))
            except Exception as e:
                logger.error(f"Erreur Google Trends pour {chunk}: {e}")
                self.stats.record_error()
                # Fallback sur données simulées
                simulated = self._simulate_trends(chunk)
                all_data = pd.concat([all_data, simulated], axis=1)

        return all_data

    def get_related_queries(self, player_name: str) -> dict:
        """Récupère les requêtes associées à un joueur."""
        if not self.pytrends:
            return self._simulate_related_queries(player_name)

        try:
            self.pytrends.build_payload([player_name], cat=294, timeframe="today 3-m")
            related = self.pytrends.related_queries()
            result = related.get(player_name, {})
            return {
                "top": result.get("top", pd.DataFrame()).to_dict("records") if isinstance(result.get("top"), pd.DataFrame) else [],
                "rising": result.get("rising", pd.DataFrame()).to_dict("records") if isinstance(result.get("rising"), pd.DataFrame) else [],
            }
        except Exception as e:
            logger.error(f"Erreur requêtes associées pour {player_name}: {e}")
            return self._simulate_related_queries(player_name)

    def _simulate_trends(self, player_names: list[str]) -> pd.DataFrame:
        """Génère des données de tendances simulées (réalistes)."""
        dates = pd.date_range(end=datetime.now(), periods=90, freq='D')
        data = {}
        for name in player_names:
            # Base + saisonnalité + bruit + pics aléatoires
            base = random.randint(30, 70)
            trend = np.linspace(0, random.randint(-10, 15), 90)
            noise = np.random.normal(0, 5, 90)
            seasonality = 10 * np.sin(np.linspace(0, 4 * np.pi, 90))

            # Pics aléatoires (matchs importants, rumeurs)
            spikes = np.zeros(90)
            for _ in range(random.randint(2, 5)):
                spike_day = random.randint(0, 89)
                spikes[spike_day] = random.randint(20, 50)

            values = base + trend + noise + seasonality + spikes
            values = np.clip(values, 0, 100).astype(int)
            data[name] = values

        return pd.DataFrame(data, index=dates)

    def _simulate_related_queries(self, player_name: str) -> dict:
        """Simule des requêtes associées."""
        team = next((p["team"] for p in TARGET_PLAYERS if p["name"] == player_name), "club")
        return {
            "top": [
                {"query": f"{player_name} transfert", "value": 100},
                {"query": f"{player_name} salaire", "value": 75},
                {"query": f"{player_name} stats", "value": 60},
                {"query": f"{player_name} {team}", "value": 55},
                {"query": f"{player_name} blessure", "value": 40},
            ],
            "rising": [
                {"query": f"{player_name} mercato 2025", "value": 300},
                {"query": f"{player_name} nouveau club", "value": 200},
            ]
        }

    def collect_all_trends(self, players: list[dict] = None) -> dict:
        """Collecte complète des tendances pour tous les joueurs."""
        if players is None:
            players = TARGET_PLAYERS

        names = [p["name"] for p in players]

        logger.info(f"Collecte des tendances pour {len(names)} joueurs...")
        trends_df = self.get_player_trends(names)

        # Requêtes associées pour chaque joueur
        related = {}
        for name in names:
            related[name] = self.get_related_queries(name)

        self.stats.print_summary()

        # Convertir l'index Timestamp en string pour la sérialisation JSON
        if not trends_df.empty:
            trends_df.index = trends_df.index.astype(str)
            trends_dict = trends_df.to_dict()
        else:
            trends_dict = {}

        return {
            "trends_timeseries": trends_dict,
            "related_queries": related,
        }

# 2. SIMULATION DE LOGS DE NAVIGATION (Web Usage Data)


class WebUsageSimulator:
    """
    Simule des logs de navigation réalistes pour le Web Usage Mining.

    Types d'utilisateurs simulés :
    - Parieurs (consultent stats, cotes, comparaisons)
    - Fans (consultent news, réseaux sociaux, merchandising)
    - Scouts/Agents (consultent profils détaillés, historiques)
    - Curieux (navigation aléatoire, pics de buzz)

    Ces données simulent ce qu'on obtiendrait de vrais logs Apache/Nginx.
    """

    PAGES = {
        "stats": ["joueur/stats", "joueur/matchs", "joueur/buts", "joueur/passes",
                   "comparaison/joueurs", "classement/ligue"],
        "news": ["actualites/transferts", "actualites/matchs", "actualites/blessures",
                  "rumeurs/mercato", "interviews/joueur"],
        "social": ["reactions/fans", "sondages/meilleur-joueur", "forum/discussion",
                    "hashtag/trending"],
        "commerce": ["maillot/acheter", "billets/match", "abonnement/club",
                      "paris/cotes", "fantasy/equipe"],
        "profil": ["joueur/profil", "joueur/historique-transferts", "joueur/valeur",
                    "agent/contact", "club/effectif"],
    }

    USER_TYPES = {
        # 3 segments distincts : parieurs, fans, curieux
        "parieur": {"stats": 0.50, "commerce": 0.35, "news": 0.10, "profil": 0.05, "social": 0.00},
        "fan":     {"news": 0.20, "social": 0.45, "commerce": 0.20, "stats": 0.05, "profil": 0.10},
        "curieux": {"news": 0.45, "social": 0.15, "stats": 0.15, "commerce": 0.05, "profil": 0.20},
    }

    # Comportement spécifique par type
    USER_BEHAVIOR = {
        "parieur": {"sessions_range": (8, 20), "duree_base": 45,  "duree_std": 15, "pct_mobile": 0.7},
        "fan":     {"sessions_range": (5, 15), "duree_base": 30,  "duree_std": 10, "pct_mobile": 0.85},
        "curieux": {"sessions_range": (1, 5),  "duree_base": 20,  "duree_std": 10, "pct_mobile": 0.5},
    }

    def __init__(self, n_users: int = 500, n_sessions: int = 2000):
        self.n_users = n_users
        self.n_sessions = n_sessions

    def generate_logs(self) -> pd.DataFrame:
        """
        Génère des logs de navigation simulés avec des profils bien distincts.

        Returns:
            DataFrame avec colonnes :
            user_id, session_id, timestamp, page, player_viewed,
            duration_seconds, user_type, device
        """
        logger.info(f"Génération de {self.n_sessions} sessions pour {self.n_users} utilisateurs...")

        # Créer les utilisateurs avec comportement distinct par type
        users = []
        for i in range(self.n_users):
            user_type = random.choices(
                list(self.USER_TYPES.keys()),
                weights=[0.30, 0.40, 0.30]
            )[0]
            behavior = self.USER_BEHAVIOR[user_type]

            # Device biaisé par type (parieurs, fans, curieux)
            if random.random() < behavior["pct_mobile"]:
                device = "mobile"
            else:
                device = random.choice(["desktop", "tablet"])

            # Nombre de sessions biaisé par type
            nb_sessions = random.randint(*behavior["sessions_range"])

            users.append({
                "user_id": hashlib.md5(f"user_{i}".encode()).hexdigest()[:10],
                "user_type": user_type,
                "device": device,
                "favorite_player": random.choice(TARGET_PLAYERS)["name"],
                "nb_sessions_target": nb_sessions,
                "duree_base": behavior["duree_base"],
                "duree_std": behavior["duree_std"],
            })

        # Générer les sessions
        logs = []
        base_date = datetime.now() - timedelta(days=90)

        for _ in range(self.n_sessions):
            user = random.choice(users)
            session_id = hashlib.md5(f"session_{random.randint(0,999999)}".encode()).hexdigest()[:12]
            session_time = base_date + timedelta(
                days=random.randint(0, 90),
                hours=random.randint(8, 23),
                minutes=random.randint(0, 59)
            )

            # Nombre de pages dépend du type
            if user["user_type"] == "parieur":
                n_pages = max(2, int(random.gauss(7, 2)))
            elif user["user_type"] == "fan":
                n_pages = max(1, int(random.gauss(5, 2)))
            else:  # curieux
                n_pages = max(1, int(random.gauss(2, 1)))
            n_pages = min(n_pages, 15)

            probs = self.USER_TYPES[user["user_type"]]

            for page_idx in range(n_pages):
                category = random.choices(
                    list(probs.keys()),
                    weights=list(probs.values())
                )[0]

                page = random.choice(self.PAGES[category])

                # Joueur consulté (biais vers le favori)
                if random.random() < 0.4:
                    player = user["favorite_player"]
                else:
                    player = random.choice(TARGET_PLAYERS)["name"]

                # Durée spécifique au type d'utilisateur
                duration = max(5, int(random.gauss(
                    user["duree_base"], user["duree_std"]
                )))

                logs.append({
                    "user_id": user["user_id"],
                    "user_type": user["user_type"],
                    "session_id": session_id,
                    "timestamp": (session_time + timedelta(seconds=page_idx * duration)).isoformat(),
                    "page_category": category,
                    "page": page,
                    "player_viewed": player,
                    "duration_seconds": duration,
                    "device": user["device"],
                })

        df = pd.DataFrame(logs)
        logger.info(f"Généré {len(df)} entrées de log ({self.n_users} utilisateurs, {self.n_sessions} sessions)")
        return df

# 3. FREQUENT PATTERN MINING (Apriori)


class PatternMiner:
    """
    Mining de patterns fréquents dans les comportements de navigation.

    Utilise l'algorithme Apriori pour identifier les associations :
    - Quelles pages sont consultées ensemble ?
    - Quels parcours mènent à l'achat (maillot, billet, pari) ?
    - Quels joueurs sont comparés ensemble ?
    """

    def __init__(self, min_support: float = 0.05, min_confidence: float = 0.3):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.stats = CollectionStats("Pattern Mining")

    def mine_page_patterns(self, logs_df: pd.DataFrame) -> dict:
        """
        Identifie les patterns fréquents de navigation par session.

        Chaque session est un "panier" de pages consultées.
        Apriori identifie les associations fréquentes.
        """
        logger.info("Mining de patterns fréquents (Apriori)...")

        # Construire les transactions (1 session = 1 transaction)
        sessions = logs_df.groupby("session_id")["page_category"].apply(list).tolist()

        # Encoder en format binaire pour Apriori
        all_categories = list(set(cat for session in sessions for cat in session))
        transactions = []
        for session in sessions:
            transaction = {cat: (cat in session) for cat in all_categories}
            transactions.append(transaction)

        transactions_df = pd.DataFrame(transactions)

        try:
            from mlxtend.frequent_patterns import apriori, association_rules

            # Apriori
            frequent_itemsets = apriori(
                transactions_df,
                min_support=self.min_support,
                use_colnames=True
            )

            if frequent_itemsets.empty:
                logger.warning("Aucun pattern fréquent trouvé. Réduction du support minimal...")
                frequent_itemsets = apriori(
                    transactions_df,
                    min_support=self.min_support / 2,
                    use_colnames=True
                )

            # Règles d'association
            if not frequent_itemsets.empty:
                rules = association_rules(
                    frequent_itemsets,
                    metric="confidence",
                    min_threshold=self.min_confidence,
                    num_itemsets=len(frequent_itemsets)
                )
            else:
                rules = pd.DataFrame()

            self.stats.record_success()

            # Conversion pour JSON
            itemsets_list = []
            for _, row in frequent_itemsets.iterrows():
                itemsets_list.append({
                    "items": list(row["itemsets"]),
                    "support": round(row["support"], 4),
                })

            rules_list = []
            for _, row in rules.iterrows():
                rules_list.append({
                    "antecedent": list(row["antecedents"]),
                    "consequent": list(row["consequents"]),
                    "support": round(row["support"], 4),
                    "confidence": round(row["confidence"], 4),
                    "lift": round(row["lift"], 4),
                })

            # Tri par lift décroissant
            rules_list.sort(key=lambda x: x["lift"], reverse=True)

            logger.info(
                f"Trouvé {len(itemsets_list)} itemsets fréquents, "
                f"{len(rules_list)} règles d'association"
            )

            return {
                "frequent_itemsets": itemsets_list,
                "association_rules": rules_list,
                "params": {
                    "min_support": self.min_support,
                    "min_confidence": self.min_confidence,
                    "nb_sessions": len(sessions),
                }
            }

        except ImportError:
            logger.warning("mlxtend non installé. Calcul manuel des fréquences...")
            return self._fallback_patterns(sessions, all_categories)

    def _fallback_patterns(self, sessions, categories) -> dict:
        """Calcul de fréquences simples sans mlxtend."""
        from itertools import combinations

        n = len(sessions)
        pair_counts = Counter()
        single_counts = Counter()

        for session in sessions:
            unique = set(session)
            for cat in unique:
                single_counts[cat] += 1
            for pair in combinations(sorted(unique), 2):
                pair_counts[pair] += 1

        itemsets = [
            {"items": [k], "support": round(v / n, 4)}
            for k, v in single_counts.most_common()
        ] + [
            {"items": list(k), "support": round(v / n, 4)}
            for k, v in pair_counts.most_common(20)
            if v / n >= self.min_support
        ]

        return {"frequent_itemsets": itemsets, "association_rules": [], "params": {}}

    def mine_player_associations(self, logs_df: pd.DataFrame) -> list[dict]:
        """
        Identifie quels joueurs sont consultés ensemble dans une même session.
        Utile pour les recommandations et la comparaison.
        """
        sessions = logs_df.groupby("session_id")["player_viewed"].apply(set).tolist()

        from itertools import combinations
        pair_counts = Counter()
        n = len(sessions)

        for session in sessions:
            if len(session) >= 2:
                for pair in combinations(sorted(session), 2):
                    pair_counts[pair] += 1

        associations = [
            {
                "joueur_1": pair[0],
                "joueur_2": pair[1],
                "co_occurrences": count,
                "support": round(count / n, 4),
            }
            for pair, count in pair_counts.most_common(20)
        ]

        logger.info(f"Top associations joueurs: {associations[:3]}")
        return associations

# 4. CLUSTERING K-MEANS (Segmentation d'Audience)


class AudienceSegmenter:
    """
    Segmentation des audiences par K-Means.

    Features par utilisateur :
    - Nombre de sessions
    - Durée moyenne par session
    - Distribution des catégories de pages (stats, news, social, commerce, profil)
    - Nombre de joueurs distincts consultés
    - Appareil préféré

    Objectif : identifier les segments parieurs / fans / curieux
    et prédire la "Fan Value".
    """

    def __init__(self, n_clusters: int = 4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.model = None
        self.stats = CollectionStats("K-Means Clustering")

    def prepare_features(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """Prépare la matrice de features pour le clustering."""
        logger.info("Préparation des features utilisateur...")

        user_features = logs_df.groupby("user_id").agg({
            "session_id": "nunique",
            "duration_seconds": ["mean", "sum"],
            "player_viewed": "nunique",
            "page_category": lambda x: x.value_counts().to_dict(),
            "user_type": "first",
            "device": "first",
        })

        # Aplatir les colonnes MultiIndex
        user_features.columns = [
            "nb_sessions", "duree_moyenne", "duree_totale",
            "nb_joueurs_consultes", "category_dist", "user_type_real", "device"
        ]

        # Extraire la distribution des catégories
        categories = ["stats", "news", "social", "commerce", "profil"]
        for cat in categories:
            user_features[f"pct_{cat}"] = user_features["category_dist"].apply(
                lambda d: d.get(cat, 0) / sum(d.values()) if sum(d.values()) > 0 else 0
            )

        # Encoder le device
        user_features["is_mobile"] = (user_features["device"] == "mobile").astype(int)

        # Features finales pour le clustering
        feature_cols = [
            "nb_sessions", "duree_moyenne", "duree_totale",
            "nb_joueurs_consultes", "is_mobile",
        ] + [f"pct_{cat}" for cat in categories]

        features_df = user_features[feature_cols].fillna(0)
        self.feature_cols = feature_cols
        self.user_features = user_features

        return features_df

    def find_optimal_k(self, features_df: pd.DataFrame,
                       k_range: range = range(2, 8)) -> int:
        """
        Détermine le nombre optimal de clusters via le score Silhouette.
        Force un minimum de K=3 pour capturer parieurs/fans/curieux.
        """
        logger.info("Recherche du K optimal (méthode Silhouette)...")

        X = self.scaler.fit_transform(features_df)
        scores = {}

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            score = silhouette_score(X, labels)
            scores[k] = round(score, 4)
            logger.info(f"  K={k}: Silhouette={score:.4f}")

        # Prendre K=3 comme cible (3 profils : parieur, fan, curieux)
        optimal_k = 3
        if 3 in scores:
            best_score = max(scores.values())
            best_k = max(scores, key=scores.get)
            if best_k != 3 and scores[best_k] > scores.get(3, 0) * 1.2:
                optimal_k = best_k

        logger.info(f"K choisi = {optimal_k} (Silhouette={scores.get(optimal_k, '?')})")
        return optimal_k, scores

    def fit_predict(self, features_df: pd.DataFrame,
                    k: int = None) -> pd.DataFrame:
        """
        Exécute le clustering K-Means.

        Returns:
            DataFrame avec les labels de cluster et les profils
        """
        if k is None:
            k = self.n_clusters

        X = self.scaler.fit_transform(features_df)

        self.model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = self.model.fit_predict(X)

        features_df = features_df.copy()
        features_df["cluster"] = labels

        # Ajouter le vrai type pour validation
        if hasattr(self, 'user_features') and "user_type_real" in self.user_features.columns:
            features_df["user_type_real"] = self.user_features["user_type_real"].values

        self.stats.record_success()
        return features_df

    def analyze_clusters(self, clustered_df: pd.DataFrame) -> list[dict]:
        """
        Analyse et interprète chaque cluster.
        Assigne un profil type à chaque segment.
        """
        logger.info("Analyse des clusters...")

        profiles = []
        for cluster_id in sorted(clustered_df["cluster"].unique()):
            cluster = clustered_df[clustered_df["cluster"] == cluster_id]

            profile = {
                "cluster_id": int(cluster_id),
                "taille": len(cluster),
                "pourcentage": round(len(cluster) / len(clustered_df) * 100, 1),
                "nb_sessions_moyen": round(cluster["nb_sessions"].mean(), 1),
                "duree_moyenne": round(cluster["duree_moyenne"].mean(), 1),
                "nb_joueurs_moyen": round(cluster["nb_joueurs_consultes"].mean(), 1),
                "pct_mobile": round(cluster["is_mobile"].mean() * 100, 1),
            }

            # Distribution des catégories
            categories = ["stats", "news", "social", "commerce", "profil"]
            cat_dist = {}
            for cat in categories:
                cat_dist[cat] = round(cluster[f"pct_{cat}"].mean() * 100, 1)
            profile["distribution_categories"] = cat_dist

            # Label du segment basé sur un scoring multi-critères
            scores = {
                "Parieur": (
                    cat_dist.get("stats", 0) * 2 +
                    cat_dist.get("commerce", 0) * 1.5 +
                    profile["nb_sessions_moyen"] * 0.5
                ),
                "Fan": (
                    cat_dist.get("social", 0) * 2 +
                    cat_dist.get("news", 0) * 1 +
                    cat_dist.get("commerce", 0) * 0.5 +
                    profile["pct_mobile"] * 0.3
                ),
                "Curieux": (
                    cat_dist.get("news", 0) * 1.5 +
                    cat_dist.get("profil", 0) * 1 +
                    (100 - profile["nb_sessions_moyen"] * 10) * 0.3
                ),
            }

            # Éviter de réutiliser un label déjà assigné
            used_labels = [p["segment_label"] for p in profiles]
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for label, score in sorted_scores:
                if label not in used_labels:
                    profile["segment_label"] = label
                    break
            else:
                profile["segment_label"] = sorted_scores[0][0]

            # Validation : comparaison avec le vrai type (si disponible)
            if "user_type_real" in cluster.columns:
                real_dist = cluster["user_type_real"].value_counts(normalize=True).to_dict()
                profile["validation_types_reels"] = {
                    k: round(v * 100, 1) for k, v in real_dist.items()
                }

            profiles.append(profile)

        return profiles

    def compute_fan_value(self, clustered_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule la "Fan Value" prédictive pour chaque utilisateur.

        Utilise le meilleur modèle ML sauvegardé par pipeline_ia.py
        (comparaison Random Forest vs SVR vs Gradient Boosting).
        Si aucun modèle n'est disponible, utilise la formule pondérée comme fallback.
        """
        df = clustered_df.copy()

        # Essayer de charger le meilleur modèle ML
        model_used = self._predict_with_best_model(df)

        if model_used:
            logger.info("Fan Value calculée avec le meilleur modèle ML")
        else:
            # Fallback : formule pondérée manuelle
            logger.info("Fan Value calculée avec la formule pondérée (aucun modèle ML trouvé)")
            self._compute_fan_value_manual(df)

        return df

    def _predict_with_best_model(self, df: pd.DataFrame) -> bool:
        """
        Charge le meilleur modèle de régression sauvegardé et prédit la Fan Value.
        Retourne True si le modèle a été utilisé, False sinon.
        """
        try:
            import joblib
            models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
            model_path = os.path.join(models_dir, "best_transfer_model.joblib")
            scaler_path = os.path.join(models_dir, "transfer_scaler.joblib")
            info_path = os.path.join(os.path.dirname(__file__), "..", "data", "ia_best_models.json")

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return False

            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)

            # Charger le nom du meilleur modèle
            model_name = model.__class__.__name__
            if os.path.exists(info_path):
                with open(info_path) as f:
                    info = json.load(f)
                # ia_best_models.json uses top-level keys (see pipeline_ia.py)
                best_info = info  # top-level: best_model, metrics, etc.
                model_name = best_info.get("best_model", model_name)
                r2 = best_info.get("metrics", {}).get("r2_test", "?")
                logger.info(f"   Modèle chargé: {model_name} (R²={r2})")

            # Préparer les features
            feature_cols = ["nb_sessions", "duree_moyenne", "duree_totale",
                            "duree_std", "duree_max", "nb_joueurs_consultes", "is_mobile",
                            "pct_stats", "pct_news", "pct_social", "pct_commerce", "pct_profil"]

            # Ajouter les colonnes manquantes
            if "duree_std" not in df.columns:
                df["duree_std"] = 0
            if "duree_max" not in df.columns:
                df["duree_max"] = df["duree_moyenne"] * 2
            if "is_mobile" not in df.columns:
                df["is_mobile"] = 0

            # Vérifier que toutes les colonnes existent
            available = [c for c in feature_cols if c in df.columns]
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                for col in missing:
                    df[col] = 0
                logger.warning(f"  Colonnes manquantes ajoutées à 0: {missing}")

            X = df[feature_cols].fillna(0).values
            X_scaled = scaler.transform(X)

            # Prédiction
            predictions = model.predict(X_scaled)
            df["fan_value"] = np.clip(predictions, 0, 100).round(2)

            # Scores composantes pour l'analyse
            def minmax(series):
                if series.max() == series.min():
                    return pd.Series(0.5, index=series.index)
                return (series - series.min()) / (series.max() - series.min())

            df["score_fidelite"] = minmax(df["nb_sessions"])
            df["score_commerce"] = minmax(df["pct_commerce"])
            df["score_engagement"] = minmax(df["duree_moyenne"])
            df["score_diversite"] = minmax(df["nb_joueurs_consultes"])

            logger.info(f"   Fan Value prédite par {model_name} pour {len(df)} utilisateurs")
            return True

        except Exception as e:
            logger.warning(f"  Erreur chargement modèle ML: {e}")
            return False

    def _compute_fan_value_manual(self, df: pd.DataFrame):
        """Calcul manuel de la Fan Value (fallback sans modèle ML)."""
        def minmax(series):
            if series.max() == series.min():
                return pd.Series(0.5, index=series.index)
            return (series - series.min()) / (series.max() - series.min())

        df["score_fidelite"] = minmax(df["nb_sessions"])
        df["score_commerce"] = minmax(df["pct_commerce"])
        df["score_engagement"] = minmax(df["duree_moyenne"])
        df["score_diversite"] = minmax(df["nb_joueurs_consultes"])

        weights = {"fidelite": 0.3, "commerce": 0.35, "engagement": 0.2, "diversite": 0.15}
        df["fan_value"] = (
            weights["fidelite"] * df["score_fidelite"] +
            weights["commerce"] * df["score_commerce"] +
            weights["engagement"] * df["score_engagement"] +
            weights["diversite"] * df["score_diversite"]
        )
        df["fan_value"] = (df["fan_value"] * 100).round(2)

# 5. FONCTION PRINCIPALE AXE 3


def run_axe3(use_trends: bool = True, n_users: int = 500,
             n_sessions: int = 2000) -> dict:
    """
    Exécute la pipeline complète de l'Axe 3 - Web Usage Mining.

    Args:
        use_trends: Activer la collecte Google Trends
        n_users: Nombre d'utilisateurs simulés
        n_sessions: Nombre de sessions simulées

    Returns:
        Dictionnaire avec trends, patterns, clusters, fan values
    """
    logger.info("=" * 60)
    logger.info("AXE 3 - WEB USAGE MINING")
    logger.info("=" * 60)

    results = {}

    # --- 1. Google Trends ---
    if use_trends:
        logger.info("\n Collecte des tendances Google Trends...")
        trends_collector = TrendsCollector()
        trends_data = trends_collector.collect_all_trends()
        export_to_json(trends_data, "axe3_trends.json")
        results["trends"] = trends_data

    # --- 2. Simulation de logs ---
    logger.info(f"\n Simulation de {n_sessions} sessions de navigation...")
    simulator = WebUsageSimulator(n_users=n_users, n_sessions=n_sessions)
    logs_df = simulator.generate_logs()
    logs_df.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "axe3_web_logs.csv"), index=False)
    export_to_json(
        logs_df.head(100).to_dict(orient="records"),
        "axe3_web_logs_sample.json"
    )
    results["logs_stats"] = {
        "total_logs": len(logs_df),
        "utilisateurs_uniques": logs_df["user_id"].nunique(),
        "sessions_uniques": logs_df["session_id"].nunique(),
    }

    # --- 3. Frequent Pattern Mining ---
    logger.info("\n Mining de patterns fréquents (Apriori)...")
    miner = PatternMiner(min_support=0.05, min_confidence=0.3)
    patterns = miner.mine_page_patterns(logs_df)
    export_to_json(patterns, "axe3_patterns.json")
    results["patterns"] = patterns

    player_assoc = miner.mine_player_associations(logs_df)
    export_to_json(player_assoc, "axe3_player_associations.json")
    results["player_associations"] = player_assoc

    logger.info("\n TOP RÈGLES D'ASSOCIATION:")
    for rule in patterns.get("association_rules", [])[:5]:
        logger.info(
            f"  {rule['antecedent']} → {rule['consequent']} "
            f"(conf={rule['confidence']}, lift={rule['lift']})"
        )

    # --- 4. Clustering K-Means ---
    logger.info("\n Segmentation d'audience (K-Means)...")
    segmenter = AudienceSegmenter()
    features_df = segmenter.prepare_features(logs_df)

    # Recherche du K optimal
    optimal_k, silhouette_scores = segmenter.find_optimal_k(features_df)
    results["silhouette_scores"] = silhouette_scores

    # Clustering avec K optimal
    clustered_df = segmenter.fit_predict(features_df, k=optimal_k)
    cluster_profiles = segmenter.analyze_clusters(clustered_df)
    export_to_json(cluster_profiles, "axe3_clusters.json")
    results["clusters"] = cluster_profiles

    logger.info("\n SEGMENTS IDENTIFIÉS:")
    for cp in cluster_profiles:
        logger.info(
            f"  Cluster {cp['cluster_id']} - {cp['segment_label']}: "
            f"{cp['taille']} users ({cp['pourcentage']}%)"
        )

    # --- 5. Fan Value ---
    logger.info("\n Calcul de la Fan Value prédictive...")
    fan_df = segmenter.compute_fan_value(clustered_df)
    fan_summary = fan_df.groupby("cluster")["fan_value"].agg(["mean", "median", "std"]).round(2)
    export_to_csv(
        fan_df[["nb_sessions", "duree_moyenne", "nb_joueurs_consultes",
                "cluster", "fan_value"]].to_dict(orient="records"),
        "axe3_fan_values.csv"
    )
    results["fan_value_summary"] = fan_summary.to_dict()

    logger.info("\n FAN VALUE PAR SEGMENT:")
    for idx, row in fan_summary.iterrows():
        logger.info(f"  Cluster {idx}: moyenne={row['mean']}, médiane={row['median']}")

    logger.info("\n Axe 3 terminé avec succès!")
    return results


if __name__ == "__main__":
    run_axe3(use_trends=False)