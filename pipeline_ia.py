"""
Scout-AI : Pipeline IA — Prédiction du Potentiel de Transfert
================================================================
Pipeline professionnelle de Machine Learning pour prédire le potentiel
de transfert des joueurs de football.

Architecture :
  1. Construction d'un dataset d'entraînement équilibré (4 catégories)
  2. Entraînement et évaluation de 3 modèles (RF, SVR, GB)
  3. Sélection du meilleur modèle par R²
  4. Prédiction sur les 10 joueurs réels (jamais vus en entraînement)

Catégories de transfert :
   Très demandé  (75-100)  — Joueur au cœur des rumeurs, forte couverture
   En hausse      (50-75)   — Potentiel montant, intérêt croissant
   Sous-évalué    (25-50)   — Talent réel mais peu médiatisé
   Stable         (0-25)    — Peu de mouvement attendu

Usage:  python pipeline_ia.py
"""

import os, sys, json, warnings, unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import DATA_DIR, TARGET_PLAYERS
from utils.helpers import setup_logger, export_to_json

logger = setup_logger("pipeline-ia")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_NAMES = [
    "mentions", "sentiment_moyen", "diversite_sources", "cote_popularite",
    "nb_rumeurs", "pagerank", "authority_score", "nb_vues",
]

def normalize_name(name):
    return unicodedata.normalize("NFC", str(name).strip())


# ============================================================
# DATASET D'ENTRAÎNEMENT (équilibré sur les 4 catégories)
# ============================================================

def build_training_data():
    """
    Construit un dataset d'entraînement équilibré avec 25 joueurs
    par catégorie (100 joueurs au total), couvrant tout le spectre 0-100.
    """
    logger.info(" Construction du dataset d'entraînement...")
    np.random.seed(42)

    # 4 catégories × 25 joueurs = 100 joueurs d'entraînement
    categories = {
        "hot":        {"target_range": (75, 100), "count": 25},
        "rising":     {"target_range": (50, 75),  "count": 25},
        "undervalued":{"target_range": (25, 50),  "count": 25},
        "stable":     {"target_range": (0, 25),   "count": 25},
    }

    # Distributions réalistes des features par catégorie
    feature_ranges = {
        "hot": {
            "mentions": (30, 55), "sentiment_moyen": (3.5, 4.8), "diversite_sources": (8, 15),
            "cote_popularite": (65, 95), "nb_rumeurs": (3, 7),
            "pagerank": (0.04, 0.12), "authority_score": (0.10, 0.25),
            "nb_vues": (800, 2000),
        },
        "rising": {
            "mentions": (15, 35), "sentiment_moyen": (3.0, 4.2), "diversite_sources": (5, 10),
            "cote_popularite": (40, 70), "nb_rumeurs": (1, 4),
            "pagerank": (0.02, 0.06), "authority_score": (0.04, 0.14),
            "nb_vues": (400, 1000),
        },
        "undervalued": {
            "mentions": (5, 18), "sentiment_moyen": (2.8, 4.0), "diversite_sources": (2, 7),
            "cote_popularite": (20, 50), "nb_rumeurs": (0, 3),
            "pagerank": (0.008, 0.03), "authority_score": (0.01, 0.07),
            "nb_vues": (100, 500),
        },
        "stable": {
            "mentions": (0, 8), "sentiment_moyen": (2.0, 3.2), "diversite_sources": (0, 4),
            "cote_popularite": (5, 25), "nb_rumeurs": (0, 1),
            "pagerank": (0.002, 0.015), "authority_score": (0.002, 0.03),
            "nb_vues": (20, 200),
        },
    }

    rows = []
    for cat_name, cat_info in categories.items():
        low, high = cat_info["target_range"]
        ranges = feature_ranges[cat_name]

        for i in range(cat_info["count"]):
            row = {}
            for feat in FEATURE_NAMES:
                f_low, f_high = ranges[feat]
                if feat in ("mentions", "nb_rumeurs", "nb_vues"):
                    row[feat] = int(np.random.uniform(f_low, f_high))
                else:
                    row[feat] = round(np.random.uniform(f_low, f_high), 4)

            # Score cible avec du bruit naturel
            row["transfer_potential"] = round(np.random.uniform(low, high), 2)
            row["categorie"] = cat_name
            rows.append(row)

    df = pd.DataFrame(rows)

    # Vérification de l'équilibre
    for cat in categories:
        n = len(df[df["categorie"] == cat])
        logger.info(f"  {cat}: {n} joueurs (cible {categories[cat]['target_range']})")

    logger.info(f"  Total: {len(df)} joueurs × {len(FEATURE_NAMES)} features")
    return df


# ============================================================
# CHARGEMENT DES 10 JOUEURS RÉELS
# ============================================================

def load_real_players():
    """Charge les features des 10 joueurs réels depuis les données scrapées."""
    logger.info("\n Chargement des 10 joueurs réels...")

    players = [{"nom": normalize_name(p["name"]), "club": p["team"]} for p in TARGET_PLAYERS]
    df = pd.DataFrame(players)

    # Axe 1 : popularité
    _merge_axe1(df)

    # Axe 2 : centralité + HITS + rumeurs
    _merge_axe2(df)

    # Axe 3 : co-occurrences + vues
    _merge_axe3(df)

    # Remplir les NaN
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0
    df[FEATURE_NAMES] = df[FEATURE_NAMES].fillna(0)

    # Log
    for _, row in df.iterrows():
        logger.info(f"  {row['nom']}: mentions={int(row['mentions'])}, "
                     f"rumeurs={int(row['nb_rumeurs'])}, "
                     f"pagerank={row['pagerank']:.4f}")

    return df


def _merge_axe1(df):
    path = os.path.join(DATA_DIR, "axe1_cotes_popularite.json")
    if not os.path.exists(path):
        logger.warning("   axe1_cotes_popularite.json non trouvé")
        return
    with open(path, encoding="utf-8") as f:
        pop_df = pd.DataFrame(json.load(f))
    if "joueur" not in pop_df.columns:
        return
    pop_df["joueur"] = pop_df["joueur"].apply(normalize_name)
    cols = ["joueur", "cote_popularite", "mentions", "sentiment_moyen", "diversite_sources"]
    cols = [c for c in cols if c in pop_df.columns]
    merged = df.merge(pop_df[cols], left_on="nom", right_on="joueur", how="left")
    if "joueur" in merged.columns:
        merged = merged.drop("joueur", axis=1)
    for col in merged.columns:
        df[col] = merged[col]
    matched = df["mentions"].notna().sum() if "mentions" in df.columns else 0
    logger.info(f"   Axe 1: {matched}/{len(df)} joueurs matchés")


def _merge_axe2(df):
    # Centralité
    path = os.path.join(DATA_DIR, "axe2_centralite.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            jc = pd.DataFrame([c for c in json.load(f) if c["type"] == "JOUEUR"])
        if not jc.empty:
            jc["noeud"] = jc["noeud"].apply(normalize_name)
            merged = df.merge(jc[["noeud", "pagerank", "betweenness", "closeness", "in_degree"]],
                              left_on="nom", right_on="noeud", how="left")
            if "noeud" in merged.columns:
                merged = merged.drop("noeud", axis=1)
            for col in ["pagerank", "betweenness", "closeness", "in_degree"]:
                if col in merged.columns:
                    df[col] = merged[col]

    # HITS authority
    path = os.path.join(DATA_DIR, "axe2_hits.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            auth = {normalize_name(a["noeud"]): a["score_authority"]
                    for a in json.load(f).get("top_authorities", []) if a.get("type") == "JOUEUR"}
        df["authority_score"] = df["nom"].map(auth)

    # Rumeurs
    path = os.path.join(DATA_DIR, "axe2_graphe.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rumeurs = [e for e in json.load(f)["edges"] if e.get("relation") == "RUMEUR"]
        rc, rf = {}, {}
        for r in rumeurs:
            j = normalize_name(r.get("joueur", ""))
            rc[j] = rc.get(j, 0) + 1
            rf[j] = max(rf.get(j, 0), r.get("fiabilite", 0))
        df["nb_rumeurs"] = df["nom"].map(rc)
        df["fiabilite_max_rumeur"] = df["nom"].map(rf)

    logger.info(f"   Axe 2: centralité + HITS + rumeurs")


def _merge_axe3(df):
    # Co-occurrences
    path = os.path.join(DATA_DIR, "axe3_player_associations.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            cooc = {}
            for a in json.load(f):
                for j in [a["joueur_1"], a["joueur_2"]]:
                    nj = normalize_name(j)
                    cooc[nj] = cooc.get(nj, 0) + a["co_occurrences"]
        df["co_occurrences_total"] = df["nom"].map(cooc)

    # Vues
    path = os.path.join(DATA_DIR, "axe3_web_logs.csv")
    if os.path.exists(path):
        logs = pd.read_csv(path)
        logs["player_viewed"] = logs["player_viewed"].apply(normalize_name)
        views = logs.groupby("player_viewed").agg(
            nb_vues=("session_id", "count"),
            sessions_uniques=("session_id", "nunique"),
            duree_moy_consultation=("duration_seconds", "mean")
        ).reset_index()
        merged = df.merge(views, left_on="nom", right_on="player_viewed", how="left")
        if "player_viewed" in merged.columns:
            merged = merged.drop("player_viewed", axis=1)
        for col in ["nb_vues", "sessions_uniques", "duree_moy_consultation"]:
            if col in merged.columns:
                df[col] = merged[col]

    logger.info(f"   Axe 3: co-occurrences + vues")


# ============================================================
# COMPARAISON DES 3 MODÈLES
# ============================================================

def train_and_compare(df_train):
    """
    Entraîne 3 modèles, évalue avec 5-fold CV et test split.
    Retourne les résultats et le meilleur modèle (par R² test).
    """
    logger.info("\n Entraînement et comparaison des 3 modèles...")

    X = df_train[FEATURE_NAMES].values
    y = df_train["transfer_potential"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=df_train["categorie"])

    logger.info(f"  Train: {len(X_tr)} | Test: {len(X_te)} | Features: {len(FEATURE_NAMES)}")

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42),
        "SVR": SVR(
            kernel="rbf", C=100, gamma="auto", epsilon=0.1),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05, min_samples_leaf=2, random_state=42),
    }

    results = {}
    for name, model in models.items():
        logger.info(f"\n  {' ' if 'Forest' in name else ' ' if 'SVR' in name else ' '} {name}...")

        # Cross-validation 5-fold
        cv_r2 = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2")
        cv_mae = -cross_val_score(model, X_tr, y_tr, cv=5, scoring="neg_mean_absolute_error")

        # Train + test
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        r2 = r2_score(y_te, y_pred)
        mae = mean_absolute_error(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))

        results[name] = {
            "r2_test": round(r2, 4),
            "mae_test": round(mae, 4),
            "rmse_test": round(rmse, 4),
            "cv_r2_mean": round(cv_r2.mean(), 4),
            "cv_r2_std": round(cv_r2.std(), 4),
            "cv_mae_mean": round(cv_mae.mean(), 4),
        }

        if hasattr(model, "feature_importances_"):
            results[name]["feature_importance"] = dict(sorted(
                {FEATURE_NAMES[i]: round(float(v), 4) for i, v in enumerate(model.feature_importances_)}.items(),
                key=lambda x: -x[1]))

        logger.info(f"    R² test    = {r2:.4f}")
        logger.info(f"    MAE test   = {mae:.4f}")
        logger.info(f"    CV R² (5f) = {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")
        logger.info(f"    CV MAE     = {cv_mae.mean():.4f}")

    # Sélection du meilleur modèle par R² test
    best_name = max(results, key=lambda x: results[x]["r2_test"])
    logger.info(f"\n   MEILLEUR MODÈLE : {best_name} (R²={results[best_name]['r2_test']})")

    # Ré-entraîner le meilleur sur TOUT le dataset d'entraînement
    best_model = type(models[best_name])(**models[best_name].get_params())
    best_model.fit(X_scaled, y)

    return results, best_name, best_model, scaler


# ============================================================
# PRÉDICTION SUR LES 10 JOUEURS RÉELS
# ============================================================

def predict_real_players(best_model, scaler, df_real):
    """Utilise le meilleur modèle pour prédire les 10 joueurs réels."""
    logger.info("\n Prédiction des 10 joueurs réels...")

    X_real = df_real[FEATURE_NAMES].fillna(0).values
    X_scaled = scaler.transform(X_real)
    predictions = best_model.predict(X_scaled)

    # Clamp entre 0 et 100
    predictions = np.clip(predictions, 0, 100)

    preds = []
    for i, (_, row) in enumerate(df_real.iterrows()):
        score = round(float(predictions[i]), 2)
        if score >= 75:
            categorie = " Très demandé"
        elif score >= 50:
            categorie = " En hausse"
        elif score >= 25:
            categorie = " Sous-évalué"
        else:
            categorie = " Stable"

        preds.append({
            "joueur": row["nom"],
            "club": row["club"],
            "prediction_ml": score,
            "categorie": categorie,
            "mentions": int(row.get("mentions", 0)),
            "sentiment": round(float(row.get("sentiment_moyen", 0)), 2),
            "diversite_sources": int(row.get("diversite_sources", 0)),
            "cote_popularite": round(float(row.get("cote_popularite", 0)), 2),
            "nb_rumeurs": int(row.get("nb_rumeurs", 0)),
            "pagerank": round(float(row.get("pagerank", 0)), 6),
            "authority_score": round(float(row.get("authority_score", 0)), 6),
            "nb_vues": int(row.get("nb_vues", 0)),
        })

    preds.sort(key=lambda x: x["prediction_ml"], reverse=True)

    logger.info("\n CLASSEMENT PRÉDICTIF — POTENTIEL DE TRANSFERT :")
    for i, p in enumerate(preds, 1):
        bar = "█" * int(p["prediction_ml"] / 5)
        logger.info(f"  {i:2d}. {p['joueur']:<22s} {p['prediction_ml']:6.2f}/100  {p['categorie']:<18s} {bar}")

    return preds


# ============================================================
# SAUVEGARDE
# ============================================================

def save_results(best_name, best_model, scaler, results, preds):
    """Sauvegarde le modèle, les prédictions et les métadonnées."""
    logger.info("\n Sauvegarde...")

    joblib.dump(best_model, os.path.join(MODELS_DIR, "best_transfer_model.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "transfer_scaler.joblib"))
    joblib.dump(FEATURE_NAMES, os.path.join(MODELS_DIR, "transfer_feature_cols.joblib"))

    export_to_json(preds, "ia_transfer_predictions.json")

    export_to_json({
        "best_model": best_name,
        "metrics": {k: v for k, v in results[best_name].items() if k != "feature_importance"},
        "all_models": {k: {m: v for m, v in r.items() if m != "feature_importance"} for k, r in results.items()},
        "feature_importance": results[best_name].get("feature_importance", {}),
        "feature_cols": FEATURE_NAMES,
        "n_training_samples": 100,
        "n_real_players": len(preds),
        "date": datetime.now().isoformat(),
        "classement": [{"rang": i+1, "joueur": p["joueur"], "score": p["prediction_ml"],
                         "categorie": p["categorie"]} for i, p in enumerate(preds)],
    }, "ia_best_models.json")

    logger.info(f"  → models/best_transfer_model.joblib ({best_name})")
    logger.info(f"  → data/ia_transfer_predictions.json ({len(preds)} joueurs)")


# ============================================================
# FIGURES
# ============================================================

def generate_figures(results, best_name, preds):
    """Génère les visualisations de la pipeline."""
    logger.info("\n Figures...")

    # 1. Comparaison R² des 3 modèles
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0e17"); ax.set_facecolor("#111827")
    names = list(results.keys())
    r2s = [results[n]["r2_test"] for n in names]
    colors = ["#3b82f6", "#10b981", "#f59e0b"]
    bars = ax.bar(names, r2s, color=colors, edgecolor="white", linewidth=0.5)
    for b, v in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
                f"{v:.4f}", ha="center", color="white", fontsize=12, fontweight="bold")
    best_idx = names.index(best_name)
    bars[best_idx].set_edgecolor("#ef4444"); bars[best_idx].set_linewidth(3)
    ax.set_title("Comparaison R² — 3 Modèles de Prédiction", color="white", fontsize=14, fontweight="bold")
    ax.set_ylabel("R² Score", color="white"); ax.tick_params(colors="white")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color("#334155")
    fig.savefig(os.path.join(FIGURES_DIR, "ia_model_comparison.png"), dpi=150, bbox_inches="tight", facecolor="#0a0e17")
    plt.close()

    # 2. Classement des 10 joueurs
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor("#0a0e17"); ax.set_facecolor("#111827")
    ps = sorted(preds, key=lambda x: x["prediction_ml"])
    js = [p["joueur"] for p in ps]
    ss = [p["prediction_ml"] for p in ps]
    bar_colors = []
    for s in ss:
        if s >= 75: bar_colors.append("#ef4444")
        elif s >= 50: bar_colors.append("#f59e0b")
        elif s >= 25: bar_colors.append("#3b82f6")
        else: bar_colors.append("#64748b")
    ax.barh(js, ss, color=bar_colors, edgecolor="white", linewidth=0.3)
    for i, (j, s) in enumerate(zip(js, ss)):
        ax.text(s + 1, i, f"{s:.1f}", va="center", color="white", fontsize=10)
    ax.set_title(f"Potentiel de Transfert — Prédit par {best_name}", color="white", fontsize=14, fontweight="bold")
    ax.set_xlabel("Score /100", color="white"); ax.tick_params(colors="white"); ax.set_xlim(0, 105)
    # Zones colorées
    ax.axvspan(75, 105, alpha=0.08, color="#ef4444")
    ax.axvspan(50, 75, alpha=0.08, color="#f59e0b")
    ax.axvspan(25, 50, alpha=0.08, color="#3b82f6")
    ax.axvspan(0, 25, alpha=0.08, color="#64748b")
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    for s in ["bottom", "left"]: ax.spines[s].set_color("#334155")
    fig.savefig(os.path.join(FIGURES_DIR, "ia_transfer_ranking.png"), dpi=150, bbox_inches="tight", facecolor="#0a0e17")
    plt.close()

    # 3. Feature importance
    imp = results[best_name].get("feature_importance", {})
    if imp:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor("#0a0e17"); ax.set_facecolor("#111827")
        si = sorted(imp.items(), key=lambda x: x[1])
        ax.barh([i[0] for i in si], [i[1] for i in si], color="#3b82f6", edgecolor="white", linewidth=0.3)
        ax.set_title(f"Importance des Features ({best_name})", color="white", fontsize=13)
        ax.tick_params(colors="white")
        for s in ["top", "right"]: ax.spines[s].set_visible(False)
        for s in ["bottom", "left"]: ax.spines[s].set_color("#334155")
        fig.savefig(os.path.join(FIGURES_DIR, "ia_feature_importance.png"), dpi=150, bbox_inches="tight", facecolor="#0a0e17")
        plt.close()


# ============================================================
# PIPELINE PRINCIPALE
# ============================================================

def run_pipeline():
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║     SCOUT-AI — Prédiction Potentiel de Transfert                 ║
    ║                                                                  ║
    ║   Entraînement : 100 joueurs (4 catégories × 25)                ║
    ║   Prédiction   : 10 joueurs réels (données scrapées)            ║
    ║   Modèles      : Random Forest vs SVR vs Gradient Boosting      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    start = datetime.now()

    # 1. Dataset d'entraînement équilibré
    df_train = build_training_data()

    # 2. Charger les 10 joueurs réels
    df_real = load_real_players()

    # 3. Entraîner et comparer les 3 modèles
    results, best_name, best_model, scaler = train_and_compare(df_train)

    # 4. Prédire les 10 joueurs réels
    preds = predict_real_players(best_model, scaler, df_real)

    # 5. Sauvegarder
    save_results(best_name, best_model, scaler, results, preds)

    # 6. Figures
    generate_figures(results, best_name, preds)

    elapsed = (datetime.now() - start).total_seconds()
    r2 = results[best_name]["r2_test"]
    print(f"\n{'=' * 60}")
    print(f"   PIPELINE TERMINÉE ({elapsed:.1f}s)")
    print(f"   Meilleur modèle : {best_name} (R² = {r2})")
    print(f"   Entraîné sur 100 joueurs (4 catégories × 25)")
    print(f"   Prédit {len(preds)} joueurs réels")
    print(f"   Modèle : models/best_transfer_model.joblib")
    print(f"{'=' * 60}")
    return results


if __name__ == "__main__":
    run_pipeline()