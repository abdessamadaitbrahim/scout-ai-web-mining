# System Architecture

## Overview

Scout-AI is a three-axis web mining and intelligence platform for football transfer market analysis.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA COLLECTION LAYER                        │
│                                                                       │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │   AXE 1          │  │   AXE 2          │  │   AXE 3          │  │
│  │ Content Mining   │  │ Structure Mining │  │ Usage Mining     │  │
│  │                  │  │                  │  │                  │  │
│  │ • NewsAPI        │  │ • NetworkX Graph │  │ • Google Trends  │  │
│  │ • Football-Data  │  │ • HITS Algorithm │  │ • Web Log Sim.   │  │
│  │ • Transfermarkt  │  │ • PageRank       │  │ • Apriori        │  │
│  │ • BERT NLP/NER   │  │ • Louvain        │  │ • K-Means        │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                     │                      │             │
└───────────┼─────────────────────┼──────────────────────┼───────────┘
            │                     │                      │
┌───────────┼─────────────────────┼──────────────────────┼───────────┐
│           │      STORAGE LAYER  │                      │            │
│  ┌────────▼──────┐  ┌──────────▼──────┐  ┌───────────▼──────┐    │
│  │   MongoDB     │  │     Neo4J        │  │  ElasticSearch   │    │
│  │ Articles, NLP │  │ Graph, HITS,     │  │ Full-text index  │    │
│  │ Stats, Logs   │  │ Communities      │  │ Sentiment search │    │
│  └───────────────┘  └─────────────────┘  └──────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
            │
┌───────────┼───────────────────────────────────────────────────────┐
│           │     ML PIPELINE LAYER                                  │
│  ┌────────▼──────────────────────────────────────────────────┐    │
│  │                   pipeline_ia.py                           │    │
│  │  Random Forest  vs  SVR  vs  Gradient Boosting            │    │
│  │  → Best model selected by R² score                        │    │
│  │  → Saves to models/best_transfer_model.joblib             │    │
│  └────────────────────────────────┬───────────────────────────┘    │
└───────────────────────────────────┼───────────────────────────────┘
                                    │
┌───────────────────────────────────┼───────────────────────────────┐
│                  DASHBOARD LAYER  │                                 │
│  ┌────────────────────────────────▼───────────────────────────┐   │
│  │            dashboard_streamlit.py (Streamlit)               │   │
│  │  Tab 1: Overview      Tab 2: Popularity Scores             │   │
│  │  Tab 3: Graph         Tab 4: HITS & Centrality             │   │
│  │  Tab 5: Segmentation  Tab 6: Transfer Predictions          │   │
│  └────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Collection** (`main.py --axe [0|1|2|3]`): Runs one or all collection axes, outputs JSON/CSV to `data/`
2. **ML Pipeline** (`pipeline_ia.py`): Reads feature data from `data/`, trains 3 models, saves best to `models/`
3. **Database Population** (`setup_docker.py`): Imports `data/` files into MongoDB, Neo4J, ElasticSearch
4. **Dashboard** (`streamlit run dashboard_streamlit.py`): Reads from databases (or falls back to `data/` files)

## Module Dependencies

```
config/settings.py          ← loaded by all modules
utils/helpers.py            ← loaded by all collectors and DB managers
collectors/axe1_*.py   ──┐
collectors/axe2_*.py   ──┤──▶ data/ (JSON/CSV output)
collectors/axe3_*.py   ──┘       │
pipeline_ia.py ◀─────────────────┤
                                  │
database/*.py ◀──────────────────┤
                                  │
dashboard_streamlit.py ◀──────────┘ (+ databases)
```
