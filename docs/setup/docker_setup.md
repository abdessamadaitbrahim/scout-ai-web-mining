# Docker & Database Setup Guide

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- `docker-compose` or `docker compose` available in your terminal

---

## Starting the Databases

The project uses three NoSQL databases managed via Docker Compose:

| Service | Port | Purpose |
|---------|------|---------|
| MongoDB 7.0 | 27017 | Primary document storage (articles, profiles, usage logs) |
| Neo4J 5 Community | 7474 (browser), 7687 (bolt) | Graph database (relational network, HITS) |
| ElasticSearch 8.12 | 9200 | Full-text search (articles, sentiment index) |

### Quick Start

```bash
# Start all three databases (detached mode)
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### Using the Helper Script

```bash
# Start databases + wait for readiness + populate with collected data
python setup_docker.py

# Options:
python setup_docker.py --up         # Start containers only
python setup_docker.py --status     # Check connection status
python setup_docker.py --populate   # Load data into already-running containers
python setup_docker.py --down       # Stop all containers
python setup_docker.py --reset      # Stop + delete volumes + restart (DESTROYS DATA)
```

---

## Environment Variables for Databases

Configure in your `.env` file:

```bash
MONGO_URI=mongodb://localhost:27017
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=                    # Leave blank — Docker runs with NEO4J_AUTH=none
ELASTIC_HOST=http://localhost:9200
```

---

## Neo4J Browser Interface

After starting containers, open your browser at:
- **http://localhost:7474** → Neo4J Browser (no authentication required by default)

You can run Cypher queries directly or import the generated script:

```cypher
// Load the generated Cypher script
:source data/axe2_import_neo4j.cypher
```

---

## ElasticSearch Compatibility Note

The project ships with two different versions:

| Component | Version |
|-----------|---------|
| `docker-compose.yml` image | `elasticsearch:8.12.0` |
| Python client in `requirements.txt` | `elasticsearch==7.17.9` |

The 7.x client is currently pinned because the code uses the deprecated `body=` parameter
style. You have two options:

**Option A (current default)**: Keep both as-is. The 7.x client communicates with ES 8.x
over HTTP with security disabled (`xpack.security.enabled: false`). Most operations work.

**Option B (recommended for production)**: Upgrade the Python client:
```bash
pip install elasticsearch>=8.0.0
```
Then update all `es.search(index=..., body=...)` calls in:
- `database/elastic_manager.py`
- `dashboard_streamlit.py`

to the v8 keyword-argument style (remove the `body=` parameter, pass fields directly).

---

## Stopping and Cleanup

```bash
# Stop containers (data preserved in Docker volumes)
docker compose down

# Stop + delete all data (IRREVERSIBLE)
docker compose down -v
```
