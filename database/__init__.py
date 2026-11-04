"""
Scout-AI Database Package
===========================
NoSQL database managers for MongoDB, Neo4J, and ElasticSearch.

Modules:
    mongo_manager     — MongoDB document storage (articles, profiles, usage data)
    neo4j_manager     — Neo4J graph database (relational network, HITS scores)
    elastic_manager   — ElasticSearch full-text indexing (articles, sentiment)
    setup_databases   — Orchestrates all three database setups

Usage:
    from database.mongo_manager import setup_mongodb
    from database.neo4j_manager import setup_neo4j
    from database.elastic_manager import setup_elasticsearch
    from database.setup_databases import setup_all
"""
