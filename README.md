# Chatbot RAG

Ce projet implémente un chatbot basé sur une architecture Retrieval-Augmented Generation (RAG).

## Fonctionnalités
- Génération d’embeddings avec Gemini (text-embedding-004)
- Stockage vectoriel avec PostgreSQL et pgvector
- Recherche de similarité avec la distance cosinus
- Exécution locale

## Structure du projet
.
├── data/
│   └── conversation.txt
├── notebook/
│   ├── prototypage.py
│   └── .env
├── src/
│   ├── .gitignore
│   └── requirements.txt
└── README.md

## Installation
1. Créer et activer un environnement virtuel
2. Installer les dépendances :
pip install -r src/requirements.txt
3. Ajouter la clé API dans notebook/.env :
GEMINI_API_KEY=your_api_key_here
4. Installer PostgreSQL et pgvector

## Exécution
python notebook/prototypage.py

## Principe de fonctionnement
- Le texte est découpé en passages
- Chaque passage est transformé en embedding
- Les embeddings sont stockés dans PostgreSQL
- Les questions utilisateur retournent les passages les plus similaires

Crée par Mariem Ben Tamansourt