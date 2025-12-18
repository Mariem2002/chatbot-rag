

# Chatbot RAG

Ce projet implémente un **chatbot basé sur l’architecture Retrieval-Augmented Generation (RAG)**, capable de générer des réponses synthétisées à partir de passages de texte pertinents.

## Fonctionnalités

* Découpage automatique des textes en passages (chunks)
* Génération d’embeddings avec Gemini (`text-embedding-004`)
* Stockage vectoriel dans PostgreSQL avec **pgvector**
* Recherche de similarité vectorielle (distance cosinus)
* Génération de réponses par le modèle Gemini à partir des passages récupérés (RAG)
* Exécution locale en boucle interactive

## Structure du projet

```
.
│   .gitignore
│   README.md
│
├───data
│       conversation.txt
│
└───src
        .env
        prototypage.py
        requirements.txt
```

## Installation

1. Créer et activer un environnement virtuel
2. Installer les dépendances :

```
pip install -r src/requirements.txt
```

3. Ajouter la clé API dans `src/.env` :

```
GEMINI_API_KEY=your_api_key_here
```

4. Installer PostgreSQL et l’extension pgvector

## Exécution

```
python src/prototypage.py
```

## Principe de fonctionnement

1. Le texte source est découpé en passages (chunks)
2. Chaque passage est transformé en embedding et stocké dans PostgreSQL
3. Lorsqu’une question utilisateur est posée :

   * Les passages les plus similaires sont récupérés via recherche vectorielle (**retrieval**)
   * Les passages récupérés sont envoyés avec la question à un modèle Gemini pour générer une réponse synthétisée (**generation**)
4. La réponse générée est affichée à l’utilisateur

## Auteur

Créé par **Mariem Ben Tamansourt**

---

