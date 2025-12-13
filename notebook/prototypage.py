# prototypage.py
import os
import psycopg
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY manquante dans .env")

client = genai.Client(api_key=GEMINI_API_KEY)
EMBEDDING_MODEL = "text-embedding-004"
EMBEDDING_DIM = 768

# Chaîne de connexion 
DB_CONN_STR = (
    f"host=localhost "
    f"port=5433 "  
    f"dbname=postgres "
    f"user=postgres "
    f"password=1234"  
)

def create_chunks(file_path: str) -> list[str]:
    """Lit et nettoie les chunks du fichier."""
    try:
     
        full_path = r"C:\Users\Mariem\Desktop\chatbot-Rag\Chatbot-RAG\data\conversation.txt"
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        chunks = [line.strip().removeprefix("    ") for line in lines 
                  if line.strip() and not line.startswith("<")]
        print(f"{len(chunks)} chunks chargés depuis {full_path}")
        return chunks
    except FileNotFoundError:
        print(f"Fichier non trouvé ! Crée data/conversation.txt avec du texte de test.")
        return []

def get_embedding(text: str) -> list[float]:
    """Calcule l'embedding avec Gemini."""
    result = client.models.embed_content(
        model=f"models/{EMBEDDING_MODEL}",
        contents=[text],
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    return result.embeddings[0].values

def save_embeddings(chunks: list[str]):
    """Crée la table et insère les embeddings."""
    with psycopg.connect(DB_CONN_STR) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Test connexion + extension
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                print("✅ Extension pgvector activée.")
            except Exception as e:
                print(f"❌ Erreur extension: {e}")
                print("Installe pgvector d'abord (voir instructions).")
                return

            # Table
            cur.execute("DROP TABLE IF EXISTS embeddings")
            cur.execute(f"""
                CREATE TABLE embeddings (
                    id SERIAL PRIMARY KEY,
                    corpus TEXT,
                    embedding VECTOR({EMBEDDING_DIM})
                )
            """)
            print(f"✅ Table créée avec VECTOR({EMBEDDING_DIM}).")

            # Insertion
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    vec = get_embedding(chunk)
                    cur.execute(
                        "INSERT INTO embeddings (corpus, embedding) VALUES (%s, %s)",
                        (chunk, vec)
                    )
                if (i + 1) % 10 == 0 or i == len(chunks) - 1:
                    print(f"→ {i + 1}/{len(chunks)} insérés")
            
            # Index HNSW
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_embeddings ON embeddings 
                USING hnsw (embedding vector_cosine_ops)
            """)
            print("✅ Index HNSW créé.")

def search_similar(query: str, limit: int = 5):
    """Recherche les chunks les plus similaires."""
    vec = get_embedding(query)
    with psycopg.connect(DB_CONN_STR) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, corpus, embedding <=> %s::vector AS distance
                FROM embeddings
                ORDER BY distance
                LIMIT %s
            """, (vec, limit))
            return cur.fetchall()

# ==================== LANCEMENT ====================
if __name__ == "__main__":
    print("🚀 Démarrage RAG avec pgvector + Gemini\n")

    # Chargement
    chunks = create_chunks("data/conversation.txt")
    if not chunks:
        print("Crée un fichier data/conversation.txt avec du texte pour tester !")
        exit(1)

    # Insertion
    save_embeddings(chunks)

    # Test interactif
    print("\n🔍 Mode recherche : Pose une question (ou 'quit')")
    while True:
        query = input("\nTa question : ").strip()
        if query.lower() in ["quit", "exit", "q"]:
            break
        results = search_similar(query)
        if results:
            print(f"\nTop résultats pour « {query } » :\n")
            for i, (id_, text, dist) in enumerate(results, 1):
          
                print(f"{i}. [dist={dist:.4f}] {text}")
        else:
            print("Aucun résultat.")
    
    print("\n✅ Terminé !")