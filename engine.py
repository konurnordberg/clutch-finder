"""Persistent semantic search engine, independent of the web interface."""
import json
import os
from functools import lru_cache
from threading import RLock

SEARCH_LOCK = RLock()
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
from core import MODEL, COLLECTION, filters, unique_hits, render_hits

@lru_cache(maxsize=1)
def resources():
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer
    manifest = json.loads((ROOT / 'chromadb/manifest.json').read_text())
    if manifest['model'] != MODEL:
        raise RuntimeError('Index and query embedding models do not match.')
    client = chromadb.PersistentClient(path=str(ROOT / 'chromadb'), settings=Settings(anonymized_telemetry=False))
    return SentenceTransformer(MODEL, device='cpu'), client.get_collection(COLLECTION, embedding_function=None)

def _search(query, team='All teams', quarter='Any quarter', clutch=False, count=5):
    if not query or not query.strip():
        return '<div class="empty">Describe a play to start searching.</div>'
    if len(query) > 1000:
        raise ValueError('Please keep your description under 1,000 characters.')
    model, collection = resources()
    where = filters(team, quarter, clutch)
    options = {'where': where} if where else {}
    vector = model.encode([query.strip()], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=min(int(count)*3, collection.count()),
                              include=['documents','metadatas','distances'], **options)
    return render_hits(unique_hits(result, int(count)))


def search(query, team='All teams', quarter='Any quarter', clutch=False, count=5):
    # Shared model/index stay in memory; serialize inference across sessions.
    with SEARCH_LOCK:
        return _search(query, team, quarter, clutch, count)
