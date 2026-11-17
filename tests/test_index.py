"""Integration checks against the real persisted index and embedding model."""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / 'chromadb/manifest.json').exists(), reason='Build the real index first')

def test_real_index_and_retrieval():
    from engine import resources
    from core import unique_hits
    model, collection = resources()
    manifest = json.loads((ROOT / 'chromadb/manifest.json').read_text())
    assert collection.count() == manifest['chunks']
    assert manifest['games'] == 13 if manifest['season_type'] == 'POST' and manifest['season'] == 2024 else True
    vector = model.encode(['An interception returned for a touchdown'], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=5, include=['documents','metadatas','distances'])
    assert all('intercept' in h['metadata']['description'].lower() and 'TOUCHDOWN' in h['metadata']['description'] for h in unique_hits(result,3))

def test_filters_really_constrain_results():
    from engine import resources, search
    from core import filters
    model, collection = resources()
    vector = model.encode(['touchdown'], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=5, where=filters('KC','4',True), include=['metadatas'])
    for meta in result['metadatas'][0]:
        assert meta['team'] == 'KC' and meta['quarter'] == 4 and meta['clutch'] == 1
    assert 'Describe a play' in search(' ')
    assert 'No plays match' in search('touchdown', team='NOT_A_TEAM')
