"""Download nflverse data, chunk plays, embed, and persist a Chroma database."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('ANONYMIZED_TELEMETRY', 'False')
from core import MODEL, COLLECTION, make_records

def build(season=2024, season_type='POST'):
    import chromadb
    import pandas as pd
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from chromadb.config import Settings
    db = ROOT / 'chromadb'
    if (db / 'manifest.json').exists():
        raise SystemExit('Index already exists. Move chromadb/ aside before rebuilding to preserve the previous index.')
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    source = f'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet'
    path = raw / f'{season}.parquet'
    if not path.exists():
        print('Downloading', source, flush=True)
        temp = path.with_suffix('.download')
        urllib.request.urlretrieve(source, temp)
        temp.replace(path)
    frame = pd.read_parquet(path)
    if season_type != 'ALL':
        frame = frame[frame.season_type == season_type]
    model = SentenceTransformer(MODEL, device='cpu')
    records = make_records(frame, model.tokenizer)
    if not records:
        raise SystemExit('No plays found for the selected season/type.')
    client = chromadb.PersistentClient(path=str(db), settings=Settings(anonymized_telemetry=False))
    # A failed build may leave a partial collection; safely replace only that collection.
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
    collection = client.create_collection(COLLECTION, configuration={'hnsw': {'space': 'cosine'}}, embedding_function=None)
    for start in range(0, len(records), 128):
        batch = records[start:start+128]
        passages = [r['text'] for r in batch]
        details = model.encode(passages, normalize_embeddings=True, batch_size=32)
        outcomes = model.encode([p.split('\n')[0] for p in passages], normalize_embeddings=True, batch_size=32)
        vectors = 0.65 * outcomes + 0.35 * details
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        collection.add(ids=[r['id'] for r in batch], documents=[r['text'] for r in batch],
                       metadatas=[r['metadata'] for r in batch], embeddings=vectors.tolist())
        print(f'Indexed {min(start+128,len(records))}/{len(records)} chunks', flush=True)
    manifest = {'model': MODEL, 'season': season, 'season_type': season_type,
                'embedding_strategy': 'Normalized blend: 65% recorded outcome-label embedding + 35% full passage embedding',
                'chunks': collection.count(), 'games': int(frame.game_id.nunique()),
                'plays': len({(r['metadata']['game_id'], r['metadata']['play_id']) for r in records}),
                'teams': sorted({r['metadata']['team'] for r in records}), 'source': source,
                'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'chunking': 'Focal play with recorded event labels; 230-token windows with 30-token overlap. Previous same-drive play retained as display-only metadata.'}
    (db / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--season', type=int, default=2024)
    parser.add_argument('--season-type', choices=['POST','REG','ALL'], default='POST')
    args = parser.parse_args()
    build(args.season, args.season_type)
