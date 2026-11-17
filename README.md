# Clutch Finder

**[Open the live app](https://konur-clutch-finder.streamlit.app/)** · [GitHub repository](https://github.com/konurnordberg/clutch-finder)

Describe an NFL play in your own words and find matching moments from the **2024-season playoffs (played January–February 2025)**. Each result includes the original play description, possession team, game date, quarter, clock, score margin, and source identifiers. This is semantic text retrieval, not a chatbot or video search.

Try “a quarterback scrambling,” “an interception returned for a touchdown,” or “a sack that forces a fumble.” Filter by possession team and quarter, or select **Late & close**: Q4/OT, at most 5 minutes remaining in that quarter, and a pre-play score margin within 8 points. The possession-team filter refers to the offense, including on interceptions.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. The committed `chromadb/` contains the precomputed index, so startup does not download NFL data or re-embed it. The public MiniLM embedding model downloads on first search (no paid API key needed). Subsequent searches reuse the loaded model. An internet connection is needed for that initial model download.

## Assignment: six steps

1. **Pick a dataset.** nflverse's 2024 play-by-play Parquet release, filtered to postseason games. Source URL, SHA-256, exact game/play/chunk counts, and teams are recorded in `chromadb/manifest.json`. Raw downloads are excluded from Git.
2. **Chunk + embed.** Sort by game/play order. Each chunk centers on one play, enriched with deterministic English labels from recorded event flags (interception, sack, touchdown, etc.). The preceding eligible play from the same game, drive, and possession team is retained as display-only metadata so its action cannot dominate the focal play embedding. Long descriptions use 230-token windows with 30-token overlap. These bounds fit MiniLM's 256-token input. Embed with `sentence-transformers/all-MiniLM-L6-v2`: blend 65% outcome-label embedding with 35% full-passage embedding and normalize the resulting 384-dimensional vector. This weights the action above terse player/yard-line notation. Queries use the same model. The weights were chosen after inspecting representative searches; this is a small qualitative check, not a held-out benchmark. This preserves play-level filters while making local drive context available.
3. **Store in Chroma.** `build_index.py` creates a disk-backed `chromadb.PersistentClient` collection with cosine distance. The app opens that existing collection. Both embeddings and source metadata are persisted.
4. **Build the app.** `app.py` provides a Streamlit interface. It embeds the query with the same model, filters candidates in Chroma, retrieves nearest chunks, and deduplicates split chunks by game/play ID. Source content is HTML-escaped. Cosine similarity is a ranking score, not a probability or factual confidence.
5. **Publish.** Deploy from this GitHub repository to **Streamlit Community Cloud** using `app.py`. The original starter specifies Hugging Face Spaces; the project owner requested Streamlit Community Cloud because Hugging Face account creation was unavailable. Confirm that hosting substitution with the instructor.
6. **Submit the link on Canvas.** After deployment, verify a search on the live app and copy the `https://…streamlit.app` URL to Canvas. Submission requires your course account; it has not been performed by this repository.

## Rebuild or expand the corpus

```bash
# Preserve the existing database before replacing it.
mv chromadb chromadb-backup
python build_index.py --season 2024 --season-type POST
# Other supported selections: --season-type REG or ALL
```

Do not commit `chromadb-backup/`. The builder deliberately refuses to replace a completed index. A partial failed build can be rerun. Full seasons take more time and storage. Rebuilding downloads the public source file and model; the source file is cached under `data/raw/`. The manifest records the data actually used. Do not run the app while rebuilding its database.

## Publish on Streamlit Community Cloud

1. Sign in at https://share.streamlit.io/ with your GitHub account.
2. Choose **Create app → deploy from GitHub**.
3. Select `konurnordberg/clutch-finder`, branch `main`, entrypoint `app.py`.
4. Under advanced settings choose **Python 3.12**, then deploy on the free Community Cloud tier.
5. Run an example search after the build completes and copy the live app URL.

No secrets, paid API, or Hugging Face account are required. The model downloads publicly on first search. The committed database avoids re-embedding data on the hosting machine. Application tests pass locally with Python 3.10; live search was also verified on Streamlit Community Cloud with Python 3.12.16. CPU-only PyTorch wheels are selected on Linux to avoid unnecessary CUDA downloads.

## Validation

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover drive boundaries, chunk bounds, late-game filter boundaries, missing/no-play records, deduplication, HTML escaping, persisted-index counts, real embedding retrieval, and actual Chroma filtering. Integration tests need the built index and the embedding model; without the index they are explicitly skipped.

## Data attribution and limitations

Data: [nflverse / nflverse-data](https://github.com/nflverse/nflverse-data), licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This project filters, combines, tokenizes, and embeds original descriptions and preserves identifiers. The data license continues to apply to the included database and descriptions; the MIT license covers original application code only. See `DATA_LICENSE.md`.

Coverage is intentionally the 2024 postseason, not every NFL game. Semantic similarity can confuse negation, incomplete passes, and related actions; scores do not prove a match. Natural-language numerical constraints are not guaranteed—use the explicit filters. Historical source descriptions can contain errors. No official NFL affiliation, video footage, live feeds, or generated play narratives are included.

## Files

- `app.py`: Streamlit interface and search
- `core.py`: chunking, metadata filters, deduplication, rendering
- `build_index.py`: reproducible data download and index build
- `chromadb/`: saved SQLite/vector index and build manifest
- `engine.py`: cached model/index and serialized inference
- `.streamlit/config.toml`: theme and hosting configuration
- `tests/`: unit and real-index integration checks
- `requirements.txt`: pinned runtime dependencies
