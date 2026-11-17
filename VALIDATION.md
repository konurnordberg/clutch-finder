# Validation

Local validation on November 17, 2026, Python 3.10, macOS ARM64.

- `HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q`: **10 passed**.
- Includes an actual Streamlit AppTest search, blank-query handling, real MiniLM inference and persisted Chroma retrieval, metadata filtering, chunk bounds/drive boundaries, HTML escaping, and recorded-outcome labels.
- Index: **1,948 plays / 1,948 chunks / 13 games**, 2024 NFL postseason. Raw source checksum and build details are in `chromadb/manifest.json`.
- Qualitative check: “An interception returned for a touchdown” returns the Eric Murray (LAC–HOU), Quan Martin (WAS–DET), and Cooper DeJean (KC–PHI) interception touchdowns as its first three hits. The source uses abbreviated player names; see original descriptions for identifiers.
- “A sack that forces a fumble” returns actual sack/fumble descriptions.
- These example queries informed the embedding weighting and are **not** an independent accuracy benchmark. Related-but-imperfect matches remain possible; the app makes no factual-confidence claim.
- Deployed to https://konur-clutch-finder.streamlit.app/ on Python 3.12.16. Verified a real search in the public browser UI: the three interception-return touchdowns appeared first, matching local results.
- Streamlit selected PyArrow 24.0.0 after flagging version 25.0.1; requirements now explicitly pin 24.0.0. All 10 tests passed again after aligning the local environment.
- Canvas submission has not been performed; submit the live app URL through the course account.
