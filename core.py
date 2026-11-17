"""Pure data preparation and retrieval helpers."""
import html
import math

MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
COLLECTION = 'nfl_plays'

def number(value, default=0):
    try:
        return float(value) if math.isfinite(float(value)) else default
    except (ValueError, TypeError):
        return default

def event_summary(row):
    """Deterministic labels from recorded outcomes, not generated claims."""
    labels = []
    for field, label in [('interception', 'interception'), ('return_touchdown', 'returned for a touchdown'),
                         ('pass_touchdown', 'passing touchdown'), ('rush_touchdown', 'rushing touchdown'),
                         ('qb_scramble', 'quarterback scrambling'), ('sack', 'quarterback sacked'),
                         ('fumble', 'fumble'), ('fumble_lost', 'fumble recovered by the opposing team'),
                         ('complete_pass', 'completed pass'), ('incomplete_pass', 'incomplete pass')]:
        if number(row.get(field)) == 1:
            labels.append(label)
    if row.get('field_goal_result') in ('made', 'missed', 'blocked'):
        labels.append('field goal ' + row['field_goal_result'])
    return '; '.join(labels) or str(row.get('play_type', 'play'))

def make_records(frame, tokenizer):
    """Chunk focal plays within drives; retain prior play as display-only context.

    Use 230-token windows with 30-token overlap, below the model's 256-token
    limit. Event labels come directly from the structured outcome flags.
    """
    records = []
    frame = frame.sort_values(['game_id', 'play_id'])
    previous = {}
    for row in frame.to_dict('records'):
        desc = row.get('desc')
        if not isinstance(desc, str) or not desc.strip() or row.get('play_type') in ('no_play', None):
            continue
        team = row.get('posteam')
        if not isinstance(team, str) or not team:
            continue
        game = str(row['game_id'])
        drive = int(number(row.get('fixed_drive', row.get('drive'))))
        quarter = int(number(row.get('qtr')))
        score = number(row.get('score_differential'))
        seconds = number(row.get('quarter_seconds_remaining'), 9999)
        heading = f"{row['away_team']} at {row['home_team']} | {row['game_date']} | {team} possession | Q{quarter} {row.get('time', '')}"
        focus = f"Play outcome: {event_summary(row)}.\nMatching play: {desc.strip()}\n{heading}"
        key = (game, drive, team)
        prior = previous.get(key, '')
        text = focus
        # Split rare long source descriptions instead of silently truncating.
        tokens = tokenizer.encode(text, add_special_tokens=False)
        windows = [tokens[i:i+230] for i in range(0, len(tokens), 200)]
        for part, window in enumerate(windows):
            metadata = {
                'game_id': game, 'play_id': int(number(row['play_id'])),
                'drive': drive, 'team': team, 'home_team': str(row['home_team']),
                'away_team': str(row['away_team']), 'date': str(row['game_date']),
                'quarter': quarter, 'clock': str(row.get('time', '')),
                'season': int(number(row['season'])), 'description': desc.strip(),
                'score_margin': score, 'previous_play': prior,
                'clutch': int(quarter >= 4 and seconds <= 300 and abs(score) <= 8),
            }
            records.append({'id': f'{game}_{metadata["play_id"]}_{part}',
                            'text': text if len(windows) == 1 else tokenizer.decode(window),
                            'metadata': metadata})
        previous[key] = desc.strip()
    return records

def filters(team='All teams', quarter='Any quarter', clutch=False):
    clauses = []
    if team != 'All teams':
        clauses.append({'team': {'$eq': team}})
    if quarter != 'Any quarter':
        clauses.append({'quarter': {'$eq': int(quarter)}})
    if clutch:
        clauses.append({'clutch': {'$eq': 1}})
    return {'$and': clauses} if len(clauses) > 1 else (clauses[0] if clauses else None)

def unique_hits(result, limit):
    hits, seen = [], set()
    for text, meta, distance in zip(result['documents'][0], result['metadatas'][0], result['distances'][0]):
        key = (meta['game_id'], meta['play_id'])
        if key not in seen:
            hits.append({'text': text, 'metadata': meta, 'similarity': 1 - distance})
            seen.add(key)
        if len(hits) == limit:
            break
    return hits

def render_hits(hits):
    if not hits:
        return '<div class="empty">No plays match these filters. Try another team or turn off late-game mode.</div>'
    cards = []
    for i, hit in enumerate(hits, 1):
        m = hit['metadata']
        esc = lambda v: html.escape(str(v))
        cards.append(f'''<article class="play-card">
        <div class="eyebrow">MATCH {i} · {esc(m['date'])} · COSINE SIMILARITY {hit['similarity']:.2f}</div>
        <h2>{esc(m['away_team'])} <span>at</span> {esc(m['home_team'])}</h2>
        <div class="tags">{esc(m['team'])} possession · Q{m['quarter']} · {esc(m['clock'])} · Score margin {m['score_margin']:+g}</div>
        <p>{esc(m['description'])}</p>
        <details><summary>View indexed passage &amp; source identifiers</summary><p>{esc(hit['text'])}</p>
        <p>Previous play in this drive: {esc(m.get('previous_play') or 'None')}</p><small>Game {esc(m['game_id'])} · Drive {m['drive']} · Play {m['play_id']}</small></details></article>''')
    return ''.join(cards)
