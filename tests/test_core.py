import pandas as pd
from core import make_records, filters, unique_hits, render_hits

class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split()
    def decode(self, tokens):
        return ' '.join(tokens)

def row(play=1, **changes):
    base = dict(game_id='2024_01_A_B', play_id=play, fixed_drive=1, desc='Runner gains five yards.',
                play_type='run', posteam='A', away_team='A', home_team='B', game_date='2025-01-01',
                qtr=4, time='02:00', quarter_seconds_remaining=120, score_differential=-7, season=2024)
    return dict(base, **changes)

def test_context_stays_in_drive_and_game():
    records = make_records(pd.DataFrame([row(1), row(2), row(3, fixed_drive=2), row(4, game_id='2024_02_A_B')]), Tokenizer())
    assert not records[0]['metadata']['previous_play']
    assert records[1]['metadata']['previous_play']
    assert not records[2]['metadata']['previous_play']
    assert not records[3]['metadata']['previous_play']

def test_exact_clutch_boundaries():
    frame = pd.DataFrame([row(1, quarter_seconds_remaining=300, score_differential=8),
                          row(2, quarter_seconds_remaining=301), row(3, score_differential=-9),
                          row(4, qtr=3), row(5, qtr=5)])
    assert [r['metadata']['clutch'] for r in make_records(frame, Tokenizer())] == [1,0,0,0,1]

def test_long_descriptions_preserved_in_bounded_windows():
    text = ' '.join(f'word{i}' for i in range(600))
    records = make_records(pd.DataFrame([row(desc=text)]), Tokenizer())
    assert all(len(r['text'].split()) <= 230 for r in records)
    assert 'word599' in records[-1]['text']
    assert len({r['id'] for r in records}) == len(records)

def test_filters():
    assert filters() is None
    assert filters('KC') == {'team': {'$eq': 'KC'}}
    assert len(filters('KC', '4', True)['$and']) == 3

def test_no_play_and_missing_team_removed():
    records = make_records(pd.DataFrame([row(1, play_type='no_play'), row(2, posteam=None), row(3)]), Tokenizer())
    assert len(records) == 1

def test_deduplication_and_html_escaping():
    meta = make_records(pd.DataFrame([row(desc='<script>alert(1)</script>')]), Tokenizer())[0]['metadata']
    result = {'documents': [['a','b']], 'metadatas': [[meta,meta]], 'distances': [[.1,.2]]}
    hits = unique_hits(result,5)
    assert len(hits) == 1
    assert '<script>' not in render_hits(hits)
    assert '&lt;script&gt;' in render_hits(hits)


def test_event_labels_use_recorded_outcomes():
    from core import event_summary
    summary = event_summary({'interception': 1, 'return_touchdown': 1, 'sack': 0})
    assert 'interception' in summary and 'returned for a touchdown' in summary
    assert 'sacked' not in summary
