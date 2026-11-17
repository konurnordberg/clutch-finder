"""Streamlit frontend for Clutch Finder."""
import json
from pathlib import Path
import streamlit as st
from engine import search

ROOT = Path(__file__).resolve().parent

def main():
    st.set_page_config(page_title='Clutch Finder', page_icon='🏈', layout='wide')
    st.markdown('<style>' + (ROOT / 'style.css').read_text() + '</style>', unsafe_allow_html=True)
    manifest_path = ROOT / 'chromadb/manifest.json'
    if not manifest_path.exists():
        st.error('The search index is missing. Run python build_index.py before starting the app.')
        st.stop()
    m = json.loads(manifest_path.read_text())
    label = {'POST':'playoffs', 'REG':'regular season', 'ALL':'regular season + playoffs'}[m['season_type']]
    st.markdown(f'''<header class="hero"><div class="eyebrow">NFL PLAY SEARCH · {m['season']} SEASON {label.upper()}</div>
<h1>Remember the moment.<br>Find the play.</h1>
<p>Describe the action in your own words. Explore real NFL plays, with the game context behind every match.</p>
<div class="eyebrow">{m['games']} GAMES / {m['plays']:,} PLAYS / CLUTCH FINDER</div></header>''', unsafe_allow_html=True)
    examples = ['An interception returned for a touchdown', 'A quarterback scrambling', 'A sack that forces a fumble']
    st.caption('TRY A SEARCH')
    for col, example in zip(st.columns(3), examples):
        if col.button(example, use_container_width=True):
            st.session_state['query'] = example
    with st.form('search_form'):
        query = st.text_input('What play are you looking for?', key='query', max_chars=1000,
                              placeholder='A quarterback escapes pressure and runs downfield')
        a,b,c = st.columns([2,2,1])
        team = a.selectbox('Possession team', ['All teams'] + m['teams'])
        quarter = b.selectbox('Quarter (5 = overtime)', ['Any quarter','1','2','3','4','5'])
        count = c.slider('Results',1,10,5)
        clutch = st.checkbox('Late & close: final 5 minutes of Q4 or OT, within 8 points before the play')
        submitted = st.form_submit_button('Find the play →', type='primary', use_container_width=True)
    if submitted:
        try:
            with st.spinner('Finding matching plays… First search loads the public embedding model.'):
                st.session_state['results'] = search(query,team,quarter,clutch,count)
                st.session_state['last_query'] = query
        except Exception:
            st.error('Search could not load. Check the server logs and network access, then try again.')
            import logging
            logging.exception('Search failed')
            st.session_state.pop('results',None)
    if 'results' in st.session_state:
        st.caption('RESULTS FOR: ' + st.session_state.get('last_query',''))
        st.markdown(st.session_state['results'],unsafe_allow_html=True)
    else:
        st.info('Describe a play above, or choose an example and press Find the play.')
    st.divider()
    st.caption('Matches are ranked by semantic similarity, not certainty. Filters apply to the matching play. Score margin is from the possession team’s perspective before the play. This is text search, not video search.')
    st.markdown('Data: [nflverse](https://github.com/nflverse/nflverse-data) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · No affiliation with the NFL.')

if __name__ == '__main__':
    main()
