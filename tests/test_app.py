from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]

def test_streamlit_search_flow():
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    assert not app.exception
    assert len(app.selectbox) == 2
    app.text_input[0].set_value('An interception returned for a touchdown')
    app.button[-1].click().run()
    assert not app.exception
    assert any('INTERCEPTED' in m.value and 'TOUCHDOWN' in m.value for m in app.markdown)
    app.text_input[0].set_value('')
    app.button[-1].click().run()
    assert any('Describe a play' in m.value for m in app.markdown)
