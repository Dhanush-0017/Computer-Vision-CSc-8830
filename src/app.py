"""
================================================================================
app.py  --  CSc 8830 Computer Vision — Course Web Application
================================================================================
Single web application hosting EVERY assignment for the course, as required by
the submission instructions ("all assignments must be accessible via that
webpage").

ARCHITECTURE — modules are AUTO-DISCOVERED.
    app.py does not hardcode a list of modules. On startup it scans the
    modules/ package, imports every file named moduleN.py, reads its metadata,
    and builds the navigation automatically. Scales to any number of modules
    with no change to this file.

TO ADD A NEW MODULE (e.g. Module 7):
    1. cp modules/_template.py modules/module7.py
    2. edit the NUMBER / TITLE / SUBTITLE / STATUS constants at the top
    3. write the render() function
    4. reload the browser — it appears in the sidebar automatically
    There is NOTHING to edit in app.py.

MODULE CONTRACT — each modules/moduleN.py must define:
    NUMBER   : int          assignment number (used for ordering)
    TITLE    : str          short name shown in the sidebar
    SUBTITLE : str          one-line description for the home page
    STATUS   : str          'complete' | 'in progress' | 'scheduled'
    render() : callable     draws the page with Streamlit

How to run:
    cd src
    streamlit run app.py
Then open http://localhost:8501 in a browser.
================================================================================
"""
import os
import sys
import pkgutil
import importlib
import traceback

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title='CSc 8830 — Computer Vision',
                   page_icon='📷', layout='wide')

STUDENT = 'Dhanush Nagarajan'
SCHOOL = 'Georgia State University'
COURSE = 'CSc 8830: Computer Vision'
REPO_URL = ''   # <- paste your GitHub URL here; shown on the home page


# ----------------------------------------------------------------------------
# Module discovery
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def discover_modules():
    """Import every modules/moduleN.py and collect its metadata.

    Returns a list of dicts sorted by module number. Import failures are
    captured rather than raised, so one broken module cannot take down the
    whole site — it appears in the sidebar with an error badge instead.
    """
    import modules as pkg
    found = []

    for info in pkgutil.iter_modules(pkg.__path__):
        name = info.name
        if not name.startswith('module'):
            continue                      # skips _template.py, helpers, etc.
        try:
            full_name = 'modules.' + name
            if full_name in sys.modules:
                mod = importlib.reload(sys.modules[full_name])
            else:
                mod = importlib.import_module(full_name)
            num = getattr(mod, 'NUMBER', None)
            if num is None:               # fall back to digits in the filename
                digits = ''.join(ch for ch in name if ch.isdigit())
                num = int(digits) if digits else 999
            found.append({
                'number': num,
                'title': getattr(mod, 'TITLE', 'Module ' + str(num)),
                'subtitle': getattr(mod, 'SUBTITLE', ''),
                'status': getattr(mod, 'STATUS', 'scheduled').lower(),
                'render': getattr(mod, 'render', None),
                'error': None,
            })
        except Exception:
            digits = ''.join(ch for ch in name if ch.isdigit())
            found.append({
                'number': int(digits) if digits else 999,
                'title': name + ' (failed to load)',
                'subtitle': '', 'status': 'error', 'render': None,
                'error': traceback.format_exc(),
            })

    return sorted(found, key=lambda m: m['number'])


BADGE = {'complete': '🟢', 'in progress': '🟡',
         'scheduled': '⚪', 'error': '🔴'}


# ----------------------------------------------------------------------------
# Home page
# ----------------------------------------------------------------------------
def home(mods):
    st.title(COURSE)
    st.subheader('Assignment Portfolio — ' + STUDENT)
    st.caption(SCHOOL)
    if REPO_URL:
        st.markdown('[Source code on GitHub](' + REPO_URL + ')')
    st.divider()

    st.markdown('This web application hosts every assignment for the course. '
                'Select a module from the sidebar.')
    st.write('')

    done = sum(1 for m in mods if m['status'] == 'complete')
    a, b, c = st.columns(3)
    a.metric('Modules published', len(mods))
    b.metric('Complete', done)
    c.metric('In progress / scheduled', len(mods) - done)
    st.write('')

    # three cards per row, wrapping automatically however many modules exist
    for row_start in range(0, len(mods), 3):
        cols = st.columns(3)
        for col, m in zip(cols, mods[row_start:row_start + 3]):
            with col:
                with st.container(border=True):
                    st.markdown('##### ' + BADGE.get(m['status'], '⚪') +
                                ' Module ' + str(m['number']))
                    st.markdown('**' + m['title'] + '**')
                    st.caption(m['subtitle'])
                    st.caption('_' + m['status'].title() + '_')

    st.divider()
    st.caption('Each module is a self-contained page under `src/modules/`. '
               'The navigation is generated automatically from the files '
               'present, so new assignments drop in without modifying the app.')


# ----------------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------------
mods = discover_modules()

st.sidebar.title('CSc 8830')
st.sidebar.caption('Computer Vision — Assignment Portfolio')

labels = ['🏠  Home'] + [
    BADGE.get(m['status'], '⚪') + '  Module ' + str(m['number']) + ' — ' + m['title']
    for m in mods]

choice = st.sidebar.radio('Navigate', labels, label_visibility='collapsed')

st.sidebar.divider()
if st.sidebar.button('🔄 Reload modules', use_container_width=True):
    discover_modules.clear()
    st.rerun()
st.sidebar.caption(STUDENT + '  \n' + SCHOOL)

if choice == labels[0]:
    home(mods)
else:
    m = mods[labels.index(choice) - 1]
    if m['error']:
        st.error('Module ' + str(m['number']) + ' failed to import.')
        st.code(m['error'])
    elif m['render'] is None:
        st.error('Module ' + str(m['number']) + ' has no render() function.')
    else:
        m['render']()
