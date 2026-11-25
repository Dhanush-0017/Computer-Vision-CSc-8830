"""
app.py -- the shell for the whole web app.

Every module is reachable from one web page. This app finds whatever lives
in modules/ and builds the site from it: a sidebar with
every module, a home page with a card per module, and one page per module.
Adding a module means adding a folder; this file doesn't change.

Each modules/moduleN/__init__.py defines:
    NUMBER, TITLE, SUBTITLE, STATUS   -- shown in the sidebar and on the cards
    LABEL (optional)                  -- e.g. 'Modules 5 & 6' (default 'Module N')
    THUMBNAIL (optional)              -- picture for the home card
    RESULTS (optional)                -- [(label, value, note), ...] key numbers
    render()                          -- draws the module's own content

The look of the site is in ui.py.

Run with:
    streamlit run app.py
"""
import importlib
import importlib.util
import os
import sys
import traceback

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(HERE, 'modules')
sys.path.insert(0, HERE)

import ui                                                    # noqa: E402
importlib.reload(ui)        # so a new push shows up without restarting the server

st.set_page_config(page_title='Computer Vision',
                   layout='wide',
                   initial_sidebar_state='expanded')

STUDENT = 'Dhanush Nagarajan'
SCHOOL = 'Georgia State University'
REPO_URL = 'https://github.com/Dhanush-0017/Computer-Vision-CSc-8830'


# ----------------------------------------------------------------------------
# Module discovery
# ----------------------------------------------------------------------------
def folder_stamp():
    """Newest change time of any file under modules/. When a push changes a
    module, this number changes and the cached list below is rebuilt."""
    newest = 0
    for root, dirs, files in os.walk(MODULES_DIR):
        for name in files:
            if name.endswith('.py'):
                t = os.path.getmtime(os.path.join(root, name))
                if t > newest:
                    newest = t
    return newest


@st.cache_resource(show_spinner=False)
def discover_modules(stamp):
    """Find every modules/moduleN/ folder and load its page.

    Each one is loaded straight from its file path instead of `import modules`,
    because on Streamlit Cloud another package had already taken the name
    "modules" and the import found nothing. If a module fails to load, only
    that module shows an error; the rest of the site still works.
    """
    found = []
    if not os.path.isdir(MODULES_DIR):
        return found

    for name in sorted(os.listdir(MODULES_DIR)):
        if not name.startswith('module'):
            continue                      # skips _template.py and anything else

        path = os.path.join(MODULES_DIR, name)
        if os.path.isdir(path):
            init = os.path.join(path, '__init__.py')
            mod_name = name
        elif name.endswith('.py'):
            init = path
            mod_name = name[:-3]
        else:
            continue
        if not os.path.exists(init):
            continue

        digits = ''.join(ch for ch in mod_name if ch.isdigit())
        try:
            full_name = 'modules.' + mod_name
            if full_name in sys.modules:
                mod = importlib.reload(sys.modules[full_name])
            else:
                spec = importlib.util.spec_from_file_location(
                    full_name, init,
                    submodule_search_locations=[path] if os.path.isdir(path) else None)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[full_name] = mod
                spec.loader.exec_module(mod)

            num = getattr(mod, 'NUMBER', None)
            if num is None:
                num = int(digits) if digits else 999
            label = getattr(mod, 'LABEL', 'Module ' + str(num))
            thumb = getattr(mod, 'THUMBNAIL', None)
            if thumb is not None:
                thumb = os.path.join(path, thumb)
            found.append({
                'key': mod_name,
                'number': num,
                'label': label,
                'title': getattr(mod, 'TITLE', label),
                'subtitle': getattr(mod, 'SUBTITLE', ''),
                'status': getattr(mod, 'STATUS', 'scheduled').lower(),
                'thumbnail': thumb,
                'results': getattr(mod, 'RESULTS', []),
                'render': getattr(mod, 'render', None),
                'error': None,
            })
        except Exception:
            sys.modules.pop('modules.' + mod_name, None)
            found.append({
                'key': mod_name, 'number': int(digits) if digits else 999,
                'label': 'Module ' + digits,
                'title': mod_name + ' (failed to load)', 'subtitle': '',
                'status': 'error', 'thumbnail': None, 'results': [],
                'render': None, 'error': traceback.format_exc(),
            })

    return sorted(found, key=lambda m: m['number'])


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def go_to(label):
    st.session_state['nav'] = label


def home_page(mods, labels):
    ui.banner(REPO_URL)
    if not mods:
        # if this ever shows up, discovery found nothing: say where it looked
        st.error('No modules were found.')
        st.write('Looked in:', MODULES_DIR)
        st.write('Exists:', os.path.isdir(MODULES_DIR))
        return

    done = 0
    for m in mods:
        if m['status'] == 'complete':
            done = done + 1
    a, b, c = st.columns(3)
    a.metric('Modules', len(mods))
    b.metric('Complete', done)
    n_results = 0
    for m in mods:
        n_results = n_results + len(m['results'])
    c.metric('Key results', n_results)
    st.write('')

    # three cards per row, however many modules there are
    for row_start in range(0, len(mods), 3):
        cols = st.columns(3)
        for col, m in zip(cols, mods[row_start:row_start + 3]):
            with col:
                with st.container(border=True):
                    ui.card(m)
                    st.button('Open ' + m['label'], key='open_' + m['key'],
                              on_click=go_to,
                              args=(labels[mods.index(m) + 1],),
                              type='primary', width='stretch')


def module_page(m):
    ui.page_header(m)
    if m['error']:
        st.error(m['label'] + ' failed to load.')
        st.code(m['error'])
    elif m['render'] is None:
        st.error(m['label'] + ' has no render() function.')
    else:
        m['render']()


# ----------------------------------------------------------------------------
# Sidebar and main
# ----------------------------------------------------------------------------
ui.inject_css()
mods = discover_modules(folder_stamp())

labels = ['Home']
for m in mods:
    labels.append(m['label'] + ': ' + m['title'])

# a link like ?page=module7 opens that module directly
if 'nav' not in st.session_state or st.session_state['nav'] not in labels:
    st.session_state['nav'] = labels[0]
    wanted = st.query_params.get('page', '')
    for m in mods:
        if m['key'] == wanted:
            st.session_state['nav'] = labels[mods.index(m) + 1]

with st.sidebar:
    ui.sidebar_brand()
    choice = st.radio('Navigate', labels, key='nav', label_visibility='collapsed')
    ui.sidebar_footer(STUDENT, SCHOOL, REPO_URL)

if choice == labels[0]:
    st.query_params.clear()
    home_page(mods, labels)
else:
    m = mods[labels.index(choice) - 1]
    st.query_params['page'] = m['key']
    module_page(m)
