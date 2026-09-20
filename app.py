"""
app.py -- the shell for the whole course portfolio site.

The assignment says every module has to be reachable from one webpage, so
instead of writing a separate app per module I made this one page that
auto-discovers whatever lives in modules/. It scans modules/, imports every
moduleN.py it finds, reads a few metadata fields off each one, and builds the
sidebar from that. So dropping in module3.py, module4.py etc. later doesn't
require touching this file at all -- it just shows up.

Each modules/moduleN.py needs to define:
    NUMBER, TITLE, SUBTITLE, STATUS   -- shown in the sidebar / home cards
    render()                          -- draws the actual page

Run with:
    streamlit run app.py
"""
import importlib.util
import os
import sys
import traceback

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(HERE, 'modules')
sys.path.insert(0, HERE)

st.set_page_config(page_title='CSc 8830 — Computer Vision',
                   page_icon='📷', layout='wide')

STUDENT = 'Dhanush Nagarajan'
SCHOOL = 'Georgia State University'
COURSE = 'CSc 8830: Computer Vision'
REPO_URL = 'https://github.com/Dhanush-0017/Computer-Vision-CSc-8830'


# ----------------------------------------------------------------------------
# Module discovery
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def discover_modules():
    """Find every modules/moduleN/ folder and load its page.

    I look at the folder on disk rather than doing `import modules`, because
    "modules" is a generic name and on Streamlit Cloud something else had
    already claimed it, so the import found nothing and the sidebar came up
    empty. Loading each one straight from its file path can't be shadowed.

    If a module fails to load, only that one gets an error badge - the rest of
    the site still works.
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
        elif name.endswith('.py'):        # still works if a module is one file
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
                # "Reload modules" should pick up edits, and a plain import is
                # a no-op once it's already in sys.modules
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
            found.append({
                'number': num,
                'title': getattr(mod, 'TITLE', 'Module ' + str(num)),
                'subtitle': getattr(mod, 'SUBTITLE', ''),
                'status': getattr(mod, 'STATUS', 'scheduled').lower(),
                'render': getattr(mod, 'render', None),
                'error': None,
            })
        except Exception:
            sys.modules.pop('modules.' + mod_name, None)
            found.append({
                'number': int(digits) if digits else 999,
                'title': mod_name + ' (failed to load)',
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

    if not mods:
        # If this ever shows up it means discovery found nothing, so print
        # where it looked instead of just showing an empty sidebar.
        st.error('No modules were found.')
        st.write('Looked in:', MODULES_DIR)
        st.write('Exists:', os.path.isdir(MODULES_DIR))
        if os.path.isdir(MODULES_DIR):
            st.write('Contents:', sorted(os.listdir(MODULES_DIR)))
        st.write('Running from:', HERE)
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
    st.caption('Each module is a self-contained folder under `modules/`. '
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
if st.sidebar.button('🔄 Reload modules', width='stretch'):
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
