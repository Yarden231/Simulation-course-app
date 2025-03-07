import streamlit as st
# Set page config without the theme argument
st.set_page_config(
    page_title="Simulation Course Platform",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Import all page functions
from utils import set_rtl
from home import show as show_home
from  goodness_of_fit import show as show_googness
from story import show_story
from intro import show as show_intro
from sampling_methods_page import show_rng_demo

from event_simulation import show_simulation_page
from compare_alternatives import show_simulation_page as alternative_page






set_rtl()

def main():
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = 'דף הבית'

    # Sidebar navigation
    st.sidebar.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)

    # Define the available pages and their corresponding functions
    pages = {
        "דף הבית": show_home,
        "הקדמה לסיפור דוגמאת הקורס": show_intro,
        "סיפור מערכת טאקו לוקו": show_story,
        "התאמת התפלגות למודל": show_googness,
        "אלגוריתמי דגימה": show_rng_demo,
        "תכנות אירועים": show_simulation_page,
        "השוואה בין חלופות": alternative_page
    }

    # Add buttons for each page in the sidebar
    for page_name, page_func in pages.items():
        if st.sidebar.button(page_name):
            st.session_state.page = page_name

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    # Display the selected page's content
    pages[st.session_state.page]()

if __name__ == "__main__":
    main()
