import streamlit as st



# Set page config without the theme argument
st.set_page_config(
    page_title="Simulation Course Platform",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)


from utils import set_rtl
# Call the set_rtl function to apply RTL styles
set_rtl()
# Import all page functions
from home import show as show_home
from food_truck import show_food_truck
from sampling_methods import show_sampling_methods
from  goodness_of_fit import show as show_googness
from Random_Number_Generators import show_random_generator
from story import show_story

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
        "סיפור מערכת טאקו לוקו": show_story,
        "התאמת התפלגות למודל": show_googness,
        "אלגוריתמי דגימה": show_sampling_methods,
        "תכנות אירועים": show_food_truck,
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
