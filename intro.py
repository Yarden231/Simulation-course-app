# home.py 
import streamlit as st

def show():
    # Apply custom CSS
    """
    Display the intro page with a custom CSS and an SVG image.
    
    """
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Path to the SVG or PNG file
    image_path = "intro.svg"  # or change to "/mnt/data/image.png" if using PNG

    # Display the image directly with Streamlit
    st.image(image_path, use_column_width=True)


if __name__ == "__main__":
    show()