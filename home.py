# home.py 
import streamlit as st

 


def show():
    # Apply custom CSS
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    # Path to the SVG file
    image_path  = "home.svg"


    # Display the image with full width
    st.markdown(
        f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh; overflow: hidden;">
            <img src="data:image/png;base64,{st.image(image_path, use_column_width=True)}" style="width: 20vw; height: auto;" />
        </div>
        """,
        unsafe_allow_html=True
    )




if __name__ == "__main__":
    show()