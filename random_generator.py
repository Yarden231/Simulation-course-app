import streamlit as st
from lcg import show_lcg
from lfsr import show_lfsr
from utils import set_ltr_sliders

def show_rng_demo():
    """Main function to display RNG demonstration."""
    # Add custom CSS for RTL support
    st.markdown("""
        <style>
            .rtl { direction: rtl; text-align: right; }
            .element-container { direction: rtl; }
            .stMarkdown { direction: rtl; }
        </style>
    """, unsafe_allow_html=True)
    
    set_ltr_sliders()
    
    # Title and introduction
    st.markdown("""
        <h1 class='rtl'>מחוללי מספרים אקראיים עבור סימולציית טאקו לוקו</h1>
        <div class='rtl'>
            <p>בסימולציית משאית המזון שלנו, אנחנו זקוקים למספרים אקראיים עבור מגוון החלטות:</p>
            <ul>
                <li>זמני הגעת לקוחות</li>
                <li>זמני הכנת מנות</li>
                <li>בחירת סוג המנה מהתפריט</li>
                <li>סבלנות הלקוחות וזמני המתנה</li>
            </ul>
            <p>לשם כך, נשתמש בשתי שיטות שונות לייצור מספרים פסאודו-אקראיים:</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different RNG methods
    tab1, tab2 = st.tabs(["מחולל קונגרואנטי לינארי (LCG)", "אוגר הזזה עם משוב לינארי (LFSR)"])
    
    with tab1:
        show_lcg()
    
    with tab2:
        show_lfsr()

if __name__ == "__main__":
    show_rng_demo()