import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import set_rtl, set_ltr_sliders
import time

def show_sampling_methods():
    st.title("אלגוריתמי דגימה - מודלים סטטיסטיים למשאית טאקו לוקו")

    st.write("""
    ### מבוא לדגימה בסימולציה
    בשלב זה של הקורס, נלמד כיצד לדגום נתונים שישמשו אותנו בסימולציית משאית הטאקו. 
    
    ### למה אנחנו צריכים לדגום?
    במשאית הטאקו שלנו יש מספר תהליכים אקראיים:
    - 🕒 **זמני הגעת לקוחות** - לא ניתן לדעת בדיוק מתי יגיע הלקוח הבא
    - ⏱️ **זמני הכנת מנות** - משתנים בהתאם למורכבות ההזמנה
    - ⌛ **זמני המתנת לקוחות** - כל לקוח מוכן להמתין זמן שונה
    
    כדי לייצג תהליכים אלו בסימולציה, נשתמש בהתפלגויות סטטיסטיות שונות.
    """)

    if 'selected_sampling' not in st.session_state:
        st.session_state.selected_sampling = None

    num_samples = st.slider("מספר דגימות", min_value=1000, max_value=1000000, value=1000, step=1000)
    update_interval = st.slider("תדירות עדכון (מספר דגימות)", 100, 1000, 100)

    st.header("בחר שיטת דגימה")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("התפלגות אחידה\nזמני המתנת לקוחות"):
            st.session_state.selected_sampling = 'uniform'
            
    with col2:
        if st.button("התפלגות נורמלית\nזמני הכנת מנות"):
            st.session_state.selected_sampling = 'normal'
            
    with col3:
        if st.button("התפלגות מעריכית\nזמני הגעת לקוחות"):
            st.session_state.selected_sampling = 'exponential'

    if st.session_state.selected_sampling == 'uniform':
        st.header("התפלגות אחידה - מודל זמני המתנת לקוחות")
        st.write("""
        ### מהי התפלגות אחידה?
        התפלגות אחידה מייצגת מצב בו כל ערך בטווח מסוים הוא בעל סיכוי שווה להופיע.
        
        ### איך זה קשור למשאית הטאקו?
        במקרה שלנו, הלקוחות מוכנים להמתין בין 5 ל-20 דקות:
        - זמן המתנה מינימלי: 5 דקות
        - זמן המתנה מקסימלי: 20 דקות
        - כל זמן המתנה בטווח זה הוא אפשרי באופן שווה
        
        ### הפונקציה המתמטית:
        """)
        st.latex(r"f(x) = \frac{1}{b-a}, \quad a \leq x \leq b")
        
        a = st.slider("זמן המתנה מינימלי (דקות)", 0.0, 10.0, 5.0)
        b = st.slider("זמן המתנה מקסימלי (דקות)", a + 0.1, 30.0, 20.0)
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: np.ones_like(x) / (b - a)
        run_sampling(lambda size: sample_uniform(a, b, size), num_samples, update_interval, 
                    "התפלגות זמני המתנה", progress_bar, plot_placeholder, 
                    qqplot_placeholder, stats_placeholder, print_samples=True, 
                    true_density=true_density)

    elif st.session_state.selected_sampling == 'normal':
        st.header("התפלגות נורמלית - מודל זמני הכנת מנות")
        st.write("""
        ### מהי התפלגות נורמלית?
        התפלגות נורמלית (או גאוסיאנית) היא התפלגות פעמון המתארת תהליכים טבעיים רבים.
        
        ### איך זה קשור למשאית הטאקו?
        זמני ההכנה של טאקו לוקוסיטו:
        - ממוצע: 5 דקות
        - סטיית תקן: כדקה אחת
        - רוב ההכנות נמשכות בין 4-6 דקות
        
        ### הפונקציה המתמטית:
        """)
        st.latex(r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}")
        
        mu = st.slider("זמן הכנה ממוצע (דקות)", 1.0, 10.0, 5.0)
        sigma = st.slider("סטיית תקן (דקות)", 0.1, 3.0, 1.0)
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: stats.norm.pdf(x, mu, sigma)
        run_sampling(lambda size: sample_normal(mu, sigma, size), num_samples, update_interval, 
                    "התפלגות זמני הכנה", progress_bar, plot_placeholder, 
                    qqplot_placeholder, stats_placeholder, print_samples=True, 
                    true_density=true_density)

    elif st.session_state.selected_sampling == 'exponential':
        st.header("התפלגות מעריכית - מודל הגעת לקוחות")
        st.write("""
        ### מהי התפלגות מעריכית?
        התפלגות מעריכית מתארת את הזמן בין אירועים אקראיים עוקבים.
        
        ### איך זה קשור למשאית הטאקו?
        הגעת לקוחות למשאית:
        - ממוצע: 10 לקוחות בשעה
        - λ = 1/6 (בממוצע לקוח כל 6 דקות)
        - זמני ההגעה בין לקוחות הם בלתי תלויים
        
        ### הפונקציה המתמטית:
        """)
        st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
        
        lambda_param = st.slider("קצב הגעה (לקוחות לשעה)", 1.0, 20.0, 10.0)
        lambda_minutes = lambda_param / 60  # Convert to per-minute rate
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: lambda_minutes * np.exp(-lambda_minutes * x)
        run_sampling(lambda size: sample_exponential(lambda_minutes, size), 
                    num_samples, update_interval, "התפלגות זמני הגעה", 
                    progress_bar, plot_placeholder, qqplot_placeholder, 
                    stats_placeholder, print_samples=True, true_density=true_density)

    st.write("""
    ### 📚 קשר לחומר הקורס
    
    בקורס סימולציה אנו לומדים כיצד:
    1. **לזהות התפלגויות מתאימות** - התאמת מודל סטטיסטי לנתונים אמיתיים
    2. **לדגום מהתפלגויות** - שימוש באלגוריתמים ליצירת מספרים אקראיים
    3. **לבדוק את טיב ההתאמה** - שימוש במבחנים סטטיסטיים ובדיקות ויזואליות
    
    ### 🎯 יישום במשאית הטאקו
    
    הדגימות שלמדנו ישמשו אותנו ב:
    - חיזוי עומסים במשאית
    - תכנון כוח אדם אופטימלי
    - שיפור זמני המתנה
    - הערכת רווחיות
    """)

if __name__ == "__main__":
    show_sampling_methods()