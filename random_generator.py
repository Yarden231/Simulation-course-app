import streamlit as st
from lcg import show_lcg
from lfsr import show_lfsr
from utils import set_ltr_sliders
import numpy as np
from plotly.subplots import make_subplots
import scipy.stats as stats
import plotly.graph_objects as go

def create_distribution_plot(x_data, y_data, hist_data=None, title="התפלגות"):
    """Create a plot with both PDF and histogram if provided."""
    fig = make_subplots(rows=1, cols=1)
    
    # Add PDF line
    fig.add_trace(
        go.Scatter(x=x_data, y=y_data, mode='lines', name='PDF',
                  line=dict(color='blue', width=2))
    )
    
    # Add histogram if provided
    if hist_data is not None:
        fig.add_trace(
            go.Histogram(x=hist_data, histnorm='probability density',
                        name='היסטוגרמה', opacity=0.7,
                        marker_color='rgba(100, 100, 255, 0.5)')
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="ערך",
        yaxis_title="צפיפות",
        height=400,
        showlegend=True,
        title_x=0.5
    )
    
    return fig

def show_sampling_methods():
    st.markdown("""
        <h1 style='text-align: right; direction: rtl;'>שיטות דגימה</h1>
        <p style='text-align: right; direction: rtl;'>
        להלן הדגמה אינטראקטיבית של שיטות שונות לדגימת מספרים אקראיים מהתפלגויות שונות
        </p>
    """, unsafe_allow_html=True)
    
    method = st.selectbox(
        'בחר שיטת דגימה',
        ['טרנספורם הופכי', 'Box-Muller', 'קבלה-דחייה', 'קומפוזיציה'],
        key='sampling_method'
    )
    
    if method == 'טרנספורם הופכי':
        show_inverse_transform()
    elif method == 'Box-Muller':
        show_box_muller()
    elif method == 'קבלה-דחייה':
        show_acceptance_rejection()
    else:
        show_composition()

def show_inverse_transform():
    st.markdown("""
        <div style='text-align: right; direction: rtl;'>
            <h2>שיטת הטרנספורם ההופכי</h2>
            <p>שיטת הטרנספורם ההופכי היא שיטה ליצירת דגימות מהתפלגות כלשהי על סמך פונקציית ההתפלגות המצטברת (CDF) ההופכית . שיטה זו שימושית במיוחד עבור התפלגויות בהן יש ביטוי אנליטי לפונקציה המצטברת ההפוכה.</p>
            <p>מטרה: עבור כל התפלגות רציפה, נוכל ליצור דגימה אקראית התואמת את ההתפלגות הזו על ידי הפעלת פונקציית ה-CDF ההופכית על משתנה אקראי אחיד  U  בטווח [0, 1].</p>
            <h3>דוגמה: דגימה מהתפלגות מעריכית</h3>
            <p>ההתפלגות המעריכית מתארת אירועים שמתרחשים במרווחים של זמן, כמו זמני המתנה בין אירועים. פונקציית ה-CDF ההופכית של ההתפלגות המעריכית מאפשרת לדגום ערכים התואמים את צורת ההתפלגות.</p>
        </div>
    """, unsafe_allow_html=True)
    
    lambda_param = st.slider('פרמטר λ', min_value=0.1, max_value=5.0, value=1.0, step=0.1, key='lambda_exp')
    n_samples = st.slider('מספר דגימות', min_value=100, max_value=10000, value=1000, step=100, key='n_samples_exp')
    
    # Generate samples
    u = np.random.uniform(0, 1, n_samples)
    samples = -np.log(1 - u) / lambda_param
    
    # Create x range for PDF
    x = np.linspace(0, max(samples), 100)
    pdf = lambda_param * np.exp(-lambda_param * x)
    
    # Create plot
    fig = create_distribution_plot(x, pdf, samples, "התפלגות מעריכית")
    st.plotly_chart(fig, use_container_width=True)
    
    st.latex(r"X = F^{-1}(U) = -\frac{\ln(1-U)}{\lambda}")

def show_box_muller():
    st.markdown("""
        <div style='text-align: right; direction: rtl;'>
            <h2>שיטת בוקס-מולר</h2>
            <p>שיטת בוקס-מולר היא שיטה פופולרית להפקת דגימות מהתפלגות נורמלית (גאוסיאנית) תוך שימוש בשני משתנים אקראיים בלתי תלויים עם התפלגות אחידה.</p>
            <p>מטרה: שיטה זו ממירה שני משתנים אקראיים אחידים,  U_1  ו- U_2 , לשני משתנים אקראיים בלתי תלויים עם התפלגות נורמלית סטנדרטית.</p>
        </div>
    """, unsafe_allow_html=True)
    
    mu = st.slider('ממוצע (μ)', min_value=-5.0, max_value=5.0, value=0.0, step=0.1, key='mu_normal')
    sigma = st.slider('סטיית תקן (σ)', min_value=0.1, max_value=5.0, value=1.0, step=0.1, key='sigma_normal')
    n_samples = st.slider('מספר דגימות', min_value=100, max_value=10000, value=1000, step=100, key='n_samples_normal')
    
    # Generate samples using Box-Muller
    u1 = np.random.uniform(0, 1, n_samples)
    u2 = np.random.uniform(0, 1, n_samples)
    
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    samples = mu + sigma * z
    
    # Create x range for PDF
    x = np.linspace(min(samples), max(samples), 100)
    pdf = stats.norm.pdf(x, mu, sigma)
    
    # Create plot
    fig = create_distribution_plot(x, pdf, samples, "התפלגות נורמלית")
    st.plotly_chart(fig, use_container_width=True)
    
    st.latex(r"X = \mu + \sigma \sqrt{-2\ln(U_1)} \cos(2\pi U_2)")

def show_acceptance_rejection():
    st.markdown("""
        <div style='text-align: right; direction: rtl;'>
            <h2>שיטת קבלה-דחייה</h2>
            <p>שיטת קבלה-דחייה משמשת ליצירת דגימות מהתפלגויות מורכבות כאשר קשה לדגום ישירות.</p>
            <p>מטרה: בדוגמה זו, נבצע דגימה מהתפלגות בי-מודלית (התפלגות עם שתי פסגות), תוך שימוש בהתפלגות הצעתית קלה לדגימה ובעלת טווח רחב מההתפלגות המבוקשת.</p>
        </div>
    """, unsafe_allow_html=True)
    
    n_samples = st.slider('מספר דגימות', min_value=100, max_value=10000, value=1000, step=100, key='n_samples_ar')
    
    # Target distribution (bimodal)
    def target_pdf(x):
        return 0.5 * (stats.norm.pdf(x, -2, 0.5) + stats.norm.pdf(x, 2, 0.5))
    
    # Proposal distribution (uniform over range)
    x_range = np.linspace(-4, 4, 100)
    M = 0.5  # Maximum value of target PDF
    
    samples = []
    while len(samples) < n_samples:
        # Generate proposal
        x = np.random.uniform(-4, 4)
        u = np.random.uniform(0, M)
        
        # Accept/reject
        if u <= target_pdf(x):
            samples.append(x)
    
    samples = np.array(samples)
    pdf = target_pdf(x_range)
    
    # Create plot
    fig = create_distribution_plot(x_range, pdf, samples, "התפלגות בי-מודלית")
    st.plotly_chart(fig, use_container_width=True)

def show_composition():
    st.markdown("""
        <div style='text-align: right; direction: rtl;'>
            <h2>שיטת קומפוזיציה</h2>
            <p>שיטת קומפוזיציה משמשת לדגימה מתערובת של התפלגויות, כאשר כל התפלגות מייצגת חלק מההתפלגות הכוללת.</p>
            <p>מטרה: בדוגמה זו, יוצרים דגימות מתערובת של שתי התפלגויות נורמליות עם מרכזים שונים, כך שההתפלגות הכוללת היא ממוצע משוקלל של שתי ההתפלגויות.</p>
        </div>
    """, unsafe_allow_html=True)
    
    p = st.slider('משקל התפלגות ראשונה', min_value=0.0, max_value=1.0, value=0.5, step=0.1, key='weight')
    n_samples = st.slider('מספר דגימות', min_value=100, max_value=10000, value=1000, step=100, key='n_samples_comp')
    
    # Generate samples
    samples = []
    for _ in range(n_samples):
        if np.random.random() < p:
            samples.append(np.random.normal(-2, 0.5))
        else:
            samples.append(np.random.normal(2, 0.5))
    
    samples = np.array(samples)
    
    # Create x range for PDF
    x = np.linspace(min(samples), max(samples), 100)
    pdf = p * stats.norm.pdf(x, -2, 0.5) + (1-p) * stats.norm.pdf(x, 2, 0.5)
    
    # Create plot
    fig = create_distribution_plot(x, pdf, samples, "תערובת התפלגויות")
    st.plotly_chart(fig, use_container_width=True)
    
    st.latex(r"f(x) = p \cdot \mathcal{N}(x; \mu_1, \sigma_1^2) + (1-p) \cdot \mathcal{N}(x; \mu_2, \sigma_2^2)")


def show_rng_demo():
        # Apply custom CSS
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    set_ltr_sliders()

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
            <p>בעמוד זה נראה כיצד ניתן לייצר מספרים אקראיים בין 0 ל- 1, ולאחר מכן להשתמש במספרים אלו כדי לייצר מספרים אקראיים שעוקבים אחר התפלגויות מסובכות יותר.</p>
            <h4>כדי לדגום מספרים המתפלגים אחיד בין 0-1, נשתמש בשתי שיטות שונות לייצור מספרים פסאודו-אקראיים:</h4>
        </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different RNG methods
    tab1, tab2 = st.tabs(["מחולל קונגרואנטי לינארי (LCG)", "רגיסטר הזזה עם משוב לינארי (LFSR)"])
    
    with tab1:
        show_lcg()
    
    with tab2:
        show_lfsr()

    
    # Additional tabs for sampling methods
    st.markdown("<div class='rtl'><h2>שיטות דגימה</h2></div>", unsafe_allow_html=True)
    st.markdown("<div class='rtl'><p>כל שיטות הדגימה שנלמד משתמשות במספרים אקראיים בטווח 0-1 כבסיס להפקת דגימות מהתפלגויות שונות, כאשר כל שיטה ממירה את המספרים האקראיים בדרך שונה ובהתאם להתפלגות היעד.</p></div>", unsafe_allow_html=True)
     # Create tabs with improved content
    tab3, tab4, tab5, tab6 = st.tabs([
        "טרנספורם הופכי",
        "box muller",
        "קומפוזיציה",
        "קבלה-דחייה"
    ])
    
    with tab3:
        show_inverse_transform()
    
    with tab4:
        show_box_muller()
    
    with tab6:
        show_acceptance_rejection()

    with tab5:
        show_composition()

if __name__ == "__main__":
    show_rng_demo()