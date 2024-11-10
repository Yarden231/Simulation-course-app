import streamlit as st
from lcg import show_lcg
from lfsr import show_lfsr
from utils import set_ltr_sliders
import numpy as np
from plotly.subplots import make_subplots
import scipy.stats as stats
import plotly.graph_objects as go

def show_code_with_explanation(title,  code):
    # Display Hebrew title with RTL
    st.markdown(f"<h3 style='text-align: right;'>{title}</h3>", unsafe_allow_html=True)
    
    # Create a container div that forces LTR for code
    st.markdown("""
        <style>
            .ltr-code {
                direction: ltr !important;
                text-align: left !important;
                unicode-bidi: bidi-override;
            }
            .ltr-code * {
                direction: ltr !important;
                text-align: left !important;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Wrap code in LTR container
    with st.container():
        st.code(code, language="python")

def create_distribution_plot(x_data, y_data, hist_data=None, title="התפלגות", bins=100):
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
                        name='היסטוגרמה', opacity=0.7, nbinsx=bins,
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
    col1, col_space, col2 = st.columns([5,1,5])

    with col1:
        st.markdown("""
            <div style='text-align: right; direction: rtl;'>
                <h2>שיטת הטרנספורם ההופכי</h2>
                <p>שיטת הטרנספורם ההופכי היא שיטה ליצירת דגימות מהתפלגות כלשהי על סמך פונקציית ההתפלגות המצטברת (CDF) ההופכית . שיטה זו שימושית במיוחד עבור התפלגויות בהן יש ביטוי אנליטי לפונקציה המצטברת ההפוכה.</p>
                <p>מטרה: עבור כל התפלגות רציפה, נוכל ליצור דגימה אקראית התואמת את ההתפלגות הזו על ידי הפעלת פונקציית ה-CDF ההופכית על משתנה אקראי אחיד  U  בטווח [0, 1].</p>
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
    

    with col2:
        # Create plot
        fig = create_distribution_plot(x, pdf, samples, "התפלגות מעריכית")
        st.plotly_chart(fig, use_container_width=True)
        
        st.latex(r"X = F^{-1}(U) = -\frac{\ln(1-U)}{\lambda}")

def show_box_muller():

    col1, col_space, col2 = st.columns([5,1,5])
    with col1:
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

    with col2: 
        # Create plot
        fig = create_distribution_plot(x, pdf, samples, "התפלגות נורמלית")
        st.plotly_chart(fig, use_container_width=True)
        
        st.latex(r"X = \mu + \sigma \sqrt{-2\ln(U_1)} \cos(2\pi U_2)")

def show_acceptance_rejection():

    col1, col_space, col2 = st.columns([5,1,5])
    with col1:
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
    with col2:
        # Create plot
        fig = create_distribution_plot(x_range, pdf, samples, "התפלגות בי-מודלית")
        st.plotly_chart(fig, use_container_width=True)
        
        st.latex(r"f(x) = p \cdot \mathcal{N}(x; \mu_1, \sigma_1^2) + (1-p) \cdot \mathcal{N}(x; \mu_2, \sigma_2^2)")

def show_composition():

    col1, col_space, col2 = st.columns([5,0.5,5])
    with col1:
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
    
    with col2:
        # Create plot
        fig = create_distribution_plot(x, pdf, samples, "תערובת התפלגויות")
        st.plotly_chart(fig, use_container_width=True)
        
        st.latex(r"f(x) = p \cdot \mathcal{N}(x; \mu_1, \sigma_1^2) + (1-p) \cdot \mathcal{N}(x; \mu_2, \sigma_2^2)")



def show_order_sampling():

    st.markdown("""
        <div style='text-align: center;'>
            <h1>3. דגימת זמני הזמנות</h1>
        </div>
    """, unsafe_allow_html=True)
    st.text(" ")
    st.text(" ")

    col_intro, col_space, col_sam = st.columns([5,1,5])

    with col_intro:
        st.markdown("""
            <div style='text-align: right; direction: rtl;'>
                <p>בדוגמה זו נדגום זמני הזמנה עבור שלושה סוגי לקוחות:</p>
                <ul>
                    <li>סוג א' (50%): זמן הזמנה אחיד בין 3-4 דקות - מתאים ללקוחות המזמינים מנות פשוטות ומהירות</li>
                    <li>סוג ב' (25%): זמן הזמנה משולש בין 4-6 דקות - מתאים ללקוחות המתלבטים או מזמינים מנות מורכבות יותר</li>
                    <li>סוג ג' (25%): זמן הזמנה קבוע של 10 דקות - מתאים להזמנות גדולות או מיוחדות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Add explanation of the distribution shape
        st.markdown("""
            <div style='text-align: right; direction: rtl;'>
                <h3>מבנה ההתפלגות הרצויה</h3>
                <p>ההתפלגות המתקבלת היא התפלגות מעורבת המורכבת משלושה חלקים:</p>
                <ul>
                    <li>חלק אחיד בין 3-4 דקות עם גובה 0.5 (50% מהלקוחות)</li>
                    <li>חלק משולש בין 4-6 דקות עם שיא ב-5 דקות (25% מהלקוחות)</li>
                    <li>נקודת מסה ב-10 דקות המייצגת 25% מהלקוחות</li>
                </ul>
                <p>להלן צורת ההתפלגות התיאורטית:</p>
            </div>
        """, unsafe_allow_html=True)

        # Create theoretical distribution plot
        def theoretical_pdf(x):
            if isinstance(x, np.ndarray):
                pdf = np.zeros_like(x, dtype=float)
                # Uniform part (3-4 minutes)
                mask1 = (3 <= x) & (x < 4)
                pdf[mask1] = 0.5
                
                # Triangular part (4-6 minutes)
                mask2 = (4 <= x) & (x < 5)
                mask3 = (5 <= x) & (x < 6)
                pdf[mask2] = (x[mask2] - 4) / 4  # Rising part of triangle
                pdf[mask3] = (6 - x[mask3]) / 4  # Falling part of triangle
                
                # Point mass at 10 minutes
                mask4 = np.isclose(x, 10)
                pdf[mask4] = 0.25
                
                return pdf
            else:
                if 3 <= x < 4:
                    return 0.5
                elif 4 <= x < 5:
                    return (x - 4) / 4
                elif 5 <= x < 6:
                    return (6 - x) / 4
                elif x == 10:
                    return 0.25
                else:
                    return 0

        # Create x range for plot
        x_range = np.linspace(2, 11, 1000)
        pdf_values = theoretical_pdf(x_range)

        # Create plot using plotly
        fig = go.Figure()
        
        # Add PDF line
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=pdf_values,
                mode='lines',
                name='פונקציית צפיפות',
                line=dict(color='blue', width=2)
            )
        )
        
        # Add point mass at x=10
        fig.add_trace(
            go.Scatter(
                x=[10],
                y=[0.25],
                mode='markers',
                name='נקודת מסה (10 דקות)',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='diamond'
                )
            )
        )

        # Update layout
        fig.update_layout(
            title='התפלגות זמני הזמנה תיאורטית',
            xaxis_title='זמן (דקות)',
            yaxis_title='צפיפות',
            height=400,
            showlegend=True,
            title_x=0.5,
            annotations=[
                dict(
                    x=3.5,
                    y=0.55,
                    xref="x",
                    yref="y",
                    text="התפלגות אחידה<br>50% מהלקוחות",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40
                ),
                dict(
                    x=5,
                    y=0.3,
                    xref="x",
                    yref="y",
                    text="התפלגות משולשת<br>25% מהלקוחות",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40
                ),
                dict(
                    x=10,
                    y=0.3,
                    xref="x",
                    yref="y",
                    text="25% מהלקוחות",
                    showarrow=True,
                    arrowhead=2,
                    ax=0,
                    ay=-40
                )
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

        # Add explanation about the sampling methods
        st.markdown("""
            <div style='text-align: right; direction: rtl;'>
                <h3>שיטות דגימה</h3>
                <p>נציג שלוש שיטות שונות לדגימת זמני הזמנה מההתפלגות הזו:</p>
                <ol>
                    <li><strong>טרנספורם הופכי:</strong> משתמש בפונקציית ההתפלגות המצטברת ההופכית</li>
                    <li><strong>קבלה-דחייה:</strong> שימוש בפונקציית עטיפה פשוטה לדגימה</li>
                    <li><strong>קומפוזיציה:</strong> דגימה מכל אחת מההתפלגויות בנפרד לפי המשקל שלה</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)

    with col_sam:

        # Continue with the existing sampling method selection and implementation...
        sampling_method = st.selectbox(
            'בחר שיטת דגימה',
            ['טרנספורם הופכי', 'קבלה-דחייה', 'קומפוזיציה'],
            key='order_sampling_method'
        )
        n_samples = st.slider('מספר דגימות', min_value=10000, max_value=100000, value=10000, step=1000, key='n_samples_order')


        if sampling_method == 'טרנספורם הופכי':
            samples = sample_inverse_transform_order(n_samples)
            
            inverse_transform_code = '''def transpose():
        u = random.uniform(0, 1)
        if 0 <= u < 0.5:
            x = 2 * u + 3
        elif 0.5 <= u < 0.625:
            x = (8 + math.sqrt(32 * u - 16)) / 2
        elif 0.625 <= u < 0.75:
            x = (12 + math.sqrt(24 - 32 * u)) / 2
        elif 0.75 <= u <= 1:
            x = 10
        return x'''
            
            show_code_with_explanation(
                "טרנספורם הופכי לדגימת זמני הזמנה",
                inverse_transform_code
            )
        
        elif sampling_method == 'קבלה-דחייה':
            samples = sample_rejection_order(n_samples)
            
            rejection_code = '''def f(x):
        if 3 <= x < 4:
            return 0.5
        elif 4 <= x < 5:
            return (x - 4) / 4
        elif 5 <= x < 6:
            return (6 - x) / 4
        elif x == 10:
            return 0.25
        else:
            return 0

    def rejection_sample():
        while True:
            y = 7 * random.uniform(0, 1) + 3
            u = random.uniform(0, 1)
            if u <= f(y) / 0.5:
                return y'''
            
            show_code_with_explanation(
                "דגימת קבלה-דחייה לזמני הזמנה",
                rejection_code
            )
        
        else:  # Composition
            samples = sample_composition_order(n_samples)
            
            composition_code = '''def composition():
    u1 = random.uniform(0, 1)
    if 0 <= u1 < 0.5:
        # Type A: Uniform between 3 and 4
        x = random.uniform(3, 4)
    elif 0.5 <= u1 < 0.75:
        # Type B: Triangular between 4 and 6
        x = random.triangular(4, 5, 6)
    else:
        # Type C: Fixed 10 minutes
        x = 10
    return x'''
            
            show_code_with_explanation(
                "שיטת הקומפוזיציה לזמני הזמנה",
                composition_code
            )





        # Create empirical PDF and theoretical PDF
        x_range = np.linspace(2, 11, 100)
        
        def theoretical_pdf(x):
            if isinstance(x, np.ndarray):
                pdf = np.zeros_like(x, dtype=float)
                mask1 = (3 <= x) & (x < 4)
                mask2 = (4 <= x) & (x < 5)
                mask3 = (5 <= x) & (x < 6)
                mask4 = np.isclose(x, 10)
                
                pdf[mask1] = 0.5
                pdf[mask2] = (x[mask2] - 4) / 4
                pdf[mask3] = (6 - x[mask3]) / 4
                pdf[mask4] = 0.25
                return pdf
            else:
                if 3 <= x < 4:
                    return 0.5
                elif 4 <= x < 5:
                    return (x - 4) / 4
                elif 5 <= x < 6:
                    return (6 - x) / 4
                elif x == 10:
                    return 0.25
                else:
                    return 0

        pdf = theoretical_pdf(x_range)
        
        # Create plot using plotly
        fig = create_distribution_plot(x_range, pdf, samples, "התפלגות זמני הזמנה")
        st.plotly_chart(fig, use_container_width=True)

        # Display statistics in cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
                <div style='text-align: right; direction: rtl; padding: 1rem; border-radius: 4px;'>
                    <h4>מדדי מרכז</h4>
                    <p>ממוצע: {np.mean(samples):.2f}</p>
                    <p>חציון: {np.median(samples):.2f}</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
                <div style='text-align: right; direction: rtl; padding: 1rem; border-radius: 4px;'>
                    <h4>מדדי פיזור</h4>
                    <p>סטיית תקן: {np.std(samples):.2f}</p>
                    <p>טווח: [{np.min(samples):.2f}, {np.max(samples):.2f}]</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
                <div style='text-align: right; direction: rtl; padding: 1rem; border-radius: 4px;'>
                    <h4>מדדי צורה</h4>
                    <p>אסימטריה: {stats.skew(samples):.2f}</p>
                    <p>קורטוזיס: {stats.kurtosis(samples):.2f}</p>
                </div>
            """, unsafe_allow_html=True)

def sample_inverse_transform_order(n):
    samples = []
    for _ in range(n):
        u = np.random.uniform(0, 1)
        if 0 <= u < 0.5:
            x = 2 * u + 3
        elif 0.5 <= u < 0.625:
            x = (8 + np.sqrt(32 * u - 16)) / 2
        elif 0.625 <= u < 0.75:
            x = (12 - np.sqrt(24 - 32 * u)) / 2
        elif 0.75 <= u <= 1:
            x = 10
        samples.append(x)
    return np.array(samples)

def sample_rejection_order(n):
    def f(x):
        if 3 <= x < 4:
            return 0.5
        elif 4 <= x < 5:
            return (x - 4) / 4
        elif 5 <= x < 6:
            return (6 - x) / 4
        elif 9.9 <= x < 10.1:
            return 0.25
        else:
            return 0

    samples = []
    while len(samples) < n:
        y = 7 * np.random.uniform(0, 1) + 3
        u = np.random.uniform(0, 1)
        if u <= f(y) / 0.5:
            samples.append(y)
    return np.array(samples)

def sample_composition_order(n):
    samples = []
    for _ in range(n):
        u1 = np.random.uniform(0, 1)
        if 0 <= u1 < 0.5:
            x = np.random.uniform(3, 4)
        elif 0.5 <= u1 < 0.75:
            x = np.random.triangular(4, 5, 6)
        else:
            x = 10
        samples.append(x)

    return np.array(samples)

def show_rng_demo():
        # Apply custom CSS
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    set_ltr_sliders()

    st.markdown("""
        <style>
            .rtl { direction: rtl; text-align: right; }
            .element-container { direction: rtl; }
            .stMarkdown { direction: rtl; }
        </style>
    """, unsafe_allow_html=True)
    
    
    # Main title
    st.markdown("<h1 style='text-align: right; '>מחוללי מספרים אקראיים ואלגוריתמי דגימה</h1>", unsafe_allow_html=True)
    

    st.markdown("""
        <div dir='rtl'>
            <h6>
                בעמוד זה נלמד כיצד לייצר מספרים אקראיים בטווח [0,1] ולהשתמש בהם כדי ליצור משתנים אקראיים מהתפלגויות שונות.
                תהליך זה הכרחי עבור סימולציית משאית המזון שלנו, שכן הוא מאפשר לנו לדמות תרחישים אקראיים כמו זמני הגעת לקוחות וזמני הכנת מנות.
            </h6>
        </div>
    """, unsafe_allow_html=True)


    st.markdown("""
        <div dir='rtl'>
            <h3>תהליך העבודה</h3>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div dir='rtl'>
                <h4>שלב 1: יצירת מספרים אקראיים בסיסיים</h4>
                <ul>
                    <li>מחולל קונגרואנטי לינארי (LCG)</li>
                    <li>רגיסטר הזזה עם משוב לינארי (LFSR)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div dir='rtl'>
                <h4>שלב 2: המרה להתפלגויות רצויות</h4>
                <ul>
                    <li>שיטת הטרנספורם ההופכי</li>
                    <li>שיטת Box-Muller</li>
                    <li>שיטת הקומפוזיציה</li>
                    <li>שיטת קבלה-דחייה</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div dir='rtl'>
                <h4>שלב 3: יישום מעשי</h4>
                <ul>
                    <li>דגימת זמני הזמנה</li>
                    <li>דוגמאות מעשיות לכל שיטה</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    

    with st.expander("1. מחוללי מספרים אקראיים ", expanded=True):

        # Title and introduction
        st.markdown("""
            <h2 class='rtl'>1. מחוללי מספרים אקראיים עבור סימולציית טאקו לוקו</h2>
            <div class='rtl'>
                <p>בסימולציית משאית המזון , אנחנו זקוקים למספרים אקראיים עבור מגוון החלטות:</p>
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

        # Add space before tabs
        st.markdown("<br>", unsafe_allow_html=True)

        # Create tabs for different RNG methods
        tab1, tab2 = st.tabs(["מחולל קונגרואנטי לינארי (LCG)", "רגיסטר הזזה עם משוב לינארי (LFSR)"])
        
        with tab1:
            show_lcg()
        
        with tab2:
            show_lfsr()

    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    with st.expander("2. הסבר על שיטות הדגימה", expanded=True):
        # Additional tabs for sampling methods
        st.markdown("<h2 class='rtl'>2. הסבר על שיטות הדגימה</h2>", unsafe_allow_html=True)
        st.markdown("<div class='rtl'><p>בדוגמה זו נדגום זמני הגעת לקוחות בסימולציית משאית המזון. <p>כל שיטות הדגימה שנלמד משתמשות במספרים אקראיים בטווח 0-1 כבסיס להפקת דגימות מהתפלגויות שונות, כאשר כל שיטה ממירה את המספרים האקראיים בדרך שונה ובהתאם להתפלגות היעד.</p></div>", unsafe_allow_html=True)
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
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    with st.expander("3. דגימת זמני הזמנות ", expanded=True):
        show_order_sampling()

if __name__ == "__main__":
    show_rng_demo()