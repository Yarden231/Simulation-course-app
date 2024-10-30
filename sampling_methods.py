
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import set_rtl
from utils import set_ltr_sliders
import time

def plot_qqplot(samples, title):
    fig, ax = plt.subplots(figsize=(6, 4))
    stats.probplot(samples, dist="norm", plot=ax)
    ax.set_title(f"{title} - QQ Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.grid(True)
    return fig

def sample_uniform(a, b, size):
    return np.random.uniform(a, b, size)

def sample_normal(mu, sigma, size):
    """Sample from normal distribution"""
    samples = np.random.normal(mu, sigma, size)
    return np.clip(samples, 2, 15)  # Clip to realistic food prep times

def sample_exponential(lambda_param, size):
    """Sample from exponential distribution"""
    samples = np.random.exponential(1/lambda_param, size)
    return np.clip(samples, 2, 15)  # Clip to realistic food prep times

def sample_composite(size):
    """Sample from mixture of two normal distributions"""
    n_simple = int(0.2 * size)
    n_complex = size - n_simple
    
    simple_orders = np.random.normal(5, 1, n_simple)
    complex_orders = np.random.normal(10, 1.5, n_complex)
    
    all_orders = np.concatenate([simple_orders, complex_orders])
    return np.clip(all_orders, 2, 15)

def plot_histogram(samples, title, distribution_func=None, true_density=None):
    """Plot histogram with better styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    bins = np.linspace(min(samples), max(samples), 30)
    ax.hist(samples, bins=bins, density=True, alpha=0.7, color='pink', label='Sampled Data')
    
    if true_density:
        x = np.linspace(min(samples), max(samples), 100)
        ax.plot(x, true_density(x), 'darkred', linewidth=2, label='True Density')

    if distribution_func:
        x = np.linspace(0, 1, 100)
        ax.plot(x, distribution_func(x), 'darkred', linewidth=2, linestyle='--', label='Target Distribution')

    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_qqplot(samples, title):
    """Plot QQ plot with better styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    stats.probplot(samples, dist="norm", plot=ax)
    
    ax.get_lines()[0].set_markerfacecolor('pink')
    ax.get_lines()[0].set_markeredgecolor('darkred')
    ax.get_lines()[1].set_color('darkred')
    
    ax.set_title(f"{title}\nQ-Q Plot")
    ax.grid(True, alpha=0.3)
    
    return fig

def display_statistics(samples):
    """Display statistics with better formatting."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>מדדי מרכז:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>ממוצע: {np.mean(samples):.2f} דקות</li>
                    <li>חציון: {np.median(samples):.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>מדדי פיזור:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>סטיית תקן: {np.std(samples):.2f} דקות</li>
                    <li>טווח: {np.min(samples):.2f} - {np.max(samples):.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)


def show_sampling_intro():
    # Main header
    st.markdown("""
        <div style="
            background-color: #1A1A1A;
            border: 1px solid #8B0000;
            border-radius: 8px;
            padding: 30px;
            margin: 20px 0;
        ">
            <!-- Title Section -->
            <div style="margin-bottom: 30px;">
                <h1 style="
                    color: #FFFFFF;
                    text-align: right;
                    font-size: 1.8rem;
                    margin-bottom: 15px;
                ">שיטות דגימה - הדרך ליצירת סימולציה מדויקת 🎲</h1>
                <p style="
                    color: #CCCCCC;
                    text-align: right;
                    line-height: 1.6;
                ">
                    כדי לדמות את פעילות משאית המזון של משפחת לוקו בצורה מדויקת, אנחנו צריכים להבין כיצד ליצור מספרים אקראיים 
                    שמתנהגים בדיוק כמו הנתונים האמיתיים שאספנו.
                </p>
            </div>

            <!-- Two Column Layout -->
            <div style="
                display: flex;
                gap: 30px;
                margin-bottom: 30px;
            ">
                <!-- Left Column -->
                <div style="flex: 1;">
                    <div style="
                        background-color: #2D2D2D;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                    ">
                        <h4 style="
                            color: #FFFFFF;
                            margin-bottom: 15px;
                            text-align: right;
                        ">למה זה חשוב?</h4>
                        <ul style="
                            color: #CCCCCC;
                            padding-right: 20px;
                            margin: 0;
                            text-align: right;
                            list-style-type: none;
                        ">
                            <li style="margin-bottom: 10px;">🎯 דיוק בחיזוי זמני המתנה</li>
                            <li style="margin-bottom: 10px;">⚡ שיפור יעילות התהליך</li>
                            <li style="margin-bottom: 10px;">📊 תכנון משמרות מדויק</li>
                            <li style="margin-bottom: 10px;">💡 קבלת החלטות מבוססות נתונים</li>
                        </ul>
                    </div>
                </div>

                <!-- Right Column -->
                <div style="flex: 2;">
                    <div style="
                        background-color: #2D2D2D;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                    ">
                        <h3 style="
                            color: #FFFFFF;
                            margin-bottom: 15px;
                            text-align: right;
                        ">מה נלמד בעמוד זה?</h3>
                        <p style="
                            color: #CCCCCC;
                            margin-bottom: 15px;
                            text-align: right;
                        ">
                            בעמוד זה נלמד את השיטות השונות לדגימת מספרים אקראיים עבור:
                        </p>
                        <ul style="
                            color: #CCCCCC;
                            padding-right: 20px;
                            margin: 0;
                            text-align: right;
                        ">
                            <li style="margin-bottom: 10px;">⏰ זמני הגעת לקוחות למשאית</li>
                            <li style="margin-bottom: 10px;">🍽️ זמני הכנת מנות שונות</li>
                            <li style="margin-bottom: 10px;">⌛ זמני המתנה מקסימליים של לקוחות</li>
                            <li style="margin-bottom: 10px;">🔄 זמני מעבר בין עמדות השירות</li>
                        </ul>
                    </div>
                </div>
            </div>

            <!-- Bottom Section -->
            <div style="
                background-color: #2D2D2D;
                padding: 20px;
                border-radius: 8px;
            ">
                <h4 style="
                    color: #FFFFFF;
                    text-align: right;
                    margin-bottom: 15px;
                ">כיצד נשתמש בשיטות הדגימה?</h4>
                <p style="
                    color: #CCCCCC;
                    text-align: right;
                    line-height: 1.6;
                    margin-bottom: 0;
                ">
                    בהמשך, נשתמש בשיטות אלו כדי ליצור סימולציה מדויקת של פעילות המשאית. 
                    הסימולציה תאפשר לנו לבחון תרחישים שונים ולקבל החלטות מושכלות לגבי תפעול המשאית.
                    נתחיל בהבנת השיטות הבסיסיות ונתקדם לשיטות מורכבות יותר.
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

from statsmodels.graphics.gofplots import qqplot
import numpy as np


def plot_histogram(samples, title, distribution_func=None, true_density=None):
    """Plot histogram with better styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(samples), max(samples), 30)
    ax.hist(samples, bins=bins, density=True, alpha=0.7, color='#8B0000', edgecolor='black', label='Sampled Data')
    
    if true_density:
        x = np.linspace(min(samples), max(samples), 100)
        ax.plot(x, true_density(x), 'darkred', linewidth=2, label='True Density')

    if distribution_func:
        x = np.linspace(0, 1, 100)
        ax.plot(x, distribution_func(x), 'darkred', linewidth=2, linestyle='--', label='Target Distribution')

    ax.set_title(title)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_qqplot(samples, title):
    """Plot QQ plot with better styling."""
    fig, ax = plt.subplots(figsize=(8, 5))
    stats.probplot(samples, dist="norm", plot=ax)
    
    ax.get_lines()[0].set_markerfacecolor('#8B0000')
    ax.get_lines()[0].set_markeredgecolor('#8B0000')
    ax.get_lines()[1].set_color('#8B0000')
    
    ax.set_title(f"{title}\nQ-Q Plot")
    ax.grid(True, alpha=0.3)
    return fig

def display_statistics(samples):
    """Display statistics with better formatting."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>מדדי מרכז:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>ממוצע: {np.mean(samples):.2f} דקות</li>
                    <li>חציון: {np.median(samples):.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>מדדי פיזור:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>סטיית תקן: {np.std(samples):.2f} דקות</li>
                    <li>טווח: {np.min(samples):.2f} - {np.max(samples):.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

def run_sampling(sampling_function, num_samples, update_interval, title, progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=False, distribution_func=None, true_density=None):
    """Run sampling with visualization updates."""
    # Generate all samples at once
    all_samples = sampling_function(num_samples)
    
    # Calculate number of iterations
    num_iterations = (num_samples + update_interval - 1) // update_interval
    
    # Process samples in batches
    samples = []
    for i in range(num_iterations):
        start_idx = i * update_interval
        end_idx = min(start_idx + update_interval, num_samples)
        
        batch_samples = all_samples[start_idx:end_idx]
        samples.extend(batch_samples)
        
        # Generate Q-Q plot
        qqplot_fig, ax = plt.subplots()
        qqplot(np.array(samples), line='s', ax=ax)
        ax.get_lines()[0].set_markerfacecolor('#8B0000')
        ax.get_lines()[0].set_markeredgecolor('#8B0000')
        ax.get_lines()[1].set_color('#8B0000')
        
        with plot_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                st.pyplot(qqplot_fig)
                plt.close(qqplot_fig)

            with col2:
                hist_fig = plot_histogram(samples, title)
                st.pyplot(hist_fig)
                plt.close(hist_fig)

        # Update statistics placeholder
        stats_placeholder.empty()
        with stats_placeholder:
            display_statistics(samples)
        
        # Update progress
        progress = min(1.0, end_idx / num_samples)
        progress_bar.progress(progress)



def show_sampling_methods():
    
        # Apply custom CSS
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    set_ltr_sliders()

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

    num_samples = st.slider("מספר דגימות", min_value=5000, max_value=1000000, value=1000, step=1000)

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
        run_sampling(lambda size: sample_uniform(a, b, size), num_samples, 100, 
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
        run_sampling(lambda size: sample_normal(mu, sigma, size), num_samples, 100, 
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
                    num_samples, 100, "התפלגות זמני הגעה", 
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