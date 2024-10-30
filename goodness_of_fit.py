import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from utils import set_rtl, set_ltr_sliders

def load_css():
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_introduction():
    # Title and main description
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>התאמת התפלגות למודל 📊</h1>
            <p>ניתוח וביצוע מבחני טיב התאמה לנתוני משאית המזון</p>
        </div>
    """, unsafe_allow_html=True)

    # Background card
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">רקע</h3>
            <p>
                על סמך תיאור הבעלים, צוות סימולציה ערך מדידות, התאים התפלגויות וערך מבחני טיב התאמה 
                לצורך שיערוך טיב התאמת ההתפלגות.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Customer Types Section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">1. הזמנות</h3>
            <p>לקוחות עם דחיפויות והעדפות שונות משפיעים על זמני עיבוד ההזמנות.</p>
        </div>
    """, unsafe_allow_html=True)

    # Create three columns for customer types
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style="background-color: #2D2D2D; padding: 20px; border-radius: 8px; border: 1px solid #8B0000;">
                <h4 style="color: #FFFFFF; text-align: center; margin-bottom: 15px;">סוג א'</h4>
                <div style="text-align: center; color: #CCCCCC;">
                    <p style="margin-bottom: 10px;">50% מהלקוחות</p>
                    <p>אחיד (3-4 דקות)</p>
                    <p class="highlight">המהיר ביותר</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="background-color: #2D2D2D; padding: 20px; border-radius: 8px; border: 1px solid #8B0000;">
                <h4 style="color: #FFFFFF; text-align: center; margin-bottom: 15px;">סוג ב'</h4>
                <div style="text-align: center; color: #CCCCCC;">
                    <p style="margin-bottom: 10px;">25% מהלקוחות</p>
                    <p>משולש (4-6 דקות)</p>
                    <p class="highlight">בינוני</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="background-color: #2D2D2D; padding: 20px; border-radius: 8px; border: 1px solid #8B0000;">
                <h4 style="color: #FFFFFF; text-align: center; margin-bottom: 15px;">סוג ג'</h4>
                <div style="text-align: center; color: #CCCCCC;">
                    <p style="margin-bottom: 10px;">25% מהלקוחות</p>
                    <p>קבוע (10 דקות)</p>
                    <p class="highlight">האיטי ביותר</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Cooking Times Section
    st.markdown("""
        <div class="custom-card rtl-content" style="margin-top: 30px;">
            <h3 class="section-header">2. זמני בישול סטוכסטיים</h3>
            <p>הזמן הנדרש להכנת כל מנה עוקב אחר התפלגות נורמלית, המשתנה לפי גודל המנה:</p>
        </div>
    """, unsafe_allow_html=True)

    # Create three columns for meal types
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style="background-color: #2D2D2D; padding: 20px; border-radius: 8px; border: 1px solid #8B0000;">
                <h4 style="color: #FFFFFF; text-align: center; margin-bottom: 15px;">ארוחה בודדת</h4>
                <div style="text-align: center; color: #CCCCCC;">
                    <p style="margin-bottom: 10px;">N(5, 1)</p>
                    <p>הכנה מהירה לשירות מותאם אישית</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style="background-color: #2D2D2D; padding: 20px; border-radius: 8px; border: 1px solid #8B0000;">
                <h4 style="color: #FFFFFF; text-align: center; margin-bottom: 15px;">מנה של 2</h4>
                <div style="text-align: center; color: #CCCCCC;">
                    <p style="margin-bottom: 10px;">N(8, 2)</p>
                    <p>זמן הכנה מאוזן לנפח בינוני</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style="background-color: #2D2D2D; padding: 20px; border-radius: 8px; border: 1px solid #8B0000;">
                <h4 style="color: #FFFFFF; text-align: center; margin-bottom: 15px;">מנה של 3</h4>
                <div style="text-align: center; color: #CCCCCC;">
                    <p style="margin-bottom: 10px;">N(10, 3)</p>
                    <p>הכנה ארוכה יותר עם יעילות אך סיכון לבישול חסר</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Arrival Times Section
    st.markdown("""
        <div class="custom-card rtl-content" style="margin-top: 30px;">
            <h3 class="section-header">3. זמני הגעה</h3>
            <div style="color: #CCCCCC;">
                <p הצוות ביצעע מדידות של זמני הגעת הלקוחות אך שכח לנתח אותם.    </p>
                <p style="margin-top: 15px;"> בעמוד זה נצלול יחד עם אוצ'ו למשימה של התאמת התפלגות ידועה לדגימות שבידינו</p>
                <p style="margin-top: 15px;">לנוחיותכם, ניתן לקבל מדגם חדש, ולבדוק כיצד באופן שיטתי מתאימים התפלגויות עבור דגימות מסוגים שונים.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)


def show_business_context():
    """Display the business context and importance of the analysis."""
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">ניתוח זמני ההגעה למשאית המזון 🚚</h3>
            <p>
                כדי לייעל את פעילות משאית המזון שלנו, עלינו להבין תחילה את דפוסי זמני ההכנה של המנות.
                המטרה היא לבנות מודל סטטיסטי מדויק שישמש אותנו בהמשך לסימולציה של פעילות המשאית.
            </p>
        </div>
        
        <div class="info-box rtl-content">
            <h4>למה זה חשוב?</h4>
            <ul class="custom-list">
                <li>🎯 נוכל לחזות טוב יותר את זמני ההמתנה של הלקוחות</li>
                <li>👥 נוכל לתכנן טוב יותר את מספר העובדים הנדרש בכל משמרת</li>
                <li>⚡ נוכל לזהות הזדמנויות לייעול תהליך ההכנה</li>
                <li>📊 נוכל לבדוק תרחישים שונים בסימולציה לפני יישומם בשטח</li>
            </ul>
        </div>
        
        <div class="custom-card rtl-content">
            <h4>תהליך הניתוח:</h4>
            <ol class="custom-list">
                <li>1️⃣ איסוף וניתוח ראשוני של נתוני זמני ההכנה</li>
                <li>2️⃣ זיהוי דפוסים והתפלגויות אפשריות</li>
                <li>3️⃣ התאמת מודל סטטיסטי לנתונים</li>
                <li>4️⃣ בדיקת טיב ההתאמה של המודל</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)

def generate_service_times(size=1000, distribution_type=None):
    """Generate realistic food preparation times."""
    np.random.seed(int(pd.Timestamp.now().timestamp()))
    
    if distribution_type is None:
        distribution_type = np.random.choice(['normal', 'uniform', 'exponential', 'mixture'])
    
    def scale_times(times, min_time=2, max_time=15):
        """Scale times to realistic food preparation times (2-15 minutes)"""
        return (times - np.min(times)) * (max_time - min_time) / (np.max(times) - np.min(times)) + min_time
    
    if distribution_type == 'normal':
        # Normal distribution for standard menu items
        mu = np.random.uniform(7, 9)  # average prep time
        sigma = np.random.uniform(1, 2)  # variation in prep time
        samples = np.random.normal(mu, sigma, size)
        samples = scale_times(samples)
        dist_info = {
            'type': 'Normal',
            'description': 'מתאים למנות סטנדרטיות עם זמן הכנה קבוע יחסית',
            'params': {'זמן ממוצע': f"{mu:.1f} דקות", 'סטיית תקן': f"{sigma:.1f} דקות"}
        }
        
    elif distribution_type == 'uniform':
        # Uniform distribution for simple items
        a = np.random.uniform(2, 5)
        b = np.random.uniform(10, 15)
        samples = np.random.uniform(a, b, size)
        dist_info = {
            'type': 'Uniform',
            'description': 'מתאים למנות פשוטות עם זמן הכנה גמיש',
            'params': {'זמן מינימלי': f"{a:.1f} דקות", 'זמן מקסימלי': f"{b:.1f} דקות"}
        }
        
    elif distribution_type == 'exponential':
        # Exponential for rush orders or complex items
        lambda_param = np.random.uniform(0.15, 0.25)
        samples = np.random.exponential(1/lambda_param, size)
        samples = scale_times(samples)
        dist_info = {
            'type': 'Exponential',
            'description': 'מתאים למנות מורכבות או הזמנות בשעות עומס',
            'params': {'קצב שירות': f"{lambda_param:.2f} לקוחות לדקה"}
        }
        
    else:  # mixture
        # Mix of regular and special orders
        prop_regular = np.random.uniform(0.6, 0.8)
        n_regular = int(size * prop_regular)
        n_special = size - n_regular
        
        regular_samples = np.random.normal(8, 1.5, n_regular)
        special_samples = np.random.exponential(2, n_special) + 5
        samples = np.concatenate([regular_samples, special_samples])
        samples = scale_times(samples)
        
        dist_info = {
            'type': 'Mixture',
            'description': 'שילוב של מנות רגילות ומנות מיוחדות',
            'params': {'אחוז מנות רגילות': f"{prop_regular*100:.0f}%"}
        }
    
    samples = np.clip(samples, 2, 15)  # Ensure realistic preparation times
    return samples, dist_info

def generate_random_samples(sample_size):
    """Generate samples from a random distribution with random parameters."""
    distribution = np.random.choice(['normal', 'uniform', 'exponential'])
    if distribution == 'normal':
        mu = np.random.uniform(-5, 5)
        sigma = np.random.uniform(0.5, 2)
        samples = np.random.normal(loc=mu, scale=sigma, size=sample_size)
        return samples, 'Normal', (mu, sigma)
    elif distribution == 'uniform':
        a = np.random.uniform(-5, 0)
        b = np.random.uniform(0.5, 5)
        samples = np.random.uniform(low=a, high=b, size=sample_size)
        return samples, 'Uniform', (a, b)
    elif distribution == 'exponential':
        lam = np.random.uniform(0.5, 2)
        samples = np.random.exponential(scale=1/lam, size=sample_size)
        return samples, 'Exponential', (lam,)
    
def display_samples(samples):
    """Display the first few samples and a simple plot of all samples."""
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header"> דגימות זמני ההכנה בעמדה</h3>
            <p> להלן מדגם מייצג של זמני ההכנה כפי שנמדדו על ידי עובד מסור של המשאית:</p>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Display first few samples in a table
        st.markdown("""
            <div class="info-box rtl-content">
                <h4>זמני ההכנה שנמדדו (בדקות):</h4>
            </div>
        """, unsafe_allow_html=True)
        
        # Create a DataFrame with the first 10 samples
        sample_df = pd.DataFrame({
            'Sample #': range(1, 1001),
            'Time (minutes)': samples.round(2)
        }).set_index('Sample #')
        
        st.dataframe(sample_df, height=200)

        # Display summary statistics with business context
        st.markdown("""
            <div class="info-box rtl-content">
                <h4> סטטיסטיקה תיאורית של דגימות זמני ההכנה שנמדדו:</h4>
                <ul class="custom-list">
                    <li>מספר מדידות: {:d}</li>
                    <li>זמן הכנה ממוצע: {:.2f} דקות</li>
                    <li>זמן הכנה מינימלי: {:.2f} דקות</li>
                    <li>זמן הכנה מקסימלי: {:.2f} דקות</li>
                    <li>סטיית תקן: {:.2f} דקות</li>
                    <li>חציון: {:.2f} דקות</li>
                </ul>
                <p>נתונים אלו מסייעים לנו להבין את טווח זמני ההכנה הטיפוסיים ואת מידת השונות בתהליך.</p>
            </div>
        """.format(
            len(samples),
            np.mean(samples),
            np.min(samples),
            np.max(samples),
            np.std(samples),
            np.median(samples)
        ), unsafe_allow_html=True)
        


    with col2:
        # Create a simple line plot of all samples
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(samples, marker='o', linestyle='None', alpha=0.5, markersize=3,color='darkred')
        plt.title('Service Times')
        plt.xlabel('Sample Number')
        plt.ylabel('Time (minutes)')
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)

def visualize_samples_and_qqplots(samples):
    """Display enhanced histograms and QQ plots in a two-column layout."""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3 class="section-header">ניתוח גרפי של ההתפלגות</h3>
                <p>להלן ניתוח גרפי של הנתונים באמצעות היסטוגרמה ותרשימי Q-Q:</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Add interpretation guide
        st.markdown("""
            <div class="info-box rtl-content">
                <h4>כיצד לפרש את הגרפים:</h4>
                <ul>
                    <li><strong>היסטוגרמה:</strong> מציגה את התפלגות זמני ההכנה. הקו הכחול מראה את אומדן צפיפות הגרעין (KDE).</li>
                    <li><strong>תרשימי Q-Q:</strong> משווים את הנתונים להתפלגויות שונות. ככל שהנקודות קרובות יותר לקו הישר, כך ההתאמה טובה יותר.</li>
                    <li><strong>רצועות אמון:</strong> האזור האפור מציין רווח בר-סמך של 95%. נקודות מחוץ לרצועה מעידות על סטייה מההתפלגות.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

        sns.histplot(data=samples, kde=False, stat='density', ax=axs[0])
        axs[0].set_title('Arrival Time Distribution')
        axs[0].set_xlabel('Time (minutes)')
        axs[0].set_ylabel('Density')

        # QQ Plots with confidence bands
        distributions = [
            ('norm', 'Normal Distribution', axs[1]),
            ('uniform', 'Uniform Distribution', axs[2]),
            ('expon', 'Exponential Distribution', axs[3])
        ]

        for dist_name, title, ax in distributions:
            qq = stats.probplot(samples, dist=dist_name, fit=True, plot=ax)
            
            x = qq[0][0]
            y = qq[0][1]
            slope, intercept = qq[1][0], qq[1][1]
            y_fit = slope * x + intercept
            
            n = len(samples)
            sigma = np.std((y - y_fit) / np.sqrt(1 - 1/n))
            conf_band = 1.96 * sigma
            
            ax.fill_between(x, y_fit - conf_band, y_fit + conf_band, alpha=0.1, color='gray')
            ax.set_title(f'Q-Q Plot - {title}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

def estimate_parameters(samples, distribution):
    """Enhanced parameter estimation with confidence intervals and visual explanation."""
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">אמידת פרמטרים לסימולציה</h3>
            <p>כדי לייצר זמני הכנה מציאותיים בסימולציה, נאמוד את הפרמטרים של ההתפלגות הנבחרת:</p>
        </div>
    """, unsafe_allow_html=True)

    if distribution == 'Normal':
        # Maximum Likelihood estimation for Normal distribution
        mu, sigma = stats.norm.fit(samples)
        
        # Calculate confidence intervals using bootstrap
        bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
        bootstrap_means = np.mean(bootstrap_samples, axis=1)
        bootstrap_stds = np.std(bootstrap_samples, axis=1)
        
        mu_ci = np.percentile(bootstrap_means, [2.5, 97.5])
        sigma_ci = np.percentile(bootstrap_stds, [2.5, 97.5])
        
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>פרמטרים של התפלגות נורמלית:</h4>
                <ul>
                    <li>ממוצע (μ): {mu:.2f} [CI: {mu_ci[0]:.2f}, {mu_ci[1]:.2f}]</li>
                    <li>סטיית תקן (σ): {sigma:.2f} [CI: {sigma_ci[0]:.2f}, {sigma_ci[1]:.2f}]</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        return mu, sigma

    elif distribution == 'Exponential':
        # Maximum Likelihood estimation for Exponential distribution
        lambda_est = 1 / np.mean(samples)
        
        # Calculate confidence interval for lambda using bootstrap
        bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
        bootstrap_lambdas = 1 / np.mean(bootstrap_samples, axis=1)
        lambda_ci = np.percentile(bootstrap_lambdas, [2.5, 97.5])
        
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>פרמטר של התפלגות מעריכית:</h4>
                <ul>
                    <li>קצב (λ): {lambda_est:.4f} [CI: {lambda_ci[0]:.4f}, {lambda_ci[1]:.4f}]</li>
                    <li>זמן ממוצע (1/λ): {1/lambda_est:.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        return lambda_est,

    elif distribution == 'Uniform':
        # Maximum Likelihood estimation for Uniform distribution
        a, b = np.min(samples), np.max(samples)
        
        # Calculate confidence intervals using bootstrap
        bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
        bootstrap_mins = np.min(bootstrap_samples, axis=1)
        bootstrap_maxs = np.max(bootstrap_samples, axis=1)
        
        a_ci = np.percentile(bootstrap_mins, [2.5, 97.5])
        b_ci = np.percentile(bootstrap_maxs, [2.5, 97.5])
        
        st.markdown(f"""
            <div class="info-box rtl-content">
                <h4>פרמטרים של התפלגות אחידה:</h4>
                <ul>
                    <li>מינימום (a): {a:.2f} [CI: {a_ci[0]:.2f}, {a_ci[1]:.2f}]</li>
                    <li>מקסימום (b): {b:.2f} [CI: {b_ci[0]:.2f}, {b_ci[1]:.2f}]</li>
                    <li>טווח: {b-a:.2f} דקות</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        return a, b

def generate_service_times(size=1000, distribution_type=None):
    """
    Generate service times from various distributions.
    If distribution_type is None, randomly select one.
    """
    # Use numpy random seed based on current timestamp
    np.random.seed(int(pd.Timestamp.now().timestamp()))
    
    if distribution_type is None:
        distribution_type = np.random.choice([
            'normal', 'uniform', 'exponential', 'mixture', 'lognormal'
        ])
    
    def scale_times(times, min_time=2, max_time=15):
        """Scale times to be between min_time and max_time minutes"""
        return (times - np.min(times)) * (max_time - min_time) / (np.max(times) - np.min(times)) + min_time
    
    if distribution_type == 'normal':
        # Normal distribution with realistic parameters
        mu = np.random.uniform(7, 9)  # mean service time
        sigma = np.random.uniform(1, 2)  # standard deviation
        samples = np.random.normal(mu, sigma, size)
        samples = scale_times(samples)
        dist_info = {'type': 'Normal', 'params': {'mu': mu, 'sigma': sigma}}
        
    elif distribution_type == 'uniform':
        # Uniform distribution between min and max times
        a = np.random.uniform(2, 5)  # minimum time
        b = np.random.uniform(10, 15)  # maximum time
        samples = np.random.uniform(a, b, size)
        dist_info = {'type': 'Uniform', 'params': {'a': a, 'b': b}}
        
    elif distribution_type == 'exponential':
        # Exponential distribution scaled to realistic times
        lambda_param = np.random.uniform(0.15, 0.25)  # rate parameter
        samples = np.random.exponential(1/lambda_param, size)
        samples = scale_times(samples)
        dist_info = {'type': 'Exponential', 'params': {'lambda': lambda_param}}
        
    elif distribution_type == 'lognormal':
        # Lognormal distribution for right-skewed times
        mu = np.random.uniform(1.8, 2.2)
        sigma = np.random.uniform(0.2, 0.4)
        samples = np.random.lognormal(mu, sigma, size)
        samples = scale_times(samples)
        dist_info = {'type': 'Lognormal', 'params': {'mu': mu, 'sigma': sigma}}
        
    elif distribution_type == 'mixture':
        # Mixture of distributions
        mixture_type = np.random.choice([
            'normal_exponential',
            'normal_uniform',
            'bimodal_normal'
        ])
        
        if mixture_type == 'normal_exponential':
            # Mix of normal (regular orders) and exponential (rush orders)
            prop_normal = np.random.uniform(0.6, 0.8)
            n_normal = int(size * prop_normal)
            n_exp = size - n_normal
            
            normal_samples = np.random.normal(8, 1.5, n_normal)
            exp_samples = np.random.exponential(2, n_exp) + 5
            samples = np.concatenate([normal_samples, exp_samples])
            dist_info = {
                'type': 'Mixture',
                'subtype': 'Normal-Exponential',
                'params': {'proportion_normal': prop_normal}
            }
            
        elif mixture_type == 'normal_uniform':
            # Mix of normal (regular orders) and uniform (special orders)
            prop_normal = np.random.uniform(0.7, 0.9)
            n_normal = int(size * prop_normal)
            n_uniform = size - n_normal
            
            normal_samples = np.random.normal(8, 1.5, n_normal)
            uniform_samples = np.random.uniform(4, 12, n_uniform)
            samples = np.concatenate([normal_samples, uniform_samples])
            dist_info = {
                'type': 'Mixture',
                'subtype': 'Normal-Uniform',
                'params': {'proportion_normal': prop_normal}
            }
            
        else:  # bimodal_normal
            # Bimodal normal for different types of orders
            prop_fast = np.random.uniform(0.5, 0.7)
            n_fast = int(size * prop_fast)
            n_slow = size - n_fast
            
            fast_samples = np.random.normal(6, 1, n_fast)
            slow_samples = np.random.normal(11, 1.5, n_slow)
            samples = np.concatenate([fast_samples, slow_samples])
            dist_info = {
                'type': 'Mixture',
                'subtype': 'Bimodal-Normal',
                'params': {'proportion_fast': prop_fast}
            }
        
        samples = scale_times(samples)
    
    # Ensure all times are positive and within realistic bounds
    samples = np.clip(samples, 2, 15)
    
    return samples, dist_info

def plot_likelihood(samples, distribution):
    """Enhanced likelihood function visualization with updated styling."""
    if distribution == 'Normal':
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(2)]

        mu_vals = np.linspace(np.mean(samples) - 3*np.std(samples), 
                             np.mean(samples) + 3*np.std(samples), 100)
        sigma_vals = np.linspace(np.std(samples) * 0.2, 
                                np.std(samples) * 2, 100)

        ll_mu = [np.sum(stats.norm.logpdf(samples, loc=mu, scale=np.std(samples))) 
                 for mu in mu_vals]
        ll_sigma = [np.sum(stats.norm.logpdf(samples, loc=np.mean(samples), scale=sigma)) 
                   for sigma in sigma_vals]

        ax1.plot(mu_vals, ll_mu, color='pink', linewidth=2)
        ax1.axvline(np.mean(samples), color='darkred', linestyle='--', alpha=0.5)
        ax1.set_title('Log-Likelihood for Mean (μ)')
        ax1.set_xlabel('μ')
        ax1.set_ylabel('Log-Likelihood')
        ax1.grid(True, alpha=0.3)

        ax2.plot(sigma_vals, ll_sigma, color='pink', linewidth=2)
        ax2.axvline(np.std(samples), color='darkred', linestyle='--', alpha=0.5)
        ax2.set_title('Log-Likelihood for Standard Deviation (σ)')
        ax2.set_xlabel('σ')
        ax2.set_ylabel('Log-Likelihood')
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

    elif distribution == 'Uniform':
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        ax1, ax2 = [fig.add_subplot(gs[0, i]) for i in range(2)]

        margin = (np.max(samples) - np.min(samples)) * 0.2
        a_vals = np.linspace(np.min(samples) - margin, np.min(samples) + margin, 100)
        b_vals = np.linspace(np.max(samples) - margin, np.max(samples) + margin, 100)

        fixed_b = np.max(samples)
        fixed_a = np.min(samples)

        ll_a = [np.sum(stats.uniform.logpdf(samples, loc=a, scale=fixed_b - a))
                if fixed_b > a else -np.inf for a in a_vals]
        ll_b = [np.sum(stats.uniform.logpdf(samples, loc=fixed_a, scale=b - fixed_a))
                if b > fixed_a else -np.inf for b in b_vals]

        ax1.plot(a_vals, ll_a, color='pink', linewidth=2)
        ax1.axvline(np.min(samples), color='darkred', linestyle='--', alpha=0.5)
        ax1.set_title('Log-Likelihood for Minimum (a)')
        ax1.set_xlabel('a')
        ax1.set_ylabel('Log-Likelihood')
        ax1.grid(True, alpha=0.3)

        ax2.plot(b_vals, ll_b, color='pink', linewidth=2)
        ax2.axvline(np.max(samples), color='darkred', linestyle='--', alpha=0.5)
        ax2.set_title('Log-Likelihood for Maximum (b)')
        ax2.set_xlabel('b')
        ax2.set_ylabel('Log-Likelihood')
        ax2.grid(True, alpha=0.3)

        st.pyplot(fig)

    elif distribution == 'Exponential':
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        lambda_vals = np.linspace(1/(2*np.mean(samples)), 2/np.mean(samples), 100)
        ll_lambda = [np.sum(stats.expon.logpdf(samples, scale=1/lambda_val)) 
                    for lambda_val in lambda_vals]

        ax.plot(lambda_vals, ll_lambda, color='pink', linewidth=2)
        ax.axvline(1/np.mean(samples), color='darkred', linestyle='--', alpha=0.5)
        ax.set_title('Log-Likelihood for Rate Parameter (λ)')
        ax.set_xlabel('λ')
        ax.set_ylabel('Log-Likelihood')
        ax.grid(True, alpha=0.3)

        st.pyplot(fig)

def perform_goodness_of_fit(samples, distribution, params):
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">בדיקת התאמת המודל</h3>
            <p>לפני שנשתמש במודל בסימולציה, חשוב לוודא שהוא אכן מתאר היטב את המציאות במשאית המזון שלנו. נבצע מבחנים סטטיסטיים כדי לבדוק את מידת ההתאמה:</p>
        </div>
    """, unsafe_allow_html=True)
    """Improved goodness of fit testing with corrected hypothesis testing."""
    
    # Calculate number of bins using Freedman-Diaconis rule
    iqr = stats.iqr(samples)
    bin_width = 2 * iqr / (len(samples) ** (1/3))
    n_bins = int(np.ceil((np.max(samples) - np.min(samples)) / bin_width))
    n_bins = max(5, min(n_bins, 50))
    
    # Perform Chi-Square Test
    observed_freq, bins = np.histogram(samples, bins=n_bins)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    
    if distribution == 'Normal':
        mu, sigma = params
        expected_probs = stats.norm.cdf(bins[1:], mu, sigma) - stats.norm.cdf(bins[:-1], mu, sigma)
        dof = len(observed_freq) - 3  # subtracting parameters estimated + 1
        theoretical_dist = stats.norm(mu, sigma)
        
    elif distribution == 'Exponential':
        lambda_param = params[0]
        expected_probs = stats.expon.cdf(bins[1:], scale=1/lambda_param) - stats.expon.cdf(bins[:-1], scale=1/lambda_param)
        dof = len(observed_freq) - 2
        theoretical_dist = stats.expon(scale=1/lambda_param)
        
    elif distribution == 'Uniform':
        a, b = params
        expected_probs = stats.uniform.cdf(bins[1:], a, b-a) - stats.uniform.cdf(bins[:-1], a, b-a)
        dof = len(observed_freq) - 3
        theoretical_dist = stats.uniform(a, b-a)
    
    expected_freq = expected_probs * len(samples)
    
    # Combine bins with expected frequency < 5
    while np.any(expected_freq < 5) and len(expected_freq) > 2:
        min_idx = np.argmin(expected_freq)
        if min_idx == 0:  # First bin
            observed_freq[0:2] = np.sum(observed_freq[0:2])
            expected_freq[0:2] = np.sum(expected_freq[0:2])
            observed_freq = np.delete(observed_freq, 1)
            expected_freq = np.delete(expected_freq, 1)
        elif min_idx == len(expected_freq) - 1:  # Last bin
            observed_freq[-2:] = np.sum(observed_freq[-2:])
            expected_freq[-2:] = np.sum(expected_freq[-2:])
            observed_freq = np.delete(observed_freq, -1)
            expected_freq = np.delete(expected_freq, -1)
        else:  # Middle bin
            observed_freq[min_idx:min_idx+2] = np.sum(observed_freq[min_idx:min_idx+2])
            expected_freq[min_idx:min_idx+2] = np.sum(expected_freq[min_idx:min_idx+2])
            observed_freq = np.delete(observed_freq, min_idx+1)
            expected_freq = np.delete(expected_freq, min_idx+1)
    
    # Perform Chi-Square test
    chi_square_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
    p_value_chi = 1 - stats.chi2.cdf(chi_square_stat, max(1, dof))
    
    # Perform Kolmogorov-Smirnov test
    if distribution == 'Normal':
        ks_stat, p_value_ks = stats.kstest(stats.zscore(samples), 'norm')
    elif distribution == 'Exponential':
        # Scale the data to standard exponential
        scaled_samples = samples * lambda_param
        ks_stat, p_value_ks = stats.kstest(scaled_samples, 'expon')
    elif distribution == 'Uniform':
        # Scale the data to standard uniform
        scaled_samples = (samples - a) / (b - a)
        ks_stat, p_value_ks = stats.kstest(scaled_samples, 'uniform')
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="info-box" style="background-color: #fff0f5; padding: 15px; border-radius: 5px;">
                <h4>Kolmogorov-Smirnov Test:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>Statistic: {ks_stat:.4f}</li>
                    <li>p-value: {p_value_ks:.4f}</li>
                    <li>Conclusion: {"Reject H0" if p_value_ks < 0.05 else "Fail to reject H0"}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="info-box" style="background-color: #fff0f5; padding: 15px; border-radius: 5px;">
                <h4>Chi-Square Test:</h4>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li>Statistic: {chi_square_stat:.4f}</li>
                    <li>Degrees of freedom: {dof}</li>
                    <li>p-value: {p_value_chi:.4f}</li>
                    <li>Conclusion: {"Reject H0" if p_value_chi < 0.05 else "Fail to reject H0"}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Visualization of the fit
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot histogram of data
    sns.histplot(data=samples, stat='density', alpha=0.5, ax=ax, label='Data', color='pink')
    
    # Plot fitted distribution
    x = np.linspace(np.min(samples), np.max(samples), 100)
    if distribution == 'Normal':
        pdf = stats.norm.pdf(x, *params)
        ax.plot(x, pdf, 'darkred', label='Fitted Normal')
    elif distribution == 'Exponential':
        pdf = stats.expon.pdf(x, scale=1/params[0])
        ax.plot(x, pdf, 'darkred', label='Fitted Exponential')
    elif distribution == 'Uniform':
        pdf = stats.uniform.pdf(x, *params)
        ax.plot(x, pdf, 'darkred', label='Fitted Uniform')
    
    ax.set_title('Distribution Fit to Data')
    ax.set_xlabel('Values')
    ax.set_ylabel('Density')
    ax.legend()
    
    st.pyplot(fig)

    show_simulation_next_steps()

def show_simulation_next_steps():
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">השלב הבא: בניית הסימולציה 🎮</h3>
            <p>כעת כשיש לנו מודל סטטיסטי מדויק לזמני ההכנה, נוכל:</p>
            <ul class="custom-list">
                <li>לייצר זמני הכנה מציאותיים בסימולציה</li>
                <li>לבדוק תרחישים שונים של עומס במשאית</li>
                <li>לבחון את ההשפעה של שינויים בתהליך העבודה</li>
                <li>לקבל החלטות מבוססות נתונים לשיפור היעילות</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

def show():
    # Call this at the start of your app
    load_css()
    #  st.markdown(get_custom_css(), unsafe_allow_html=True)
    # Show introduction section
    show_introduction()
    # Header section with business context
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>ניתוח זמני שירות - עמדת הכנת המנות 👨‍🍳</h1>
            <p>התאמת מודל סטטיסטי לזמני הכנת מנות במשאית</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Business context explanation
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">מטרת הניתוח הסטטיסטי</h3>
            <p>
                כדי לבנות סימולציה מדויקת של פעילות משאית המזון, עלינו להבין תחילה את דפוסי זמני ההכנה של המנות.
                דרך ניתוח הנתונים נוכל:
            </p>
            <ul class="custom-list">
                <li>🎯 לחזות טוב יותר את זמני ההמתנה של הלקוחות</li>
                <li>👥 לתכנן טוב יותר את מספר העובדים הנדרש בכל משמרת</li>
                <li>⚡ לזהות הזדמנויות לייעול תהליך ההכנה</li>
                <li>📊 לבדוק תרחישים שונים בסימולציה לפני יישומם בשטח</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    # Generate new samples
    if 'samples' not in st.session_state or st.button('יצירת מדגם חדש'):
        samples, dist_info = generate_service_times()
        st.session_state.samples = samples
        st.session_state.dist_info = dist_info
        
            # Display the samples
    

    
    samples = st.session_state.samples
    display_samples(samples)

    

    #st.markdown(f"""
    #<div class="info-box rtl-content">
        #<p>התפלגות אמיתית (למטרות בדיקה): {dist_info['type']}</p>
        #{'<p>תת-סוג: ' + dist_info.get('subtype', 'N/A') + '</p>' if 'subtype' in dist_info else ''}
        #<p>פרמטרים: {dist_info['params']}</p>
    #</div>
    #""", unsafe_allow_html=True)

    # Analysis section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">ניתוח התפלגות הנתונים</h3>
            <p>כעת נבחן את התפלגות הנתונים באמצעות כלים סטטיסטיים כדי לבחור את המודל המתאים ביותר לסימולציה:</p>
        </div>
    """, unsafe_allow_html=True)

    visualize_samples_and_qqplots(samples)

    # Distribution selection with business context
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">בחירת התפלגות מתאימה</h3>
            <p>
                בהתבסס על הניתוח הגרפי, נבחר את ההתפלגות שמתארת בצורה הטובה ביותר את זמני ההכנה במשאית.
                כל התפלגות מתאימה לתרחיש עסקי שונה:
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Create three columns for the distribution buttons with business context
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class="metric-container rtl-content">
                <h4>התפלגות נורמלית</h4>
                <p>מתאימה למנות סטנדרטיות עם זמן הכנה קבוע יחסית</p>
            </div>
        """, unsafe_allow_html=True)
        normal_button = st.button("בחר התפלגות נורמלית")

    with col2:
        st.markdown("""
            <div class="metric-container rtl-content">
                <h4>התפלגות אחידה</h4>
                <p>מתאימה למנות פשוטות עם זמן הכנה גמיש</p>
            </div>
        """, unsafe_allow_html=True)
        uniform_button = st.button("בחר התפלגות אחידה")

    with col3:
        st.markdown("""
            <div class="metric-container rtl-content">
                <h4>התפלגות מעריכית</h4>
                <p>מתאימה למנות מורכבות או הזמנות בשעות עומס</p>
            </div>
        """, unsafe_allow_html=True)
        exp_button = st.button("בחר התפלגות מעריכית")

    # Handle distribution selection
    distribution_choice = None
    if normal_button:
        distribution_choice = 'Normal'
    elif uniform_button:
        distribution_choice = 'Uniform'
    elif exp_button:
        distribution_choice = 'Exponential'

    if distribution_choice:
        params = estimate_parameters(samples, distribution_choice)
        plot_likelihood(samples, distribution_choice)
        perform_goodness_of_fit(samples, distribution_choice, params)

# To show the app, call the show() function
if __name__ == "__main__":
    show()
