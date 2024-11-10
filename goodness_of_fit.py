import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from utils import set_rtl, set_ltr_sliders
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go



def load_css():
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def show_introduction():
    # Title and main description
    st.markdown("""
        <div class="custom-header rtl-content">
            <h1>התאמת התפלגות למודל 📉</h1>
        </div>
    """, unsafe_allow_html=True)

    # Background card
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 class="section-header">רקע</h3>
            <p>
                התאמת התפלגויות מדויקות היא שלב מכריע בתהליך הסימולציה של פעילות משאית המזון. על ידי בחינה של זמני הגעה, הכנה ועיבוד הזמנות, אנו מסוגלים ליצור מודל סטטיסטי המשקף את תפקוד המשאית בצורה ריאליסטית. תהליך זה מאפשר לנו להבין לעומק את דפוסי הפעילות היומיומיים, לחזות את זמני ההמתנה, ולבחון שיפורים בתהליך – כל זאת מתוך נתונים שנאספו מהשטח.
            </p>
            <p>
                ניתוח והתאמת התפלגויות אינם מתמצים רק בסטטיסטיקה תיאורית; הם כוללים גם בדיקות של התאמת הנתונים להתפלגויות שונות. תהליך זה כולל בדיקת שונות הנתונים ובחירת מודלים שמתאימים למבנה ההזמנות והמנות, החל ממנות בודדות ועד ארוחות גדולות יותר. התוצאה היא תשתית מודלים שמאפשרת לנו לבצע תכנון יעיל של המשאבים, ולהתאים את פעילות המשאית לצורכי הלקוחות.
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="custom-header rtl-content">
            <h4>בהמשך לתשאול של חולייסיטו בדף האפליקציה הקודם, אוצ'ו לוקו עבד עם צוות סימולציה, מדד ומידל את עמדת ההזמנות ועמדת הבישול, להלן המידול:</h4>
        </div>
    """, unsafe_allow_html=True)


        # Create tabs for different RNG methods
    tab1, tab2 = st.columns(2)
    
    with tab1:
            
        # Customer Types Section
        st.markdown("""
            <div style= 'background-color: #2D2D2D;padding: 15px; border-radius: 5px; height: 100%;'>
                <div class="custom-card rtl-content">
                    <h3 class="section-header">1. הזמנות</h3>
                    <p>הזמנות שונות מתקבלות מלקוחות בעלי צרכים ודחיפויות מגוונות, מה שמשפיע ישירות על זמני עיבוד ההזמנות. הגדרת סוגי הלקוחות ותיאור ההתפלגויות לכל סוג מסייעים לדייק את חיזוי זמני השירות והעיבוד בסימולציה.</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Create three columns for customer types
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
                    <h4 style="text-align: center; margin-bottom: 15px;">סוג א'</h4>
                    <div style="text-align: center; color: #CCCCCC;">
                        <p style="margin-bottom: 10px;">50% מהלקוחות</p>
                        <p>אחיד (3-4 דקות)</p>
                        <p class="highlight">המהיר ביותר</p>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
                    <h4 style=" text-align: center; margin-bottom: 15px;">סוג ב'</h4>
                    <div style="text-align: center; color: #CCCCCC;">
                        <p style="margin-bottom: 10px;">25% מהלקוחות</p>
                        <p>משולש (4-6 דקות)</p>
                        <p class="highlight">בינוני</p>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
                    <h4 style=" text-align: center; margin-bottom: 15px;">סוג ג'</h4>
                    <div style="text-align: center; color: #CCCCCC;">
                        <p style="margin-bottom: 10px;">25% מהלקוחות</p>
                        <p>קבוע (10 דקות)</p>
                        <p class="highlight">האיטי ביותר</p>
                    </div>
            """, unsafe_allow_html=True)
    
    with tab2:

        # Cooking Times Section
        st.markdown("""
            <div style= 'background-color: #2D2D2D;padding: 15px; border-radius: 5px; height: 100%;'>
            <div class="custom-card rtl-content">
                <h3>2. זמני בישול סטוכסטיים</h3>
                <p>הזמן הנדרש להכנת כל מנה משתנה בהתאם לגודלה ועוקב אחר התפלגות נורמלית. התאמת התפלגות לזמני ההכנה מאפשרת לנו לייצג בצורה אמינה את השונות בתהליך הבישול ולחשב את זמני ההמתנה הצפויים.</p>
            </div>
        """, unsafe_allow_html=True)

        # Create three columns for meal types
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""

                    <h4 style=" text-align: center; margin-bottom: 15px;">ארוחה בודדת</h4>
                    <div style="text-align: center; color: #CCCCCC;">
                        <p style="margin-bottom: 10px;">N(5, 1)</p>
                        <p>הכנה מהירה לשירות מותאם אישית</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""

                    <h4 style=" text-align: center; margin-bottom: 15px;">מנה של 2</h4>
                    <div style="text-align: center; color: #CCCCCC;">
                        <p style="margin-bottom: 10px;">N(8, 2)</p>
                        <p>זמן הכנה מאוזן לנפח בינוני</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""

                    <h4 style=" text-align: center; margin-bottom: 15px;">מנה של 3</h4>
                    <div style="text-align: center; color: #CCCCCC;">
                        <p style="margin-bottom: 10px;">N(10, 3)</p>
                        <p>הכנה ארוכה יותר עם יעילות אך סיכון לבישול חסר</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)



    # Arrival Times Section
    st.markdown("""
        <div class="custom-card rtl-content" style="margin-top: 10px;">
            <h4>בעמוד זה נעזור לאוצ'ו לוקו לסיים את מלאכתו ונתאים התפלגות לזמני הגעת הלקוחות במשאית המזון.</h4>
            <h2>3. זמני הגעה</h2>
            <p>הצוות ביצע מדידות של זמני הגעת הלקוחות למשאית המזון, אך מדידות אלו טרם נותחו. הבנת דפוסי הגעת הלקוחות תסייע לנו לזהות זמני שיא וצווארי בקבוק, ולתכנן את המשאבים בהתאם לצורך.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

def generate_arrival_times(size=1000):
    """Generate arrival times based on exponential distribution eith lambda parameter = 6."""
    samples = np.random.exponential(scale=1/6, size=size)
    dist_info = ('Exponential', (6,))
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
    # Display sample data and allow the user to fit distributions
    st.text(" ")
    st.text(" ")
    st.markdown("""
        <div class="custom-card rtl-content">
            <h2>תוצאות הדגימה הנוכחית:</h2>
            <p>הנתונים שנאספו מוצגים להלן. בחרו את ההתפלגות המתאימה ביותר עבור הנתונים וודעו אם התאמתכם מייצגת בצורה מדויקת את דפוסי ההגעה של הלקוחות.</p>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns
    col1, col2 = st.columns([1, 1])

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
        
        st.dataframe(sample_df, height=250)

        


    with col2:
        fig = go.Figure()

        # Add scatter plot for service times
        fig.add_trace(
            go.Scatter(
                x=list(range(len(samples))),
                y=samples,
                mode='markers',
                marker=dict(
                    color='#8B0000',  # Dark red color
                    size=6,
                    opacity=0.6
                ),
                name="זמני שירות"
            )
        )

        # Update layout to match styling
        fig.update_layout(
            title="זמני שירות",
            xaxis_title="מספר מדגם",
            yaxis_title="זמן (בדקות)",
            height=400,
            title_x=0.5,
            xaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(200, 200, 200, 0.2)')
        )

        # Display plot
        st.plotly_chart(fig, use_container_width=True)

        # Display summary statistics with business context
    st.markdown("""
        <div class="info-box rtl-content">
            <h4 style='text-align: center; margin-bottom: 20px;'>סטטיסטיקה תיאורית של דגימות זמני ההגעה שנמדדו</h4>
        </div>
    """, unsafe_allow_html=True)

    # Create three columns for statistics
    col1, col2, col3 = st.columns(3)

    # First column
    with col1:
        st.markdown(f"""
            <div style= 'background-color: #2D2D2D;padding: 15px; border-radius: 5px; height: 100%;'>
                <h5 style='text-align: center; color: #FF4B4B; margin-bottom: 15px;'>מדדי מרכז</h5>
                <div style='text-align: right;'>
                    <p><strong>מספר מדידות:</strong> {len(samples):d}</p>
                    <p><strong>ממוצע:</strong> {np.mean(samples):.2f} דקות</p>
                    <p><strong>חציון:</strong> {np.median(samples):.2f} דקות</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Second column
    with col2:
        st.markdown(f"""
            <div style='background-color: #2D2D2D; padding: 15px; border-radius: 5px; height: 100%;'>
                <h5 style='text-align: center; margin-bottom: 15px;'>מדדי פיזור</h5>
                <div style='text-align: right;'>
                    <p><strong>סטיית תקן:</strong> {np.std(samples):.2f} דקות</p>
                    <p><strong>טווח בין-רבעוני:</strong> {np.percentile(samples, 75) - np.percentile(samples, 25):.2f} דקות</p>
                    <p><strong>שונות:</strong> {np.var(samples):.2f}</p>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Third column
    with col3:
        st.markdown(f"""
            <div style='background-color: #2D2D2D; padding: 15px; border-radius: 5px; height: 100%;'>
                <h5 style='text-align: center; color: #FF4B4B; margin-bottom: 15px;'>ערכי קיצון</h5>
                <div style='text-align: right;'>
                    <p><strong>מינימום:</strong> {np.min(samples):.2f} דקות</p>
                    <p><strong>מקסימום:</strong> {np.max(samples):.2f} דקות</p>
                    <p><strong>טווח:</strong> {np.max(samples) - np.min(samples):.2f} דקות</p>
                </div>
            </div>
        """, unsafe_allow_html=True)


def visualize_samples_and_qqplots(samples):
    """Display enhanced histograms and Q-Q plots using Plotly for consistent styling."""
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")
    # Explanation Section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3 style="padding-bottom: 3rem; color: #452b2b;">כעת נבחן את התפלגות הנתונים באמצעות כלים סטטיסטיים כדי לבחור את המודל המתאים ביותר לסימולציה:</h3>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1,3])

    with col1:
        # Instructions Section
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>כיצד לפרש את הגרפים:</h3>
                    <p><strong>היסטוגרמה:</strong> מציגה את התפלגות זמני ההכנה, ומאפשרת לבחון את צורת ההתפלגות.</p>
            </div>
        """, unsafe_allow_html=True)

                # Instructions Section
        st.markdown("""
            <div class="custom-card rtl-content">
                    <p><strong>תרשימי Q-Q:</strong> משווים את הנתונים להתפלגויות שונות, כשהתאמה גבוהה מתבטאת בקו ישר.</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="custom-card rtl-content">
                    <p><strong>רצועת אמון:</strong>האזור האפור מייצג רווח בר-סמך של 95% להתפלגות הנתונים.</p>
            </div>
        """, unsafe_allow_html=True)


    with col2:
        # Create main figure with subplots for histogram and Q-Q plots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("התפלגות זמני הגעה", "התפלגות נורמלית", "התפלגות אחידה", "התפלגות מעריכית"),
            vertical_spacing=0.15
        )

        # Add Histogram (similar to PDF plot in `create_distribution_plot`)
        fig.add_trace(
            go.Histogram(
                x=samples,
                histnorm='probability density',
                name='היסטוגרמה',
                marker=dict(color='rgba(139, 0, 0, 0.5)')
            ),
            row=1, col=1
        )

        # Q-Q Plots for different distributions
        distributions = [
            ('norm', 'Normal Distribution', 1, 2),
            ('uniform', 'Uniform Distribution', 2, 1),
            ('expon', 'Exponential Distribution', 2, 2)
        ]

        for dist_name, dist_title, row, col in distributions:
            qq = stats.probplot(samples, dist=dist_name)
            x = qq[0][0]  # theoretical quantiles
            y = qq[0][1]  # sample quantiles
            slope, intercept = qq[1][0], qq[1][1]
            y_fit = slope * x + intercept

            # Add Q-Q scatter plot
            fig.add_trace(
                go.Scatter(
                    x=x, y=y, mode='markers', name=f'{dist_title}',
                    marker=dict(color='#452b2b', size=5),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add Q-Q fit line
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_fit, mode='lines', name=f'{dist_title} Fit Line',
                    line=dict(color='#452b2b', width=2),
                    showlegend=False
                ),
                row=row, col=col
            )

            # Add 95% confidence interval band
            n = len(samples)
            sigma = np.std((y - y_fit) / np.sqrt(1 - 1/n))
            conf_band = 1.96 * sigma
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_fit + conf_band, mode='lines', name='Upper Confidence',
                    line=dict(color='gray', width=1, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=y_fit - conf_band, mode='lines', name='Lower Confidence',
                    line=dict(color='gray', width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(200, 200, 200, 0.2)',
                    showlegend=False
                ),
                row=row, col=col
            )

        # Layout adjustments
        fig.update_layout(
            height=700,
            title_text="ניתוח גרפי של התפלגות הנתונים",
            title_x=0.5,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

def estimate_parameters(samples, distribution):

    
    # Section header with explanation
    """
    Estimate parameters for a given distribution using Maximum Likelihood Estimation (MLE)
    and compute 95% confidence intervals using bootstrapping.

    Parameters
    ----------
    samples : array-like
        Sample data from the distribution of interest
    distribution : str
        Name of the distribution (either 'Normal', 'Exponential', or 'Uniform')

    Returns
    -------
    parameters : tuple
        Estimated parameters for the given distribution (e.g., mean and standard deviation
        for a Normal distribution, rate for an Exponential distribution, or minimum and
        maximum values for a Uniform distribution)
    """
    
    st.markdown("""
    <div class="custom-card rtl-content">
        <h1 class="section-header" style="color: #8B0000;">אמידת פרמטרים לסימולציה</h1>
        <p>כדי לייצר זמני הכנה מציאותיים בסימולציה, נבצע אמידה של הפרמטרים המרכזיים של ההתפלגות הנבחרת ונחשב רווחי בר-סמך לכל אחד מהם:</p>
    </div>
    """, unsafe_allow_html=True)
        
    # Two-column layout for results and visuals
    col1, col2 = st.columns([0.8, 0.2])
    
    if distribution == 'Normal':
        with col1:
            # Parameter estimation using MLE for Normal distribution
            mu, sigma = stats.norm.fit(samples)
            
            # Bootstrap confidence intervals for mean and standard deviation
            bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
            bootstrap_means = np.mean(bootstrap_samples, axis=1)
            bootstrap_stds = np.std(bootstrap_samples, axis=1)
            mu_ci = np.percentile(bootstrap_means, [2.5, 97.5])
            sigma_ci = np.percentile(bootstrap_stds, [2.5, 97.5])
            
            plot_likelihood(samples, distribution)  # Visual plot of likelihood
        
        with col2:
            # Display parameter estimates and confidence intervals
            st.markdown(f"""
                <div class="info-box rtl-content">
                    <h4>פרמטרים עבור התפלגות נורמלית:</h4>
                    <ul>
                        <li><strong>ממוצע (μ):</strong> {mu:.2f} <span style="color:gray;">[CI: {mu_ci[0]:.2f}, {mu_ci[1]:.2f}]</span></li>
                        <li><strong>סטיית תקן (σ):</strong> {sigma:.2f} <span style="color:gray;">[CI: {sigma_ci[0]:.2f}, {sigma_ci[1]:.2f}]</span></li>
                    </ul>
                    <p>ממוצע משקף את זמן ההכנה הממוצע, בעוד סטיית התקן מציינת את רמת השונות בזמני ההכנה.</p>
                </div>
            """, unsafe_allow_html=True)
        
        return mu, sigma

    elif distribution == 'Exponential':
        with col1:
            # Parameter estimation for Exponential distribution
            lambda_est = 1 / np.mean(samples)
            
            # Bootstrap confidence interval for lambda
            bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
            bootstrap_lambdas = 1 / np.mean(bootstrap_samples, axis=1)
            lambda_ci = np.percentile(bootstrap_lambdas, [2.5, 97.5])
            
            plot_likelihood(samples, distribution)
        
        with col2:
            st.markdown(f"""
                <div class="info-box rtl-content">
                    <h4>פרמטרים עבור התפלגות מעריכית:</h4>
                    <ul>
                        <li><strong>קצב (λ):</strong> {lambda_est:.4f} <span style="color:gray;">[CI: {lambda_ci[0]:.4f}, {lambda_ci[1]:.4f}]</span></li>
                        <li><strong>זמן ממוצע (1/λ):</strong> {1/lambda_est:.2f} דקות</li>
                    </ul>
                    <p>פרמטר הקצב (λ) מציין את התדירות המשוערת של אירועים, כמו זמני הכנה בתקופות עומס.</p>
                </div>
            """, unsafe_allow_html=True)
        
        return lambda_est,

    elif distribution == 'Uniform':
        with col1:
            # Parameter estimation for Uniform distribution
            a, b = np.min(samples), np.max(samples)
            
            # Bootstrap confidence intervals for min and max values
            bootstrap_samples = np.random.choice(samples, size=(1000, len(samples)), replace=True)
            bootstrap_mins = np.min(bootstrap_samples, axis=1)
            bootstrap_maxs = np.max(bootstrap_samples, axis=1)
            a_ci = np.percentile(bootstrap_mins, [2.5, 97.5])
            b_ci = np.percentile(bootstrap_maxs, [2.5, 97.5])
            
            plot_likelihood(samples, distribution)
        
        with col2:
            st.markdown(f"""
                <div class="info-box rtl-content">
                    <h4>פרמטרים עבור התפלגות אחידה:</h4>
                    <ul>
                        <li><strong>מינימום (a):</strong> {a:.2f} <span style="color:gray;">[CI: {a_ci[0]:.2f}, {a_ci[1]:.2f}]</span></li>
                        <li><strong>מקסימום (b):</strong> {b:.2f} <span style="color:gray;">[CI: {b_ci[0]:.2f}, {b_ci[1]:.2f}]</span></li>
                        <li><strong>טווח:</strong> {b-a:.2f} דקות</li>
                    </ul>
                    <p>התפלגות אחידה מתארת טווחי הכנה צפויים, ומתאימה לתנאים בהם זמני הכנה הם קבועים.</p>
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
    """Enhanced likelihood function visualization with Plotly and consistent styling."""

    if distribution == 'Normal':
        # Set up figure with two subplots for μ and σ likelihoods
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("פונקציית לוג הנראות עבור הממוצע (μ)", "פונקציית לוג הנראות עבור סטיית התקן (σ)"),
            horizontal_spacing=0.15
        )

        # Range of parameter values
        mu_vals = np.linspace(np.mean(samples) - 3 * np.std(samples), np.mean(samples) + 3 * np.std(samples), 100)
        sigma_vals = np.linspace(0.2 * np.std(samples), 2 * np.std(samples), 100)

        # Log-Likelihood calculations
        ll_mu = [np.sum(stats.norm.logpdf(samples, loc=mu, scale=np.std(samples))) for mu in mu_vals]
        ll_sigma = [np.sum(stats.norm.logpdf(samples, loc=np.mean(samples), scale=sigma)) for sigma in sigma_vals]

        # Plot Log-Likelihood for Mean (μ)
        fig.add_trace(
            go.Scatter(
                x=mu_vals, y=ll_mu, mode='lines', name='Log-Likelihood for μ',
                line=dict(color='#452b2b', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[np.mean(samples)], y=[max(ll_mu)], mode='markers', name='Estimated μ',
                marker=dict(color='gray', size=8, symbol='x')
            ),
            row=1, col=1
        )

        # Plot Log-Likelihood for Standard Deviation (σ)
        fig.add_trace(
            go.Scatter(
                x=sigma_vals, y=ll_sigma, mode='lines', name='Log-Likelihood for σ',
                line=dict(color='#452b2b', width=2)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[np.std(samples)], y=[max(ll_sigma)], mode='markers', name='Estimated σ',
                marker=dict(color='gray', size=8, symbol='x')
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            height=400,
            title_text="פונקציות לוג הנראות עבור פרמטרים של התפלגות נורמלית",
            title_x=0.5,
            showlegend=False
        )
        fig.update_xaxes(title_text="μ", row=1, col=1)
        fig.update_yaxes(title_text="Log-Likelihood", row=1, col=1)
        fig.update_xaxes(title_text="σ", row=1, col=2)
        fig.update_yaxes(title_text="Log-Likelihood", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

    elif distribution == 'Uniform':
        # Set up figure with two subplots for a and b likelihoods
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("פונקציית לוג הנראות עבור מינימום (a)", "פונקציית לוג הנראות עבור מקסימום (b)"),
            horizontal_spacing=0.15
        )

        # Range of parameter values
        margin = (np.max(samples) - np.min(samples)) * 0.2
        a_vals = np.linspace(np.min(samples) - margin, np.min(samples) + margin, 100)
        b_vals = np.linspace(np.max(samples) - margin, np.max(samples) + margin, 100)
        fixed_b = np.max(samples)
        fixed_a = np.min(samples)

        # Log-Likelihood calculations
        ll_a = [np.sum(stats.uniform.logpdf(samples, loc=a, scale=fixed_b - a)) if fixed_b > a else -np.inf for a in a_vals]
        ll_b = [np.sum(stats.uniform.logpdf(samples, loc=fixed_a, scale=b - fixed_a)) if b > fixed_a else -np.inf for b in b_vals]

        # Plot Log-Likelihood for Minimum (a)
        fig.add_trace(
            go.Scatter(
                x=a_vals, y=ll_a, mode='lines', name='Log-Likelihood for a',
                line=dict(color='#452b2b', width=2)
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=[np.min(samples)], y=[max(ll_a)], mode='markers', name='Estimated a',
                marker=dict(color='gray', size=8, symbol='x')
            ),
            row=1, col=1
        )

        # Plot Log-Likelihood for Maximum (b)
        fig.add_trace(
            go.Scatter(
                x=b_vals, y=ll_b, mode='lines', name='Log-Likelihood for b',
                line=dict(color='#452b2b', width=2)
            ),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[np.max(samples)], y=[max(ll_b)], mode='markers', name='Estimated b',
                marker=dict(color='gray', size=8, symbol='x')
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            height=400,
            title_text="פונקציות לוג הנראות עבור פרמטרים של התפלגות אחידה",
            title_x=0.5,
            showlegend=False
        )
        fig.update_xaxes(title_text="a", row=1, col=1)
        fig.update_yaxes(title_text="Log-Likelihood", row=1, col=1)
        fig.update_xaxes(title_text="b", row=1, col=2)
        fig.update_yaxes(title_text="Log-Likelihood", row=1, col=2)

        st.plotly_chart(fig, use_container_width=True)

    elif distribution == 'Exponential':
        # Set up figure for λ likelihood
        fig = go.Figure()

        # Range of parameter values for λ
        lambda_vals = np.linspace(1 / (2 * np.mean(samples)), 2 / np.mean(samples), 100)
        ll_lambda = [np.sum(stats.expon.logpdf(samples, scale=1 / lambda_val)) for lambda_val in lambda_vals]

        # Plot Log-Likelihood for Rate Parameter (λ)
        fig.add_trace(
            go.Scatter(
                x=lambda_vals, y=ll_lambda, mode='lines', name='Log-Likelihood for λ',
                line=dict(color='#452b2b', width=2)
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[1 / np.mean(samples)], y=[max(ll_lambda)], mode='markers', name='Estimated λ',
                marker=dict(color='gray', size=8, symbol='x')
            )
        )

        # Update layout
        fig.update_layout(
            height=400,
            title="פונקציית לוג הנראות עבור פרמטר הקצב (λ)",
            xaxis_title="λ",
            yaxis_title="Log-Likelihood",
            title_x=0.5,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

def perform_goodness_of_fit(samples, distribution, params):
    # Header with explanation for goodness-of-fit testing
    """
    Perform goodness-of-fit tests for a given distribution and sample data.

    Parameters
    ----------
    samples : array-like
        The sample data to be tested.
    distribution : str
        The distribution to test against. Can be one of 'Normal', 'Exponential',
        or 'Uniform'.
    params : tuple
        The parameters for the given distribution.

    Notes
    -----
    This function performs two tests: the Kolmogorov-Smirnov test and the Chi-Square
    test. The results of each test are displayed in a styled card, and a visualization
    of the fit is also displayed.
    """

    st.markdown("""
        <div class="custom-card rtl-content">
            <h1 class="section-header" style="color: #8B0000;">בדיקת התאמת המודל</h1>
            <p>לפני שימוש במודל בסימולציה, נוודא שהוא מתאר היטב את המציאות. נבצע מבחני טיב התאמה לבדיקת מידת ההתאמה:</p>
        </div>
    """, unsafe_allow_html=True)

    # Calculate bins for histogram
    iqr = stats.iqr(samples)
    bin_width = 2 * iqr / (len(samples) ** (1/3))
    n_bins = max(5, min(int(np.ceil((np.max(samples) - np.min(samples)) / bin_width)), 50))

    # Perform Chi-Square Test
    observed_freq, bins = np.histogram(samples, bins=n_bins)
    bin_midpoints = (bins[:-1] + bins[1:]) / 2
    
    if distribution == 'Normal':
        mu, sigma = params
        expected_probs = stats.norm.cdf(bins[1:], mu, sigma) - stats.norm.cdf(bins[:-1], mu, sigma)
        dof = len(observed_freq) - 3
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
        if min_idx == 0:
            observed_freq[0:2] = np.sum(observed_freq[0:2])
            expected_freq[0:2] = np.sum(expected_freq[0:2])
            observed_freq = np.delete(observed_freq, 1)
            expected_freq = np.delete(expected_freq, 1)
        elif min_idx == len(expected_freq) - 1:
            observed_freq[-2:] = np.sum(observed_freq[-2:])
            expected_freq[-2:] = np.sum(expected_freq[-2:])
            observed_freq = np.delete(observed_freq, -1)
            expected_freq = np.delete(expected_freq, -1)
        else:
            observed_freq[min_idx:min_idx+2] = np.sum(observed_freq[min_idx:min_idx+2])
            expected_freq[min_idx:min_idx+2] = np.sum(expected_freq[min_idx:min_idx+2])
            observed_freq = np.delete(observed_freq, min_idx+1)
            expected_freq = np.delete(expected_freq, min_idx+1)
    
    # Chi-Square test
    chi_square_stat = np.sum((observed_freq - expected_freq) ** 2 / expected_freq)
    p_value_chi = 1 - stats.chi2.cdf(chi_square_stat, max(1, dof))
    
    # Kolmogorov-Smirnov test
    if distribution == 'Normal':
        ks_stat, p_value_ks = stats.kstest(stats.zscore(samples), 'norm')
    elif distribution == 'Exponential':
        scaled_samples = samples * lambda_param
        ks_stat, p_value_ks = stats.kstest(scaled_samples, 'expon')
    elif distribution == 'Uniform':
        scaled_samples = (samples - a) / (b - a)
        ks_stat, p_value_ks = stats.kstest(scaled_samples, 'uniform')
    
    # Display results in styled cards
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
            <div class="custom-back-card rtl-content" style="background-color: #3f0000; padding: 15px; border-radius: 8px;">
                <h4>מבחן Kolmogorov-Smirnov</h4>
                <ul style="list-style-type: none; padding-left: 0; color: white;">
                    <li><strong>סטטיסטיקה:</strong> {ks_stat:.4f}</li>
                    <li><strong>ערך-p:</strong> {p_value_ks:.4f}</li>
                    <li><strong>מסקנה:</strong> {"דחיית H0" if p_value_ks < 0.05 else "כשלון לדחות H0"}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
        st.markdown(f"""
            <div class="custom-back-card rtl-content" style="background-color: #3f0000; padding: 15px; border-radius: 8px;">
                <h4>מבחן Chi-Square</h4>
                <ul style="list-style-type: none; padding-left: 0; color: white;">
                    <li><strong>סטטיסטיקה:</strong> {chi_square_stat:.4f}</li>
                    <li><strong>דרגות חופש:</strong> {dof}</li>
                    <li><strong>ערך-p:</strong> {p_value_chi:.4f}</li>
                    <li><strong>מסקנה:</strong> {"דחיית H0" if p_value_chi < 0.05 else "כשלון לדחות H0"}</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Visualization of the fit
    with col2:
        fig = go.Figure()

        # Histogram of the data
        fig.add_trace(
            go.Histogram(
                x=samples,
                histnorm='probability density',
                name='נתונים',
                marker=dict(color='rgba(139, 0, 0, 0.5)')
            )
        )

        # Fitted distribution line
        x = np.linspace(np.min(samples), np.max(samples), 100)
        if distribution == 'Normal':
            pdf = stats.norm.pdf(x, *params)
            fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='התפלגות נורמלית מותאמת', line=dict(color='darkred', width=2)))
        elif distribution == 'Exponential':
            pdf = stats.expon.pdf(x, scale=1/params[0])
            fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='התפלגות מעריכית מותאמת', line=dict(color='darkred', width=2)))
        elif distribution == 'Uniform':
            pdf = stats.uniform.pdf(x, *params)
            fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='התפלגות אחידה מותאמת', line=dict(color='darkred', width=2)))

        # Layout adjustments
        fig.update_layout(
            title="התאמת התפלגות לנתונים",
            xaxis_title="ערכים",
            yaxis_title="צפיפות",
            height=400,
            showlegend=True,
            title_x=0.5
        )

        st.plotly_chart(fig, use_container_width=True)


def create_styled_card(title, content, border_color="#453232"):
    st.markdown(
        f"""
        <div style="
            background-color: #2D2D2D;
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        ">
            <h3 style="
                color: #FFFFFF;
                margin-bottom: 15px;
                text-align: right;
                font-size: 1.2rem;
            ">{title}</h3>
            <div style="
                color: #FFFFFF;
                text-align: right;
            ">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )



def show():
    """
    This function shows the interactive simulation page.

    It displays the introduction section, instructions for using the interactive tool, and the objective of the simulation.
    It also displays the three distribution options and the button to select each distribution.
    Additionally, it displays the goodness of fit results if a distribution is selected.

    Parameters:
    None

    Returns:
    None
    """
    samples = None
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Show introduction section
    show_introduction()
    st.text(" ")
    st.text(" ")
    st.text(" ")
    st.text(" ")

    # Instructions for Interactive Tool
    st.markdown("""
        <div class="custom-card rtl-content">
            <h3>בעמוד זה נחקור את השלבים הבאים:</h3>
            <ol class="custom-list">
                <li><b>יצירת מדגם חדש:</b> לחיצה על אחד הכפתורים  מטה, תפיק דגימה עדכנית של זמני הגעה, או דגימה מהתפלגות רנדומלית עבור המשך תרגול .</li>
                <li><b>התאמת התפלגות:</b> עבור כל מדגם, הכלי מציע מספר אפשרויות התאמת התפלגות (כגון התפלגות נורמלית, אחידה או מעריכית), כך שניתן לבחור את ההתפלגות המשקפת בצורה הטובה ביותר את דפוסי ההגעה.</li>
                <li><b>בדיקת טיב ההתאמה:</b> הכלי מבצע בדיקה של טיב ההתאמה, כדי לוודא שההתפלגות שנבחרה מתאימה למאפייני המדגם, ובכך מאפשר ייצוג מדויק בסימולציה.</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)


    st.text(" ")
    st.text(" ")

    col_arrival_times, col_service_times = st.columns(2)

    # Initialize session state for samples if not already present
    if 'samples' not in st.session_state:
        st.session_state.samples = None
    if 'dist_info' not in st.session_state:
        st.session_state.dist_info = None

    with col_arrival_times:
        arrival_button = st.button('יצירת מדגם מזמני ההגעה')
        if arrival_button: 
            st.session_state.samples, st.session_state.dist_info = generate_arrival_times()

    with col_service_times:
        random_button = st.button('יצירת מדגם מהתפלגות רנדומלית')
        if random_button: 
            st.session_state.samples, st.session_state.dist_info = generate_service_times()

    # Use samples from session state
    samples = st.session_state.samples

    if samples is not None:
        display_samples(samples)
        visualize_samples_and_qqplots(samples)

        # Distribution selection section with business context
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3 class="section-header">בחירת התפלגות מתאימה</h3>
                <p>
                    על בסיס הניתוח הגרפי, יש לבחור את ההתפלגות המשקפת באופן המדויק ביותר את זמני ההכנה במשאית המזון.
                    כל התפלגות מתאימה לסוג שונה של תרחיש עסקי, ומאפשרת לנו לחזות את זמני ההמתנה בצורה מיטבית:
                </p>
            </div>
        """, unsafe_allow_html=True)

        # Create three columns for the distribution options
        col1, col2, col3 = st.columns(3)

        # Store distribution choice in session state
        if 'distribution_choice' not in st.session_state:
            st.session_state.distribution_choice = None
        if 'goodness_of_fit_results' not in st.session_state:
            st.session_state.goodness_of_fit_results = None

        with col1:
            st.markdown("""
                <div class="custom-card rtl-content" style="background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #452b2b;">
                    <h4>התפלגות נורמלית</h4>
                    <p>מתאימה עבור מנות סטנדרטיות עם זמן הכנה עקבי יחסית, כמו הזמנות רגילות בימים ללא עומס.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("בחר התפלגות נורמלית", key="normal"):
                st.session_state.distribution_choice = 'Normal'

        with col2:
            st.markdown("""
                <div class="custom-card rtl-content" style="background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #452b2b;">
                    <h4>התפלגות אחידה</h4>
                    <p>מתאימה עבור מנות פשוטות עם טווח זמן הכנה גמיש, המתאימות לתנאים משתנים.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("בחר התפלגות אחידה", key="uniform"):
                st.session_state.distribution_choice = 'Uniform'

        with col3:
            st.markdown("""
                <div class="custom-card rtl-content" style="background-color: #1E1E1E; padding: 15px; border-radius: 8px; border: 1px solid #452b2b;">
                    <h4>התפלגות מעריכית</h4>
                    <p>מתאימה עבור מנות מורכבות או הזמנות שמתקבלות בשעות עומס, כשהזמן מתארך ככל שהעומס גובר.</p>
                </div>
            """, unsafe_allow_html=True)
            if st.button("בחר התפלגות מעריכית", key="exponential"):
                st.session_state.distribution_choice = 'Exponential'

        # Perform goodness of fit test if distribution is selected
        if st.session_state.distribution_choice and st.session_state.samples is not None:
            params = estimate_parameters(st.session_state.samples, st.session_state.distribution_choice)
            st.session_state.goodness_of_fit_results = perform_goodness_of_fit(
                st.session_state.samples, 
                st.session_state.distribution_choice, 
                params
            )
            
            # Display goodness of fit results
            if st.session_state.goodness_of_fit_results:
                st.markdown("""
                    <div class="custom-card rtl-content">
                        <h3>תוצאות בדיקת טיב ההתאמה</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                # Display the results in a formatted way
                st.write(st.session_state.goodness_of_fit_results)

 

       
# To show the app, call the show() function
if __name__ == "__main__":
    show()
