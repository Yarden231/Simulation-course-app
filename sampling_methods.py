import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import set_rtl
from utils import set_ltr_sliders
import time
from statsmodels.graphics.gofplots import qqplot


class LFSR:
    def __init__(self, seed, taps):
        """מאתחל את LFSR (רישום היסט מונע משוב) ליצירת מספרים אקראיים בעזרת ערך התחלתי ומיקומי XOR"""
        self.state = seed
        self.taps = taps
        self.nbits = len(seed)

    def next(self):
        """מייצר את המצב הבא של ה-LFSR בעזרת פעולות XOR"""
        xor = 0
        for t in self.taps:
            xor ^= int(self.state[t - 1])
        self.state = str(xor) + self.state[:-1]
        return self.state

    def random(self):
        """ממיר את המצב הנוכחי למספר אקראי בין 0 ל-1"""
        self.next()
        return int(self.state, 2) / (2 ** self.nbits)

class LCG:
    def __init__(self, seed, a=1664525, c=1013904223, m=2**32):
        """מאתחל את LCG (מחולל קונגרואנציאלי ליניארי) על פי פרמטרים קבועים ליצירת מספרים אקראיים"""
        self.state = seed
        self.a = a
        self.c = c
        self.m = m

    def next(self):
        """מחשב את המצב הבא של ה-LCG לפי נוסחת מחולל קונגרואנציאלי"""
        self.state = (self.a * self.state + self.c) % self.m
        return self.state

    def random(self):
        """ממיר את המצב הנוכחי למספר אקראי בין 0 ל-1"""
        return self.next() / self.m


# Update the create_styled_card function with better spacing
def create_styled_card(title, content, border_color= '#2D2D2D' ):

    """
    Creates a styled card with a title and content, aligned to the right.

    Parameters
    ----------
    title : str
        The title of the card
    content : str
        The content of the card
    border_color : str, optional
        The color of the border (default is `#2D2D2D`)

    Returns
    -------
    None
    """
    

    st.markdown(
        f"""
        <div style="
            background-color: #1E1E1E;
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 10px;
            margin: 25px 0;  /* Increased margin */
        ">
            <h3 style="
                color: #FFFFFF;
                margin-bottom: 10px;  /* Increased margin */
                text-align: right;
                font-size: 1.2rem;
            ">{title}</h3>
            <div style="
                color: #FFFFFF;
                text-align: right;
                line-height: 1.6;  /* Added line height */
            ">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Update the create_styled_card function with better spacing
def create_styled_card_left(title, content, border_color= '#2D2D2D' ):
    """
    Creates a styled card with a title and content, aligned to the left.

    Parameters
    ----------
    title : str
        The title of the card
    content : str
        The content of the card
    border_color : str, optional
        The color of the border (default is `#2D2D2D`)

    Returns
    -------
    None
    """

    st.markdown(
        f"""
        <div style="
            background-color: #1E1E1E;
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 10px;
            margin: 25px 0;  /* Increased margin */
        ">
            <h3 style="
                color: #FFFFFF;
                margin-bottom: 10px;  /* Increased margin */
                text-align: left;
                font-size: 1.2rem;
            ">{title}</h3>
            <div style="
                color: #FFFFFF;
                text-align: left;
                line-height: 1.6;  /* Added line height */
            ">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_station_grid():
    stations = [
        ("👥", "הזמנה"),
        ("👨‍🍳", "הכנה"),
        ("📦", "אריזה")
    ]
    
    cols = st.columns(3)
    for idx, (emoji, name) in enumerate(stations):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="
                    background-color: #2D2D2D;
                    border: 1px solid #8B0000;
                    border-radius: 8px;
                    padding: 10px;
                    text-align: center;
                    height: 100%;
                ">
                    <div style="font-size: 2rem; margin-bottom: 10px;">{emoji}</div>
                    <h4 style="color: #FFFFFF; margin: 0; font-size: 1.1rem;">{name}</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

def create_sampling_methods_grid():
    """
    Creates a grid of 3 columns with 3 cards each, explaining a different sampling method.

    The sampling methods are:

    1. טרנספורם הופכי - a method for sampling from the exponential distribution, used to sample customer arrival times.
    2. דגימת קבלה-דחייה - a method for sampling from a complex distribution, used to sample preparation times for different dishes.
    3. שיטת הקומפוזיציה - a method for sampling customer waiting times based on different levels of patience, by combining distributions.

    The cards are styled with a red border and white text.
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_styled_card(
            "טרנספורם הופכי",
            "שיטה לדגימת מספרים אקראיים מהתפלגות המעריכית, המשמשת לדגימת זמני הגעת לקוחות."
        )
    with col2:
        create_styled_card(
            "דגימת קבלה-דחייה",
            "שיטה לדגימת מספרים מהתפלגות מורכבת, כגון דגימת זמני הכנה שונים למנות שונות."
        )
    with col3:
        create_styled_card(
            "שיטת הקומפוזיציה",
            "שיטה לדגימת זמני המתנה של לקוחות לפי רמות סבלנות שונות, על ידי שילוב של התפלגויות."
        )

def plot_qq(samples, title):
    """
    Plot a Q-Q plot of the given samples, with a given title.

    Parameters
    ----------
    samples : array_like
        The samples to plot.
    title : str
        The title of the plot.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.
    """
    
    fig, ax = plt.subplots(figsize=(8, 5))
    stats.probplot(samples, dist="norm", plot=ax)
    
    ax.get_lines()[0].set_markerfacecolor('#8B0000')
    ax.get_lines()[0].set_markeredgecolor('#8B0000')
    ax.get_lines()[1].set_color('#8B0000')
    
    ax.set_title(f"{title}\nQ-Q Plot")
    ax.grid(True, alpha=0.3)
    return fig

def run_sampling(sampling_function, num_samples, update_interval, title, plot_placeholder, stats_placeholder,  distribution_func=None, true_density=None):

    # Generate all samples at once
    """
    Run a sampling function, display results in batches, and update Streamlit 
    placeholders for a Q-Q plot and histogram, as well as statistics.

    Parameters
    ----------
    sampling_function : callable
        The sampling function to run, which takes a single argument of the number
        of samples to generate.
    num_samples : int
        The number of samples to generate.
    update_interval : int
        The number of samples to generate at a time before updating the Streamlit
        placeholders.
    title : str
        The title for the combined figure.
    plot_placeholder : streamlit.container.Container
        The Streamlit container to display the figure in.
    stats_placeholder : streamlit.container.Container
        The Streamlit container to display the statistics in.
    distribution_func : callable, optional
        The distribution function to use for the true density of the sample data.
        If None, the true density is not plotted.
    true_density : callable, optional
        The true density of the sample data. If None, the true density is not plotted.
    """
    
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
        
        # Create a single figure with 1x2 grid for Q-Q plot and histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Consistent size for both subplots
        
        # Generate Q-Q plot in ax1
        qqplot(np.array(samples), line='s', ax=ax1)
        ax1.get_lines()[0].set_markerfacecolor('#8B0000')
        ax1.get_lines()[0].set_markeredgecolor('#8B0000')
        ax1.get_lines()[1].set_color('#8B0000')
        ax1.set_title("Q-Q Plot")
        ax1.set_xlabel("Theoretical Quantiles")
        ax1.set_ylabel("Sample Quantiles")
        ax1.grid(True, alpha=0.3)
        
        # Generate histogram in ax2
        bins = np.linspace(min(samples), max(samples), 30)
        ax2.hist(samples, bins=bins, density=True, color='#8B0000', edgecolor='black', alpha=0.7, label='Sampled Data')
        if true_density:
            x = np.linspace(min(samples), max(samples), 100)
            ax2.plot(x, true_density(x), 'darkred', linewidth=2, label='True Density')
        ax2.set_title("Sample Histogram")
        ax2.set_xlabel("Value")
        ax2.set_ylabel("Density")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Add main title to the combined figure
        fig.suptitle(title, fontsize=16, weight='bold')

        # Display the combined figure in Streamlit
        with plot_placeholder.container():
            st.pyplot(fig)
            plt.close(fig)

        # Update statistics placeholder
        stats_placeholder.empty()
        with stats_placeholder:
            display_statistics(samples)
        
def show_sampling_intro():
    create_styled_card(
        "אלגוריתמי דגימה  🎲",
        """
        דגימה היא תהליך קריטי ליצירת סימולציות המסייעות בקבלת החלטות עסקיות. בעמוד זה נלמד את השיטות השונות לדגימת מספרים אקראיים
        אשר מסייעות בסימולציה של תהליכי שירות, כמו תכנון זמני המתנה של לקוחות ותפעול יעיל של משמרות.
        """
    )


def display_inverse_transform_method():
    
    create_styled_card(
        "טרנספורם הופכי - דגימת זמני הגעה",
        """
        שיטת הטרנספורם ההופכי מאפשרת דגימה מהתפלגות מעריכית, אשר מתארת את זמני ההגעה של לקוחות למשאית. 
        בעזרת נוסחת ההפוך של ההתפלגות, אנו ממירים מספרים אקראיים בהתפלגות אחידה למספרים המתאימים להתפלגות המעריכית.
        """
    )
    
    # Displaying LaTeX equations outside of HTML block
    st.markdown(
        """
        <div dir="rtl" style="text-align: right;">
            <ul style="list-style-type: none; padding-right: 20px;">
                <li>פונקציית הצפיפות:</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.latex(r"f(x) = \lambda e^{-\lambda x}")
    
    st.markdown(
        """
        <div dir="rtl" style="text-align: right;">
            <ul style="list-style-type: none; padding-right: 20px;">
                <li>פונקציה מצטברת:</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.latex(r"F(x) = 1 - e^{-\lambda x}")
    
    st.markdown(
        """
        <div dir="rtl" style="text-align: right;">
            <ul style="list-style-type: none; padding-right: 20px;">
                <li>טרנספורם הופכי:</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.latex(r"x = -\frac{\ln(1-U)}{\lambda}")

    st.markdown(
        """
        <div dir="rtl" style="text-align: right;">
            <ul style="list-style-type: none; padding-right: 20px;">
                <li>כאשר U הוא מספר אקראי בין 0 ל-1.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )



    lambda_param = st.slider("קצב הגעה (לקוחות לשעה)", 1.0, 20.0, 10.0)
    num_samples = st.slider("מספר דגימות", 100, 10000, 1000)

    if st.button("הרץ סימולציה"):
        samples = run_inverse_transform_simulation(lambda_param, num_samples)
        plot_histogram(samples, "התפלגות זמני הגעה")

def display_rejection_method():
    create_styled_card(
        "דגימת קבלה-דחייה - זמני הכנת מנות",
        """
        שיטה זו משמשת לדגימת זמני הכנה למנות בעלות זמני הכנה משתנים. בעזרת פונקציית מעטפת המכסה את ההתפלגות הרצויה,
        ניתן לדגום מתוך התפלגות המייצגת זמני הכנה שונים. 
        
        <ul dir="rtl" style="text-align: right;">
            <li>מנה מהירה: התפלגות אחידה (3-4 דקות)</li>
            <li>מנה רגילה: התפלגות משולשית (4-6 דקות)</li>
            <li>מנה מורכבת: זמן קבוע (10 דקות)</li>
        </ul>
        """
    )

    num_samples = st.slider("מספר דגימות", 100, 10000, 1000)
    if st.button("הרץ סימולציה"):
        samples = run_rejection_simulation(num_samples)
        plot_histogram(samples, "זמני הכנת מנות")

def display_composition_method():
    create_styled_card(
        "שיטת הקומפוזיציה - זמני המתנה",
        """
        שיטה זו משמשת לדגימת זמני המתנה של לקוחות בעלי רמות סבלנות שונות. לדוגמה:
        
        <ul dir="rtl" style="text-align: right;">
            <li>לקוחות בעלי סבלנות נמוכה (30%): 5-10 דקות</li>
            <li>לקוחות בעלי סבלנות בינונית (40%): 10-15 דקות</li>
            <li>לקוחות בעלי סבלנות גבוהה (30%): 15-20 דקות</li>
        </ul>
        השיטה משלבת בין מספר התפלגויות פשוטות כדי לייצר התפלגות מורכבת המייצגת את זמני ההמתנה.
        """
    )

    num_samples = st.slider("מספר דגימות", 100, 10000, 1000)
    if st.button("הרץ סימולציה"):
        samples = run_composition_simulation(num_samples)
        plot_histogram(samples, "זמני המתנה של לקוחות")

def composition_sample_wait_time():
    """Generate a sample based on customer patience levels."""
    patience_level = np.random.choice(
        ["low", "medium", "high"], 
        p=[0.3, 0.4, 0.3]  # Probabilities for each patience level
    )

    if patience_level == "low":
        # Low patience: Sample uniformly between 5 and 10 minutes
        sample = np.random.uniform(5, 10)
    elif patience_level == "medium":
        # Medium patience: Sample uniformly between 10 and 15 minutes
        sample = np.random.uniform(10, 15)
    elif patience_level == "high":
        # High patience: Sample uniformly between 15 and 20 minutes
        sample = np.random.uniform(15, 20)

    return sample

def display_statistics(samples):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_styled_card(
            "מדדי מרכז",
            f"""
            <div dir="rtl" style="text-align: right;">
                ממוצע: {np.mean(samples):.2f} דקות<br>
                חציון: {np.median(samples):.2f} דקות
            </div>
            """
        )
    
    with col2:
        create_styled_card(
            "מדדי פיזור",
            f"""
            <div dir="rtl" style="text-align: right;">
                סטיית תקן: {np.std(samples):.2f}<br>
                טווח: {np.min(samples):.2f} - {np.max(samples):.2f}
            </div>
            """
        )
    
    with col3:
        create_styled_card(
            "מדדי צורה",
            f"""
            <div dir="rtl" style="text-align: right;">
                אסימטריה: {stats.skew(samples):.2f}<br>
                קורטוזיס: {stats.kurtosis(samples):.2f}
            </div>
            """
        )


    
    st.markdown(
        """
        <h2 dir="rtl" style="text-align: right; margin: 30px 0 20px;">בחר שיטת דגימה להמחשה:</h2>
        """,
        unsafe_allow_html=True
    )
    
    method = st.radio("", [
        "טרנספורם הופכי - זמני הגעת לקוחות",
        "דגימת קבלה-דחייה - זמני הכנת מנות",
        "שיטת הקומפוזיציה - זמני המתנה"
    ], index=0)

    if "טרנספורם הופכי" in method:
        display_inverse_transform_method()
    elif "קבלה-דחייה" in method:
        display_rejection_method()
    elif "קומפוזיציה" in method:
        display_composition_method()

def run_inverse_transform_simulation(lambda_param, num_samples):
    # Simulate exponential distribution using inverse transform
    u = np.random.uniform(0, 1, num_samples)
    samples = -np.log(1-u) / lambda_param
    return samples

def run_rejection_simulation(num_samples):
    """Simulate using rejection sampling"""
    samples = []

    for i in range(num_samples):
        sample = rejection_sample_prep_time()
        samples.append(sample)

    
    return np.array(samples)

def run_composition_simulation(num_samples):
    """Simulate using composition method"""
    samples = []
    
    for i in range(num_samples):
        sample = composition_sample_wait_time()
        samples.append(sample)

    
    return np.array(samples)

def plot_histogram(samples, title):
    # Plot histogram with theoretical density
    plt.figure(figsize=(6, 4))
    plt.hist(samples, bins=30, density=True, alpha=0.6, color='darkred', edgecolor='black')
    
    # Theoretical density overlay
    x = np.linspace(0, np.max(samples), 1000)
    plt.plot(x, stats.expon.pdf(x, scale=1/np.mean(samples)), 'r-', lw=2, label='Theoretical PDF')
    plt.xlabel("Arrival Time")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    st.pyplot(plt)

def display_statistics(samples):
    """Display comprehensive statistics in three columns"""
    
    # Create three columns
    col1, col2, col3 = st.columns(3)
    
    # First column - Central Tendency
    with col1:
        st.markdown(f"""
            <div style="
                background-color: #2D2D2D;
                border: 1px solid #8B0000;
                border-radius: 8px;
                padding: 20px;
                height: 100%;
                font-family: 'Rubik', sans-serif;
            ">
                <h4 style="
                    color: #FFFFFF;
                    text-align: right;
                    margin-bottom: 15px;
                    font-size: 1.2rem;
                    border-bottom: 1px solid #8B0000;
                    padding-bottom: 10px;
                ">מדדי מרכז</h4>
                <div style="
                    color: #CCCCCC;
                    text-align: right;
                    font-size: 1rem;
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        margin-bottom: 10px;
                    ">
                        <span>ממוצע:</span>
                        <span>{np.mean(samples):.2f}</span>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                    ">
                        <span>חציון:</span>
                        <span>{np.median(samples):.2f}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Second column - Dispersion
    with col2:
        st.markdown(f"""
            <div style="
                background-color: #2D2D2D;
                border: 1px solid #8B0000;
                border-radius: 8px;
                padding: 20px;
                height: 100%;
                font-family: 'Rubik', sans-serif;
            ">
                <h4 style="
                    color: #FFFFFF;
                    text-align: right;
                    margin-bottom: 15px;
                    font-size: 1.2rem;
                    border-bottom: 1px solid #8B0000;
                    padding-bottom: 10px;
                ">מדדי פיזור</h4>
                <div style="
                    color: #CCCCCC;
                    text-align: right;
                    font-size: 1rem;
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        margin-bottom: 10px;
                    ">
                        <span>סטיית תקן:</span>
                        <span>{np.std(samples):.2f}</span>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                    ">
                        <span>טווח:</span>
                        <span>{np.min(samples):.2f} - {np.max(samples):.2f}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Third column - Shape
    with col3:
        st.markdown(f"""
            <div style="
                background-color: #2D2D2D;
                border: 1px solid #8B0000;
                border-radius: 8px;
                padding: 20px;
                height: 100%;
                font-family: 'Rubik', sans-serif;
            ">
                <h4 style="
                    color: #FFFFFF;
                    text-align: right;
                    margin-bottom: 15px;
                    font-size: 1.2rem;
                    border-bottom: 1px solid #8B0000;
                    padding-bottom: 10px;
                ">מדדי צורה</h4>
                <div style="
                    color: #CCCCCC;
                    text-align: right;
                    font-size: 1rem;
                ">
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        margin-bottom: 10px;
                    ">
                        <span>אסימטריה:</span>
                        <span>{stats.skew(samples):.2f}</span>
                    </div>
                    <div style="
                        display: flex;
                        justify-content: space-between;
                    ">
                        <span>קורטוזיס:</span>
                        <span>{stats.kurtosis(samples):.2f}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

#good version of display_random_number_generators
def display_random_number_generators():
    create_styled_card(
        "אלגוריתמים ליצירת מספרים פסאודו-אקראיים 🎲",
        """
        <div dir="rtl" style="text-align: right;">
        מספרים פסאודו-אקראיים הם הבסיס לכל סימולציה. להלן שתי שיטות נפוצות ליצירת מספרים אקראיים בין 0 ל-1:
        </div>
        """,
        
    )


    # הוספת דוגמה אינטראקטיבית
    st.markdown("""
        <h3 dir="rtl" style="text-align: right; margin: 30px 0 20px;">
            הדגמה אינטראקטיבית
        </h3>
    """, unsafe_allow_html=True)
    
    generator_type = st.radio("בחר סוג מחולל:", ["LCG", "LFSR"])
    num_samples = st.slider("מספר דגימות להצגה:", 1, 20, 10)
    
    if generator_type == "LCG":
        lcg = LCG(seed=12345)
        numbers = [lcg.random() for _ in range(num_samples)]
        show_random_numbers(numbers, "LCG")
    else:
        lfsr = LFSR("1010", [1, 3])  # 4-bit LFSR with taps at positions 1 and 3
        numbers = [lfsr.random() for _ in range(num_samples)]
        show_random_numbers(numbers, "LFSR")

def display_generator_state(generator_type, last_step, iteration):
    """Display the current state of the generator with improved layout."""
    if generator_type == "LCG":
        content = f"""
        <div dir="ltr" style="text-align: left;">
            <strong>Current State:</strong> {last_step['old_state']}<br>
            <strong>Calculation:</strong> {last_step['calculation']}<br>
            <strong>New State:</strong> {last_step['next_state']}<br>
            <strong>Random Number:</strong> {last_step['random_value']:.4f}
        </div>
        """
    else:
        content = f"""
        <div dir="ltr" style="text-align: left;">
            <strong>Current Bit State:</strong> {last_step['old_state']}<br>
            <strong>XOR Result:</strong> {last_step['xor_result']}<br>
            <strong>New Bit State:</strong> {last_step['next_state']}<br>
            <strong>Random Number:</strong> {last_step['random_value']:.4f}
        </div>
        """
    
    create_styled_card_left(
        f"Current State - Iteration {iteration}",
        content=content,
        
    )

# expiramental version
def display_interactive_sampling():

        # Create three columns for customer types
    col1, col2 = st.columns(2)

    with col1:

        # הסברים על סוגי המחוללים
        create_styled_card(
            "Linear Congruential Generator (LCG) - מחולל ליניארי מודולרי",
            """
            <div dir="rtl" style="text-align: right;">
            LCG הוא אלגוריתם נפוץ ליצירת רצף מספרים אקראיים. הוא משתמש בנוסחה רקורסיבית ליצירת רצף מספרים בתחום מוגדר, על בסיס ערך ראשוני (seed).
            <ul>
                <li><strong>הגדרת פרמטרים:</strong> ל-LCG יש 4 פרמטרים - a (הכפל), c (הזזה), m (מודולו), ו-seed (הערך הראשוני).</li>
                <li><strong>חישוב מצב חדש:</strong> מצב חדש מחושב באמצעות הנוסחה Xn+1 = (a ⋅ Xn + c) mod m, כאשר Xn הוא המצב הנוכחי ו-Xn+1 הוא המצב הבא.</li>
                <li><strong>חישוב ערך אקראי:</strong> כדי לקבל מספר אקראי בתחום (0, 1), מחלקים את המצב הנוכחי ב-m, כלומר: Random Value = Xn / m.</li>
                <li><strong>חזרה:</strong> חזרה על השלב הקודם ליצירת רצף מספרים אקראיים.</li>
            </ul>
            <strong>פסאודו-קוד:</strong>
            <pre>
        initialize(seed, a, c, m)
        X = seed
        while True:
            X = (a * X + c) % m
            random_value = X / m
            yield random_value
                </pre>
                </div>
                """
        )

    with col2:

        create_styled_card(
            "Linear Feedback Shift Register (LFSR) - רישום משמרת ליניארי",
            """
            <div dir="rtl" style="text-align: right;">
            ה-LFSR הוא מחולל אקראי שמתבסס על מצבי ביטים והחזרת מצב בתדירות קבועה. LFSR משתמש ב"ברזים" (taps) כדי לבצע פעולות XOR בין ביטים שונים במצב.
            <ul>
                <li><strong>הגדרת מצב ראשוני וברזים:</strong> מציבים את המצב הראשוני ומגדירים את מיקומי הברזים (מיקומי הביטים עליהם נבצע XOR).</li>
                <li><strong>חישוב XOR:</strong> מבצעים XOR בין הביטים הנבחרים (הברזים) ומאחסנים את התוצאה.</li>
                <li><strong>הזזת מצב הביטים:</strong> מזיזים את כל הביטים ב-1 ימינה, ומוסיפים את תוצאת ה-XOR כביט השמאלי החדש.</li>
                <li><strong>חישוב ערך אקראי:</strong> ממירים את המצב הנוכחי של הביטים לערך מספרי.</li>
                <li><strong>חזרה:</strong> חוזרים על התהליך עבור כל איטרציה כדי לייצר רצף של מספרים אקראיים.</li>
            </ul>
            <strong>פסאודו-קוד:</strong>
            <pre>
    initialize(seed, taps)
    state = seed
    while True:
        xor_result = 0
        for tap in taps:
            xor_result ^= state[tap - 1]
        state = (state >> 1) | (xor_result << (len(state) - 1))
        random_value = convert_to_decimal(state)
        yield random_value
            </pre>
            </div>
            """
        )

    # Initialize session state keys if they do not exist
    if 'random_generator' not in st.session_state:
        st.session_state.random_generator = None
        st.session_state.samples = []
        st.session_state.current_iteration = 0
        st.session_state.steps = []
        st.session_state.generator_type = "LCG"  # Default to LCG

    # Allow user to select generator type, and update session state
    generator_type = st.radio("Choose Generator Type:", ["LCG", "LFSR"], index=0)

    # Update session state only if generator type has changed or generator is None
    if st.session_state.random_generator is None or st.session_state.generator_type != generator_type:
        st.session_state.generator_type = generator_type
        if generator_type == "LCG":
            st.session_state.random_generator = LCG(seed=12345)
        else:
            st.session_state.random_generator = LFSR("1010", [1, 3])
        # Reset iteration and sample history
        st.session_state.samples = []
        st.session_state.current_iteration = 0
        st.session_state.steps = []

    # Set up layout for generator states and sampling steps
    col1, col2, col3 = st.columns([2, 2, 1])

    # Placeholder for previous step
    previous_step_placeholder = col1.empty()

    # Placeholder for current state
    current_state_placeholder = col2.empty()

    # Set up placeholders for plot and statistics
    plot_col, stats_col = st.columns([4, 1])

    with col3:
        if st.button("Sample Next Number", key="sample_button"):
            # Ensure random_generator is initialized
            if st.session_state.random_generator is not None:
                st.session_state.current_iteration += 1
                if generator_type == "LCG":
                    old_state = st.session_state.random_generator.state
                    next_state = st.session_state.random_generator.next()
                    random_value = st.session_state.random_generator.random()
                    
                    step = {
                        'iteration': st.session_state.current_iteration,
                        'old_state': old_state,
                        'calculation': f"({st.session_state.random_generator.a} × {old_state} + {st.session_state.random_generator.c}) mod {st.session_state.random_generator.m}",
                        'next_state': next_state,
                        'random_value': random_value
                    }
                else:
                    old_state = st.session_state.random_generator.state
                    xor_result = 0
                    for t in st.session_state.random_generator.taps:
                        xor_result ^= int(old_state[t - 1])
                    
                    next_state = st.session_state.random_generator.next()
                    random_value = st.session_state.random_generator.random()
                    
                    step = {
                        'iteration': st.session_state.current_iteration,
                        'old_state': old_state,
                        'xor_result': xor_result,
                        'next_state': next_state,
                        'random_value': random_value
                    }
                
                # Append step to session state and update samples
                if len(st.session_state.steps) > 0:
                    previous_step = st.session_state.steps[-1]
                    with previous_step_placeholder:
                        display_generator_state(st.session_state.generator_type, previous_step, st.session_state.current_iteration - 1)
                
                st.session_state.steps.append(step)
                st.session_state.samples.append(random_value)
                
                # Display current state
                with current_state_placeholder:
                    display_generator_state(st.session_state.generator_type, step, st.session_state.current_iteration)

    # Display plot in the left column and summary in the right column
    if st.session_state.samples:
        with plot_col:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Histogram
            ax1.hist(st.session_state.samples, bins=min(20, len(st.session_state.samples)), 
                    color='#8B0000', alpha=0.7, edgecolor='black')
            ax1.set_title('Random Numbers Histogram')
            ax1.set_xlabel('Value')
            ax1.set_ylabel('Frequency')
            
            # Trace Plot
            ax2.plot(range(len(st.session_state.samples)), st.session_state.samples, 
                    marker='o', markersize=4, linestyle='-', color='#8B0000')
            ax2.set_title('Random Numbers Trace Plot')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Value')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            st.pyplot(fig)

    # Display statistical summary in the right column
    if len(st.session_state.samples) > 1:
        with stats_col:
            create_styled_card(
                "Statistical Summary",
                f"""
                <div dir="ltr" style="text-align: left;">
                    <strong>Sample Count:</strong> {len(st.session_state.samples)}<br>
                    <strong>Mean:</strong> {np.mean(st.session_state.samples):.4f}<br>
                    <strong>Median:</strong> {np.median(st.session_state.samples):.4f}<br>
                    <strong>Standard Deviation:</strong> {np.std(st.session_state.samples):.4f}<br>
                    <strong>Minimum:</strong> {min(st.session_state.samples):.4f}<br>
                    <strong>Maximum:</strong> {max(st.session_state.samples):.4f}
                </div>
                """,
                
            )

def display_random_number_generators():
    create_styled_card(
        "אלגוריתמים ליצירת מספרים פסאודו-אקראיים 🎲",
        """
        <div dir="rtl" style="text-align: right;">
        מספרים פסאודו-אקראיים הם הבסיס לכל סימולציה. להלן שתי שיטות נפוצות ליצירת מספרים אקראיים בין 0 ל-1:
        </div>
        """,
        
    )

    # הצגת ההדגמה האינטראקטיבית
    display_interactive_sampling()

def rejection_sample_prep_time():
    """
    Performs rejection sampling for preparation time based on customer type and order complexity.
    Uses envelope function t(x) = 0.5 for the rejection sampling algorithm.
    """
    def f(x):
        """Target probability density function for order times."""
        if 3 <= x < 4:
            return 0.5  # Type A customers (50%)
        elif 4 <= x < 5:
            return (x - 4) / 4  # Type B customers (25%)
        elif 5 <= x < 6:
            return (6 - x) / 4  # Type B customers (25%)
        elif x == 10:
            return 0.25  # Type C customers (25%)
        else:
            return 0

    while True:
        # Generate a candidate y between 3 and 10
        y = 7 * np.random.uniform(0, 1) + 3
        u = np.random.uniform(0, 1)
        
        # Accept y if u <= f(y) / t(y), where t(y) = 0.5 is our envelope function
        if u <= f(y) / 0.5:
            return y

def inverse_transform_prep_time():
    """
    Implements inverse transform sampling for preparation time based on the composite distribution.
    """
    u = np.random.uniform(0, 1)
    
    # Calculate x based on the value of u
    if 0 <= u < 0.5:
        x = 2 * u + 3
    elif 0.5 <= u < 0.625:
        x = (8 + np.sqrt(32 * u - 16)) / 2
    elif 0.625 <= u < 0.75:
        x = (12 + np.sqrt(24 - 32 * u)) / 2
    elif 0.75 <= u <= 1:
        x = 10
    else:
        x = None
    
    return x

def composition_prep_time():
    """
    Implements composition sampling for preparation time based on customer types.
    """
    u1 = np.random.uniform(0, 1)
    
    if 0 <= u1 < 0.5:
        # Type A customers: Uniform between 3 and 4 minutes
        x = np.random.uniform(3, 4)
    elif 0.5 <= u1 < 0.75:
        # Type B customers: Triangular between 4 and 6, mode at 5
        x = np.random.triangular(4, 6, 5)
    else:
        # Type C customers: Fixed 10 minutes
        x = 10
    
    return x

def display_order_time_sampling():
    """
    Displays a Streamlit app for simulating order times using inverse transform sampling, rejection sampling, and composition sampling.
    The app displays a histogram and Q-Q plot of the samples, as well as statistical measures of central tendency, dispersion, and shape.
    """
    st.markdown("""
        <div class="custom-card rtl-content">
            <h2>דגימת זמני הזמנה</h2>
            <p>
                המערכת מדמה זמני הזמנה עבור שלושה סוגי לקוחות:
                <ul>
                    <li>סוג א' (50%): זמן הזמנה אחיד בין 3-4 דקות</li>
                    <li>סוג ב' (25%): זמן הזמנה משולש בין 4-6 דקות</li>
                    <li>סוג ג' (25%): זמן הזמנה קבוע של 10 דקות</li>
                </ul>
            </p>
        </div>
    """, unsafe_allow_html=True)

    sampling_method = st.radio(
        "בחר שיטת דגימה:",
        ["טרנספורם הופכי", "דגימת קבלה-דחייה", "שיטת הקומפוזיציה"],
        key="order_sampling"
    )

    num_samples = st.slider("מספר דגימות:", 100, 5000, 1000, key="order_samples")

    if st.button("הרץ סימולציה", key="order_simulation"):
        if sampling_method == "טרנספורם הופכי":
            samples = [inverse_transform_prep_time() for _ in range(num_samples)]
        elif sampling_method == "דגימת קבלה-דחייה":
            samples = [rejection_sample_prep_time() for _ in range(num_samples)]
        else:  # Composition
            samples = [composition_prep_time() for _ in range(num_samples)]

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram
        counts, bins, _ = ax1.hist(samples, bins=30, density=True, alpha=0.7, color='#8B0000', edgecolor='black')
        ax1.set_title('התפלגות זמני הזמנה')
        ax1.set_xlabel('זמן (דקות)')
        ax1.set_ylabel('צפיפות')
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        stats.probplot(samples, dist="norm", plot=ax2)
        ax2.get_lines()[0].set_markerfacecolor('#8B0000')
        ax2.get_lines()[0].set_markeredgecolor('#8B0000')
        ax2.get_lines()[1].set_color('#8B0000')
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)

        # Display statistics
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>סטטיסטיקה תיאורית</h3>
            </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
                <div class="stat-box">
                    <h4>מדדי מרכז</h4>
                    <p>ממוצע: {:.2f}</p>
                    <p>חציון: {:.2f}</p>
                </div>
            """.format(np.mean(samples), np.median(samples)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="stat-box">
                    <h4>מדדי פיזור</h4>
                    <p>סטיית תקן: {:.2f}</p>
                    <p>טווח: {:.2f} - {:.2f}</p>
                </div>
            """.format(np.std(samples), min(samples), max(samples)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="stat-box">
                    <h4>מדדי צורה</h4>
                    <p>אסימטריה: {:.2f}</p>
                    <p>קורטוזיס: {:.2f}</p>
                </div>
            """.format(stats.skew(samples), stats.kurtosis(samples)), unsafe_allow_html=True)


def show_sampling_methods():
    # Apply custom CSS
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    set_rtl()
    set_ltr_sliders()

    #display_random_number_generators()

    show_sampling_intro()
    create_sampling_methods_grid()
    
    
        # הצגת המחוללים האקראיים


    st.markdown(
        """
        <h2 dir="rtl" style="text-align: right; margin: 30px 0 20px;">בחר שיטת דגימה להמחשה:</h2>
        """,
        unsafe_allow_html=True
    )
    
    method = st.radio("", [
        "טרנספורם הופכי - זמני הגעת לקוחות",
        "דגימת קבלה-דחייה - זמני הכנת מנות",
        "שיטת הקומפוזיציה - זמני המתנה"
    ], index=0)

    if "טרנספורם הופכי" in method:
        display_inverse_transform_method()
    elif "קבלה-דחייה" in method:
        display_rejection_method()
    elif "קומפוזיציה" in method:
        display_composition_method()    

if __name__ == "__main__":
    show_sampling_methods()