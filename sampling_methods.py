
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from utils import set_rtl
from utils import set_ltr_sliders
import time
# Call the set_rtl function to apply RTL styles
set_rtl()

def sample_uniform(a, b, size):
    return np.random.uniform(a, b, size)

def sample_exponential(lambda_param, size):
    return np.random.exponential(1/lambda_param, size)

def sample_normal(mu, sigma, size):
    return np.random.normal(mu, sigma, size)

def sample_composite_distribution(size):
    normal_1 = np.random.normal(0, 1, size)
    normal_2 = np.random.normal(3, 1, size)
    mask = np.random.rand(size) < 0.2
    return np.where(mask, normal_1, normal_2)

def sample_acceptance_rejection(size):
    samples = []
    while len(samples) < size:
        x = np.random.random()
        y = np.random.random() * 3
        if y <= f(x):
            samples.append(x)
    return np.array(samples)

def f(x):
    return 3 * x ** 2

def plot_histogram(samples, title, distribution_func=None, true_density=None):
    fig, ax = plt.subplots(figsize=(6, 4))  # Fixed figure size
    ax.hist(samples, bins=100, density=True, alpha=0.7, label='Sampled Data')
    ax.set_title(f"{title} (Number of samples: {len(samples)})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    
    if true_density:
        x = np.linspace(min(samples), max(samples), 100)
        ax.plot(x, true_density(x), 'r-', lw=2, label='True Density Function')
    
    if distribution_func:
        x = np.linspace(0, 1, 100)
        ax.plot(x, distribution_func(x), 'g--', lw=2, label='Target Distribution')

    ax.legend(loc='upper right')  # Fixed legend location
    ax.set_xlim([min(samples), max(samples)])  # Set axis limits
    ax.set_ylim(0, 2.0)  # Fixed y-axis limit for consistency
    ax.grid(True)  # Add grid for clarity
    return fig

def plot_qqplot(samples, title):
    fig, ax = plt.subplots(figsize=(6, 4))  # Fixed figure size
    stats.probplot(samples, dist="norm", plot=ax)
    ax.set_title(f"{title} - QQ Plot")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.grid(True)  # Add grid for clarity
    return fig

def display_statistics(samples):
    mean = np.mean(samples)
    median = np.median(samples)
    std_dev = np.std(samples)
    min_val = np.min(samples)
    max_val = np.max(samples)
    
    st.write(f"**Mean:** {mean:.2f}")
    st.write(f"**Median:** {median:.2f}")
    st.write(f"**Standard Deviation:** {std_dev:.2f}")
    st.write(f"**Minimum Value:** {min_val:.2f}")
    st.write(f"**Maximum Value:** {max_val:.2f}")

def run_sampling(sampling_function, num_samples, update_interval, title, progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples, distribution_func=None, true_density=None):
    # Generate all samples at once
    all_samples = sampling_function(num_samples)
    
    # Simulate real-time updates by splitting samples into batches
    samples = []
    for i in range(0, num_samples, update_interval):
        batch_samples = all_samples[i:i+update_interval]
        samples.extend(batch_samples)
        
        # Update histograms and QQ plots side by side
        with plot_placeholder.container():
            col1, col2 = st.columns(2)
            with col1:
                fig = plot_histogram(samples, title, distribution_func, true_density)
                st.pyplot(fig)
                plt.close(fig)
            with col2:
                qqplot_fig = plot_qqplot(samples, title)
                st.pyplot(qqplot_fig)
                plt.close(qqplot_fig)

        # Update statistics
        stats_placeholder.empty()
        with stats_placeholder:
            display_statistics(samples)
        
        # Print sample values
        if print_samples:
            st.write(f"**Sample values (first {min(10, len(samples))} values):** {samples[:10]}")
        
        # Simulate progress in real-time
        progress_bar.progress((i + update_interval) / num_samples)
        
        # Delay to simulate real-time sampling (optional)
        #time.sleep(0.01)

def show_sampling_methods():
    st.title("הדגמה של שיטות דגימה שונות")

    st.write("בדף זה נלמד על שיטות דגימה שונות, ונראה דוגמאות כיצד ניתן לייצר דגימות באמצעות Python.")

    if 'selected_sampling' not in st.session_state:
        st.session_state.selected_sampling = None

    # Move the slider for sample size inside the main page area
    st.subheader("בחר מספר דגימות ודגום התפלגות")
    num_samples = st.slider("מספר דגימות", min_value=1000, max_value=1000000, value=1000, step=1000)
    update_interval = st.slider("תדירות עדכון (מספר דגימות)", 100, 1000, 100)

    st.header("בחר שיטת דגימה")
    set_ltr_sliders() 
    if st.button("התפלגות אחידה"):
        st.session_state.selected_sampling = 'uniform'
    if st.button("התפלגות נורמלית"):
        st.session_state.selected_sampling = 'normal'
    if st.button("התפלגות מעריכית"):
        st.session_state.selected_sampling = 'exponential'
    if st.button("התפלגות מורכבת"):
        st.session_state.selected_sampling = 'composite'
    if st.button("שיטת הקבלה-דחייה"):
        st.session_state.selected_sampling = 'acceptance_rejection'

    if st.session_state.selected_sampling == 'uniform':
        st.header("1. התפלגות אחידה")
        st.latex(r"f(x) = \frac{1}{b-a}, \quad a \leq x \leq b")
        a = st.slider("ערך מינימלי (a)", 0.0, 1.0, 0.0)
        b = st.slider("ערך מקסימלי (b)", a + 0.1, 1.0, 1.0)
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: np.ones_like(x) / (b - a)
        run_sampling(lambda size: sample_uniform(a, b, size), num_samples, update_interval, "Uniform Distribution", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, true_density=true_density)

    elif st.session_state.selected_sampling == 'normal':
        st.header("2. התפלגות נורמלית")
        st.latex(r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}")
        mu = st.slider("ממוצע (μ)", -10.0, 10.0, 0.0)
        sigma = st.slider("סטיית תקן (σ)", 0.1, 5.0, 1.0)
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: stats.norm.pdf(x, mu, sigma)
        run_sampling(lambda size: sample_normal(mu, sigma, size), num_samples, update_interval, "Normal Distribution", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, true_density=true_density)

    elif st.session_state.selected_sampling == 'exponential':
        st.header("3. התפלגות מעריכית")
        st.latex(r"f(x) = \lambda e^{-\lambda x}, \quad x \geq 0")
        lambda_param = st.slider("פרמטר למבדא", 0.1, 5.0, 1.0)
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: lambda_param * np.exp(-lambda_param * x)
        run_sampling(lambda size: sample_exponential(lambda_param, size), num_samples, update_interval, "Exponential Distribution", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, true_density=true_density)

    elif st.session_state.selected_sampling == 'composite':
        st.header("4. התפלגות מורכבת")
        st.latex(r"f(x) = 0.2 \cdot N(0, 1) + 0.8 \cdot N(3, 1)")
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        true_density = lambda x: 0.2 * stats.norm.pdf(x, 0, 1) + 0.8 * stats.norm.pdf(x, 3, 1)
        run_sampling(lambda size: sample_composite_distribution(size), num_samples, update_interval, "Composite Distribution", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, true_density=true_density)

    elif st.session_state.selected_sampling == 'acceptance_rejection':
        st.header("5. שיטת הקבלה-דחייה")
        st.latex(r"f(x) = 3x^2, \quad 0 \leq x \leq 1")
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        qqplot_placeholder = st.empty()
        stats_placeholder = st.empty()
        run_sampling(lambda size: sample_acceptance_rejection(size), num_samples, update_interval, "Acceptance-Rejection Method", progress_bar, plot_placeholder, qqplot_placeholder, stats_placeholder, print_samples=True, distribution_func=f)

if __name__ == "__main__":
    show_sampling_methods()