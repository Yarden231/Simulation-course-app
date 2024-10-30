import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import set_ltr_sliders, set_rtl

def mid_square_method(seed, n):
    """Generate random numbers using the Mid-Square method, with improved handling."""
    results = []
    for i in range(n):
        # Square the seed and ensure it's zero-padded to at least 8 digits
        z_squared = str(seed ** 2).zfill(8)
        
        # If seed becomes zero (all numbers zero out), break early
        if seed == 0:
            break

        # Take the middle 4 digits (for a 4-digit system)
        mid = len(z_squared) // 2
        next_seed = int(z_squared[mid-2:mid+2])

        # If we end up in a repetitive zero sequence, stop early
        if next_seed == 0:
            break

        # Generate U_i (the uniform number in [0, 1])
        u = next_seed / 10000.0
        results.append((seed, u))

        # Update the seed for the next iteration
        seed = next_seed

    return results

def lcg_method(a, c, m, seed, n):
    """Generate random numbers using Linear Congruential Generator with NumPy vectorization."""
    # Pre-allocate arrays for performance
    Z = np.zeros(n, dtype=int)  # Array to store Z_i values
    U = np.zeros(n, dtype=float)  # Array to store U_i values

    # Set the initial seed
    Z[0] = seed

    # Generate Z values using vectorized recurrence relation
    for i in range(1, n):
        Z[i] = (a * Z[i - 1] + c) % m

    # Generate U values (normalized Z values between [0, 1])
    U = Z / m

    # Return the results as a list of tuples
    return list(zip(Z, U))


def plot_histogram_of_samples(ui_values, method_name):
    """Plot histogram of generated random numbers (U_i values)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot histogram of all U_i values
    sns.histplot(ui_values, bins=20, kde=True, ax=ax, color='orange')
    ax.set_title(f"{method_name}: Histogram of All Generated U_i")
    ax.set_xlabel('U_i')
    ax.set_ylabel('Frequency')

    st.pyplot(fig)

def show_random_generator():
    set_ltr_sliders() 
    # Streamlit UI
    st.title("Random Number Generators")

    # Mid-Square Method
    st.subheader("Mid-Square Method")
    seed_ms = st.slider("Enter a 4-digit seed for Mid-Square Method:", min_value=1000, max_value=9999, value=1234)
    n_ms = st.slider("How many numbers to generate?", min_value=100, max_value=100000, value=10000)

    if st.button("Generate Mid-Square Numbers"):
        mid_square_results = mid_square_method(seed_ms, n_ms)
        st.write("Mid-Square Method Results (First 10 numbers):")
        for i, (zi, ui) in enumerate(mid_square_results):  # Show first 10 results for brevity
            st.write(f"Z_{i} = {zi}, U_{i} = {ui:.4f}")

        # Extract U_i values for the histogram
        ui_values = [ui for _, ui in mid_square_results]
        if ui_values:
            plot_histogram_of_samples(ui_values, "Mid-Square Method")
        else:
            st.write("Insufficient valid numbers generated due to repetitive zeros.")

    # Linear Congruential Generator (LCG)
    st.subheader("Linear Congruential Generator (LCG)")
    n_lcg = st.slider("How many numbers to generate (LCG)?", min_value=100, max_value=1000000, value=10000)
    a = st.slider("Enter multiplier (a):", min_value=1, value=5)
    c = st.slider("Enter increment (c):", min_value=0, value=3)
    m = st.slider("Enter modulus (m):", min_value=2, value=16)
    seed_lcg = st.slider("Enter the seed (Z0):", min_value=0, value=7)

    if st.button("Generate LCG Numbers"):
        lcg_results = lcg_method(a, c, m, seed_lcg, n_lcg)
        st.write("LCG Results (First 10 numbers):")
        for i, (zi, ui) in enumerate(lcg_results):  # Show first 10 results for brevity
            st.write(f"Z_{i} = {zi}, U_{i} = {ui:.4f}")

        # Extract U_i values for the histogram
        ui_values = [ui for _, ui in lcg_results]
        plot_histogram_of_samples(ui_values, "LCG Method")

if __name__ == "__main__":
    show_random_generator()
