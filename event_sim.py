import streamlit as st
import numpy as np
from scipy.stats import t
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import graphviz
import random
import math
from typing import Tuple, List, Dict
from graphviz import Digraph
import pandas as pd

def create_station_grid():
    stations = [
        ("🤭",  "אחוז לקוחות שהושלם שירותם (נמקסם)"),
        ("😡", "אחוז לקוחות שעזבו (נמזער)"),
        ("🍲", "אחוז המנות שלא בושלו כראוי (נמזער)")
    ]
    
    cols = st.columns(3)
    for idx, (emoji, name) in enumerate(stations):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="
                    background-color: #2D2D2D;
                    border: 1px solid #453232;
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

def run_simulation(extra_employee=None):
    """
    Simulates the food truck operation and returns performance metrics.
    This is a simplified version for demonstration. In practice, use your full simulation.
    """
    # Simplified simulation for demonstration
    # In practice, use your actual simulation code
    if extra_employee:
        served = random.uniform(75, 85)
        left = random.uniform(8, 15)
        undercooked = random.uniform(3, 8)
    else:
        served = random.uniform(65, 75)
        left = random.uniform(15, 25)
        undercooked = random.uniform(8, 15)
    
    return served, left, undercooked

def calculate_required_repetitions(data_series, initial_n, alpha, relative_precision):
    """Calculate required number of additional repetitions needed."""
    corrected_alpha = alpha / 6  # Bonferroni correction
    t_score = t.ppf(1 - corrected_alpha / 2, df=initial_n - 1)
    
    mean_data = np.mean(data_series)
    std_data = np.std(data_series, ddof=1)
    
    # Calculate current half-width
    delta_0 = t_score * (std_data / np.sqrt(initial_n))
    
    # Calculate target half-width
    delta_t = relative_precision * np.mean(mean_data)
    
    return max(0, int(np.ceil(initial_n * (delta_0 / delta_t) ** 2)) - initial_n)

def initial_analysis(initial_n, alpha, relative_precision, extra_employee):
    """Perform initial analysis of the simulation with given parameters."""
    # Data collection for current and alternative scenarios
    current_served, current_left, current_undercooked = [], [], []
    alternative_served, alternative_left, alternative_undercooked = [], [], []

    # Run initial simulations
    for _ in range(initial_n):
        # Current scenario
        served, left, undercooked = run_simulation()
        current_served.append(served)
        current_left.append(left)
        current_undercooked.append(undercooked)
        
        # Alternative scenario
        served, left, undercooked = run_simulation(extra_employee=extra_employee)
        alternative_served.append(served)
        alternative_left.append(left)
        alternative_undercooked.append(undercooked)

    # Calculate required repetitions and confidence intervals
    current_results = [
        calculate_required_repetitions(data, initial_n, alpha, relative_precision)
        for data in [current_served, current_left, current_undercooked]
    ]
    
    alternative_results = [
        calculate_required_repetitions(data, initial_n, alpha, relative_precision)
        for data in [alternative_served, alternative_left, alternative_undercooked]
    ]

    return (
        (current_served, current_left, current_undercooked),
        (alternative_served, alternative_left, alternative_undercooked),
        current_results,
        alternative_results
    )

def create_process_diagram() -> graphviz.Digraph:
    """Create a Graphviz diagram showing the food truck process flow."""
    dot = Digraph(comment="Simplified Busy Food Truck Simulation", 
                  graph_attr={
                      'bgcolor': '#1E1E1E',  # Dark grey background
                      'fontcolor': 'white',  # White text for readability
                      'fontsize': '16'  # Increase overall font size
                  },
                  node_attr={
                      'style': 'filled',
                      'fillcolor': 'white',  # White nodes
                      'fontcolor': 'black',  # Black font
                      'color': 'black',  # Black border
                      'fontsize': '16',  # Larger node text
                      'width': '1.5',  # Wider nodes
                      'height': '0.8'  # Taller nodes
                  },
                  edge_attr={
                      'color': 'white',  # White edges
                      'fontcolor': 'white',  # White edge labels
                      'fontsize': '12'  # Larger edge label text
                  })

    # Define the nodes for main steps of the process
    dot.node('A', 'Customer Arrival')
    dot.node('B', 'Order Station')
    dot.node('C', 'Meal Preparation')
    dot.node('D', 'Pickup Station')
    dot.node('E', 'Customer Departure - Success')
    dot.node('L', 'Customer Departure - Timeout')

    # Add detailed edges for the simplified logic
    # Arrival to Order Station
    dot.edge('A', 'B', 'Arrives (Exponential Interval)')

    # Order Station to Meal Preparation
    dot.edge('B', 'C', 'Order Placed (Type A, B, or C)')

    # Meal Preparation Choices based on batch
    dot.edge('C', 'D', 'Meal Ready (Batch Cooking)')
    dot.edge('C', 'L', 'Leaves if Timeout Exceeded')
    dot.edge('B', 'L', 'Leaves if Timeout Exceeded')

    # Pickup Station to Departure (Successful Order)
    dot.edge('D', 'E', 'Meal Picked Up (Uniform 2-4 mins) coocked or undercoocked')
    dot.edge('D', 'L', 'Leaves if Timeout Exceeded')
    
    return dot

def run_extended_analysis(initial_data_current: tuple, 
                        initial_data_alternative: tuple,
                        alpha: float) -> Dict:
    """Run extended analysis comparing current and alternative scenarios."""
    if initial_data_current is None or initial_data_alternative is None:
        return {}
        
    corrected_alpha = alpha / 6
    
    # Calculate pairwise confidence intervals
    results = {}
    measures = ['served', 'left', 'undercooked']
    objectives = ['maximize', 'minimize', 'minimize']
    
    for i, (measure, objective) in enumerate(zip(measures, objectives)):
        try:
            current_data = np.array(initial_data_current[i])
            alternative_data = np.array(initial_data_alternative[i])
            
            # Ensure both arrays have the same length by truncating to the shorter length
            min_length = min(len(current_data), len(alternative_data))
            current_data = current_data[:min_length]
            alternative_data = alternative_data[:min_length]
            
            # Calculate differences
            differences = current_data - alternative_data
            mean_diff = np.mean(differences)
            std_diff = np.std(differences, ddof=1)
            n = len(differences)
            
            # Calculate CI
            t_score = t.ppf(1 - corrected_alpha / 2, df=n - 1)
            margin_of_error = t_score * (std_diff / np.sqrt(n))
            ci_lower = mean_diff - margin_of_error
            ci_upper = mean_diff + margin_of_error
            
            # Determine preference
            if objective == 'maximize':
                if ci_lower > 0:
                    preference = "מצב קיים עדיף"
                elif ci_upper < 0:
                    preference = "חלופה עדיפה"
                else:
                    preference = "אין העדפה מובהקת"
            else:  # minimize
                if ci_upper < 0:
                    preference = "מצב קיים עדיף"
                elif ci_lower > 0:
                    preference = "חלופה עדיפה"
                else:
                    preference = "אין העדפה מובהקת"
            
            results[measure] = {
                'mean_diff': mean_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'preference': preference,
                'n_samples': n
            }
        except Exception as e:
            st.error(f"Error analyzing {measure}: {str(e)}")
            results[measure] = {
                'mean_diff': 0,
                'ci_lower': 0,
                'ci_upper': 0,
                'preference': "שגיאה בניתוח",
                'n_samples': 0
            }
    
    return results

def calculate_additional_runs(current_data, alternative_data, reps_current, reps_alternative):
    """Calculate the maximum number of additional runs needed across all metrics."""
    max_additional_runs = max(
        max(rep[0] for rep in reps_current),
        max(rep[0] for rep in reps_alternative)
    )
    return max_additional_runs

def run_extended_simulation(initial_n, additional_runs, extra_employee):
    """
    Run additional simulation runs and combine with initial data.
    
    Parameters:
    -----------
    initial_n : int
        Number of initial runs
    additional_runs : int
        Number of additional runs needed
    extra_employee : str
        Location of extra employee (order/prep/pickup)
        
    Returns:
    --------
    tuple
        Two tuples containing the current and alternative scenario data
    """
    # Initialize lists for storing results
    current_served, current_left, current_undercooked = [], [], []
    alt_served, alt_left, alt_undercooked = [], [], []
    
    # Run the additional simulations
    for _ in range(additional_runs):
        # Run current scenario
        served, left, undercooked = run_simulation()
        current_served.append(served)
        current_left.append(left)
        current_undercooked.append(undercooked)
        
        # Run alternative scenario
        served, left, undercooked = run_simulation(extra_employee=extra_employee)
        alt_served.append(served)
        alt_left.append(left)
        alt_undercooked.append(undercooked)
    
    return (current_served, current_left, current_undercooked), \
           (alt_served, alt_left, alt_undercooked)


# לא משומשת
def update_simulation_section(current_data, alternative_data, reps_current, reps_alternative, alpha, extra_employee):
    """Add a section to run additional simulations if needed."""
    
    # Initialize session state if needed
    if 'additional_runs_completed' not in st.session_state:
        st.session_state.additional_runs_completed = False
    if 'updated_current' not in st.session_state:
        st.session_state.updated_current = current_data
    if 'updated_alternative' not in st.session_state:
        st.session_state.updated_alternative = alternative_data

    st.markdown("<h3 style='text-align: right;'>הרצות נוספות נדרשות</h3>", unsafe_allow_html=True)
    
    # Calculate maximum additional runs needed
    max_additional = max(
        max(rep[0] for rep in reps_current),
        max(rep[0] for rep in reps_alternative)
    )
    
    if max_additional > 0 and not st.session_state.additional_runs_completed:
        st.markdown(f"""
            <div style='text-align: right; direction: rtl; 
                  background-color: #ff9800; 
                  color: white; 
                  padding: 1rem; 
                  border-radius: 0.5rem;
                  margin: 1rem 0;'>
                <strong>שים לב:</strong> נדרשות {max_additional} הרצות נוספות להשגת רמת הדיוק הרצויה
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("בצע הרצות נוספות"):
            with st.spinner('מבצע הרצות נוספות...'):
                # Run additional simulations
                current_additional, alternative_additional = run_extended_simulation(
                    len(current_data[0]), max_additional, extra_employee
                )
                
                # Combine initial and additional data
                st.session_state.updated_current = (
                    list(current_data[0]) + list(current_additional[0]),
                    list(current_data[1]) + list(current_additional[1]),
                    list(current_data[2]) + list(current_additional[2])
                )
                
                st.session_state.updated_alternative = (
                    list(alternative_data[0]) + list(alternative_additional[0]),
                    list(alternative_data[1]) + list(alternative_additional[1]),
                    list(alternative_data[2]) + list(alternative_additional[2])
                )
                
                st.session_state.additional_runs_completed = True
                st.rerun()
    
    if st.session_state.additional_runs_completed:
        # Create comparison visualization
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("תוצאות מעודכנות - מצב קיים", "תוצאות מעודכנות - חלופה")
        )
        
        metrics = ["שירות הושלם", "לקוחות שעזבו", "מנות לא מבושלות"]
        
        # Plot updated current scenario
        current_means = [np.mean(data) for data in st.session_state.updated_current]
        current_stds = [np.std(data) / np.sqrt(len(data)) for data in st.session_state.updated_current]
        
        fig.add_trace(
            go.Bar(
                name="מצב קיים",
                x=metrics,
                y=current_means,
                error_y=dict(
                    type='data',
                    array=[t.ppf(1 - alpha/2, df=len(data)-1) * std 
                          for data, std in zip(st.session_state.updated_current, current_stds)]
                ),
                marker_color='rgb(55, 83, 109)'
            ),
            row=1, col=1
        )
        
        # Plot updated alternative scenario
        alt_means = [np.mean(data) for data in st.session_state.updated_alternative]
        alt_stds = [np.std(data) / np.sqrt(len(data)) for data in st.session_state.updated_alternative]
        
        fig.add_trace(
            go.Bar(
                name="חלופה",
                x=metrics,
                y=alt_means,
                error_y=dict(
                    type='data',
                    array=[t.ppf(1 - alpha/2, df=len(data)-1) * std 
                          for data, std in zip(st.session_state.updated_alternative, alt_stds)]
                ),
                marker_color='rgb(26, 118, 255)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="תוצאות לאחר הרצות נוספות",
            title_x=0.5
        )
        
        st.plotly_chart(fig)
        
        st.markdown("""
            <div style='text-align: right; direction: rtl;
                  background-color: #4CAF50;
                  color: white;
                  padding: 1rem;
                  border-radius: 0.5rem;
                  margin: 1rem 0;'>
                הסימולציה הושלמה בהצלחה עם ההרצות הנוספות. 
                התוצאות כעת מדויקות יותר עם רווחי סמך מעודכנים.
            </div>
        """, unsafe_allow_html=True)
    
    return (st.session_state.updated_current if st.session_state.additional_runs_completed else current_data,
            st.session_state.updated_alternative if st.session_state.additional_runs_completed else alternative_data)

def update_simulation_results(current_data: tuple, alternative_data: tuple,
                            alpha: float, relative_precision: float, extra_employee: str):
    """Update simulation results with additional runs if needed."""
    
    # Initialize session state if needed
    if 'simulation_state' not in st.session_state:
        st.session_state.simulation_state = {
            'additional_runs_completed': False,
            'initial_current': current_data,
            'initial_alternative': alternative_data,
            'final_current': current_data,
            'final_alternative': alternative_data,
            'showing_results': False
        }
    
    # Get current state
    state = st.session_state.simulation_state
    
    # Calculate required additional runs
    current_required, alternative_required = calculate_total_required_runs(
        state['final_current'], state['final_alternative'], alpha, relative_precision
    )
    
    max_current = max(current_required)
    max_alternative = max(alternative_required)
    
    # Show initial results
    metrics = ["שירות הושלם", "לקוחות שעזבו", "מנות לא מבושלות"]
    
    def create_results_visualization(current_data, alternative_data, title=""):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("מצב קיים", "חלופה מוצעת")
        )
        
        # Plot current scenario
        current_means = [np.mean(data) for data in current_data]
        current_stds = [np.std(data, ddof=1) / np.sqrt(len(data)) for data in current_data]
        
        fig.add_trace(
            go.Bar(
                name="מצב קיים",
                x=metrics,
                y=current_means,
                error_y=dict(
                    type='data',
                    array=[t.ppf(1 - alpha/2, df=len(data)-1) * std 
                          for data, std in zip(current_data, current_stds)]
                ),
                marker_color='rgb(55, 83, 109)'
            ),
            row=1, col=1
        )
        
        # Plot alternative scenario
        alt_means = [np.mean(data) for data in alternative_data]
        alt_stds = [np.std(data, ddof=1) / np.sqrt(len(data)) for data in alternative_data]
        
        fig.add_trace(
            go.Bar(
                name="חלופה",
                x=metrics,
                y=alt_means,
                error_y=dict(
                    type='data',
                    array=[t.ppf(1 - alpha/2, df=len(data)-1) * std 
                          for data, std in zip(alternative_data, alt_stds)]
                ),
                marker_color='rgb(26, 118, 255)'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            title_text=title,
            title_x=0.5
        )
        
        return fig
    
    # Show current results
    st.plotly_chart(create_results_visualization(
        state['final_current'], 
        state['final_alternative'],
        "תוצאות נוכחיות"
    ))
    
    # Run additional simulations if needed
    if (max_current > 0 or max_alternative > 0) and not state['additional_runs_completed']:
        st.markdown(f"""
            <div style='text-align: right; direction: rtl; 
                  background-color: #420518; 
                  color: white; 
                  padding: 1rem; 
                  border-radius: 0.5rem;
                  margin: 1rem 0;'>
                <strong>נדרשות הרצות נוספות:</strong>
                <br>מצב קיים: {max_current} הרצות
                <br>חלופה: {max_alternative} הרצות
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("בצע הרצות נוספות"):
            with st.spinner('מבצע הרצות נוספות...'):
                # Run additional simulations for current scenario
                if max_current > 0:
                    new_current = run_complete_simulation(
                        len(state['final_current'][0]), max_current
                    )
                    state['final_current'] = tuple(
                        list(old) + list(new)
                        for old, new in zip(state['final_current'], new_current)
                    )
                
                # Run additional simulations for alternative scenario
                if max_alternative > 0:
                    new_alternative = run_complete_simulation(
                        len(state['final_alternative'][0]), max_alternative, extra_employee
                    )
                    state['final_alternative'] = tuple(
                        list(old) + list(new)
                        for old, new in zip(state['final_alternative'], new_alternative)
                    )
                
                state['additional_runs_completed'] = True
                state['showing_results'] = True
                st.rerun()
    
    # Show final results if completed
    if state['additional_runs_completed'] and state['showing_results']:
        st.markdown("<h3 style='text-align: right;'>תוצאות סופיות</h3>", unsafe_allow_html=True)
        
        st.plotly_chart(create_results_visualization(
            state['final_current'],
            state['final_alternative'],
            "תוצאות סופיות לאחר כל ההרצות"
        ))
        
        st.markdown("""
            <div style='text-align: right; direction: rtl;
                  background-color: #4CAF50;
                  color: white;
                  padding: 1rem;
                  border-radius: 0.5rem;
                  margin: 1rem 0;'>
                הסימולציה הושלמה בהצלחה עם כל ההרצות הנדרשות.
                התוצאות הסופיות מוצגות עם רווחי סמך מעודכנים.
            </div>
        """, unsafe_allow_html=True)
    
    return state['final_current'], state['final_alternative']

def calculate_total_required_runs(current_data: tuple, alternative_data: tuple, 
                                alpha: float, relative_precision: float) -> Tuple[List[int], List[int]]:
    """Calculate total required runs for both scenarios based on current data."""
    corrected_alpha = alpha / 6  # Bonferroni correction
    
    def calc_required_n(data_series):
        n = len(data_series)
        mean = np.mean(data_series)
        std = np.std(data_series, ddof=1)
        t_score = t.ppf(1 - corrected_alpha / 2, df=n - 1)
        
        # Calculate required sample size
        target_hw = relative_precision * abs(mean)
        required_n = math.ceil((std * t_score / target_hw) ** 2)
        
        return max(0, required_n - n)
    
    # Calculate required runs for each metric in current scenario
    current_required = [
        calc_required_n(metric_data)
        for metric_data in current_data
    ]
    
    # Calculate required runs for each metric in alternative scenario
    alternative_required = [
        calc_required_n(metric_data)
        for metric_data in alternative_data
    ]
    
    return current_required, alternative_required

def run_complete_simulation(initial_n: int, required_runs: int, 
                          extra_employee: str = None) -> Tuple[List[float], List[float], List[float]]:
    """Run a complete simulation for the specified number of runs."""
    served, left, undercooked = [], [], []
    
    for _ in range(required_runs):
        s, l, u = run_simulation(extra_employee)
        served.append(s)
        left.append(l)
        undercooked.append(u)
    
    return served, left, undercooked

def process_additional_runs(current_data, alternative_data, max_additional_current, 
                          max_additional_alternative, extra_employee, alpha):
    """Process additional simulation runs and update data."""

    repitition_needed = max(max_additional_current, max_additional_alternative)
    # Run additional simulations for current scenario
    if repitition_needed > 0:
        new_current = run_complete_simulation(
            len(current_data[0]), repitition_needed
        )
        current_data = tuple(
            list(old) + list(new)
            for old, new in zip(current_data, new_current)
        )
    
        # Run additional simulations for alternative scenario
        new_alternative = run_complete_simulation(
            len(alternative_data[0]), max_additional_alternative, extra_employee
        )
        alternative_data = tuple(
            list(old) + list(new)
            for old, new in zip(alternative_data, new_alternative)
        )
    
    # Run final analysis
    final_results = run_extended_analysis(
        current_data, 
        alternative_data,
        alpha
    )
    
    return current_data, alternative_data, final_results

def initial_analysis(initial_n, alpha, relative_precision, extra_employee):
    """Perform initial analysis of the simulation with given parameters."""
    # Data collection for current and alternative scenarios
    current_served, current_left, current_undercooked = [], [], []
    alternative_served, alternative_left, alternative_undercooked = [], [], []

    # Run initial simulations
    for _ in range(initial_n):
        # Current scenario
        served, left, undercooked = run_simulation()
        current_served.append(served)
        current_left.append(left)
        current_undercooked.append(undercooked)
        
        # Alternative scenario
        served, left, undercooked = run_simulation(extra_employee=extra_employee)
        alternative_served.append(served)
        alternative_left.append(left)
        alternative_undercooked.append(undercooked)

    # Calculate required repetitions, confidence intervals, and relative precision
    current_results = [
        calculate_required_repetitions(data, initial_n, alpha, relative_precision)
        for data in [current_served, current_left, current_undercooked]
    ]
    
    alternative_results = [
        calculate_required_repetitions(data, initial_n, alpha, relative_precision)
        for data in [alternative_served, alternative_left, alternative_undercooked]
    ]
    
    # Calculate relative precision (γ) for each metric
    current_relative_precisions = [
        calculate_relative_precision(data, alpha, initial_n)
        for data in [current_served, current_left, current_undercooked]
    ]
    
    alternative_relative_precisions = [
        calculate_relative_precision(data, alpha, initial_n)
        for data in [alternative_served, alternative_left, alternative_undercooked]
    ]

    return (
        (current_served, current_left, current_undercooked),
        (alternative_served, alternative_left, alternative_undercooked),
        current_results,
        alternative_results,
        current_relative_precisions,
        alternative_relative_precisions
    )

def calculate_relative_precision(data, alpha, initial_n):
    """Calculate relative precision γ for a given data."""
    mean = np.mean(data)
    std_error = np.std(data, ddof=1) / np.sqrt(initial_n)
    confidence_interval_width = t.ppf(1 - alpha / 2, df=initial_n - 1) * std_error
    gamma = confidence_interval_width / mean
    relative_precision = gamma / (1 + gamma)
    return relative_precision




def show_simulation_page():
    st.title("השוואה בין חלופות")

    # Initialize session state if not already done
    if 'simulation_state' not in st.session_state:
        st.session_state.simulation_state = {
            'initialized': False,
            'additional_runs_completed': False,
            'current_data': None,
            'alternative_data': None,
            'running_additional': False,
            'reps_current': None,
            'reps_alternative': None,
            'final_results': None,
            'extra_employee': None,
            'initial_runs': 20,
            'alpha': 0.05,
            'show_results': False,  # New flag to control results visibility
            'run_additional_sims': False
        }

    # Display initial content
    st.markdown("""
        <div style='text-align: right; direction: rtl;'>
            <h4> לאחר שיצרנו מודל סימולציה שמדמה את מערכת טאקו לוקו, בעמוד זה נבחן חלופות שונות עבור השמה של עובד נוסף באחת מעמדות משאית המזון.  בכדי להבין במובהקות סטטיסטית נתונה, מהי החלופה המצנחת, נצטרך לבצע את השלבים הבאים:</h4>
        </div>
    """, unsafe_allow_html=True)
    st.text(" ")

 
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            """
            <div style="border-radius: 8px; padding: 15px;">
                <h4 style="color: #FFFFFF; text-align: center;">1️⃣ חישוב רווח סמך</h4>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.latex(r"\bar{X} = \frac{1}{n_0} \sum_{i=1}^{n_0} X_i")
        st.latex(r"S = \sqrt{\frac{1}{n_0-1} \sum_{i=1}^{n_0} (X_i - \bar{X})^2}")
        st.latex(r"CI = \bar{X} \pm t_{n_0-1, 1-\alpha/6} \cdot \frac{s}{\sqrt{n_0}}")
    
    with cols[1]:
        st.markdown(
            """
            <div style="border-radius: 8px; padding: 15px;">
                <h4 style="color: #FFFFFF; text-align: center;">2️⃣ בדיקת דיוק יחסי</h4>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.latex(r"\frac{\gamma}{1+\gamma} = \frac{\text{CI width}}{\bar{X}}")
    
    with cols[2]:
        st.markdown(
            """
            <div style="border-radius: 8px; padding: 15px;">
                <h4 style="text-align: center;">3️⃣ חישוב ריצות נוספות</h4>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.latex(r"n^* = n_0 \cdot \left(\frac{\text{CI current}}{\text{CI desired}}\right)^2")

    st.markdown("### הסבר מפורט 📝")


    st.markdown("""
    <div style="color: #FFFFFF; text-align: right; direction: rtl; font-size: 18px;">
        <p>מתחילים עם n_0 ריצות התחלתיות (לרוב 15-20 ריצות).</p>
        <ul style="margin-right: 20px;">
            <li>עבור מדד בודד משתמשים ברמת המובהקות α.</li>
            <li>עבור k מדדים משתמשים באי-שוויון בונפרוני:</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.latex(r"\alpha = \sum_{i=1}^{k*d} \alpha_i")
    st.latex(r"\alpha_i = \frac{\alpha}{k \cdot d}")

    st.markdown("""
    <ul style="color: #FFFFFF; text-align: right; direction: rtl;">
        <li>כאשר d מוגדר להיות מספר החלופות.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style="color: #FFFFFF; text-align: right; direction: rtl;">
        <li>עבור כל אחד מהמדדים בכל אחת מהחלוקות:</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style="margin-right: 20px; color: #FFFFFF; text-align: right; direction: rtl;">
        <li>אם מעוניינים בדיוק יחסי γ, מחשבים האם:</li>
    </ul>
    """, unsafe_allow_html=True)

    st.latex(r"\frac{\gamma}{1+\gamma} \geq \frac{t_{n-1, 1-\alpha_i} \cdot \frac{s}{\sqrt{n}}}{\bar{X}}")

    st.markdown("""
    <ul style="color: #FFFFFF; text-align: right; direction: rtl;">
        <li>אם הדיוק היחסי של אחד או יותר  לא מספק:</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style="margin-right: 20px; color: #FFFFFF; text-align: right; direction: rtl;">
        <li>מחשבים רק עבור המדדים שלא עמדו בתנאי את מספר הריצות הנדרש לפי הנוסחא:</li>
    </ul>
    """, unsafe_allow_html=True)

    st.latex(r"n^* = n \cdot \left(\frac{\text{CI current}}{\bar{X}\frac{\gamma}{1+\gamma}}\right)^2")

    st.markdown("""
    <ul style="margin-right: 20px; color: #FFFFFF; text-align: right; direction: rtl;">
        <li>מבצעים ריצות נוספות כדי להשלים למספר הריצות המקסימאלי מבין כל המדדים שלא עמדו בתנאי.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style="margin-right: 20px; color: #FFFFFF; text-align: right; direction: rtl;">
        <li>בודקים מחדש את רמת הדיוק היחסי עד להשגת הדיוק הרצוי.</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style="color: #FFFFFF; text-align: right; direction: rtl;">
        <li>במקרה של מערכת Non-Terminating:</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <ul style="margin-right: 20px; color: #FFFFFF; text-align: right; direction: rtl;">
        <li>יש להוסיף שלב מקדים של קביעת זמן חימום.</li>
        <li>להחליט האם להשתמש בשיטת Replication/Deletion או Batch Means.</li>
    </ul>
    """, unsafe_allow_html=True)



    st.text(" ")
    st.text(" ")
    st.text(" ")

    # Display initial content
    st.markdown("""
        <div style='text-align: right; direction: rtl;'>            
            <h4>המדדים שמעניינים את חולייסיטו ואוצ'ו לוקו הם:</h4>
        </div>
    """, unsafe_allow_html=True)
    
    create_station_grid()
    st.text(" ")
    st.text(" ")
    st.text(" ")
    
    #st.markdown("<h3 style='text-align: right;'>תרשים זרימת התהליך</h3>", unsafe_allow_html=True)
    #dot = create_process_diagram()
    #st.graphviz_chart(dot)
    
    st.text(" ")
    st.text(" ")
    st.text(" ")


    st.markdown("<h2 style='text-align: center;'>כעת נבחר את העמדה אליה נרצה לצרף עובד נוסף ונריץ את סימולצית המצב הקיים אל מול החלופה</h2>", unsafe_allow_html=True)

    # Simulation Parameters
    st.markdown("<h3 style='text-align: right;'>הגדרות סימולציה</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        employee_location = st.radio(
            "מיקום העובד הנוסף",
            ["עמדת הזמנות", "עמדת הכנה", "עמדת איסוף"],
            key="employee_location",
            index=["עמדת הזמנות", "עמדת הכנה", "עמדת איסוף"].index(st.session_state.simulation_state.get('employee_location', "עמדת הזמנות"))
        )

        precision = st.number_input(
            "רמת דיוק יחסי (γ)",
            min_value=0.01,
            max_value=0.1,
            value=st.session_state.simulation_state.get('alpha', 0.05),
            step=0.01,
            key="precision_input"
        )

    with col2:
        initial_runs = st.number_input(
            "מספר הרצות התחלתי",
            min_value=10,
            max_value=100,
            value=st.session_state.simulation_state.get('initial_runs', 20),
            step=5,
            key="initial_runs_input"
        )

        alpha = st.number_input(
            "רמת מובהקות (α)",
            min_value=0.01,
            max_value=0.1,
            value=st.session_state.simulation_state.get('alpha', 0.05),
            step=0.01,
            key="alpha_input"
        )
        


    # Map Hebrew location names to English
    location_map = {
        "עמדת הזמנות": "order",
        "עמדת הכנה": "prep",
        "עמדת איסוף": "pickup"
    }
    extra_employee = location_map[employee_location]

    # Buttons for running and resetting simulation
    col_run, col_reset = st.columns(2)
    
    with col_run:
        run_simulation = st.button("הרץ סימולציה", key="run_simulation_button")
    
    with col_reset:
        reset_simulation = st.button("אפס סימולציה", key="reset_simulation_button")

    # Reset simulation if reset button is pressed
    if reset_simulation:
        st.session_state.simulation_state = {
            'initialized': False,
            'additional_runs_completed': False,
            'current_data': None,
            'alternative_data': None,
            'running_additional': False,
            'reps_current': None,
            'reps_alternative': None,
            'final_results': None,
            'extra_employee': None,
            'initial_runs': 20,
            'alpha': 0.05,
            'show_results': False,
            'run_additional_sims': False
        }
    

    # Run initial simulation when run button is pressed
    if run_simulation:
        with st.spinner('מריץ סימולציה התחלתית...'):
            # Run the initial analysis to get the current and alternative data
            current_data, alternative_data, reps_current, reps_alternative, relative_precision_current, relative_precision_alternative = initial_analysis(
                initial_runs, alpha, precision, extra_employee
            )
            
            # Update session state with the data
            st.session_state.simulation_state.update({
                'initialized': True,
                'current_data': current_data,
                'alternative_data': alternative_data,
                'reps_current': reps_current,
                'reps_alternative': reps_alternative,
                'relative_precision_current': relative_precision_current,  # Store relative precision for current scenario
                'relative_precision_alternative': relative_precision_alternative,  # Store relative precision for alternative scenario
                'extra_employee': extra_employee,
                'initial_runs': initial_runs,
                'alpha': alpha,
                'additional_runs_completed': False,
                'final_results': None,
                'show_results': True  # Set to True when simulation is run
            })
         
    # Show results only if simulation has been run
    if st.session_state.simulation_state.get('show_results', False):

        current_data = st.session_state.simulation_state['current_data']
        alternative_data = st.session_state.simulation_state['alternative_data']
        reps_current = st.session_state.simulation_state['reps_current']
        reps_alternative = st.session_state.simulation_state['reps_alternative']
        current_relative_precisions = st.session_state.simulation_state['relative_precision_current']
        alternative_relative_precisions = st.session_state.simulation_state['relative_precision_alternative']

        # Show initial results
        st.markdown("<h3 style='text-align: right;'>תוצאות הסימולציה</h3>", unsafe_allow_html=True)

     
        # Create initial visualization for metrics
        metrics = ["שירות הושלם", "לקוחות שעזבו", "מנות לא מבושלות"]
        
        current_means = [np.mean(data) for data in current_data]
        current_stds = [np.std(data, ddof=1) / np.sqrt(len(data)) for data in current_data]
        alt_means = [np.mean(data) for data in alternative_data]
        alt_stds = [np.std(data, ddof=1) / np.sqrt(len(data)) for data in alternative_data]

        current_conf_intervals = [t.ppf(1 - alpha / 2, df=len(data) - 1) * std for data, std in zip(current_data, current_stds)]
        alt_conf_intervals = [t.ppf(1 - alpha / 2, df=len(data) - 1) * std for data, std in zip(alternative_data, alt_stds)]


        # הצגת הנוסחה המקורית
        relative_precision = precision/(1+precision)
        st.markdown("<h5 style='text-align: right;'>    נבדוק את הדיוק היחסי שהתקבל עבור כל המדדים בכל החלופות  המדדים לפי הנוסחא:</h5>", unsafe_allow_html=True)
        st.latex(r"\frac{\gamma}{1+\gamma} \geq \frac{t_{n-1, 1-\alpha_i} \cdot \frac{s}{\sqrt{n}}}{\bar{X}}")

        # הצגת הנוסחה המוצבת
        st.markdown("<h5 style='text-align: right;'>מציאת הערך איתו נבדוק את התנאי:</h5>", unsafe_allow_html=True)
        st.latex(
            rf"\frac{{{precision}}}{{1+{precision}}} = {relative_precision:.4f}"
        )
        


        st.markdown("<h5 style='text-align: right;'>רמות הדיוק עבור כל המדדים:</h5>", unsafe_allow_html=True)
    
        col1, col2 = st.columns([2,3])
   
        
        with col1:
            st.write("")
            st.write("")
            st.write("")

            # Create a list to hold the data for the table
            table_data = []

            # Loop through all the metrics and values
            for i, (metric, current_precision, alternative_precision) in enumerate(zip(metrics, current_relative_precisions, alternative_relative_precisions)):
                # Get required runs for current and alternative scenarios
                current_required = "-" if current_precision <= relative_precision else str(reps_current[i])+" הרצות"
                alternative_required = "-" if alternative_precision <= relative_precision else str(reps_alternative[i])+" הרצות"
                
                # Add both precision and required runs
                table_data.append([
                    metric, 
                    f"{current_precision:.4f}<br>({current_required})", 
                    f"{alternative_precision:.4f}<br>({alternative_required})"
                ])

            # Create a DataFrame from the table data
            df = pd.DataFrame(table_data, columns=['', 'מצב קיים', 'חלופה'])

            def color_cell(value):
                """עיצוב תא לפי התנאי."""
                # Extract the precision value from the cell (before the <br>)
                precision = float(value.split("<br>")[0])
                if precision <= relative_precision:
                    return f"background-color: #d4edda; color: black;"  # ירוק
                else:
                    return f"background-color: #f8d7da; color: black;"  # אדום

            # עיצוב הטבלה
            styled_table = (
                df.style
                .applymap(color_cell, subset=['מצב קיים', 'חלופה'])  # עיצוב לפי הערכים
                .set_table_styles([
                    {'selector': 'th', 'props': [('text-align', 'right')]},
                    {'selector': 'td', 'props': [('white-space', 'pre-line')]}  # Allow line breaks in cells
                ])
                .set_properties(**{'text-align': 'right'})  # יישור ימין לטקסט
            )

            # הצגת הטבלה
            st.write(styled_table.to_html(escape=False), unsafe_allow_html=True)



            # Calculate required additional runs
            max_additional_current = max(rep for rep in reps_current)
            max_additional_alternative = max(rep for rep in reps_alternative)

            # Show additional runs section if needed
            if max(max_additional_current, max_additional_alternative) > 0:
                st.markdown(f"""
                    <div style='text-align: right; direction: rtl; 
                        background-color: #420518; 
                        color: white; 
                        padding: 1rem; 
                        border-radius: 0.5rem;
                        margin: 1rem 0;'>
                        <strong>נדרשות הרצות נוספות:</strong>
                        <br>מצב קיים: {max_additional_current} הרצות
                        <br>חלופה: {max_additional_alternative} הרצות
                    </div>
                """, unsafe_allow_html=True)

        with col2:
                
            fig = go.Figure()

            fig.add_trace(
                go.Bar(
                    name="מצב קיים",
                    x=metrics,
                    y=current_means,
                    error_y=dict(type='data', array=current_conf_intervals),
                    marker_color='rgb(55, 83, 109)'
                )
            )

            fig.add_trace(
                go.Bar(
                    name="חלופה",
                    x=metrics,
                    y=alt_means,
                    error_y=dict(type='data', array=alt_conf_intervals),
                    marker_color='rgb(26, 118, 255)'
                )
            )

            fig.update_layout(
                barmode='group',
                height=500,
                title_text="השוואת מדדי ביצוע",
                font=dict(size=30),
                title_x=0.5,
                xaxis_title="מדדים",
                yaxis_title="ממוצע",
                showlegend=True
            )

            st.plotly_chart(fig)

        
        repitition_needed = max(max_additional_current, max_additional_alternative)
        # Run additional simulations for current scenario
        if repitition_needed > 0:
            new_current = run_complete_simulation(
                len(current_data[0]), repitition_needed
            )
            current_data = tuple(
                list(old) + list(new)
                for old, new in zip(current_data, new_current)
            )
        
            # Run additional simulations for alternative scenario
            new_alternative = run_complete_simulation(
                len(alternative_data[0]), max_additional_alternative, extra_employee
            )
            alternative_data = tuple(
                list(old) + list(new)
                for old, new in zip(alternative_data, new_alternative)
            )

        if not st.session_state.simulation_state['additional_runs_completed']:
            if st.button("בצע הרצות נוספות", key="additional_runs_button"):
                with st.spinner('מבצע הרצות נוספות...'):
                    try:
                        # Run the initial analysis to get the current and alternative data
                        current_data, alternative_data, reps_current, reps_alternative, relative_precision_current, relative_precision_alternative = initial_analysis(
                            initial_runs, alpha, precision, extra_employee
                        )
                        
                        # Update session state with the data
                        st.session_state.simulation_state.update({
                            'initialized': True,
                            'current_data': current_data,
                            'alternative_data': alternative_data,
                            'reps_current': reps_current,
                            'reps_alternative': reps_alternative,
                            'relative_precision_current': relative_precision_current,  # Store relative precision for current scenario
                            'relative_precision_alternative': relative_precision_alternative,  # Store relative precision for alternative scenario
                            'extra_employee': extra_employee,
                            'initial_runs': initial_runs,
                            'alpha': alpha,
                            'additional_runs_completed': True,
                            'final_results': None,
                            'show_results': True  # Set to True when simulation is run
                        })
                        

                        st.success("ההרצות הנוספות הושלמו בהצלחה!")
                        
                        
                    except Exception as e:
                        st.error(f"שגיאה בביצוע ההרצות הנוספות: {str(e)}")

        # Show final results if additional runs are completed
        if st.session_state.simulation_state['additional_runs_completed']:
            results = st.session_state.simulation_state.get('current_data', {})
            
            if results:
                # Create final visualization with updated data
                current_data = st.session_state.simulation_state['current_data']
                alternative_data = st.session_state.simulation_state['alternative_data']
                n_samples = max(max_additional_alternative, max_additional_current)

                st.markdown(f"""
                    <div style='text-align: right; direction: rtl;'>
                        <strong> השוואת מדדי ביצוע עבור {n_samples} הרצות:</strong>
                    </div>
                """, unsafe_allow_html=True)


                current_means = [np.mean(data) for data in current_data]
                current_stds = [np.std(data, ddof=1) / np.sqrt(len(data)) for data in current_data]
                alt_means = [np.mean(data) for data in alternative_data]
                alt_stds = [np.std(data, ddof=1) / np.sqrt(len(data)) for data in alternative_data]

                current_errors = [t.ppf(1 - alpha / 2, df=len(data) - 1) * std for data, std in zip(current_data, current_stds)]
                alt_errors = [t.ppf(1 - alpha / 2, df=len(data) - 1) * std for data, std in zip(alternative_data, alt_stds)]

                # Calculate relative precision for both scenarios
                current_relative_precisions = [(t.ppf(1 - alpha / 2, df=len(data) - 1) * std) / mean 
                                            for data, std, mean in zip(current_data, current_stds, current_means)]
                alternative_relative_precisions = [(t.ppf(1 - alpha / 2, df=len(data) - 1) * std) / mean 
                                                for data, std, mean in zip(alternative_data, alt_stds, alt_means)]


                
                
                col1, col2 = st.columns([2,3])
            
                
                with col1:
                    st.write("")
                    st.write("")
                    st.write("")

                    metrics = ["שירות הושלם", "לקוחות שעזבו", "מנות לא מבושלות"]
                    relative_precision = precision/(1+precision)

                    # Create table data with relative precision
                    table_data = []
                    for i, metric in enumerate(metrics):
                        current_achieved = current_relative_precisions[i]
                        alternative_achieved = alternative_relative_precisions[i]
                        
                        # Format the cells with precision values
                        current_cell = f"{current_achieved:.4f}"
                        alternative_cell = f"{alternative_achieved:.4f}"
                        
                        table_data.append([metric, current_cell, alternative_cell])

                    # Create DataFrame
                    df = pd.DataFrame(table_data, columns=['', 'מצב קיים', 'חלופה'])

                    def color_cell(value):
                        """Style cell based on relative precision threshold."""
                        try:
                            precision_val = float(value)
                            if precision_val <= relative_precision:
                                return f"background-color: #d4edda; color: black;"
                            return f"background-color: #f8d7da; color: black;"
                        except:
                            return ""

                    # Style the table
                    styled_table = (
                        df.style
                        .applymap(color_cell, subset=['מצב קיים', 'חלופה'])
                        .set_table_styles([
                            {'selector': 'th', 'props': [('text-align', 'right')]},
                            {'selector': 'td', 'props': [('text-align', 'right')]}
                        ])
                    )

                    st.write(styled_table.to_html(escape=False), unsafe_allow_html=True)

                with col2:
                    fig = go.Figure()

                    fig.add_trace(
                        go.Bar(
                            name="מצב קיים",
                            x=metrics,
                            y=current_means,
                            error_y=dict(type='data', array=current_errors),
                            marker_color='rgb(55, 83, 109)'
                        )
                    )

                    fig.add_trace(
                        go.Bar(
                            name="חלופה",
                            x=metrics,
                            y=alt_means,
                            error_y=dict(type='data', array=alt_errors),
                            marker_color='rgb(26, 118, 255)'
                        )
                    )

                    fig.update_layout(
                        barmode='group',
                        height=500,
                        title_text="השוואת מדדי ביצוע (לאחר הרצות נוספות)",
                        font=dict(size=30),
                        title_x=0.5,
                        xaxis_title="מדדים",
                        yaxis_title="ממוצע",
                        showlegend=True
                    )

                    st.plotly_chart(fig)

                # Display final analysis
                st.markdown("<h3 style='text-align: right;'>ניתוח סופי</h3>", unsafe_allow_html=True)
                
                measure_names = {
                    'served': '🤭לקוחות ששורתו',
                    'left': '😡לקוחות שעזבו',
                    'undercooked': '🍲מנות לא מבושלות'
                }
                results = run_extended_analysis(current_data, alternative_data, alpha)
                col1, col2, col3 = st.columns(3)
                for (measure, data), col in zip(results.items(), [col1, col2, col3]):
                    with col:
                        st.markdown(f"""
                            <div style='
                                text-align: right;
                                direction: rtl;
                                padding: 1rem;
                                background-color: #420518;
                                border-radius: 0.5rem;
                                height: 100%;
                                color: white;
                            '>
                                <h4 style='color: #1f77b4; margin-bottom: 1rem;'>{measure_names[measure]}</h4>
                                <p style='background-color: #420518; padding: 0.25rem;'>  
                                    <strong>הפרש ממוצע:   </strong>{data['mean_diff']:.2f}
                                </p>
                                <p style='background-color: #420518; padding: 0.25rem;'>
                                    <strong>רווח סמך:   </strong>[{data['ci_lower']:.2f}, {data['ci_upper']:.2f}]
                                </p>
                                <p style='background-color: #420518; padding: 0.25rem;'>
                                    <strong>מסקנה:   </strong>{data['preference']}
                                </p>
                            </div>
                        """, unsafe_allow_html=True)







if __name__ == "__main__":
    show_simulation_page()