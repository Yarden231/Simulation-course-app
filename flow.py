import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
import pandas as pd
import seaborn as sns
from utils import set_rtl, set_ltr_sliders

import heapq
import numpy as np
from scipy.stats import t

def show():

    # Load custom CSS
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        set_ltr_sliders()

    # Header section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h1 class="section-header">סימולציית זרימה - משאית המזון 🚚</h1>
            <p>ניתוח התנהגות המערכת באמצעות סימולציית אירועים בדידים</p>
        </div>
    """, unsafe_allow_html=True)

    # Simulation Parameters Section
    st.markdown("""
        <div class="custom-card rtl-content">
            <h2 class="section-header">הגדרת פרמטרים לסימולציה</h2>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>תצורת המערכת</h3>
            </div>
        """, unsafe_allow_html=True)
        
        extra_employee = st.selectbox(
            'בחר תצורת עובדים:',
            ['מצב נוכחי', 'עובד נוסף בהזמנות', 'עובד נוסף בהכנה', 'עובד נוסף באריזה'],
            index=0,
            key='employee_config'
        )

        simulation_time = st.slider(
            'משך הסימולציה (דקות):',
            min_value=60,
            max_value=300,
            value=300,
            step=30,
            key='sim_time'
        )

    with col2:
        st.markdown("""
            <div class="custom-card rtl-content">
                <h3>פרמטרים סטטיסטיים</h3>
            </div>
        """, unsafe_allow_html=True)
        
        confidence_level = st.slider(
            'רמת ביטחון:',
            min_value=0.90,
            max_value=0.99,
            value=0.95,
            step=0.01,
            key='conf_level'
        )

        initial_runs = st.number_input(
            'מספר הרצות התחלתי:',
            min_value=10,
            max_value=100,
            value=20,
            step=5,
            key='initial_runs'
        )

    # Run Simulation Button
    if st.button('הרץ סימולציה', key='run_sim'):
        with st.spinner('מריץ סימולציה...'):
            # Run initial simulations
            results = run_initial_simulations(
                initial_runs=initial_runs,
                simulation_time=simulation_time,
                extra_employee=extra_employee
            )
            
            # Display results
            display_simulation_results(results, confidence_level)

def run_initial_simulations(initial_runs, simulation_time, extra_employee):
    """Run the initial set of simulations and collect results."""
    results = {
        'served': [],
        'left': [],
        'undercooked': []
    }
    
    extra_employee_map = {
        'מצב נוכחי': None,
        'עובד נוסף בהזמנות': 'order',
        'עובד נוסף בהכנה': 'prep',
        'עובד נוסף באריזה': 'pickup'
    }
    
    employee_config = extra_employee_map[extra_employee]
    
    for _ in range(initial_runs):
        food_truck = EventBasedFoodTruck(extra_employee=employee_config)
        served, left, undercooked = run_simulation(food_truck, simulation_time)
        results['served'].append(served)
        results['left'].append(left)
        results['undercooked'].append(undercooked)
    
    return results

def display_simulation_results(results, confidence_level):
    """Display the simulation results with visualizations."""
    
    # Calculate statistics and confidence intervals
    stats = calculate_statistics(results, confidence_level)
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        display_metric_card(
            "לקוחות ששורתו",
            stats['served']['mean'],
            stats['served']['ci'],
            "green"
        )
    
    with col2:
        display_metric_card(
            "לקוחות שעזבו",
            stats['left']['mean'],
            stats['left']['ci'],
            "red"
        )
    
    with col3:
        display_metric_card(
            "מנות לא מבושלות",
            stats['undercooked']['mean'],
            stats['undercooked']['ci'],
            "yellow"
        )

    # Create and display visualization
    fig = create_results_visualization(results)
    st.pyplot(fig)

def calculate_statistics(results, confidence_level):
    """Calculate mean and confidence intervals for simulation results."""
    stats = {}
    alpha = 1 - confidence_level
    
    for metric in results:
        data = np.array(results[metric])
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        n = len(data)
        t_value = t.ppf(1 - alpha/2, n-1)
        ci_margin = t_value * (std / np.sqrt(n))
        
        stats[metric] = {
            'mean': mean,
            'ci': (mean - ci_margin, mean + ci_margin)
        }
    
    return stats

def display_metric_card(title, mean, ci, color):
    """Display a metric card with mean and confidence interval."""
    color_map = {
        "green": "#4CAF50",
        "red": "#F44336",
        "yellow": "#FFC107"
    }
    
    st.markdown(f"""
        <div style="
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid {color_map[color]};
            text-align: center;
        ">
            <h4 style="color: white; margin-bottom: 10px;">{title}</h4>
            <div style="font-size: 24px; color: {color_map[color]};">{mean:.1f}%</div>
            <div style="font-size: 12px; color: #888;">
                CI: [{ci[0]:.1f}%, {ci[1]:.1f}%]
            </div>
        </div>
    """, unsafe_allow_html=True)

def create_results_visualization(results):
    """Create visualization of simulation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribution plots
    sns.boxplot(data=[
        results['served'],
        results['left'],
        results['undercooked']
    ], ax=ax1)
    ax1.set_xticklabels(['שורתו', 'עזבו', 'לא מבושל'])
    ax1.set_title('התפלגות התוצאות')
    
    # Correlation plot
    df = pd.DataFrame(results)
    sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', ax=ax2)
    ax2.set_title('מטריצת קורלציות')
    
    plt.tight_layout()
    return fig


# Function to calculate required sample sizes and confidence intervals with Bonferroni-corrected alpha
def calculate_required_repetitions(served_totals, left_totals, undercooked_totals, initial_n, alpha, relative_precision):
    corrected_alpha = alpha / 6
    t_score = t.ppf(1 - corrected_alpha / 2, df=initial_n - 1)

    def repetitions_needed(data):
        mean_data = np.mean(data)
        std_data = np.std(data, ddof=1)
        delta_0 = t_score * (std_data / np.sqrt(initial_n))
        delta_t = relative_precision * mean_data
        return max(0, int(np.ceil(initial_n * (delta_0 / delta_t) ** 2)) - initial_n)

    additional_served = repetitions_needed(served_totals)
    additional_left = repetitions_needed(left_totals)
    additional_undercooked = repetitions_needed(undercooked_totals)

    ci_served = t_score * (np.std(served_totals, ddof=1) / np.sqrt(initial_n))
    ci_left = t_score * (np.std(left_totals, ddof=1) / np.sqrt(initial_n))
    ci_undercooked = t_score * (np.std(undercooked_totals, ddof=1) / np.sqrt(initial_n))

    return (additional_served, ci_served), (additional_left, ci_left), (additional_undercooked, ci_undercooked)

# Function to calculate confidence intervals for the difference between current and alternative scenarios
def calculate_pairwise_CI(current_data, alternative_data, alpha=0.05):
    differences = np.array(current_data) - np.array(alternative_data)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)

    t_score = t.ppf(1 - alpha / 2, df=n - 1)
    margin_of_error = t_score * (std_diff / np.sqrt(n))
    CI_lower = mean_diff - margin_of_error
    CI_upper = mean_diff + margin_of_error

    return mean_diff, CI_lower, CI_upper, margin_of_error

# Function to run additional repetitions
def run_additional_repetitions(extra_employee, num_repetitions):
    served_totals, left_totals, undercooked_totals = [], [], []

    for _ in range(num_repetitions):
        served, left, undercooked = run_simulation(extra_employee=extra_employee)
        served_totals.append(served)
        left_totals.append(left)
        undercooked_totals.append(undercooked)

    return served_totals, left_totals, undercooked_totals

# Run additional repetitions and calculate CI for differences
def extended_analysis(initial_data_current, initial_data_alternative, additional_reps_current, additional_reps_alternative, alpha):
    # Apply Bonferroni correction
    corrected_alpha = alpha / 6

    # Determine maximum number of additional repetitions needed across all measures and scenarios
    max_repetitions = max(
        *[rep[0] for rep in additional_reps_current],
        *[rep[0] for rep in additional_reps_alternative]
    )

    # Run additional repetitions if needed
    if max_repetitions > 0:
        # Extend initial data with additional repetitions for the current scenario
        extra_served, extra_left, extra_undercooked = run_additional_repetitions(None, max_repetitions)
        initial_data_current[0].extend(extra_served)
        initial_data_current[1].extend(extra_left)
        initial_data_current[2].extend(extra_undercooked)

        # Extend initial data with additional repetitions for the alternative scenario
        extra_served_alt, extra_left_alt, extra_undercooked_alt = run_additional_repetitions('prep', max_repetitions)
        initial_data_alternative[0].extend(extra_served_alt)
        initial_data_alternative[1].extend(extra_left_alt)
        initial_data_alternative[2].extend(extra_undercooked_alt)

    # Calculate pairwise confidence intervals for each measure
    served_diff_mean, served_CI_lower, served_CI_upper, _ = calculate_pairwise_CI(initial_data_current[0], initial_data_alternative[0], corrected_alpha)
    left_diff_mean, left_CI_lower, left_CI_upper, _ = calculate_pairwise_CI(initial_data_current[1], initial_data_alternative[1], corrected_alpha)
    undercooked_diff_mean, undercooked_CI_lower, undercooked_CI_upper, _ = calculate_pairwise_CI(initial_data_current[2], initial_data_alternative[2], corrected_alpha)

    # Interpretation of results for each measure
    def interpret_results(diff_mean, CI_lower, CI_upper, measure_name, objective='minimize'):
        if objective == 'maximize':
            # Prefer alternative if entire CI is positive (current - alternative > 0), meaning current performs worse
            if CI_lower > 0:
                result = " Current state is preferred."
            elif CI_upper < 0:
                result = "Alternative  is preferred."
            else:
                result = "No preference; results are inconclusive."
        else:  # For measures to minimize
            # Prefer alternative if entire CI is negative (current - alternative < 0), meaning alternative performs worse
            if CI_upper < 0:
                result = "Current state is preferred."
            elif CI_lower > 0:
                result = "Alternative is preferred."
            else:
                result = "No preference; results are inconclusive."

        print(f"\n{measure_name}: Mean Difference: {diff_mean:.2f}, CI: [{CI_lower:.2f}, {CI_upper:.2f}] - {result}")

    # Display interpretation for each measure with the correct objective
    interpret_results(served_diff_mean, served_CI_lower, served_CI_upper, "Customers Served", objective='maximize')
    interpret_results(left_diff_mean, left_CI_lower, left_CI_upper, "Customers Who Left", objective='minimize')
    interpret_results(undercooked_diff_mean, undercooked_CI_lower, undercooked_CI_upper, "Undercooked Meals", objective='minimize')

# Run initial simulations for both scenarios and calculate CIs
def initial_analysis(initial_n, alpha, relative_precision, extra_employee):
    # Data collection for current and alternative scenarios
    current_served, current_left, current_undercooked = [], [], []
    alternative_served, alternative_left, alternative_undercooked = [], [], []

    for _ in range(initial_n):
        served, left, undercooked = run_simulation()
        current_served.append(served)
        current_left.append(left)
        current_undercooked.append(undercooked)

        served, left, undercooked = run_simulation(extra_employee=extra_employee)
        alternative_served.append(served)
        alternative_left.append(left)
        alternative_undercooked.append(undercooked)

    # Calculate repetitions and confidence intervals for the current and alternative states
    additional_reps_current = calculate_required_repetitions(
        current_served, current_left, current_undercooked, initial_n, alpha, relative_precision
    )
    additional_reps_alternative = calculate_required_repetitions(
        alternative_served, alternative_left, alternative_undercooked, initial_n, alpha, relative_precision
    )

    # Plot initial confidence intervals for each measure
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Current scenario plot
    ax[0].bar(['Served', 'Left', 'Undercooked'],
               [np.mean(current_served), np.mean(current_left), np.mean(current_undercooked)],
               yerr=[additional_reps_current[0][1], additional_reps_current[1][1], additional_reps_current[2][1]],
               capsize=10)
    ax[0].set_title('Current Scenario')
    ax[0].set_ylabel('Mean with CI')

    # Alternative scenario plot
    ax[1].bar(['Served', 'Left', 'Undercooked'],
               [np.mean(alternative_served), np.mean(alternative_left), np.mean(alternative_undercooked)],
               yerr=[additional_reps_alternative[0][1], additional_reps_alternative[1][1], additional_reps_alternative[2][1]],
               capsize=10)
    ax[1].set_title(f"Alternative Scenario (extra employee at '{extra_employee}' station)")
    ax[1].set_ylabel('Mean with CI')

    plt.tight_layout()
    plt.show()

    # Display additional repetitions needed
    print("Current Scenario:")
    print(f"  Additional repetitions needed (served): {additional_reps_current[0][0]}")
    print(f"  Additional repetitions needed (left): {additional_reps_current[1][0]}")
    print(f"  Additional repetitions needed (undercooked): {additional_reps_current[2][0]}\n")

    print("Alternative Scenario:")
    print(f"  Additional repetitions needed (served): {additional_reps_alternative[0][0]}")
    print(f"  Additional repetitions needed (left): {additional_reps_alternative[1][0]}")
    print(f"  Additional repetitions needed (undercooked): {additional_reps_alternative[2][0]}")

    # Return initial data and additional repetitions needed
    return (current_served, current_left, current_undercooked), \
           (alternative_served, alternative_left, alternative_undercooked), \
           additional_reps_current, additional_reps_alternative

# Generalized class for the food truck simulation
class EventBasedFoodTruck:
    def __init__(self, extra_employee=None):
        # Initialize the simulation variables
        self.current_time = 0
        self.event_queue = []  # Priority queue for events
        self.extra_employee = extra_employee  # Allows an additional employee at one station
        # Define the max capacity for each station based on where the extra employee is assigned
        self.max_order_stations = 2 if self.extra_employee == 'order' else 1
        self.max_prep_stations = 2 if self.extra_employee == 'prep' else 1
        self.max_pickup_stations = 2 if self.extra_employee == 'pickup' else 1
        self.order_station_busy = 0
        self.prep_station_busy = 0
        self.pickup_station_busy = 0
        self.batch = []  # Queue for orders waiting to be processed
        self.left_count = 0  # Count of visitors who left due to long wait
        self.undercooked_count = 0  # Count of undercooked meals in 3-meal batches
        self.wait_times = []  # List to store waiting times of visitors (includes both served and left)
        self.visitors = {}  # Dictionary to store information about each visitor
        self.customers_arrived = 0  # Track total arrivals
        self.customers_served = 0  # Track successfully served customers
        self.three_meal_batches = 0  # Count of 3-meal batches prepared

    # Function to schedule a new event
    def schedule_event(self, time, event_type, visitor_id):
        heapq.heappush(self.event_queue, (time, event_type, visitor_id))

    # Function to process the next event in the queue
    def process_event(self):
        if not self.event_queue:
            return False
        time, event_type, visitor_id = heapq.heappop(self.event_queue)
        self.current_time = time

        # Check patience for all customers each time an event is processed
        self.check_patience()

        if event_type == 'arrival':
            self.handle_arrival(visitor_id)
        elif event_type == 'order_end':
            self.handle_order_end(visitor_id)
        elif event_type == 'prep_end':
            self.handle_prep_end(visitor_id)
        elif event_type == 'pickup_end':
            self.handle_pickup_end(visitor_id)
        elif event_type == 'monitor':
            self.monitor()
        return True

    # Function to check if any customer exceeds their patience threshold
    def check_patience(self):
        for visitor_id in list(self.visitors.keys()):
            if visitor_id in self.batch or visitor_id in self.visitors:
                arrival_time = self.visitors[visitor_id]['arrival_time']
                max_wait_time = self.visitors[visitor_id]['max_wait_time']
                wait_time = self.current_time - arrival_time
                if wait_time > max_wait_time:
                    self.left_count += 1
                    self.wait_times.append(wait_time)
                    if visitor_id in self.batch:
                        self.batch.remove(visitor_id)
                    del self.visitors[visitor_id]  # Remove from tracking
                    continue

    # Handle the arrival of a new visitor
    def handle_arrival(self, visitor_id):
        self.customers_arrived += 1
        max_wait_time = np.random.uniform(5, 20)
        self.visitors[visitor_id] = {
            'arrival_time': self.current_time,
            'max_wait_time': max_wait_time
        }
        if self.order_station_busy < self.max_order_stations:
            self.order_station_busy += 1
            self.schedule_event(self.current_time, 'order_end', visitor_id)
        else:
            self.batch.append(visitor_id)

    # Handle the end of the ordering process
    def handle_order_end(self, visitor_id):
        self.order_station_busy -= 1
        if visitor_id not in self.visitors:  # If customer already left, skip
            return

        # Ensure the event is not re-scheduled unnecessarily
        if 'order_complete' in self.visitors[visitor_id]:
            return

        # Mark as order completed
        self.visitors[visitor_id]['order_complete'] = True

        batch_size = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
        self.visitors[visitor_id]['batch_size'] = batch_size
        if batch_size == 1:
            service_time = np.random.normal(5, 1)  # Single meal prep time
        elif batch_size == 2:
            service_time = np.random.normal(8, 2)  # Batch of 2 prep time
        else:
            service_time = np.random.normal(10, 3)  # Batch of 3 prep time
            self.three_meal_batches += 1
            if np.random.rand() < 0.5:
                self.undercooked_count += 1

        if self.prep_station_busy < self.max_prep_stations:
            self.prep_station_busy += 1
            self.schedule_event(self.current_time + service_time, 'prep_end', visitor_id)
        else:
            self.batch.append(visitor_id)

        # Check if there are more orders to process
        if len(self.batch) > 0 and self.order_station_busy < self.max_order_stations:
            self.order_station_busy += 1
            self.schedule_event(self.current_time, 'order_end', self.batch.pop(0))

    # Handle the end of the food preparation process
    def handle_prep_end(self, visitor_id):
        self.prep_station_busy -= 1
        if visitor_id not in self.visitors:  # If customer already left, skip
            return

        if self.pickup_station_busy < self.max_pickup_stations:
            self.pickup_station_busy += 1
            self.schedule_event(self.current_time, 'pickup_end', visitor_id)
        else:
            self.batch.append(visitor_id)

        # Check if there are more prep tasks to process
        if len(self.batch) > 0 and self.prep_station_busy < self.max_prep_stations:
            self.prep_station_busy += 1
            self.schedule_event(self.current_time, 'prep_end', self.batch.pop(0))

    # Handle the end of the order pickup process
    def handle_pickup_end(self, visitor_id):
        self.pickup_station_busy -= 1
        if visitor_id not in self.visitors:  # If customer already left, skip
            return

        wait_time = self.current_time - self.visitors[visitor_id]['arrival_time']
        self.wait_times.append(wait_time)
        self.customers_served += 1
        del self.visitors[visitor_id]  # Remove from tracking once served

        # Check if there are more pickups to process
        if len(self.batch) > 0 and self.pickup_station_busy < self.max_pickup_stations:
            self.pickup_station_busy += 1
            self.schedule_event(self.current_time, 'pickup_end', self.batch.pop(0))

    # Monitor function to periodically check the queue size
    def monitor(self):
        self.schedule_event(self.current_time + 1, 'monitor', None)


def run_simulation(food_truck, simulation_time):
    """Run a single simulation for the specified time."""
    visitor_count = 0
    
    # Schedule initial arrival
    interarrival_time = np.random.exponential(6)
    food_truck.schedule_event(interarrival_time, 'arrival', visitor_count)
    visitor_count += 1
    
    # Run simulation
    while food_truck.current_time < simulation_time:
        if not food_truck.process_event():
            break
            
        # Schedule next arrival
        if food_truck.event_queue and food_truck.event_queue[0][1] == 'arrival':
            interarrival_time = np.random.exponential(6)
            next_arrival_time = food_truck.current_time + interarrival_time
            food_truck.schedule_event(next_arrival_time, 'arrival', visitor_count)
            visitor_count += 1
    
    # Calculate metrics
    total_customers = food_truck.customers_served + food_truck.left_count
    if total_customers == 0:
        return 0, 0, 0
        
    served_percentage = (food_truck.customers_served / total_customers) * 100
    left_percentage = (food_truck.left_count / total_customers) * 100
    undercooked_percentage = (food_truck.undercooked_count / food_truck.three_meal_batches) * 100 if food_truck.three_meal_batches > 0 else 0
    
    return served_percentage, left_percentage, undercooked_percentage


if __name__ == "__main__":
    show()