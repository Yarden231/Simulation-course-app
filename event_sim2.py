# Import necessary libraries
import streamlit as st
import numpy as np
from scipy.stats import t
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import graphviz
import random
import math
from typing import Tuple, List, Dict, NamedTuple
from dataclasses import dataclass
from enum import Enum

# Define data structures
class EmployeeLocation(Enum):
    ORDER = "order"
    PREP = "prep"
    PICKUP = "pickup"

@dataclass
class SimulationConfig:
    initial_runs: int
    alpha: float
    relative_precision: float
    extra_employee: EmployeeLocation

class SimulationResults(NamedTuple):
    served: List[float]
    left: List[float]
    undercooked: List[float]

class AnalysisResults(NamedTuple):
    mean_diff: float
    ci_lower: float
    ci_upper: float
    preference: str
    n_samples: int

# Simulation logic
class FoodTruckSimulator:
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.metrics = ["שירות הושלם", "לקוחות שעזבו", "מנות לא מבושלות"]
        
    def run_single_simulation(self, extra_employee: bool = False) -> Tuple[float, float, float]:
        """Run a single iteration of the simulation"""
        if extra_employee:
            served = random.uniform(75, 85)
            left = random.uniform(8, 15)
            undercooked = random.uniform(3, 8)
        else:
            served = random.uniform(65, 75)
            left = random.uniform(15, 25)
            undercooked = random.uniform(8, 15)
        
        return served, left, undercooked

    def run_initial_simulations(self) -> Tuple[SimulationResults, SimulationResults]:
        """Run initial set of simulations for both scenarios"""
        current_results = [[], [], []]
        alternative_results = [[], [], []]

        for _ in range(self.config.initial_runs):
            served, left, undercooked = self.run_single_simulation()
            current_results[0].append(served)
            current_results[1].append(left)
            current_results[2].append(undercooked)

            served, left, undercooked = self.run_single_simulation(True)
            alternative_results[0].append(served)
            alternative_results[1].append(left)
            alternative_results[2].append(undercooked)

        return (SimulationResults(*current_results), 
                SimulationResults(*alternative_results))

    def calculate_required_repetitions(self, data: List[float]) -> Tuple[int, float]:
        """Calculate required additional repetitions for a metric"""
        n = len(data)
        corrected_alpha = self.config.alpha / 6  # Bonferroni correction
        t_score = t.ppf(1 - corrected_alpha / 2, df=n - 1)
        
        mean_data = np.mean(data)
        std_data = np.std(data, ddof=1)
        
        current_hw = t_score * (std_data / np.sqrt(n))
        target_hw = self.config.relative_precision * abs(mean_data)
        
        additional_n = max(0, math.ceil((std_data * t_score / target_hw) ** 2 - n))
        return additional_n, current_hw

# Visualization
class SimulationVisualizer:
    @staticmethod
    def create_comparison_plot(current_data: SimulationResults, 
                             alternative_data: SimulationResults,
                             alpha: float,
                             title: str = "השוואת מדדי ביצוע") -> go.Figure:
        """Create comparison visualization between current and alternative scenarios"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("מצב קיים", "חלופה מוצעת")
        )
        
        metrics = ["שירות הושלם", "לקוחות שעזבו", "מנות לא מבושלות"]
        
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

# Main application
def show_simulation_page():
    st.title("סימולציית אירועים בדידים")
    
    # Setup configuration
    col1, col2 = st.columns(2)
    
    with col1:
        employee_location = st.selectbox(
            "מיקום העובד הנוסף",
            ["עמדת הזמנות", "עמדת הכנה", "עמדת איסוף"]
        )
        
        initial_runs = st.number_input(
            "מספר הרצות התחלתי",
            min_value=10,
            max_value=100,
            value=20,
            step=5
        )
    
    with col2:
        alpha = st.number_input(
            "רמת מובהקות (α)",
            min_value=0.01,
            max_value=0.1,
            value=0.05,
            step=0.01
        )
        
        precision = st.number_input(
            "דיוק יחסי נדרש",
            min_value=0.01,
            max_value=0.2,
            value=0.05,
            step=0.01
        )
    
    # Initialize simulation
    location_map = {
        "עמדת הזמנות": EmployeeLocation.ORDER,
        "עמדת הכנה": EmployeeLocation.PREP,
        "עמדת איסוף": EmployeeLocation.PICKUP
    }
    
    config = SimulationConfig(
        initial_runs=initial_runs,
        alpha=alpha,
        relative_precision=precision,
        extra_employee=location_map[employee_location]
    )
    
    simulator = FoodTruckSimulator(config)
    visualizer = SimulationVisualizer()
    
    # Run simulation when requested
    if st.button("הרץ סימולציה"):
        with st.spinner('מריץ סימולציה התחלתית...'):
            current_results, alternative_results = simulator.run_initial_simulations()
            
            # Show initial results
            fig = visualizer.create_comparison_plot(
                current_results,
                alternative_results,
                alpha
            )
            st.plotly_chart(fig)
            
            # Calculate additional runs needed
            additional_runs_current = max(
                simulator.calculate_required_repetitions(data)[0]
                for data in current_results
            )
            
            additional_runs_alternative = max(
                simulator.calculate_required_repetitions(data)[0]
                for data in alternative_results
            )
            
            if additional_runs_current > 0 or additional_runs_alternative > 0:
                st.warning(
                    f"נדרשות הרצות נוספות:\n"
                    f"מצב קיים: {additional_runs_current}\n"
                    f"חלופה: {additional_runs_alternative}"
                )

if __name__ == "__main__":
    main()