import simpy
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EventType(Enum):
    """Enum for different types of events in the simulation"""
    ARRIVAL = "arrival"
    ORDER_START = "order_start"
    ORDER_COMPLETE = "order_complete"
    PREP_START = "prep_start"
    PREP_COMPLETE = "prep_complete"
    PICKUP_START = "pickup_start"
    PICKUP_COMPLETE = "pickup_complete"
    CUSTOMER_LEFT = "customer_left"
    UNDERCOOKED = "undercooked"

@dataclass
class Customer:
    """Represents a customer in the system"""
    id: int
    arrival_time: float
    order_time: Optional[float] = None
    prep_time: Optional[float] = None
    pickup_time: Optional[float] = None
    total_time: Optional[float] = None
    has_left: bool = False
    order_undercooked: bool = False

@dataclass
class SimulationMetrics:
    """Tracks various metrics during simulation"""
    wait_times: Dict[str, List[float]] = field(default_factory=lambda: {
        'order': [], 'prep': [], 'pickup': [], 'total': []
    })
    queue_sizes: Dict[str, List[int]] = field(default_factory=lambda: {
        'order': [], 'prep': [], 'pickup': [], 'total': []
    })
    customers_left: int = 0
    customers_served: int = 0
    undercooked_orders: int = 0
    total_customers: int = 0

@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation"""
    order_capacity: int
    prep_capacity: int
    pickup_capacity: int
    order_time_range: Tuple[float, float]
    prep_time_params: Tuple[float, float]  # mean and std for normal distribution
    pickup_time_range: Tuple[float, float]
    leave_probability: float
    undercook_probability: float = 0.1

class FoodTruckSimulation:
    """Main simulation class for the food truck"""
    
    def __init__(self, env: simpy.Environment, config: SimulationConfig):
        self.env = env
        self.config = config
        self.metrics = SimulationMetrics()
        self.customers: Dict[int, Customer] = {}
        self.events: List[Dict] = []
        
        # Initialize resources
        self.order_station = simpy.Resource(env, capacity=config.order_capacity)
        self.prep_station = simpy.Resource(env, capacity=config.prep_capacity)
        self.pickup_station = simpy.Resource(env, capacity=config.pickup_capacity)
        
        # Initialize batch processing
        self.prep_batch: List[Customer] = []
        self.batch_process = env.process(self.process_prep_batch())
        
        # Start monitoring process
        self.env.process(self.monitor_queues())

    def log_event(self, customer_id: int, event_type: EventType, time: float):
        """Log events for analysis"""
        event = {
            'customer_id': customer_id,
            'event_type': event_type.value,
            'time': time,
            'queue_sizes': {
                'order': len(self.order_station.queue),
                'prep': len(self.prep_station.queue),
                'pickup': len(self.pickup_station.queue)
            }
        }
        self.events.append(event)
        logger.debug(f"Event logged: {event}")

    def process_customer(self, customer: Customer):
        """Process a single customer through the system"""
        # Check if customer leaves before ordering
        if np.random.random() < self.config.leave_probability:
            self.metrics.customers_left += 1
            customer.has_left = True
            self.log_event(customer.id, EventType.CUSTOMER_LEFT, self.env.now)
            return

        # Order process
        with self.order_station.request() as request:
            yield request
            order_start = self.env.now
            self.log_event(customer.id, EventType.ORDER_START, order_start)
            
            # Simulate order time
            order_time = np.random.uniform(*self.config.order_time_range)
            yield self.env.timeout(order_time)
            
            customer.order_time = self.env.now - order_start
            self.metrics.wait_times['order'].append(customer.order_time)
            self.log_event(customer.id, EventType.ORDER_COMPLETE, self.env.now)

        # Add to prep batch
        self.prep_batch.append(customer)

    def process_prep_batch(self):
        """Process food preparation in batches"""
        while True:
            if len(self.prep_batch) >= 3 or (len(self.prep_batch) > 0 and self.env.now % 5 == 0):
                current_batch = self.prep_batch[:3]
                self.prep_batch = self.prep_batch[3:]
                
                with self.prep_station.request() as request:
                    yield request
                    prep_start = self.env.now
                    
                    # Log prep start for batch
                    for customer in current_batch:
                        self.log_event(customer.id, EventType.PREP_START, prep_start)
                    
                    # Simulate preparation time
                    mean, std = self.config.prep_time_params
                    prep_time = max(1, np.random.normal(mean, std))
                    yield self.env.timeout(prep_time)
                    
                    # Process each customer in batch
                    for customer in current_batch:
                        customer.prep_time = self.env.now - prep_start
                        self.metrics.wait_times['prep'].append(customer.prep_time)
                        
                        # Check for undercooked orders
                        if np.random.random() < self.config.undercook_probability:
                            customer.order_undercooked = True
                            self.metrics.undercooked_orders += 1
                            self.log_event(customer.id, EventType.UNDERCOOKED, self.env.now)
                        
                        self.log_event(customer.id, EventType.PREP_COMPLETE, self.env.now)
                        self.env.process(self.process_pickup(customer))
            
            yield self.env.timeout(1)

    def process_pickup(self, customer: Customer):
        """Handle customer pickup process"""
        with self.pickup_station.request() as request:
            yield request
            pickup_start = self.env.now
            self.log_event(customer.id, EventType.PICKUP_START, pickup_start)
            
            # Simulate pickup time
            pickup_time = np.random.uniform(*self.config.pickup_time_range)
            yield self.env.timeout(pickup_time)
            
            customer.pickup_time = self.env.now - pickup_start
            customer.total_time = self.env.now - customer.arrival_time
            
            self.metrics.wait_times['pickup'].append(customer.pickup_time)
            self.metrics.wait_times['total'].append(customer.total_time)
            self.metrics.customers_served += 1
            
            self.log_event(customer.id, EventType.PICKUP_COMPLETE, self.env.now)

    def monitor_queues(self):
        """Monitor queue sizes over time"""
        while True:
            self.metrics.queue_sizes['order'].append(len(self.order_station.queue))
            self.metrics.queue_sizes['prep'].append(len(self.prep_station.queue))
            self.metrics.queue_sizes['pickup'].append(len(self.pickup_station.queue))
            self.metrics.queue_sizes['total'].append(
                len(self.order_station.queue) + 
                len(self.prep_station.queue) + 
                len(self.pickup_station.queue)
            )
            yield self.env.timeout(1)

def create_visualization(metrics: SimulationMetrics):
    """Create visualization of simulation results"""
    fig = go.Figure()
    
    # Queue sizes over time
    times = list(range(len(metrics.queue_sizes['order'])))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=metrics.queue_sizes['order'],
        name='תור הזמנות',
        line=dict(color='#FF6B6B')
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=metrics.queue_sizes['prep'],
        name='תור הכנה',
        line=dict(color='#4ECDC4')
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=metrics.queue_sizes['pickup'],
        name='תור איסוף',
        line=dict(color='#45B7D1')
    ))
    
    fig.add_trace(go.Scatter(
        x=times,
        y=metrics.queue_sizes['total'],
        name='סה"כ בתור',
        line=dict(color='#96CEB4', width=3)
    ))
    
    fig.update_layout(
        title='גודל התורים לאורך זמן',
        xaxis_title='זמן (דקות)',
        yaxis_title='מספר לקוחות בתור',
        template='plotly_dark',
        font=dict(family='Helvetica'),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=True
    )
    
    return fig

def run_simulation(config: SimulationConfig, simulation_time: int, arrival_rate: float):
    """Run the complete simulation"""
    env = simpy.Environment()
    food_truck = FoodTruckSimulation(env, config)
    
    def customer_generator():
        customer_id = 0
        while True:
            customer_id += 1
            food_truck.metrics.total_customers += 1
            
            # Create new customer
            customer = Customer(id=customer_id, arrival_time=env.now)
            food_truck.customers[customer_id] = customer
            food_truck.log_event(customer_id, EventType.ARRIVAL, env.now)
            
            # Start customer process
            env.process(food_truck.process_customer(customer))
            
            # Wait for next customer
            yield env.timeout(np.random.exponential(arrival_rate))
    
    env.process(customer_generator())
    env.run(until=simulation_time)
    
    return food_truck

def display_simulation_results(metrics: SimulationMetrics):
    """Display comprehensive simulation results in Streamlit"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="סה\"כ לקוחות",
            value=metrics.total_customers,
            delta=f"+{metrics.customers_served} שורתו"
        )
    
    with col2:
        st.metric(
            label="זמן המתנה ממוצע",
            value=f"{np.mean(metrics.wait_times['total']):.1f} דקות",
            delta=f"מקס׳ {np.max(metrics.wait_times['total']):.1f}"
        )
    
    with col3:
        st.metric(
            label="אחוז עזיבה",
            value=f"{(metrics.customers_left/metrics.total_customers)*100:.1f}%",
            delta=f"{metrics.customers_left} לקוחות"
        )

def show_food_truck():
    """Main Streamlit application"""
    st.title("סימולציית משאית המזון")
    
    # Configuration inputs
    config = SimulationConfig(
        order_capacity=st.slider("כמות עמדות הזמנה", 1, 5, 2),
        prep_capacity=st.slider("כמות עמדות הכנה", 1, 5, 3),
        pickup_capacity=st.slider("כמות עמדות איסוף", 1, 5, 2),
        order_time_range=(
            st.slider("זמן הזמנה מינימלי", 1, 5, 2),
            st.slider("זמן הזמנה מקסימלי", 5, 15, 8)
        ),
        prep_time_params=(
            st.slider("זמן הכנה ממוצע", 3, 15, 8),
            st.slider("סטיית תקן זמן הכנה", 0.5, 5.0, 2.0)
        ),
        pickup_time_range=(1, 3),
        leave_probability=st.slider("הסתברות לעזיבה", 0.0, 0.5, 0.1),
        undercook_probability=0.1
    )
    
    simulation_time = st.slider("זמן סימולציה (דקות)", 60, 480, 120)
    arrival_rate = st.slider("זמן ממוצע בין הגעות (דקות)", 1, 20, 5)
    
    if st.button("הפעל סימולציה"):
        with st.spinner("מריץ סימולציה..."):
            food_truck = run_simulation(config, simulation_time, arrival_rate)
            
            # Display results
            display_simulation_results(food_truck.metrics)
            
            # Display visualization
            st.plotly_chart(
                create_visualization(food_truck.metrics),
                use_container_width=True
            )

if __name__ == "__main__":
    show_food_truck()