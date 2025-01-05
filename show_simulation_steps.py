import streamlit as st
import numpy as np
import plotly.graph_objects as go
import heapq
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional

class Event:
    def __init__(self, time: float):
        self.time = time

    def __lt__(self, other):
        return self.time < other.time

    def handle(self, sim):
        raise NotImplementedError("Handle method must be implemented by subclasses")

class CustomerArrivalEvent(Event):
    def __init__(self, time: float, customer_id: int):
        super().__init__(time)
        self.customer_id = customer_id
        
    def handle(self, sim):
        # Create new customer
        sim.state.customers_arrived += 1
        max_wait_time = np.random.uniform(5, 20)
        
        # Record customer arrival
        sim.state.visitors[self.customer_id] = {
            'arrival_time': self.time,
            'max_wait_time': max_wait_time
        }

        # Schedule reneging event
        reneging_time = self.time + max_wait_time
        sim.schedule_event(RenegingEvent(reneging_time, self.customer_id))

        # Schedule next arrival
        next_arrival_time = self.time + np.random.exponential(5.5)
        sim.schedule_event(CustomerArrivalEvent(next_arrival_time, sim.state.customers_arrived))

        # Handle current arrival
        if sim.state.order_station_busy < 1:
            sim.state.order_station_busy += 1
            order_service_time = np.random.normal(2, 1)  # Mean 3 minutes with 0.5 std
            order_completion_time = self.time + max(0.5, order_service_time)  # Minimum 0.5 minutes
            sim.schedule_event(OrderCompletionEvent(order_completion_time, self.customer_id))
        else:
            sim.state.order_queue.append(self.customer_id)

class OrderCompletionEvent(Event):
    def __init__(self, time: float, customer_id: int):
        super().__init__(time)
        self.customer_id = customer_id

    def handle(self, sim):
        if self.customer_id not in sim.state.visitors:
            return

        sim.state.order_station_busy -= 1
        
        # Calculate batch size and service time
        batch_size = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
        service_time = self._calculate_service_time(batch_size)
        
        if batch_size == 3:
            sim.state.three_meal_batches += 1
            if np.random.rand() < 0.5:
                sim.state.undercooked_count += 1

        # Move to prep station
        if sim.state.prep_station_busy < 1:
            sim.state.prep_station_busy += 1
            sim.schedule_event(PrepCompletionEvent(max(0.5,self.time+np.random.normal(2, 1)), self.customer_id))
        else:
            sim.state.prep_queue.append(self.customer_id)

        # Process next in order queue
        if sim.state.order_queue and sim.state.order_station_busy < 1:
            next_customer = sim.state.order_queue.pop(0)
            sim.state.order_station_busy += 1
            order_service_time = np.random.normal(2, 3)
            next_order_completion_time = self.time + max(0.5, order_service_time)
            sim.schedule_event(OrderCompletionEvent(next_order_completion_time, next_customer))

    def _calculate_service_time(self, batch_size: int) -> float:
        if batch_size == 1:
            return max(0, np.random.normal(5, 1))
        elif batch_size == 2:
            return max(0, np.random.normal(8, 2))
        else:
            return max(0, np.random.normal(10, 3))

class PrepCompletionEvent(Event):
    def __init__(self, time: float, customer_id: int):
        super().__init__(time)
        self.customer_id = customer_id

    def handle(self, sim):
        if self.customer_id not in sim.state.visitors:
            return

        sim.state.prep_station_busy -= 1

        if sim.state.pickup_station_busy < 1:
            sim.state.pickup_station_busy += 1
            sim.schedule_event(PickupCompletionEvent(self.time+np.random.uniform(1,2), self.customer_id))
        else:
            sim.state.pickup_queue.append(self.customer_id)

        if sim.state.prep_queue and sim.state.prep_station_busy < 1:
            next_customer = sim.state.prep_queue.pop(0)
            sim.state.prep_station_busy += 1
            sim.schedule_event(PrepCompletionEvent(max(0.5,self.time+np.random.normal(2, 4)), next_customer))

class PickupCompletionEvent(Event):
    def __init__(self, time: float, customer_id: int):
        super().__init__(time)
        self.customer_id = customer_id

    def handle(self, sim):
        if self.customer_id not in sim.state.visitors:
            return

        sim.state.pickup_station_busy -= 1
        sim.state.customers_served += 1
        del sim.state.visitors[self.customer_id]

        if sim.state.pickup_queue and sim.state.pickup_station_busy < 1:
            next_customer = sim.state.pickup_queue.pop(0)
            sim.state.pickup_station_busy += 1
            sim.schedule_event(PickupCompletionEvent(self.time+np.random.uniform(1,2), next_customer))

class RenegingEvent(Event):
    def __init__(self, time: float, customer_id: int):
        super().__init__(time)
        self.customer_id = customer_id

    def handle(self, sim):
        # Check if customer is still in the system
        if self.customer_id not in sim.state.visitors:
            return
        
        # Customer leaves the system
        sim.state.left_count += 1
        
        # Remove from appropriate queue
        if self.customer_id in sim.state.order_queue:
            sim.state.order_queue.remove(self.customer_id)
        elif self.customer_id in sim.state.prep_queue:
            sim.state.prep_queue.remove(self.customer_id)
        elif self.customer_id in sim.state.pickup_queue:
            sim.state.pickup_queue.remove(self.customer_id)
            
        del sim.state.visitors[self.customer_id]

@dataclass
class SimulationState:
    current_time: float = 0
    event_queue: List[Event] = None
    order_station_busy: int = 0
    prep_station_busy: int = 0
    pickup_station_busy: int = 0
    order_queue: List[int] = None
    prep_queue: List[int] = None
    pickup_queue: List[int] = None
    left_count: int = 0
    undercooked_count: int = 0
    customers_arrived: int = 0
    customers_served: int = 0
    three_meal_batches: int = 0
    visitors: Dict = None
    order_queue_history: List[int] = None
    prep_queue_history: List[int] = None
    pickup_queue_history: List[int] = None
    time_history: List[float] = None

    def __post_init__(self):
        self.event_queue = []
        self.order_queue = []
        self.prep_queue = []
        self.pickup_queue = []
        self.visitors = {}
        self.order_queue_history = []
        self.prep_queue_history = []
        self.pickup_queue_history = []
        self.time_history = []

class FoodTruckSimulation:
    def __init__(self):
        self.state = SimulationState()
        self.initialize_simulation()

    def initialize_simulation(self):
        # Schedule first arrival
        first_arrival_time = np.random.exponential(6)
        self.schedule_event(CustomerArrivalEvent(first_arrival_time, 0))

    def schedule_event(self, event: Event):
        heapq.heappush(self.state.event_queue, event)

    def process_next_event(self) -> bool:
        if not self.state.event_queue:
            return False

        # Get next event and update simulation time
        event = heapq.heappop(self.state.event_queue)
        previous_time = self.state.current_time
        self.state.current_time = event.time
        
        # Handle the event
        event.handle(self)

        # Update history only if time has changed
        if self.state.current_time != previous_time:
            self.state.order_queue_history.append(len(self.state.order_queue))
            self.state.prep_queue_history.append(len(self.state.prep_queue))
            self.state.pickup_queue_history.append(len(self.state.pickup_queue))
            self.state.time_history.append(self.state.current_time)

        return True

def show_food_truck_simulation():
    if 'simulation' not in st.session_state:
        st.session_state.simulation = FoodTruckSimulation()

    st.title("סימולציה של משאית המזון - הרצה צעד אחר צעד")

    st.title("לחץ על כפתור הצעד הבא כדי להתקדם לזמן האירוע הבא בסימולציה")

    # Create three columns for main metrics
    col1, col2, col3 = st.columns([1, 1, 2])
    
    
    sim_state = st.session_state.simulation.state
    
    with col1:

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.markdown(
            f"""
            <div style="border-radius: 8px; padding: 15px; ">
                <h1 style="color: #FFFFFF; text-align: center;">⏱️ שעון הסימולציה</h1>
                <h1 style="color: #FFFFFF; text-align: center;">{sim_state.current_time:.2f}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        st.markdown("<h3 style='text-align: right;'>משתנים סוכמים</h3>", unsafe_allow_html=True)
        
        st.markdown(
            f"""
            <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                <h4 style="color: #FFFFFF; text-align: center;">👥 לקוחות שהגיעו</h4>
                <h2 style="color: #FFFFFF; text-align: center;">{sim_state.customers_arrived}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                <h4 style="color: #FFFFFF; text-align: center;">✅ לקוחות ששורתו</h4>
                <h2 style="color: #FFFFFF; text-align: center;">{sim_state.customers_served}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                <h4 style="color: #FFFFFF; text-align: center;">😤 לקוחות שעזבו</h4>
                <h2 style="color: #FFFFFF; text-align: center;">{sim_state.left_count}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                <h4 style="color: #FFFFFF; text-align: center;">🍲 מנות לא מבושלות</h4>
                <h2 style="color: #FFFFFF; text-align: center;">{sim_state.undercooked_count}</h2>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col2:

        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")

        # Next Event Information
        st.markdown("<h3 style='text-align: right;'>האירוע הבא</h3>", unsafe_allow_html=True)
        if sim_state.event_queue:
            next_event = sim_state.event_queue[0]
            event_type = next_event.__class__.__name__.replace('Event', '')
            st.markdown(
                f"""
                <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                    <h4 style="color: #FFFFFF; text-align: right;">זמן: {next_event.time:.2f}</h4>
                    <h4 style="color: #FFFFFF; text-align: right;">סוג: {event_type}</h4>
                    <h4 style="color: #FFFFFF; text-align: right;">מזהה לקוח: {getattr(next_event, 'customer_id', 'N/A')}</h4>
                </div>
                """, 
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                    <p style="color: #FFFFFF; text-align: center;">אין אירועים נוספים בתור</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
        # Station Status Section
        st.markdown("<h3 style='text-align: right;'>מצב התחנות</h3>", unsafe_allow_html=True)
        
        st.markdown(
            f"""
            <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                <h2 style="color: #FFFFFF; text-align: center;">עמדת הזמנות</h2>
                <h1 style="color: #FFFFFF; text-align: center;">{'✅' if sim_state.order_station_busy == 0 else '⛔'}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                <h2 style="color: #FFFFFF; text-align: center;">עמדת הכנה</h2>
                <h1 style="color: #FFFFFF; text-align: center;">{'✅' if sim_state.prep_station_busy == 0 else '⛔'}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div style="border-radius: 8px; padding: 15px; background-color: #420518;">
                <h2 style="color: #FFFFFF; text-align: center;">עמדת איסוף</h2>
                <h1 style="color: #FFFFFF; text-align: center;">{'✅' if sim_state.pickup_station_busy == 0 else '⛔'}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )



    with col3:
        # Queue Length Charts
        st.markdown("<h3 style='text-align: right;'>אורכי תורים</h3>", unsafe_allow_html=True)
        
        # Create DataFrame for queue lengths
        queue_data = pd.DataFrame({
            'זמן': sim_state.time_history,
            'תור הזמנות': sim_state.order_queue_history,
            'תור הכנה': sim_state.prep_queue_history,
            'תור איסוף': sim_state.pickup_queue_history
        })

        # Create step plot using Plotly
        fig = go.Figure()
        
        # Add traces for each queue
        fig.add_trace(go.Scatter(
            x=queue_data['זמן'],
            y=queue_data['תור הזמנות'],
            name='תור הזמנות',
            mode='lines',
            line=dict(shape='hv', width=2)  # 'hv' creates horizontal-then-vertical steps
        ))
        
        fig.add_trace(go.Scatter(
            x=queue_data['זמן'],
            y=queue_data['תור הכנה'],
            name='תור הכנה',
            mode='lines',
            line=dict(shape='hv', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=queue_data['זמן'],
            y=queue_data['תור איסוף'],
            name='תור איסוף',
            mode='lines',
            line=dict(shape='hv', width=2)
        ))

        # Update layout
        fig.update_layout(
            title='אורכי תורים לאורך זמן',
            xaxis_title='זמן',
            yaxis_title='מספר אנשים בתור',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )

        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', zeroline=False)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', zeroline=False)

        # Display plot
        st.plotly_chart(fig, use_container_width=True)

        # Display metrics for current queue lengths
        col11, col22, col33 = st.columns(3)
        with col11:
            st.metric(
                label="תור הזמנות",
                value=len(sim_state.order_queue),
                delta=len(sim_state.order_queue) - sim_state.order_queue_history[-2] if len(sim_state.order_queue_history) > 1 else None
            )
        
        with col22:
            st.metric(
                label="תור הכנה",
                value=len(sim_state.prep_queue),
                delta=len(sim_state.prep_queue) - sim_state.prep_queue_history[-2] if len(sim_state.prep_queue_history) > 1 else None
            )
        
        with col33:
            st.metric(
                label="תור איסוף",
                value=len(sim_state.pickup_queue),
                delta=len(sim_state.pickup_queue) - sim_state.pickup_queue_history[-2] if len(sim_state.pickup_queue_history) > 1 else None
            )

        # Create bar chart for queue distribution
        st.markdown("<h3 style='text-align: right;'>התפלגות הלקוחות בתורים</h3>", unsafe_allow_html=True)
        queue_distribution = pd.DataFrame({
            'תור': ['הזמנות', 'הכנה', 'איסוף'],
            'כמות לקוחות': [
                len(sim_state.order_queue),
                len(sim_state.prep_queue),
                len(sim_state.pickup_queue)
            ]
        })
        st.bar_chart(queue_distribution.set_index('תור'))

    # Control Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("צעד הבא", key="next_step"):
            st.session_state.simulation.process_next_event()
    
    with col2:
        if st.button("אפס סימולציה", key="reset_simulation"):
            st.session_state.simulation = FoodTruckSimulation()

if __name__ == "__main__":
    show_food_truck_simulation()