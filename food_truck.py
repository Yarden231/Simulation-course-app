import simpy
import numpy as np
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils import set_rtl  # RTL setting function
import time  # Import the time module
from utils import set_ltr_sliders
from PIL import Image
# Call the set_rtl function to apply RTL styles
set_rtl()

class FoodTruck:
    def __init__(self, env, order_time_min, order_time_max, config):
        self.env = env
        self.event_log = []
        self.order_station = simpy.Resource(env, capacity=config['order_capacity'])
        self.prep_station = simpy.Resource(env, capacity=config['prep_capacity'])
        self.pickup_station = simpy.Resource(env, capacity=config['pickup_capacity'])
        self.batch = []
        self.left_count = 0
        self.left_before_ordering = 0
        self.total_visitors = 0
        self.undercooked_count = 0
        self.wait_times = {'order': [], 'prep': [], 'pickup': [], 'total': []}
        self.queue_sizes = {'order': [], 'prep': [], 'pickup': [], 'total': []}
        self.left_over_time = []
        self.left_before_ordering_over_time = []
        self.undercooked_over_time = []
        self.total_visitors_over_time = []
        self.order_time_min = order_time_min
        self.order_time_max = order_time_max

    def log_event(self, customer_id, event_type, time):
        self.event_log.append({'customer_id': customer_id, 'event': event_type, 'time': time})


    def process_service(self, station, visitor, service_time_range):
        with station.request() as req:
            yield req
            start_time = self.env.now
            service_time = np.random.uniform(*service_time_range)
            yield self.env.timeout(service_time)
        end_time = self.env.now
        return start_time, end_time

    def order_service(self, visitor):
        arrival_time = self.env.now
        self.log_event(visitor['name'], 'arrival', arrival_time)
        start_time, end_time = yield from self.process_service(self.order_station, visitor, (self.order_time_min, self.order_time_max))
        self.wait_times['order'].append(end_time - arrival_time)
        self.log_event(visitor['name'], 'order_complete', end_time)
        return end_time - start_time

    def prep_service(self, visitors):
        prep_start = self.env.now
        with self.prep_station.request() as req:
            yield req
            for visitor in visitors:
                self.log_event(visitor['name'], 'preparing', self.env.now)
            service_time = np.random.normal(5, 1)
            yield self.env.timeout(service_time)
        prep_end = self.env.now
        undercooked = np.random.rand(len(visitors)) < 0.3
        self.undercooked_count += sum(undercooked)
        for i, visitor in enumerate(visitors):
            visitor['prep_time'] = prep_end - prep_start
            self.wait_times['prep'].append(visitor['prep_time'])
            self.log_event(visitor['name'], 'prep_complete', prep_end)
            if undercooked[i]:
                self.log_event(visitor['name'], 'undercooked', prep_end)
        for visitor in visitors:
            self.env.process(self.pickup_service(visitor))

    def pickup_service(self, visitor):
        pickup_start = self.env.now
        start_time, end_time = yield from self.process_service(self.pickup_station, visitor, (2, 4))
        self.log_event(visitor['name'], 'exit', end_time)
        pickup_time = end_time - pickup_start
        self.wait_times['pickup'].append(pickup_time)
        total_time = end_time - visitor['arrival_time']
        self.wait_times['total'].append(total_time)

    def process_batch(self):
        while self.batch:
            batch_size = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
            visitors_to_process = self.batch[:batch_size]
            del self.batch[:batch_size]
            yield self.env.process(self.prep_service(visitors_to_process))

    def monitor(self):
        while True:
            self.queue_sizes['order'].append(len(self.order_station.queue))
            self.queue_sizes['prep'].append(len(self.prep_station.queue))
            self.queue_sizes['pickup'].append(len(self.pickup_station.queue))
            total_queue_size = len(self.order_station.queue) + len(self.prep_station.queue) + len(self.pickup_station.queue)
            self.queue_sizes['total'].append(total_queue_size)
            self.left_over_time.append(self.left_count)
            self.left_before_ordering_over_time.append(self.left_before_ordering)
            self.undercooked_over_time.append(self.undercooked_count)
            self.total_visitors_over_time.append(self.total_visitors)
            yield self.env.timeout(1)

# Simulation function with real-time updates
def run_simulation(sim_time, arrival_rate, order_time_min, order_time_max, leave_probability, config):
    # Create a SimPy environment and FoodTruck object
    env = simpy.Environment()
    food_truck = FoodTruck(env, order_time_min, order_time_max, config)
    
    # Start the customer arrival and monitoring processes
    env.process(arrival_process(env, food_truck, arrival_rate, leave_probability))
    env.process(food_truck.monitor())
    
    # Run the simulation for the given time
    env.run(until=sim_time)
    
    return food_truck  # Return the completed FoodTruck object after the simulation ends


# Customer arrival process
def arrival_process(env, food_truck, arrival_rate, leave_probability):
    visitor_count = 0
    while True:
        yield env.timeout(np.random.exponential(arrival_rate))
        visitor_count += 1
        env.process(visitor(env, visitor_count, food_truck, leave_probability))

# Visitor service process
def visitor(env, name, food_truck, leave_probability):
    food_truck.total_visitors += 1
    arrival_time = env.now
    if np.random.random() < leave_probability:
        food_truck.left_before_ordering += 1
        food_truck.left_count += 1
        food_truck.log_event(name, 'left_before_ordering', arrival_time)
        return
    order_time = yield env.process(food_truck.order_service({'name': name, 'arrival_time': arrival_time}))
    food_truck.batch.append({'name': name, 'arrival_time': arrival_time, 'order_time': order_time})
    food_truck.log_event(name, 'ordered', arrival_time)
    if len(food_truck.batch) >= 1:
        env.process(food_truck.process_batch())

# Plot real-time queue animation
def plot_real_time_queues(food_truck, step):
    df = pd.DataFrame({
        'Time': range(len(food_truck.queue_sizes['order'])),
        'Order Queue': food_truck.queue_sizes['order'],
        'Prep Queue': food_truck.queue_sizes['prep'],
        'Pickup Queue': food_truck.queue_sizes['pickup'],
        'Total Queue': food_truck.queue_sizes['total']
    })

    # Get current queue sizes for the title
    current_order_queue = df['Order Queue'].iloc[step]
    current_prep_queue = df['Prep Queue'].iloc[step]
    current_pickup_queue = df['Pickup Queue'].iloc[step]
    current_total_queue = df['Total Queue'].iloc[step]

    fig = go.Figure(data=[
        go.Bar(x=['Order Queue', 'Prep Queue', 'Pickup Queue', 'Total Queue'], 
               y=[current_order_queue, current_prep_queue, current_pickup_queue, current_total_queue],
               marker=dict(color=['blue', 'green', 'red', 'black']))
    ])

    # Update the layout with dynamic title including current queue sizes
    fig.update_layout(
        title=f"Queue Status at Step {step}: \n Order Queue={current_order_queue}, \n Prep Queue={current_prep_queue}, \n Pickup Queue={current_pickup_queue}, \n Total Queue={current_total_queue}",
        xaxis_title="Queue Type",
        yaxis_title="Queue Size",
        yaxis=dict(range=[0, max(df['Total Queue'])])  # Set y-axis limit based on total queue sizes
    )
    return fig

# Plot queue sizes over time after the simulation
def plot_queue_sizes_over_time(food_truck):
    df = pd.DataFrame({
        'Time': range(len(food_truck.queue_sizes['order'])),
        'Order Queue': food_truck.queue_sizes['order'],
        'Prep Queue': food_truck.queue_sizes['prep'],
        'Pickup Queue': food_truck.queue_sizes['pickup'],
        'Total Queue': food_truck.queue_sizes['total']
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Order Queue'], mode='lines', name='Order Queue', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Prep Queue'], mode='lines', name='Prep Queue', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Pickup Queue'], mode='lines', name='Pickup Queue', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Total Queue'], mode='lines', name='Total Queue', line=dict(color='black', width=4)))

    fig.update_layout(
        title="Queue Sizes Over Time",
        xaxis_title="Time",
        yaxis_title="Queue Size",
        legend_title="Queue Type"
    )
    return fig


# Main Streamlit app
def show_food_truck():
    set_ltr_sliders()  # Inject the CSS to ensure LTR behavior for the sliders

    # Load custom CSS
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    set_ltr_sliders()

    # Main header with custom styling
    st.markdown("""
        <div class="main-header rtl-content">
            <h1>🚚 סימולציית משאית המזון</h1>
            <p class="subtitle">ניתוח וסימולציה של תהליכי שירות באמצעות תכנות מבוסס אירועים</p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs for different sections
    tabs = st.tabs(["מבוא", "הגדרות סימולציה", "תוצאות"])

    # Introduction Tab
    with tabs[0]:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
                <div class="info-card rtl-content">
                    <h3>מהי סימולציה מבוססת אירועים? 🎯</h3>
                    <p>
                        תכנות מבוסס אירועים היא שיטה המאפשרת לדמות מערכות מורכבות באמצעות רצף של אירועים 
                        המתרחשים לאורך זמן. במקרה של משאית המזון, אנו מדמים:
                    </p>
                    <ul>
                        <li>הגעת לקוחות בזמנים אקראיים</li>
                        <li>תהליכי הזמנה והכנת מזון</li>
                        <li>ניהול תורים ומשאבים</li>
                        <li>איסוף הזמנות והתנהגות לקוחות</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Load and display event flow diagram
            try:
                image = Image.open("event_flow_diagram.png")
                st.image(image, caption="תרשים זרימת אירועים", use_column_width=True)
            except:
                st.warning("לא נמצא תרשים זרימה")

        # Process description
        st.markdown("""
            <div class="process-card rtl-content">
                <h3>תהליך העבודה במשאית 🔄</h3>
                <div class="process-grid">
                    <div class="process-item">
                        <h4>1. הגעת לקוחות</h4>
                        <p>לקוחות מגיעים בהתפלגות פואסונית</p>
                    </div>
                    <div class="process-item">
                        <h4>2. הזמנה</h4>
                        <p>ביצוע הזמנה בעמדת השירות</p>
                    </div>
                    <div class="process-item">
                        <h4>3. הכנה</h4>
                        <p>הכנת המנה במטבח</p>
                    </div>
                    <div class="process-item">
                        <h4>4. איסוף</h4>
                        <p>איסוף ההזמנה המוכנה</p>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Simulation Settings Tab
    with tabs[1]:
        st.markdown("""
            <div class="settings-header rtl-content">
                <h2>הגדרות הסימולציה ⚙️</h2>
                <p>התאם את הפרמטרים לפי הצרכים שלך</p>
            </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
                <div class="settings-section rtl-content">
                    <h3>פרמטרי זמן</h3>
                </div>
            """, unsafe_allow_html=True)
            
            sim_time = st.slider("⏱️ זמן סימולציה (דקות)", 100, 10000, 100)
            arrival_rate = st.slider("⌛ זמן ממוצע בין הגעות (דקות)", 5, 20, 1)
            order_time_min = st.slider("📝 זמן הזמנה מינימלי (דקות)", 1, 5, 1)
            order_time_max = st.slider("📝 זמן הזמנה מקסימלי (דקות)", 5, 10, 1)

        with col2:
            st.markdown("""
                <div class="settings-section rtl-content">
                    <h3>קיבולת עמדות</h3>
                </div>
            """, unsafe_allow_html=True)
            
            config = {
                'order_capacity': st.slider("🛎️ עמדות הזמנה", 1, 5, 1),
                'prep_capacity': st.slider("👨‍🍳 עמדות הכנה", 1, 5, 1),
                'pickup_capacity': st.slider("📦 עמדות איסוף", 1, 5, 1)
            }
            
            leave_probability = st.slider("🚶‍♂️ הסתברות לעזיבה", 0.0, 0.5, 0.1)

        # Simulation control buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 הפעל סימולציה", use_container_width=True):
                with st.spinner("מריץ סימולציה..."):
                    # Your simulation logic here
                    food_truck = run_simulation(sim_time, arrival_rate, 
                                             order_time_min, order_time_max, 
                                             leave_probability, config)
                    
                    # Real-time visualization placeholder
                    chart_placeholder = st.empty()
                    
                    # Update visualization in real-time
                    for step in range(len(food_truck.queue_sizes['order'])):
                        chart = plot_real_time_queues(food_truck, step)
                        chart_placeholder.plotly_chart(chart, use_container_width=True)
                        time.sleep(0.1)
                    
                    st.success("✅ הסימולציה הושלמה בהצלחה!")
    # Results Tab
    with tabs[2]:
        st.markdown("""
            <div class="results-header rtl-content">
                <h2>תוצאות הסימולציה 📊</h2>
                <p>ניתוח מדדי ביצוע וגרפים</p>
            </div>
        """, unsafe_allow_html=True)  # Fixed parameter name

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="זמן המתנה ממוצע", value="12.5 דקות", delta="-2.1 דקות")
        with col2:
            st.metric(label="אחוז תפוסה", value="85%", delta="5%")
        with col3:
            st.metric(label="לקוחות שעזבו", value="15%", delta="-3%")
        with col4:
            st.metric(label="יעילות המערכת", value="92%", delta="7%")

        # Charts
        st.markdown("<br>", unsafe_allow_html=True)  # Fixed parameter name
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_queue_size_chart(), use_container_width=True)
        with col2:
            st.plotly_chart(create_utilization_chart(), use_container_width=True)

def create_queue_size_chart():
    """Create a sample queue size chart"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[4, 3, 2, 4], name='תור הזמנות'))
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[2, 4, 5, 3], name='תור הכנה'))
    fig.update_layout(
        title='גודל תורים לאורך זמן',
        title_x=0.5,
        yaxis_title='גודל התור',
        xaxis_title='זמן (דקות)',
        font=dict(size=14)
    )
    return fig

def create_utilization_chart():
    """Create a sample utilization chart"""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['הזמנות', 'הכנה', 'איסוף'],
        y=[75, 85, 65],
        marker_color=['#FF9999', '#66B2FF', '#99FF99']
    ))
    fig.update_layout(
        title='ניצולת עמדות',
        title_x=0.5,
        yaxis_title='אחוז ניצולת',
        yaxis_range=[0, 100],
        font=dict(size=14)
    )
    return fig

# Add to your CSS file - now with corrected markdown calls
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        color: #FFFFFF;
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        color: #CCCCCC;
        font-size: 1.2rem;
    }
    
    .info-card, .process-card, .settings-section {
        background-color: #1E1E1E;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #3D3D3D;
    }
    
    .process-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .process-item {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    
    .process-item h4 {
        color: #FFFFFF;
        margin-bottom: 0.5rem;
    }
    
    .process-item p {
        color: #CCCCCC;
        margin: 0;
    }
    
    .settings-header, .results-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Improve slider appearance */
    .stSlider {
        padding: 1rem 0;
    }
    
    /* Style metrics */
    .stMetric {
        background-color: #2D2D2D;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #3D3D3D;
    }

    /* RTL support for specific elements */
    .rtl-content {
        direction: rtl;
        text-align: right;
    }

    /* Improve tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #2D2D2D;
        border-radius: 8px;
    }

    /* Improve button styling */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #45a049 0%, #4CAF50 100%);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)  # Fixed parameter name

if __name__ == "__main__":
    show_food_truck()