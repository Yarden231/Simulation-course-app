import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from food_truck import run_simulation

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from utils import set_rtl

# Call the set_rtl function to apply RTL styles
set_rtl()
def create_queue_animation(food_truck):
    df = pd.DataFrame({
        'Time': range(len(food_truck.queue_sizes['order'])),
        'Order Queue': food_truck.queue_sizes['order'],
        'Prep Queue': food_truck.queue_sizes['prep'],
        'Pickup Queue': food_truck.queue_sizes['pickup'],
        'Total Queue': food_truck.queue_sizes['total']
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Order Queue'], mode='lines', name='Order Queue'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Prep Queue'], mode='lines', name='Prep Queue'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Pickup Queue'], mode='lines', name='Pickup Queue'))
    fig.add_trace(go.Scatter(x=df['Time'], y=df['Total Queue'], mode='lines', name='Total Queue', line=dict(color='black', width=4)))

    fig.update_layout(title="Queue Sizes Over Time", xaxis_title="Time", yaxis_title="Queue Size", legend_title="Queue Type")
    return fig, df  # Return the DataFrame for further use

def create_real_time_animation(df, speed=1.0):
    # Calculate the maximum queue size for setting the Y-axis range
    max_queue_size = df[['Order Queue', 'Prep Queue', 'Pickup Queue', 'Total Queue']].max().max()

    # Create the frames for the animation
    frames = []
    for i in range(len(df)):
        frames.append(go.Frame(
            data=[
                go.Bar(x=['Order Queue', 'Prep Queue', 'Pickup Queue', 'Total Queue'], 
                       y=[df['Order Queue'].iloc[i], df['Prep Queue'].iloc[i], df['Pickup Queue'].iloc[i], df['Total Queue'].iloc[i]],
                       marker=dict(color=['blue', 'green', 'red', 'black']))
            ],
            name=str(i),
            layout=go.Layout(
                annotations=[
                    go.layout.Annotation(
                        text=f"Simulation Time: {df['Time'].iloc[i]}",
                        x=0.5, y=1.1, xref="paper", yref="paper",
                        showarrow=False, font=dict(size=16)
                    )
                ]
            )
        ))

    # Create the figure and add the frames
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title="Real-time Simulation of Food Truck Operations",
            xaxis_title="Queue Type",
            yaxis_title="Number of People",
            yaxis=dict(range=[0, max_queue_size]),  # Set Y-axis range
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    {"label": "Play", 
                     "method": "animate", 
                     "args": [None, {"frame": {"duration": int(1000 / speed), "redraw": True}, 
                                    "fromcurrent": True, "transition": {"duration": int(300 / speed)}}]},
                    {"label": "Pause", 
                     "method": "animate", 
                     "args": [[None], {"frame": {"duration": 0, "redraw": False}, 
                                      "mode": "immediate", "transition": {"duration": 0}}]},
                    {"label": "Reset", 
                     "method": "animate", 
                     "args": [[None], {"frame": {"duration": 0, "redraw": True}, 
                                      "mode": "immediate", "transition": {"duration": 0}}]}
                ]
            )]
        ),
        frames=frames
    )

    return fig


def show_food_truck():
    st.title("סימולציית משאית מזון")

    st.header("הגדרות סימולציה")
    sim_time = st.slider("זמן סימולציה (דקות)", 1000, 10000, 5000)
    arrival_rate = st.slider("זמן ממוצע בין הגעות לקוחות (דקות)", 5, 20, 10)
    order_time_min = st.slider("זמן הזמנה מינימלי (דקות)", 1, 5, 3)
    order_time_max = st.slider("זמן הזמנה מקסימלי (דקות)", 5, 10, 7)
    leave_probability = st.slider("הסתברות לעזיבה לפני הזמנה", 0.0, 0.5, 0.1)
    
    config = {
        'order_capacity': st.slider("כמות עמדות בהזמנה", 1, 5, 1),
        'prep_capacity': st.slider("כמות עמדות בהכנה", 1, 5, 1),
        'pickup_capacity': st.slider("כמות עמדות באיסוף", 1, 5, 1)
    }

    if st.button("הפעל סימולציה"):
        with st.spinner("מריץ סימולציה..."):
            logger = EventLogger()
            food_truck = run_simulation(sim_time, arrival_rate, order_time_min, order_time_max, leave_probability, config, logger)
            if food_truck:
                st.success("הסימולציה הושלמה!")
                st.session_state.food_truck = food_truck  # שמירת התוצאות ב-session state
            else:
                st.error("הייתה בעיה בהרצת הסימולציה.")
        
        st.subheader("תוצאות הסימולציה זמינות. ניתן לשנות מהירות הצגה למטה.")
    
    if 'food_truck' in st.session_state:
        food_truck = st.session_state.food_truck
        
        st.header("הגדרות הצגת האנימציה")
        speed = st.slider("מהירות הצגת האנימציה (פי)", 0.1, 10.0, 1.0)
        
        st.subheader("גודל התור לאורך הזמן")
        queue_animation, queue_data = create_queue_animation(food_truck)
        st.plotly_chart(queue_animation, use_container_width=True)
        
        st.subheader("אנימציה בזמן אמת של התורים בסימולציה")
        real_time_animation = create_real_time_animation(queue_data, speed)
        st.plotly_chart(real_time_animation, use_container_width=True)
    
    st.write("""
    #### חקרו את הסימולציה
    נסו וראו כיצד משתנים שונים משפיעים על ביצועי משאית המזון. התאימו את ההגדרות וההרצה לסימולציות שונות!
    """)

if __name__ == "__main__":
    show_food_truck()



