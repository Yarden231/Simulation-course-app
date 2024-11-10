import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
from utils import set_ltr_sliders

def generate_next_number(lcg_modulus, a, c, lcg_seed):
    """Generate next number in LCG sequence."""
    next_num = (a * lcg_seed + c) % lcg_modulus
    # Normalize to [0,1]
    return next_num / (lcg_modulus - 1)

def create_animated_plots():
    """Create empty figure for animation."""
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=['התפלגות המספרים', 'רצף המספרים לאורך זמן']
    )
    
    # Initialize empty traces
    fig.add_trace(
        go.Scatter(
            x=[],
            y=[],
            mode='lines+markers',
            name='רצף מספרים',
            line=dict(color='#1f77b4')
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Histogram(
            x=[],
            nbinsx=30,
            name='היסטוגרמה',
            marker_color='#1f77b4'
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text='ניתוח המספרים המיוצרים',
        title_x=0.5,
        title_xanchor='center',
        font=dict(size=14),
        xaxis=dict(range=[-1, 20]),  # Initial x-axis range for trace plot
        xaxis2=dict(range=[0, 1]),   # x-axis range for histogram
        yaxis=dict(range=[0, 1])     # y-axis range for trace plot
    )
    
    # Update axes labels
    fig.update_xaxes(title_text='אינדקס', row=1, col=1)
    fig.update_yaxes(title_text='ערך', row=1, col=1)
    fig.update_xaxes(title_text='ערך', row=2, col=1)
    fig.update_yaxes(title_text='תדירות', row=2, col=1)
    
    return fig

def show_lcg():
    """
    Demonstrates the Linear Congruential Generator (LCG) by generating and visualizing 
    a sequence of pseudo-random numbers.

    This function sets up a Streamlit interface to allow users to input parameters 
    for the LCG: modulus, multiplier, increment, seed, number of values to generate, 
    and delay between numbers. It then animates the generation of these numbers, 
    updating a plot and displaying statistics such as mean, median, minimum, and maximum 
    in real-time.

    The page includes an explanation of the LCG algorithm and its equation, 
    with inputs and explanations provided in Hebrew.
    """
    set_ltr_sliders()

    st.markdown('<h2 style="text-align: right; direction: rtl;">Linear Congruential Generator (LCG)</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div dir="rtl" style="text-align: right;">
            <p>
            מחולל המספרים הפסאודו-אקראיים מסוג Linear Congruential Generator (LCG) ידוע בפשטותו ובמהירותו, 
            מה שהביא לשימוש נרחב שלו במערכות מחשב. 
            המחולל יוצר סדרה של מספרים פסאודו-אקראיים באמצעות משוואה לינארית כדלקמן:
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"X_{n+1} = (aX_n + c) \bmod m")
        


        st.markdown("""
        <div dir="rtl" style="text-align: right;">
            <ul style="list-style-type: none; padding-right: 20px;">
                <li>• X<sub>n</sub> הוא המספר הפסאודו-אקראי הנוכחי</li>
                <li>• a (המכפיל) ו-c (התוספת) הם קבועים</li>
                <li>• m (המודולו) הוא הגבול העליון של המספרים המיוצרים (לא כולל)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Parameters input with Hebrew labels
        st.markdown('<div dir="rtl" style="text-align: right;">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            lcg_modulus = st.number_input('מודולו (m)', min_value=2**16, max_value=2**32, value=2**16, format="%d")
            a = st.number_input('מכפיל (a)', min_value=1, max_value=2**31, value=1597, format="%d")
            c = st.number_input('תוספת (c)', min_value=0, max_value=lcg_modulus-1, value=51749, format="%d")
        
        with col2:
            lcg_seed = st.number_input('ערך התחלתי', value=42, min_value=0, max_value=lcg_modulus-1, format="%d")
            lcg_count = st.number_input('מספר הערכים לייצור', min_value=1000, max_value=10000, value=1000, step=100 ,format="%d")
            lcg_delay = st.slider('השהיה בין מספרים (שניות)', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button('צור מספרים'):
            # Initialize plots
            plot_spot = st.empty()
            fig = create_animated_plots()
            
            # Initialize statistics spots
            stats_container = st.container()
            col1, col2, col3, col4 = stats_container.columns(4)
            mean_spot = col1.empty()
            median_spot = col2.empty()
            min_spot = col3.empty()
            max_spot = col4.empty()
            
            # Generate and display numbers one by one
            numbers = []
            current = lcg_seed
            
            for i in range(lcg_count):
                # Generate next number
                new_number = generate_next_number(lcg_modulus, a, c, current)
                numbers.append(new_number)
                current = int(new_number * (lcg_modulus - 1))
                
                # Update trace plot data
                fig.data[0].x = list(range(len(numbers)))
                fig.data[0].y = numbers
                
                # Update histogram data
                fig.data[1].x = numbers
                
                # Update x-axis range for trace plot if needed
                if i > 18:  # After first 20 points
                    fig.update_xaxes(range=[i-19, i+1], row=1, col=1)
                
                # Update plots
                plot_spot.plotly_chart(fig, use_container_width=True)
                
                # Update statistics
                if numbers:
                    mean_spot.metric("ממוצע", f"{np.mean(numbers):.4f}")
                    median_spot.metric("חציון", f"{np.median(numbers):.4f}")
                    min_spot.metric("מינימום", f"{min(numbers):.4f}")
                    max_spot.metric("מקסימום", f"{max(numbers):.4f}")
                
                time.sleep(lcg_delay)

if __name__ == "__main__":
    show_lcg()
