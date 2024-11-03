import streamlit as st
from pylfsr import LFSR
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

def create_animated_plots():
    """Create empty figure template for animation."""
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
    
    fig.update_layout(
        height=700,
        showlegend=False,
        title_text='ניתוח המספרים המיוצרים',
        title_x=0.5,
        title_xanchor='center',
        font=dict(size=14),
        xaxis=dict(range=[-1, 20]),
        xaxis2=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1])
    )
    
    fig.update_xaxes(title_text='אינדקס', row=1, col=1)
    fig.update_yaxes(title_text='ערך', row=1, col=1)
    fig.update_xaxes(title_text='ערך', row=2, col=1)
    fig.update_yaxes(title_text='תדירות', row=2, col=1)
    
    return fig

def generate_normalized_sequence(lfsr, length):
    """Generate sequence and normalize to [0,1]."""
    sequence = []
    max_val = 2**len(lfsr.initstate) - 1
    
    for _ in range(length):
        # Get current state as an integer
        current_val = int(''.join(map(str, lfsr.state)), 2)
        # Normalize to [0,1]
        sequence.append(current_val / max_val)
        lfsr.next()
        
    return sequence

def show_lfsr():
    """Main function to display LFSR demonstration interface."""
    # Add Hebrew title with RTL support
    st.markdown('<h1 style="text-align: center; direction: ltr;">Linear Feedback Shift Register (LFSR)</h1>', 
                unsafe_allow_html=True)
    

    # Create main container for content
    with st.container():
        # Add Hebrew explanation with RTL support and hyperlink
        st.markdown("""
        <div dir="rtl" style="text-align: right;">
            <p>
            רגיסטר הזזה עם משוב לינארי (LFSR) <a href="https://he.wikipedia.org/wiki/LFSR">למידע נוסף</a>
            הוא רגיסטר הזזה שבו ביט הקלט הוא פונקציה לינארית של שניים או יותר ממצביו הקודמים.</p>
            <p>
            הרגיסטר מייצר רצף של סיביות פסאודו-אקראיות באמצעות הנוסחה:
            </p>
        </div>
        """, unsafe_allow_html=True)

        
        # Display mathematical formula
        st.latex(r"b_{i+n} = c_{n-1}b_{i+n-1} \oplus c_{n-2}b_{i+n-2} \oplus \cdots \oplus c_0b_i")
        
        # Add formula explanation in Hebrew with RTL support and enhanced formatting
        st.markdown("""
        <div dir="rtl" style="text-align: right; font-size: 16px; font-family: 'Arial';">
            <ul style="list-style-type: none; padding-right: 20px;">
                <li><b>b<sub>i</sub></b> הוא הביט הנוכחי,</li>
                <li><b>c<sub>i</sub></b> הם מקדמי המשוב (0 או 1),</li>
                <li><b>⊕</b> מייצג פעולת <b>XOR</b>.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Parameters input with Hebrew labels
        st.markdown('<div dir="rtl" style="text-align: right;">', unsafe_allow_html=True)
        # Create two-column layout for inputs
        col1_lfsr, col2_lfsr = st.columns(2)
        
        with col1_lfsr:
            lfsr_count = st.number_input('מספר הערכים לייצור', min_value=1000, max_value=10000, value=1000, format="%d")
            lfsr_n_bits = st.number_input('מספר סיביות ברגיסטר', 
                                min_value=2, max_value=16, value=8, 
                                key="lfsr_n_bits_lfsr")
                                
            lfsr_seed = st.number_input('ערך התחלתי', 
                                min_value=1, max_value=2**lfsr_n_bits-1, value=15,
                                key="lfsr_seed_lfsr")
                                
        with col2_lfsr:
            available_taps = list(range(1, lfsr_n_bits+1))
            taps = st.multiselect(
                '(בחרו שניים) מיקומי משוב',
                options=available_taps,
                default=[lfsr_n_bits, lfsr_n_bits-2] if lfsr_n_bits >= 3 else [lfsr_n_bits],
                key="taps_lfsr"
            )
            
            lfsr_delay = st.slider('השהיה בין ערכים (שניות)', 
                            min_value=0.0, max_value=1.0, value=0.1, step=0.1,
                            key="lfsr_delay_lfsr")
        
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button('צור מספרים', key='generate_lfsr'):
            if len(taps) < 2:
                st.error('יש לבחור לפחות שני מיקומי משוב')
                return
                
            # Initialize LFSR
            init_state = [int(x) for x in format(lfsr_seed, f'0{lfsr_n_bits}b')]
            lfsr = LFSR(initstate=init_state, fpoly=taps)
            
            # Create plot
            plot_spot = st.empty()
            fig = create_animated_plots()
            
            # Statistics containers
            stats_container = st.container()
            col1_lfsr, col2_lfsr, col3, col4 = stats_container.columns(4)
            mean_spot = col1_lfsr.empty()
            median_spot = col2_lfsr.empty()
            min_spot = col3.empty()
            max_spot = col4.empty()
            print(lfsr_count)
            # Generate and display sequence
            sequence = generate_normalized_sequence(lfsr, lfsr_count)
            current_sequence = []
            
            i = 0
            for value in sequence:
                i+=1
                current_sequence.append(value)
                
                # Update plots
                fig.data[0].x = list(range(len(current_sequence)))
                fig.data[0].y = current_sequence
                fig.data[1].x = current_sequence
                
                plot_spot.plotly_chart(fig, use_container_width=True)

                                # Update x-axis range for trace plot if needed
                if i > 18:  # After first 20 points
                    fig.update_xaxes(range=[i-19, i+1], row=1, col=1)

                # Update statistics
                if current_sequence:
                    mean_spot.metric("ממוצע", f"{np.mean(current_sequence):.4f}")
                    median_spot.metric("חציון", f"{np.median(current_sequence):.4f}")
                    min_spot.metric("מינימום", f"{min(current_sequence):.4f}")
                    max_spot.metric("מקסימום", f"{max(current_sequence):.4f}")
                    
                time.sleep(lfsr_delay)
            
            # Display LFSR properties
            if st.button("בדוק תכונות", key="test_properties"):
                result = lfsr.test_properties(verbose=1)
                
            # Display usage example
            st.markdown("""
                <div dir="rtl" style="text-align: right;">
                    <h3>שימוש בסימולציית טאקו לוקו:</h3>
                    <ul>
                        <li>טאקו לוקוסיטו (50%): מספר בין 0 ל-0.5</li>
                        <li>טאקו לוקוסיצ'ימו (25%): מספר בין 0.5 ל-0.75</li>
                        <li>מתקטאקו (25%): מספר בין 0.75 ל-1</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    show_lfsr()