import streamlit as st


def create_styled_card(title, content, border_color="#453232"):
    st.markdown(
        f"""
        <div style="
            background-color: #2D2D2D;
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        ">
            <h3 style="
                color: #FFFFFF;
                margin-bottom: 15px;
                text-align: right;
                font-size: 1.2rem;
            ">{title}</h3>
            <div style="
                color: #FFFFFF;
                text-align: right;
            ">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def create_station_grid():
    stations = [
        ("👥", "הזמנה"),
        ("👨‍🍳", "הכנה"),
        ("📦", "אריזה")
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

def show_menu():
    # Title with custom styling
    st.markdown("<h1 style='text-align: right; color: white; font-size: 24px; margin-bottom: 30px;'>🍽️ תפריט ״לוקו טאקו״</h1>", unsafe_allow_html=True)

    # Menu items data
    menu_items = [
        {
            'emoji': '🌮',
            'name': 'טאקו לוקוסיטו',
            'prep_time': '3-4 דקות',
            'percentage': '50% מההזמנות'
        },
        {
            'emoji': '🌯',
            'name': 'טאקו לוקוסיצ׳ימו',
            'prep_time': '4-6 דקות',
            'percentage': '25% מההזמנות'
        },
        {
            'emoji': '🥙',
            'name': 'מתקטאקו',
            'prep_time': '10 דקות',
            'percentage': '25% מההזמנות',
            'warning': 'תלונות על בישול חסר ב-30% מהמקרים'
        }
    ]

    # Loop through menu items
    for item in menu_items:
        with st.container():
            col1, col2 = st.columns([0.15, 0.85])
            
            # Emoji column
            with col1:
                st.markdown(f"<div style='font-size: 40px; text-align: center;'>{item['emoji']}</div>", 
                          unsafe_allow_html=True)
            
            # Details column
            with col2:
                st.markdown(
                    f"""
                    <div style='background-color: #2D2D2D; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                        <div style='color: white; font-size: 10px; font-weight: bold; margin-bottom: 10px; text-align: right;'>
                            {item['name']}
                        </div>
                        <div style='color: #CCCCCC; text-align: right;'>
                            זמן הכנה: {item['prep_time']}
                        </div>
                        <div style='color: #CCCCCC; text-align: right;'>
                            {item['percentage']}
                        </div>
                        {f"<div style='color: #FF4444; text-align: right; margin-top: 10px;'>⚠️ {item['warning']}</div>" if 'warning' in item else ''}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

            # Add spacing between items
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

# Update the create_styled_card function with better spacing
def create_styled_card(title, content, border_color="#453232"):
    st.markdown(
        f"""
        <div style="
            background-color: #1E1E1E;
            border: 1px solid {border_color};
            border-radius: 8px;
            padding: 10px;
            margin: 25px 0;  /* Increased margin */
        ">
            <h3 style="
                color: #FFFFFF;
                margin-bottom: 10px;  /* Increased margin */
                text-align: right;
                font-size: 1.2rem;
            ">{title}</h3>
            <div style="
                color: #FFFFFF;
                text-align: right;
                line-height: 1.6;  /* Added line height */
            ">{content}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Update the show_story function with proper image handling and spacing
def show_story():
    # Apply custom CSS
    with open('.streamlit/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    # Additional custom CSS for spacing
    st.markdown("""
        <style>
        .main > div {
            direction: rtl;
        }
        .stMarkdown {
            color: #FFFFFF !important;
            margin-bottom: 10px;  /* Added spacing */
        }
        h1, h2 {
            margin-top: 40px !important;  /* Added top spacing for headers */
            margin-bottom: 30px !important;  /* Added bottom spacing for headers */
        }
        p {
            line-height: 1.6 !important;  /* Improved line height for paragraphs */
            margin-bottom: 15px !important;  /* Added spacing between paragraphs */
        }
        </style>
    """, unsafe_allow_html=True)

    # Header with more spacing
    st.markdown("""
        <h1 style='
            text-align: center; 
            color: #FFFFFF; 
            margin-bottom: 3rem;
            margin-top: 2rem;
        '>פרויקט הסימולציה של אוצ׳ו לוקו</h1>
    """, unsafe_allow_html=True)

    # Introduction with improved spacing
    create_styled_card(
        "הקדמה",
        """
        <p style='margin-bottom: 10px;'>אוצ׳ו לוקו החל את פרויקט הסימולציה של חייו במשאית המזון המשפחתית האהובה. 
        </p>
        
        <p style='margin-bottom: 10px;'>אוצ׳ו מתכנן לחקור ולהבין מהם קצבי השירות בכל עמדה ומהם קצבי הגעת הלקוחות למשאית המזון. בעזרת הקצבים שיאסוף, יוכל דרך פרויקט סימולציה קפדני שיבצע אוצ׳ו, להבין מהם צווארי הבקבוק במערכת שלו וכך ידע לנהל את עובדיו בצורה האופטימלית.</p>
        
        <p>בכדי להבין את קצבי השירות ב-״לוקו טאקו״, אוצ׳ו תשאל את אביו חולייסיטו, וביקש שיספר לו על התנהלות הלקוחות והעובדים במשאיתו האהובה.</p>
        """
    )

    # Questioning summary header
    st.markdown("""
        <h2 style='
            color: #FFFFFF; 
            text-align: right;
            margin-top: 40px;
            margin-bottom: 30px;
        '>להלן סיכום התשאול:</h2>
    """, unsafe_allow_html=True)

    # Correct way to display SVG in Streamlit
    try:
        with open("story.svg", "r") as file:
            svg_content = file.read()
            st.markdown(f"""
                <div style="
                    display: flex;
                    justify-content: center;
                    margin: 30px 0;
                    background-color: #2D2D2D;
                    padding: 10px;
                    border-radius: 8px;
                ">
                    {svg_content}
                </div>
            """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("קובץ ה-SVG לא נמצא. יש לוודא שהקובץ נמצא בתיקייה הנכונה.")

    # Rest of your code remains the same, but add spacing between sections
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)

    # Summary header with spacing
    st.markdown("""
        <h2 style='
            color: #FFFFFF; 
            text-align: right;
            margin-top: 40px;
            margin-bottom: 30px;
        '>אוצ'ו לוקו סיכם עבורכם את הפרטים באופן מסודר:</h2>
    """, unsafe_allow_html=True)

    # Service Stations with spacing
    st.markdown("""
        <h2 style='
            color: #FFFFFF; 
            text-align: right;
            margin-top: 40px;
            margin-bottom: 30px;
        '>עמדות השירות</h2>
    """, unsafe_allow_html=True)
    
    create_station_grid()
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)

    # Operating Hours with spacing
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
    create_styled_card(
        "⏰ שעות פעילות",
        """
        <div style='margin-bottom: 15px;'>המשאית פועלת בין השעות 12:00-17:00</div>
        <div>ממוצע של 10 לקוחות בשעה</div>
        """
    )

    # Menu section with spacing
    st.markdown("<div style='margin: 30px 0;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <h2 style='
            color: #FFFFFF; 
            text-align: right;
            margin-top: 40px;
            margin-bottom: 30px;
        '>🍽️ תפריט ״לוקו טאקו״</h2>
    """, unsafe_allow_html=True)
    
    # Menu items
    menu_items = [
        {
            'emoji': '🌮',
            'name': 'טאקו לוקוסיטו',
            'prep_time': '4-6 דקות',
            'percentage': '50% מההזמנות'
        },
        {
            'emoji': '🌯',
            'name': 'טאקו לוקוסיצ׳ימו',
            'prep_time': '10 דקות',
            'percentage': '25% מההזמנות'
        },
        {
            'emoji': '🥙',
            'name': 'מתקטאקו',
            'prep_time': 'משתנה',
            'percentage': '25% מההזמנות',
            'warning': 'תלונות על בישול חסר ב-30% מהמקרים'
        }
    ]

    # Loop through menu items with spacing
    for item in menu_items:
        with st.container():
            col1, col2 = st.columns([0.15, 0.85])
            
            # Emoji column
            with col1:
                st.markdown(f"""
                    <div style='
                        font-size: 40px; 
                        text-align: center;
                        margin-top: 10px;
                    '>{item['emoji']}</div>
                """, unsafe_allow_html=True)
            
            # Details column
            with col2:
                st.markdown(
                    f"""
                    <div style='
                        background-color: #2D2D2D; 
                        padding: 10px; 
                        border-radius: 8px; 
                        margin-bottom: 10px;
                        border: 1px solid #453232;
                    '>
                        <div style='
                            color: white; 
                            font-size: 10px; 
                            font-weight: bold; 
                            margin-bottom: 15px; 
                            text-align: right;
                        '>
                            {item['name']}
                        </div>
                        <div style='
                            color: #CCCCCC; 
                            text-align: right;
                            margin-bottom: 10px;
                        '>
                            זמן הכנה: {item['prep_time']}
                        </div>
                        <div style='
                            color: #CCCCCC; 
                            text-align: right;
                            margin-bottom: 10px;
                        '>
                            {item['percentage']}
                        </div>
                        {f"<div style='color: #FF4444; text-align: right; margin-top: 15px;'>⚠️ {item['warning']}</div>" if 'warning' in item else ''}
                    </div>
                    """, 
                    unsafe_allow_html=True
                )

    # Customer Patience with spacing
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    create_styled_card(
        "⏱️ סבלנות לקוחות",
        """
        <div style='margin-bottom: 15px;'>לקוחות מוכנים להמתין בין 5 ל-20 דקות לפני עזיבה</div>
        <div style='color: #CCCCCC; font-size: 0.9rem;'>התשלום מתבצע רק בעת איסוף ההזמנה</div>
        """
    )

    # Service Process with spacing
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    create_styled_card(
        "🔄 תהליך השירות",
        """
        <ol style='
            margin: 0; 
            padding-right: 10px;
            line-height: 1.6;
        '>
            <li style='margin-bottom: 10px;'>כל לקוח מתחיל בדלפק ההזמנות</li>
            <li style='margin-bottom: 10px;'>ההזמנה עוברת לעמדת הבישול</li>
            <li style='margin-bottom: 10px;'>לאחר הבישול, המנה עוברת לאריזה</li>
            <li>הלקוח מקבל את הזמנתו בעמדת האיסוף</li>
        </ol>
        """
    )

    # Management Challenge with spacing
    st.markdown("<div style='margin: 40px 0;'></div>", unsafe_allow_html=True)
    create_styled_card(
        "🎯 אתגר ניהולי",
        """
        <div style='margin-bottom: 15px;'>המנהל שוקל להוסיף עובד נוסף ומעוניין להבין לאיזו עמדה כדאי להוסיף אותו, אם בכלל.</div>
        <div style='color: #CCCCCC;'>החלטה זו תתבסס על תוצאות הסימולציה.</div>
        """
    )

    # Final spacing at the bottom of the page
    st.markdown("<div style='margin: 50px 0;'></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    show_story()