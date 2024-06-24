import streamlit as st
from streamlit_gsheets import GSheetsConnection
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
from streamlit_folium import st_folium
import plotly.express as px
import math
from numerize import numerize
from PIL import Image

st.set_page_config(
    page_title="CityVista: Explore, Compare, Relocate'",
    layout="wide",
    initial_sidebar_state="expanded")

def roundup(x):
    return math.ceil(x / 10.0) * 10
conn = st.connection("gsheets", type=GSheetsConnection)

# Load the CSV data
@st.cache_data(ttl=43200)
def load_data():
    df = conn.read(usecols=list(range(1, 16)),
        nrows=195,)
    data = df
    data['CityState'] =  data['CITY'].map(str) + ', ' + data['STATE'].map(str) 
    return data

def main():
    data = load_data()
    col_data = data.copy(deep=True)
    
    # Set up the sidebar
    st.sidebar.title('Filters')
    
    # Get unique states from the data
    states = data['STATE'].unique().tolist()
    states.sort(key=str.lower)
    unique_states = ['All'] + states
    
    selected_state = st.sidebar.selectbox('Select State:', unique_states)
    
    if selected_state != 'All':
        data = data[data['STATE'] == selected_state]
    
    score_options = ['TRANSIT_SCORE', 'WALK_SCORE', 'BIKE_SCORE', 'AVG_CITY_SCORE']
    selected_score = st.sidebar.selectbox('Choose a score:', score_options)
    if len(data.index) == 1:
        selected_score_range = st.sidebar.slider(f'{selected_score} Range', min_value=data[selected_score].min(), max_value=data[selected_score].max()+1, value=(data[selected_score].min(), data[selected_score].max()+1),disabled=True)
        median_rent_range = st.sidebar.slider('Median Rent Range', min_value=data['MEDIAN_RENT'].min(), max_value=data['MEDIAN_RENT'].max()+1, value=(data['MEDIAN_RENT'].min(), data['MEDIAN_RENT'].max()+1),disabled=True)
    else:
        selected_score_range = st.sidebar.slider(f'{selected_score} Range', min_value=data[selected_score].min(), max_value=data[selected_score].max(), value=(data[selected_score].min(), data[selected_score].max()))
        median_rent_range = st.sidebar.slider('Median Rent Range', min_value=data['MEDIAN_RENT'].min(), max_value=data['MEDIAN_RENT'].max(), value=(data['MEDIAN_RENT'].min(), data['MEDIAN_RENT'].max()))
    
    show_col_calc = st.sidebar.toggle("Show Cost of living calculator", value=False)

    st.sidebar.divider()

    show_basic_interactions = st.sidebar.toggle("Show Basic Interactions", value=False)

    if show_basic_interactions:
        st.sidebar.markdown("""
        ### Basic interactions
        - **Scatter Plot**: Compare rent against chosen getting around metric to find your ideal city.
        - **Map Visualization**: Dive into cities across the US with interactive map.
        - **City Details**: Discover insights about each city, including population, median rent, and more.
        - **Cost of Living Calculator**: Plan your budget and salary needs for a seamless relocation.
        """)

    show_future_updates = st.sidebar.toggle("Show Future Updates", value=False)

    if show_future_updates:
        st.sidebar.markdown("""
        ### Future Updates
        - **Rent Options**: Choose rent by home type and beds.
        - **Climate Risk**: Risk prone and prepardness by city.
        - **Adding tax calculation**: Post tax salary comaprison.
        - **Want to suggest something?**: Message me.
        """)
        
    show_enjoying_tool = st.sidebar.toggle("Enjoying this tool?", value=False)
    
    if show_enjoying_tool:
        st.sidebar.markdown("""
        Support 🫶 \n
        Let's connect on LinkedIn, X or share with a friend. \n
        Greatest support = Job referral
        """)
        
    st.sidebar.markdown(
        """
        [Pipeline Overview](https://github.com/Haikoitoh/CityVista/blob/main/ETL%20Pipeline.jpeg) | [dbt Lineage](https://github.com/Haikoitoh/CityVista/blob/main/dbt%20Lineage.jpeg)
        """
    )
    
    st.sidebar.title("Contact")
    st.sidebar.info(
        """
        Sumeet Badgujar:\n
        [LinkedIn](https://www.linkedin.com/in/sumeetbadgujar) | [GitHub](https://github.com/Haikoitoh) | [Twitter](https://twitter.com/SumeetBadgujar)
        """
    )
    
    
    # Filter the data based on user inputs
    if len(data.index) == 1:
        filtered_data = data[(data['MEDIAN_RENT'].between(median_rent_range[0], median_rent_range[1])) &
                         (data[selected_score])]
    else:
        filtered_data = data[(data['MEDIAN_RENT'].between(median_rent_range[0], median_rent_range[1])) &
                            (data[selected_score].between(selected_score_range[0], selected_score_range[1]))]
    
    
    # Display title
    st.title('CityVista: Explore, Compare, Relocate')
    
    # Scatter plot
    st.subheader(f"{selected_score} vs Median Rent for Cities")
    #st.write(f"{selected_score} vs Median Rent for Cities")
    scatter_plot_container = st.container(border=True)
    with scatter_plot_container:
        with st.expander("Set Scatter plot lines"):
            a,row1_col1, x,row1_col2,y, row1_col3 = st.columns([0.2,1,1,1,1,1])
            with row1_col1:
                rent_line = st.slider(
                        "Median Rent", min_value=filtered_data['MEDIAN_RENT'].min(), max_value=filtered_data['MEDIAN_RENT'].max(), value=1500, step=100
                    )
            with row1_col2:
                score_line = st.slider(
                        f"{selected_score}", min_value=round(filtered_data[selected_score].min()), max_value=round(filtered_data[selected_score].max()), value=roundup((filtered_data[selected_score].max()/2)), step=5
                    )
            with row1_col3:
                size_options = ['POPULATION', 'NUMBER_PARKS']
                selected_size = st.selectbox('Choose a size:', size_options)
    
        #plotly plot
        if selected_score == 'AVG_CITY_SCORE':
            fig = px.scatter(filtered_data, x='MEDIAN_RENT', y=selected_score, size=selected_size, hover_data=['CITY','TRANSIT_SCORE', 'WALK_SCORE', 'BIKE_SCORE'],color='CITY', height = 450)
        else:
            fig = px.scatter(filtered_data, x='MEDIAN_RENT', y=selected_score, size=selected_size, hover_data=['CITY'],color='CITY', height = 450)
        
        # Add reference lines
        #Score line
        fig.add_shape(type="line",
            x0 = filtered_data['MEDIAN_RENT'].min(), y0 = score_line, x1 = filtered_data['MEDIAN_RENT'].max(), y1 = score_line,
            line=dict(color="Red",width=2, dash="dashdot"),
        )
        #rent line
        fig.add_shape(type="line",
            x0 = rent_line, y0 = filtered_data[selected_score].min(), x1 = rent_line, y1 = filtered_data[selected_score].max(),
            line=dict(color="RoyalBlue",width=2, dash="dashdot"),
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig,use_container_width=True)
    
    second_row_container = st.container(border=True)
    map_container = st.container(border=True)
    city_details_container = st.container(border=True)
    # Interactive map
    with second_row_container:
        row2_col1, blank2, row2_col2 = st.columns([1,0.1,1])
        row3_col1,blank3, row3_col2 = st.columns([1,0.1,1])
        with row2_col1:
            st.subheader('Interactive Map')
            if filtered_data.shape[0] > 0:
                recenter_button = st.button('Recenter Map')
        with row2_col2:
            st.subheader('City Details')
            sort_options = [selected_score, 'NUMBER_PARKS', 'MEDIAN_RENT', 'PCT_10_MIN_WALK_PARK']
            selected_sort = st.selectbox('Sort by:', sort_options)
        with map_container:
            with row3_col1:
            
            #st.write("This is an interactive map showing city locations.")
    
                if filtered_data.shape[0] > 0:
                    #recenter_button = st.button('Recenter Map')
                    m = folium.Map(location=[np.mean(data['LATITUDE']), np.mean(data['LONGITUDE'])], zoom_start=2.5, tiles='OpenStreetMap',min_zoom=2.5)
    
                    for i in range(len(filtered_data)):
                        Metric = ['Avg City Score','Transit Score','Walk Score','Bike Score']
                        Value = [filtered_data.iloc[i]['AVG_CITY_SCORE'],filtered_data.iloc[i]['TRANSIT_SCORE'],
                                 filtered_data.iloc[i]['WALK_SCORE'],filtered_data.iloc[i]['BIKE_SCORE']]
                        df = pd.DataFrame({'Value': Value}, index=Metric)
                        html = df.to_html(
                                classes="table table-striped table-hover table-condensed table-responsive"
                            )
                        popup = folium.Popup(html)
    
                        popup_content = f"City: {filtered_data.iloc[i]['CITY']}<br>Population:"
                        folium.Marker(
                            location=[filtered_data.iloc[i]['LATITUDE'], filtered_data.iloc[i]['LONGITUDE']],
                            popup = html,
                            tooltip = filtered_data.iloc[i]['CITY']
                        ).add_to(m)
                    
                    if recenter_button:
                        #m.location = [np.mean(filtered_data['LATITUDE']), np.mean(filtered_data['LONGITUDE'])]
                        sw = filtered_data[['LATITUDE', 'LONGITUDE']].min().values.tolist()
                        ne = filtered_data[['LATITUDE', 'LONGITUDE']].max().values.tolist()
                        m.fit_bounds([sw, ne]) 
                    sw = filtered_data[['LATITUDE', 'LONGITUDE']].min().values.tolist()
                    ne = filtered_data[['LATITUDE', 'LONGITUDE']].max().values.tolist()
    
                    m.fit_bounds([sw, ne]) 
                
                    # Display the map
                    #folium_static(m)
                    st_folium(m,height = 500, use_container_width = True,returned_objects=[])
                else:
                    st.write("No data available to display on the map.")
    
        with city_details_container:
            with row3_col2:
            # Scrollable window
                city_details_container = st.container(height=510)
                sorted_data = filtered_data.sort_values(by=[selected_sort],ascending=False)
                with city_details_container:
                    for index, row in sorted_data.head(15).iterrows():
                        with st.container():
                            st.subheader(f"{row['CITY']}")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            pop = numerize.numerize(row['POPULATION'])
                            st.subheader(f"{pop}")
                            st.caption("Population")
                        with col2:
                            st.subheader(f"{row[selected_score]}")
                            st.caption(f"{selected_score}")
                        with col3:
                            st.subheader(f"{row['MEDIAN_RENT']:,}")
                            st.caption("Median Rent")
                        with col4:
                            st.subheader(f"{row['NUMBER_PARKS']}")
                            st.caption("Number of Parks")
                        with col5:
                            st.subheader(f"{row['PCT_10_MIN_WALK_PARK']}%")
                            st.caption("Living within 10 min walk of park")
                        st.divider()
    
    col1_container = st.container(border=True)
    if show_col_calc:
        with col1_container:
            st.subheader('Calculate Cost Of Living')
            st.markdown("🏡 Planning a move? Cost of living calculator is your key to financial foresight. Evaluate the standard of living and salary requirements of your potential new home with ease.")
            st.markdown("Set realistic expectations or explore alternatives with confidence. Your next move, simplified. 🚚💰")
            row5_col1,x, row5_col2,y = st.columns([1,0.5,1,0.2])
            with row5_col1:
                current_city_option1 = st.selectbox(
                " **Current City**",
                (col_data['CityState'].values.tolist()),
                index = None,
                placeholder = "Select a city",
                    )
                st.caption('#')
                new_city_option1 = st.selectbox(
                " **New City**",
                (col_data['CityState'].values.tolist()),
                index = None,
                placeholder = "Select a city",
                    )
                st.caption('#')
                current_income1 = st.number_input(' **Pre-tax household income**',placeholder=50000,value=50000, step=100)
                st.caption('#')
                #st.write(current_income)
            with row5_col2:
                with st.columns(3)[1]:
                    st.subheader('#')
                    home_image = Image.open('icons/home.png')
                    if None not in (current_city_option1, new_city_option1) :
                        st.image(home_image, width=128)
                if None not in (current_city_option1, new_city_option1) :
                    base_cost_of_living_index1 = col_data[col_data["CityState"] == current_city_option1]["COST_OF_LIVING_INDEX"].iloc[0]
                    desired_cost_of_living_index1 = col_data[col_data["CityState"] == new_city_option1]["COST_OF_LIVING_INDEX"].iloc[0]
                    #st.write(base_cost_of_living_index,desired_cost_of_living_index)
                    ratio1 = desired_cost_of_living_index1 / base_cost_of_living_index1
        
                    # Calculate adjusted income
                    adjusted_income1 = round(current_income1 * ratio1)
                    #st.markdown(f"**To maintain your standard of living in {new_city_option1}, you'll need a household income of:**")
                    st.markdown(f"<h5 style='text-align: center; '>To maintain your standard of living in {new_city_option1}, you'll need a household income of:</h5>", unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center; '>$ {adjusted_income1:,} </h3>", unsafe_allow_html=True)
                    pct_ratio1 = round(100 * (ratio1 - 1),2)
                    high_low1 = 'higher' if ratio1 > 1 else 'lower'
                    st.markdown(f"<h6 style='text-align: center; '>The cost of living in {new_city_option1} is {pct_ratio1}% {high_low1} than in {current_city_option1}</h6>", unsafe_allow_html=True)
                
            blank5, row6_1,row6_2,row6_3,row6_4,row6_5,row6_6,row6_7,row6_8 = st.columns([0.4,1,0.8,1,0.8,1,0.8,1,0.4])
        
            with row6_1:
                container1 = st.container()
                container1.markdown(
                    """
                    <style>
                    .centered {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
        
                # Centered image
                income_image = Image.open('icons/income.png')
                if None not in (current_city_option1, new_city_option1) :
                    container1.image(income_image, width = 128)
                
                    diff = current_income1 - adjusted_income1
                    if diff > 0 : 
                        less_more = 'less'
                    else:
                        diff = abs(diff)
                        less_more = 'more'
                    # Centered markdown
                    container1.markdown(f"The total income needed is **${diff:,} {less_more}** than your current household income")
        
            with row6_3:
                container2 = st.container()
                container2.markdown(
                    """
                    <style>
                    .centered {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
        
                # Centered image
                spending_image = Image.open('icons/spending.png')
                if None not in (current_city_option1, new_city_option1) :
                    container2.image(spending_image, width = 128)
                    diff = current_income1 - adjusted_income1
                    if diff > 0 : 
                        less_more = 'less'
                    else:
                        diff = abs(diff)
                        less_more = 'more'
                    # Centered markdown
                    container2.markdown(f"The cost of living in **{new_city_option1}** is **{pct_ratio1}% {high_low1}** than your current city")
                
            with row6_5:
                container3 = st.container()
                container3.markdown(
                    """
                    <style>
                    .centered {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
        
                # Centered image
                park_image = Image.open('icons/park.png')
                if None not in (current_city_option1, new_city_option1) :
                    container3.image(park_image, width = 128)
                    cur_park = col_data[col_data["CityState"] == current_city_option1]["PARKS_AREA"].iloc[0] 
                    new_park = col_data[col_data["CityState"] == new_city_option1]["PARKS_AREA"].iloc[0]
                    diff = round((new_park - cur_park)/new_park * 100,2)
                    if diff > 0 : 
                        less_more = 'less'
                    else:
                        diff = abs(diff)
                        less_more = 'more'
                        # Centered markdown
                    container3.markdown(f"The total park area is **{diff}% {less_more}** than your current city")
        
            with row6_7:
                container4 = st.container()
                container4.markdown(
                    """
                    <style>
                    .centered {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        height: 100vh;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
        
                # Centered image
                rent_image = Image.open('icons/rent.png')
                if None not in (current_city_option1, new_city_option1) :
                    container4.image(rent_image, width = 128)
                    cur_rent = col_data[col_data["CityState"] == current_city_option1]["MEDIAN_RENT"].iloc[0] 
                    new_rent = col_data[col_data["CityState"] == new_city_option1]["MEDIAN_RENT"].iloc[0]
                    diff = round((new_rent - cur_rent)/new_park * 100,2)
                    if diff > 0 : 
                        less_more = 'less'
                    else:
                        diff = abs(diff)
                        less_more = 'more'
                    # Centered markdown
                    container4.markdown(f"Median rent tends to cost **{diff}% {less_more}**")

if __name__ == "__main__":
    main()
