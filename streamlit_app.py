import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import json

@st.cache
def load_data():
    data = pd.read_csv(r"climate_change_indicators.csv")  # Update with your file path
    with open(r"countries.geojson") as f:  # Update with your file path
        geojson_data = json.load(f)
    return data, geojson_data

data, geojson_data = load_data()

tomato_colors = [  #Define tomato colors
    [0.0, 'rgb(255, 245, 238)'],
    [0.2, 'rgb(255, 228, 225)'],
    [0.4, 'rgb(255, 182, 193)'],
    [0.6, 'rgb(255, 160, 122)'],
    [0.8, 'rgb(255, 127, 80)'],
    [1.0, 'rgb(255, 99, 71)']
]
# Title of the Dashboard
st.title('Climate Change Dashboard')

# KPIs Section
st.header('Key Rates')
with st.container():
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.metric(label="Heating Rate per Year", value="0.0242°C")

    with kpi2:
        st.metric(label="Heating Rate per Decade", value="0.224°C")

    with kpi3:
        st.metric(label="Acceleration of Yearly Rate", value="0.00032°C")

# Figures Section
st.header('Visualizations')
with st.container():
    fig_col1, fig_col2 = st.columns(2)


    # Placeholder for Figures
    with fig_col1:
        st.subheader("Mean Temperature Change Over the Years")
        mean_temp_change = data.loc[:, '1961':'2022'].mean()
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=mean_temp_change.index,
            y=mean_temp_change.values,
            mode='lines+markers',
            name='Mean Temperature Change',
            line=dict(color='royalblue')
        ))
        fig1.update_layout(
            title='Mean Temperature Change Over the Years',
            xaxis_title='Year',
            yaxis_title='Temperature Change (°C)',
            template='plotly_dark'
        )
        st.plotly_chart(fig1)

    with fig_col2:
        st.subheader("Temperature Change Forecast")
        # Calculate the mean temperature change for each year
        mean_temp_change = data.loc[:, '1961':'2022'].mean()

        # Prepare data for Exponential Smoothing
        years = mean_temp_change.index.astype(int).values.reshape(-1, 1)
        temperature_change = mean_temp_change.values
        years_series = pd.Series(temperature_change, index=years.flatten())

        # Fit Exponential Smoothing model
        es_model = ExponentialSmoothing(years_series, trend='add', seasonal=None, seasonal_periods=None).fit()

        # Forecast future temperatures for the next 10 years
        forecast_years = np.arange(2023, 2033)
        forecast = es_model.forecast(len(forecast_years))

        fig6 = go.Figure()
        fig6.add_trace(go.Scatter(
            x=years.flatten(),
            y=temperature_change,
            mode='lines+markers',
            name='Observed',
            line=dict(color='royalblue')
        ))
        fig6.add_trace(go.Scatter(
            x=np.append(years.flatten(), forecast_years),
            y=np.append(temperature_change, forecast),
            mode='lines',
            name='Forecast',
            line=dict(dash='dash', color='tomato')
        ))
        fig6.update_layout(
            title='Temperature Change Forecast (1961-2032)',
            xaxis_title='Year',
            yaxis_title='Temperature Change (°C)',
            template='plotly_dark'
        )
        st.plotly_chart(fig6)

# Additional Figures
with st.container():
    fig_col3, fig_col4 = st.columns(2)


    with fig_col3:
        st.subheader("Top N Countries with Highest Temperature Increase")
        n = st.slider('Select Top N Countries', 1, 50, 10)  # Slider for selecting top N countries
        data['Temp_Increase'] = data['2022'] - data['1961']
        top_countries = data[['Country', 'Temp_Increase']].sort_values(by='Temp_Increase', ascending=False).head(n)
        top_countries = top_countries.sort_values(by='Temp_Increase')
        fig2 = px.bar(top_countries, 
                    x='Temp_Increase', 
                    y='Country', 
                    orientation='h', 
                    color='Temp_Increase',
                    color_continuous_scale=tomato_colors,
                    title=f'Top {n} Countries with Highest Temperature Increase (1961-2022)')
        fig2.update_layout(
            xaxis_title='Temperature Increase (°C)',
            yaxis_title='Country',
            template='plotly_dark'
        )
        st.plotly_chart(fig2)

    with fig_col4:
        st.subheader("Temperature Trends in Northern vs Southern Hemisphere")
        northern_hemisphere = [
            'Afghanistan, Islamic Rep. of', 'Albania', 'Algeria', 'Andorra, Principality of', 'Angola', 'Armenia, Rep. of',
            'Austria', 'Azerbaijan, Rep. of', 'Bahamas, The', 'Bahrain, Kingdom of', 'Bangladesh', 'Belarus, Rep. of',
            'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brunei Darussalam',
            'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Cayman Islands',
            'Central African Rep.', 'Chad', 'China, P.R.: Hong Kong', 'China, P.R.: Macao', 'China, P.R.: Mainland', 'Colombia',
            'Comoros, Union of the', 'Congo, Dem. Rep. of the', 'Congo, Rep. of', 'Costa Rica', 'Croatia, Rep. of', 'Cuba',
            'Cyprus', 'Czech Rep.', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Rep.', 'Ecuador', 'Egypt, Arab Rep. of',
            'El Salvador', 'Equatorial Guinea, Rep. of', 'Eritrea, The State of', 'Estonia, Rep. of', 'Eswatini, Kingdom of',
            'Ethiopia, The Federal Dem. Rep. of', 'Finland', 'France', 'Gabon', 'Gambia, The', 'Georgia', 'Germany', 'Ghana',
            'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana',
            'Haiti', 'Holy See', 'Honduras', 'Hungary', 'Iceland', 'India', 'Iran, Islamic Rep. of', 'Iraq', 'Ireland',
            'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan, Rep. of', 'Kuwait', 'Kyrgyz Rep.',
            'Lao People\'s Dem. Rep.', 'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
            'Malaysia', 'Maldives', 'Malta', 'Mauritania, Islamic Rep. of', 'Mauritius', 'Mexico', 'Moldova, Rep. of', 'Monaco',
            'Mongolia', 'Montenegro', 'Morocco', 'Myanmar', 'Nepal', 'Netherlands, The', 'Nicaragua', 'Niger', 'Nigeria',
            'North Macedonia, Republic of', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Philippines', 'Poland, Rep. of', 'Portugal',
            'Puerto Rico', 'Qatar', 'Romania', 'Russian Federation', 'Rwanda', 'San Marino, Rep. of', 'Saudi Arabia', 'Senegal',
            'Serbia, Rep. of', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovak Rep.', 'Slovenia, Rep. of', 'Somalia', 'Spain',
            'Sri Lanka', 'St. Kitts and Nevis', 'St. Lucia', 'St. Vincent and the Grenadines', 'Sudan', 'Suriname', 'Sweden',
            'Switzerland', 'Syrian Arab Rep.', 'Taiwan Province of China', 'Tajikistan, Rep. of', 'Thailand', 'Timor-Leste, Dem. Rep. of',
            'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom',
            'United States', 'Uruguay', 'Uzbekistan, Rep. of', 'Venezuela, Rep. Bolivariana de', 'Vietnam', 'West Bank and Gaza',
            'Western Sahara', 'Yemen, Rep. of', 'Zambia', 'Zimbabwe'
        ]

        southern_hemisphere = [
            'American Samoa', 'Antigua and Barbuda', 'Argentina', 'Aruba, Kingdom of the Netherlands', 'Australia', 'Botswana',
            'Brazil', 'Chile', 'Colombia', 'Cook Islands', 'Ecuador', 'Falkland Islands (Malvinas)', 'Fiji, Rep. of',
            'French Polynesia', 'Indonesia', 'Kiribati', 'Madagascar, Rep. of', 'Malawi', 'Marshall Islands, Rep. of the',
            'Mauritius', 'Mayotte', 'Mozambique, Rep. of', 'Namibia', 'Nauru, Rep. of', 'New Caledonia', 'New Zealand', 'Niue',
            'Norfolk Island', 'Papua New Guinea', 'Paraguay', 'Peru', 'Pitcairn Islands', 'Samoa', 'São Tomé and Príncipe, Dem. Rep. of',
            'Solomon Islands', 'South Africa', 'South Sudan, Rep. of', 'St. Helena', 'St. Pierre and Miquelon', 'Suriname',
            'Tokelau', 'Tonga', 'Tuvalu', 'Uruguay', 'Vanuatu', 'Wallis and Futuna Islands'
        ]

        data['Hemisphere'] = data['Country'].apply(lambda x: 'Northern' if x in northern_hemisphere else ('Southern' if x in southern_hemisphere else 'Other'))
        mean_temp_change_north = data[data['Hemisphere'] == 'Northern'].loc[:, '1961':'2022'].mean()
        mean_temp_change_south = data[data['Hemisphere'] == 'Southern'].loc[:, '1961':'2022'].mean()
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=mean_temp_change_north.index,
            y=mean_temp_change_north.values,
            mode='lines+markers',
            name='Northern Hemisphere',
            line=dict(color='royalblue')
        ))
        fig3.add_trace(go.Scatter(
            x=mean_temp_change_south.index,
            y=mean_temp_change_south.values,
            mode='lines+markers',
            name='Southern Hemisphere',
            line=dict(color='tomato')
        ))
        fig3.update_layout(
            title='Temperature Trends in Northern vs Southern Hemisphere (1961-2022)',
            xaxis_title='Year',
            yaxis_title='Temperature Change (°C)',
            template='plotly_dark'
        )
        st.plotly_chart(fig3)

# One more figure in full width

with st.container():
    st.subheader("3D Globe Visualization of Average Mean Temperature Change")
    mean_temp_change_all_years = data[['Country']].copy()
    mean_temp_change_all_years['Temperature Change'] = data.loc[:, '1961':'2022'].mean(axis=1)

    # Plotting the average mean temperature change on a 3D globe
    fig_globe = px.choropleth(mean_temp_change_all_years,
                            geojson=geojson_data,
                            locations='Country',
                            featureidkey='properties.ADMIN',
                            color='Temperature Change',
                            hover_name='Country',
                            projection='orthographic',
                            color_continuous_scale=tomato_colors,
                            title='Average Mean Temperature Change (1961-2022)')

    fig_globe.update_geos(
        fitbounds="locations",
        visible=True
    )

    fig_globe.update_layout(
        geo=dict(
            bgcolor='rgba(0,0,0,0)',
            showland=True,
            showcountries=True
        ))

    st.plotly_chart(fig_globe)
