import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import json

st.set_page_config(layout="wide")

@st.cache_data
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

mean_temp_change = data.loc[:, '1961':'2022'].mean()

# Prepare data for linear regression
years = mean_temp_change.index.astype(int).values.reshape(-1, 1)
temperature_change = mean_temp_change.values

# Fit linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(years, temperature_change)

# Predict temperature change
temperature_change_pred = linear_regressor.predict(years)

# Plot the observed and predicted temperature change
figL = go.Figure()

# Add observed data
figL.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change,
    mode='lines+markers',
    name='Observed',
    line=dict(color='royalblue')
))

# Add linear fit
figL.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change_pred,
    mode='lines',
    name='Linear Fit',
    line=dict(dash='dash', color='tomato')
))

figL.update_layout(
    title='Observed vs Predicted Temperature Change (1961-2022)',
    xaxis_title='Year',
    yaxis_title='Temperature Change (°C)',
    template='plotly_dark'
)

# Calculate the mean temperature change for each decade
data_decades = data.loc[:, '1961':'2022']

# Create a function to map years to decades
def year_to_decade(year):
    return f"{year // 10 * 10}s"

# Group by decades and calculate the mean temperature change for each decade
data_decades.columns = data_decades.columns.astype(int)
data_decades = data_decades.groupby(year_to_decade, axis=1).mean()

# Calculate the rate of heating per decade
rate_of_heating_per_decade = data_decades.diff(axis=1).mean(axis=0)

# Prepare data for plotting
rate_of_heating_per_decade_df = rate_of_heating_per_decade.reset_index()
rate_of_heating_per_decade_df.columns = ['Decade', 'Rate of Heating']

average_rate_of_heating = rate_of_heating_per_decade.mean()

# Plot the rate of heating per decade using plotly
fig = px.bar(rate_of_heating_per_decade_df, 
             x='Decade', 
             y='Rate of Heating',
             title='Rate of Heating Per Decade (1960s-2020s)',
             labels={'Rate of Heating': 'Temperature Change (°C)'},
             template='plotly_dark',
             color_discrete_sequence=['royalblue'])

# Add an average line to the bar chart
fig.add_trace(go.Scatter(
    x=rate_of_heating_per_decade_df['Decade'],
    y=[average_rate_of_heating] * len(rate_of_heating_per_decade_df),
    mode='lines',
    name='Average Rate of Heating',
    line=dict(color='firebrick', width=2, dash='dash')
))


# Fit linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(years, temperature_change)
temperature_change_pred_linear = linear_regressor.predict(years)

# Fit quadratic regression model
quadratic_regressor = np.poly1d(np.polyfit(years.flatten(), temperature_change, 2))
temperature_change_pred_quad = quadratic_regressor(years.flatten())

# Calculate R^2 and MSE for linear model
r2_linear = linear_regressor.score(years, temperature_change)
mse_linear = mean_squared_error(temperature_change, temperature_change_pred_linear)

# Calculate R^2 and MSE for quadratic model
r2_quad = np.corrcoef(temperature_change, temperature_change_pred_quad)[0, 1]**2
mse_quad = mean_squared_error(temperature_change, temperature_change_pred_quad)

# Extract the quadratic term coefficient (rate of acceleration)
quadratic_coefficients = np.polyfit(years.flatten(), temperature_change, 2)
rate_of_acceleration = quadratic_coefficients[0]

# Print the results
print(f"Linear Model R^2: {r2_linear:.4f}, MSE: {mse_linear:.4f}")
print(f"Quadratic Model R^2: {r2_quad:.4f}, MSE: {mse_quad:.4f}")
print(f"Rate of Acceleration (Quadratic Term Coefficient): {rate_of_acceleration:.6f}")

# Create an interactive plotly plot
figQ = go.Figure()

# Add observed data
figQ.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change,
    mode='lines+markers',
    name='Observed',
    line=dict(color='royalblue')
))

# Add linear fit
figQ.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change_pred_linear,
    mode='lines',
    name='Linear Fit',
    line=dict(dash='dash', color='tomato')
))

# Add quadratic fit
figQ.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change_pred_quad,
    mode='lines',
    name='Quadratic Fit',
    line=dict(dash='dot', color='yellowgreen')
))

figQ.update_layout(
    title='Observed vs Predicted Temperature Change (1961-2022)',
    xaxis_title='Year',
    yaxis_title='Temperature Change (°C)',
    template='plotly_dark'
)

# Define CSS styles
st.markdown(
    """
    <style>
    .kpi-container {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .kpi-item {
        text-align: center;
        font-size: 1.5em;
    }
    </style>
    """,
    unsafe_allow_html=True
)




# KPIs Section
st.header('Key Rates')
with st.container():
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown('<div class="kpi-item">', unsafe_allow_html=True)
        st.metric(label="Heating Rate per Year", value="0.0242°C")
        with st.expander("See Linear Fit"):
            st.plotly_chart(figL, use_container_width=True)  
        st.markdown('</div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown('<div class="kpi-item">', unsafe_allow_html=True)
        st.metric(label="Heating Rate per Decade", value="0.224°C")
        with st.expander("See Heating Rate per Decade"):
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown('<div class="kpi-item">', unsafe_allow_html=True)
        st.metric(label="Acceleration of Yearly Rate", value="0.00032°C")
        with st.expander("See Quadratic Fit"):
            st.plotly_chart(figQ, use_container_width=True)  
        st.markdown('</div>', unsafe_allow_html=True)

# Figures Section
st.header('Visualizations')
with st.container():
    fig_col1, fig_col2 = st.columns(2)


    # Placeholder for Figures
    with fig_col1:
        st.subheader(" ")
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
            height=600,
            title='Mean Temperature Change Over the Years',
            xaxis_title='Year',
            yaxis_title='Temperature Change (°C)',
            template='plotly_dark'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with fig_col2:
        st.subheader(" ")
        # Calculate the mean temperature change for each year
        mean_temp_change = data.loc[:, '1961':'2022'].mean()

        # Prepare data for Exponential Smoothing
        years = mean_temp_change.index.astype(int).values.reshape(-1, 1)
        temperature_change = mean_temp_change.values
        years_series = pd.Series(temperature_change, index=years.flatten())

        # Fit Exponential Smoothing model
        es_model = ExponentialSmoothing(years_series, trend='add', seasonal=None, seasonal_periods=None).fit()

        # Forecast future temperatures for the next 20 years
        forecast_years = np.arange(2023, 2043)
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
            height=600,
            title='Temperature Change Forecast',
            xaxis_title='Year',
            yaxis_title='Temperature Change (°C)',
            template='plotly_dark'
        )
        st.plotly_chart(fig6, use_container_width=True)

# Additional Figures
with st.container():
    fig_col3, fig_col4 = st.columns(2)


    with fig_col3:
        st.subheader(" ")
    
        # Initial plot with default n value
        data['Temp_Increase'] = data['2022']
        n = 10  # Default value for top N countries
        top_countries = data[['Country', 'Temp_Increase']].sort_values(by='Temp_Increase', ascending=False).head(n)
        top_countries = top_countries.sort_values(by='Temp_Increase')
        
        fig2 = px.bar(top_countries, 
                      x='Temp_Increase', 
                      y='Country', 
                      orientation='h', 
                      color='Temp_Increase',
                      color_continuous_scale=tomato_colors,
                      title=f'Top {n} Countries with Highest Temperature Increase')
        fig2.update_layout(
            height=600,
            xaxis_title='Temperature Increase (°C)',
            yaxis_title='Country',
            template='plotly_dark'
        )
        chart = st.plotly_chart(fig2, use_container_width=True)
    
        # Move the slider to the bottom
        n = st.slider('Select Top N Countries', 1, 250, 10, key='n_slider')
    
        # Update the chart based on slider value
        top_countries = data[['Country', 'Temp_Increase']].sort_values(by='Temp_Increase', ascending=False).head(n)
        top_countries = top_countries.sort_values(by='Temp_Increase')
        
        fig2 = px.bar(top_countries, 
                      x='Temp_Increase', 
                      y='Country', 
                      orientation='h', 
                      color='Temp_Increase',
                      color_continuous_scale=tomato_colors,
                      title=f'Top {n} Countries with Highest Temperature Increase')
        fig2.update_layout(
            height=600,
            xaxis_title='Temperature Increase (°C)',
            yaxis_title='Country',
            template='plotly_dark'
        )
        chart.plotly_chart(fig2, use_container_width=True)

    
           
    
           

    with fig_col4:
        st.subheader(" ")
        northern_hemisphere = [
            'Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Armenia',
            'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus',
            'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana', 'Brunei Darussalam',
            'Bulgaria', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Cayman Islands',
            'Central African Republic', 'Chad', 'China', 'Colombia',
            'Comoros', 'Congo', 'Costa Rica', 'Croatia', 'Cuba',
            'Cyprus', 'Czech Republic', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
            'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Eswatini',
            'Ethiopia', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana',
            'Gibraltar', 'Greece', 'Greenland', 'Grenada', 'Guadeloupe', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana',
            'Haiti', 'Holy See', 'Honduras', 'Hungary', 'Iceland', 'India', 'Iran', 'Iraq', 'Ireland',
            'Isle of Man', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan',
            'Laos', 'Latvia', 'Lebanon', 'Liberia', 'Libya', 'Liechtenstein', 'Lithuania', 'Luxembourg',
            'Malaysia', 'Maldives', 'Malta', 'Mauritania', 'Mauritius', 'Mexico', 'Moldova', 'Monaco',
            'Mongolia', 'Montenegro', 'Morocco', 'Myanmar', 'Nepal', 'Netherlands', 'Nicaragua', 'Niger', 'Nigeria',
            'North Macedonia', 'Norway', 'Oman', 'Pakistan', 'Panama', 'Philippines', 'Poland', 'Portugal',
            'Puerto Rico', 'Qatar', 'Romania', 'Russia', 'Rwanda', 'San Marino', 'Saudi Arabia', 'Senegal',
            'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Somalia', 'Spain',
            'Sri Lanka', 'St. Kitts and Nevis', 'St. Lucia', 'St. Vincent and the Grenadines', 'Sudan', 'Suriname', 'Sweden',
            'Switzerland', 'Syria', 'Taiwan', 'Tajikistan', 'Thailand', 'Timor-Leste',
            'Togo', 'Trinidad and Tobago', 'Tunisia', 'Turkmenistan', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom',
            'United States', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Vietnam', 'West Bank and Gaza',
            'Western Sahara', 'Yemen', 'Zambia', 'Zimbabwe'
        ]

        southern_hemisphere = [
            'American Samoa', 'Antigua and Barbuda', 'Argentina', 'Aruba', 'Australia', 'Botswana',
            'Brazil', 'Chile', 'Colombia', 'Cook Islands', 'Ecuador', 'Falkland Islands', 'Fiji',
            'French Polynesia', 'Indonesia', 'Kiribati', 'Madagascar', 'Malawi', 'Marshall Islands',
            'Mauritius', 'Mayotte', 'Mozambique', 'Namibia', 'Nauru', 'New Caledonia', 'New Zealand', 'Niue',
            'Norfolk Island', 'Papua New Guinea', 'Paraguay', 'Peru', 'Pitcairn Islands', 'Samoa', 'São Tomé and Príncipe',
            'Solomon Islands', 'South Africa', 'South Sudan', 'St. Helena', 'St. Pierre and Miquelon', 'Suriname',
            'Tokelau', 'Tonga', 'Tuvalu', 'Uruguay', 'Vanuatu', 'Wallis and Futuna'
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
            height=600,
            title="Temperature Trends in Northern vs Southern Hemisphere",
            xaxis_title='Year',
            yaxis_title='Temperature Change (°C)',
            template='plotly_dark'
        )
        st.plotly_chart(fig3, use_container_width=True)

# One more figure in full width

with st.container():
    st.subheader(" ")
    # Calculate the temp change for 2022
    temp_change_2022 = data[['Country']].copy()
    temp_change_2022['Temperature Change'] = data['2022']


    # Define a custom tomato-like color scale
    tomato_colors = [
        [0.0, 'rgb(255, 248, 247)'],
        [0.2, 'rgb(255, 228, 225)'],
        [0.4, 'rgb(255, 182, 193)'],
        [0.6, 'rgb(255, 160, 122)'],
        [0.8, 'rgb(255, 127, 80)'],
        [1.0, 'rgb(255, 99, 71)']
    ]


    # Plotting the average mean temperature change on a 3D globe
    fig_globe = px.choropleth(temp_change_2022,
                        geojson=geojson_data,
                        locations='Country',
                        featureidkey='properties.ADMIN',
                        color='Temperature Change',
                        hover_name='Country',
                        projection='orthographic',
                        color_continuous_scale=tomato_colors,
                        title='Temperature Change by 2022')

    fig_globe.update_geos(
    fitbounds="locations",
    visible=True
    )

    fig_globe.update_layout(
    height=600,
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        showland=True,
        showcountries=True
    ))
    st.plotly_chart(fig_globe, use_container_width=True)

