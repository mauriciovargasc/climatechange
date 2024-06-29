# %%
# Import necessary packages
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the cleaned and normalized dataset
file_path = r"C:\Users\puert\OneDrive\Documents\Professional\projects\climatechangeKaggle\climate_change_indicators.csv"
data = pd.read_csv(file_path)

# Load the GeoJSON file for the 3D globe map
geojson_path = r"C:\Users\puert\OneDrive\Documents\Professional\projects\climatechangeKaggle\countries.geojson"
with open(geojson_path) as f:
    geojson_data = json.load(f)

# Display the first few rows of the DataFrame
data.head()

# Check for the structure of the data
data.info()

# Initial statistics summary
data.describe()

# %% [markdown]
# I. Analyzing Temperature Trends Over the Years

# %%
# Calculate the mean temperature change for each year
mean_temp_change = data.loc[:, '1961':'2022'].mean()

# Plot the mean temperature change over the years
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=mean_temp_change.index,
    y=mean_temp_change.values,
    mode='lines+markers',
    name='Mean Temperature Change',
    line=dict(color='royalblue')
))

fig.update_layout(
    title='Mean Temperature Change Over the Years',
    xaxis_title='Year',
    yaxis_title='Temperature Change (°C)',
    template='plotly_dark'
)

fig.show()

# %% [markdown]
# The plot shows an obvious upward trend in mean temperature change, showing the reality of climate change

# %% [markdown]
# II. Identify Countries with Highest Temperature Increase

# %%
# Calculate the temperature increase for 2022 since its the progress up until now for heating.
data['Temp_Increase'] = data['2022'] 

# Identify the top 10 countries with the highest temperature increase
top_countries = data[['Country', 'Temp_Increase']].sort_values(by='Temp_Increase', ascending=False).head(10)

# Define a custom tomato-like color scale
tomato_colors = [
    [0.0, 'rgb(255, 245, 238)'],
    [0.2, 'rgb(255, 228, 225)'],
    [0.4, 'rgb(255, 182, 193)'],
    [0.6, 'rgb(255, 160, 122)'],
    [0.8, 'rgb(255, 127, 80)'],
    [1.0, 'rgb(255, 99, 71)']
]

# Function to plot the temperature increase for the top N countries
def plot_top_n_countries(data, n=10):
    top_countries = data[['Country', 'Temp_Increase']].sort_values(by='Temp_Increase', ascending=False).head(n)
    top_countries = top_countries.sort_values(by='Temp_Increase')  # Sort for plotting
    fig = px.bar(top_countries, 
                 x='Temp_Increase', 
                 y='Country', 
                 orientation='h', 
                 color='Temp_Increase',
                 color_continuous_scale=tomato_colors,
                 title=f'Top {n} Countries with Highest Temperature Increase')
    
    fig.update_layout(
        xaxis_title='Temperature Increase (°C)',
        yaxis_title='Country',
        template='plotly_dark'
    )
    
    fig.show()

# Example usage: plot the top 10 countries
plot_top_n_countries(data, n=10)


# %% [markdown]
# The bar plot demonstrates the top 10 countries by highester temperature increase. SHowing which countries are disproportionally affected.

# %% [markdown]
# III. Compare Temperature Trends Between Hemispheres

# %%
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


# Add hemisphere information to the data
data['Hemisphere'] = np.where(data['Country'].isin(northern_hemisphere), 'Northern', 
                              np.where(data['Country'].isin(southern_hemisphere), 'Southern', 'Other'))

# Calculate mean temperature change for each hemisphere over the years
mean_temp_change_north = data[data['Hemisphere'] == 'Northern'].loc[:, '1961':'2022'].mean()
mean_temp_change_south = data[data['Hemisphere'] == 'Southern'].loc[:, '1961':'2022'].mean()

# Plot temperature trends for both hemispheres
# Create an interactive plotly plot for temperature trends
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=mean_temp_change_north.index,
    y=mean_temp_change_north.values,
    mode='lines+markers',
    name='Northern Hemisphere',
    line=dict(color='royalblue')
))

fig.add_trace(go.Scatter(
    x=mean_temp_change_south.index,
    y=mean_temp_change_south.values,
    mode='lines+markers',
    name='Southern Hemisphere',
    line=dict(color='tomato')
))

fig.update_layout(
    title='Temperature Trends in Northern vs Southern Hemisphere (1961-2022)',
    xaxis_title='Year',
    yaxis_title='Temperature Change (°C)',
    template='plotly_dark'
)

fig.show()


# %% [markdown]
# The plot shows two lines, one for northern hemisphere temperature changes and another for southern hemisphere temperature changes. The plot shows and upward trend in both hemispheres, however the northern region seems to be heating up at a faster rate.

# %% [markdown]
# IV. Analyze the Rate of Heating and Its Acceleration

# %%
# Prepare data for linear regression
years = mean_temp_change.index.astype(int).values.reshape(-1, 1)
temperature_change = mean_temp_change.values

# Fit linear regression model
linear_regressor = LinearRegression()
linear_regressor.fit(years, temperature_change)

# Predict temperature change
temperature_change_pred = linear_regressor.predict(years)

# Plot the observed and predicted temperature change
fig = go.Figure()

# Add observed data
fig.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change,
    mode='lines+markers',
    name='Observed',
    line=dict(color='royalblue')
))

# Add linear fit
fig.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change_pred,
    mode='lines',
    name='Linear Fit',
    line=dict(dash='dash', color='tomato')
))

fig.update_layout(
    title='Observed vs Predicted Temperature Change (1961-2022)',
    xaxis_title='Year',
    yaxis_title='Temperature Change (°C)',
    template='plotly_dark'
)

fig.show()

# Calculate the slope of the regression line (rate of heating)
rate_of_heating = linear_regressor.coef_[0]
rate_of_heating


# %% [markdown]
# The linear regression model shows an upward trend in temperature change. Additionally, the rate of heating was approximately 0.0242°C per year.

# %%
#What is the rate of heating per decade
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

fig.show()

# Print the average mean temperature change for all decades combined

print(f'Average Rate of Heating Per Decade: {average_rate_of_heating:.4f} °C')


# %% [markdown]
# The bar plot shows the rate of heating per decade from the 1960s to the 2020s.
# The world sems to be heating at a rate of 0.224°C per decade.
# 

# %% [markdown]
# V. Check for Acceleration in the Rate of Heating

# %%
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
fig = go.Figure()

# Add observed data
fig.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change,
    mode='lines+markers',
    name='Observed',
    line=dict(color='royalblue')
))

# Add linear fit
fig.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change_pred_linear,
    mode='lines',
    name='Linear Fit',
    line=dict(dash='dash', color='tomato')
))

# Add quadratic fit
fig.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change_pred_quad,
    mode='lines',
    name='Quadratic Fit',
    line=dict(dash='dot', color='yellowgreen')
))

fig.update_layout(
    title='Observed vs Predicted Temperature Change (1961-2022)',
    xaxis_title='Year',
    yaxis_title='Temperature Change (°C)',
    template='plotly_dark'
)

fig.show()


# %% [markdown]
# The plot compares the linear and quadratic fits against the observed data. Using R^2 and MSE on top of what the plot shows we can confirm that there is an acceleration on the rate of temperature change. Each year, the temperature change increases by an approximate 0.00032°C more than it did the previous year. 

# %% [markdown]
# VI. Forecast Future Temperature Changes

# %%
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

# Create an interactive plotly plot
fig = go.Figure()

# Add observed data
fig.add_trace(go.Scatter(
    x=years.flatten(),
    y=temperature_change,
    mode='lines+markers',
    name='Observed',
    line=dict(color='royalblue')
))

# Add forecast data
fig.add_trace(go.Scatter(
    x=np.append(years.flatten(), forecast_years),
    y=np.append(temperature_change, forecast),
    mode='lines',
    name='Forecast',
    line=dict(dash='dash', color='tomato')
))

fig.update_layout(
    title='Temperature Change Forecast (1961-2032)',
    xaxis_title='Year',
    yaxis_title='Temperature Change (°C)',
    template='plotly_dark'
)

fig.show()

# %% [markdown]
# The plot shows the forecasted temperature up to the year 2032. We are forecasted to pass the 1.5°C threshold by 2025.

# %%


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
geo=dict(
    bgcolor='rgba(0,0,0,0)',
    showland=True,
    showcountries=True
))

# %% [markdown]
# The globe visualization shows a heatmap of temperature change progress up to 2022. It shows which countries have had the highest temperature change with regards to 1951-1980 baseline.


