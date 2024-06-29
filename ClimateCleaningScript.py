# %% [markdown]
# Cleaning Data Summary:
# 
# Handling Missing Values:
# 
# Missing ISO2 codes were identified and corrected.
# Missing temperature data were identified, and a combination of linear interpolation followed by forward and backward fill was applied to handle the missing values.
# 
# Data Types Conversion:
# Temperature data columns were converted to numeric types to facilitate interpolation and further analysis.
# 
# Normalization of Year Columns:
# Year columns were renamed to remove the 'F' prefix for consistency and ease of analysis.
# Verification:
# 
# Normalization of country:
# Matched country names by ISO3 identification to match the country names used in the countries.geojson file
# 
# The cleaned dataset was verified to ensure no remaining missing values in the temperature data.

# %%
import pandas as pd
import json

# Load the dataset
file_path = r"C:\Users\puert\OneDrive\Documents\Professional\projects\climatechangeKaggle\climate_change_indicators.csv"
data = pd.read_csv(file_path)

# Load the GeoJSON file
geojson_path = r"C:\Users\puert\OneDrive\Documents\Professional\projects\climatechangeKaggle\countries.geojson"
with open(geojson_path) as f:
    geojson_data = json.load(f)


# Display the first few rows of the dataset
data.head()


# %%
# Check for missing values and data types
data.info()
missing_values = data.isnull().sum()

print(missing_values[missing_values > 0])


# %%
# Display rows with missing ISO2 codes
missing_iso2 = data[data['ISO2'].isnull()]
missing_iso2

# %%
# Fill in missing ISO2 codes
data.loc[data['Country'] == 'Namibia', 'ISO2'] = 'NA'

# Optionally assign a placeholder for 'World'
data.loc[data['Country'] == 'World', 'ISO2'] = 'WL'  # Placeholder

# Display the updated rows
updated_iso2 = data[data['Country'].isin(['Namibia', 'World'])]
updated_iso2


# %%
# Calculate the number of missing values for temperature data per country
missing_temp_data = data.iloc[:, 9:].isnull().sum(axis=1)

# Add a column for missing temperature data count
data['Missing_Temp_Data_Count'] = missing_temp_data

# Display countries with missing temperature data
countries_with_missing_temp_data = data[data['Missing_Temp_Data_Count'] > 0]
countries_with_missing_temp_data


# %%
# Convert temperature columns to numeric types
temperature_columns = data.columns[9:-1]
data[temperature_columns] = data[temperature_columns].apply(pd.to_numeric, errors='coerce')

# Interpolate missing temperature data
data[temperature_columns] = data[temperature_columns].interpolate(method='linear', axis=1)

# Forward fill and backward fill to handle remaining missing values
data[temperature_columns] = data[temperature_columns].fillna(method='ffill', axis=1).fillna(method='bfill', axis=1)

# Verify if there are any remaining missing values
remaining_missing_values = data[temperature_columns].isnull().sum().sum()
remaining_missing_values


# %%
data.rename(columns=lambda x: x[1:] if x.startswith('F') else x, inplace=True)

# Verify the column names
data.columns

# %%

# Create a mapping from ISO3 to country names from the new GeoJSON file
iso3_to_country = {feature['properties']['ISO_A3']: feature['properties']['ADMIN'] for feature in geojson_data['features']}

# Map the ISO3 codes to country names in the climate change dataset
data['Country'] = data['ISO3'].map(iso3_to_country)

# Verify the normalization
normalized_countries = data[['ISO3', 'Country']].drop_duplicates().sort_values(by='ISO3')
normalized_countries

# %%
# Save the cleaned dataset
cleaned_file_path = r"C:\Users\puert\OneDrive\Documents\Professional\projects\climatechangeKaggle\climate_change_indicators.csv"
data.to_csv(cleaned_file_path, index=False)



