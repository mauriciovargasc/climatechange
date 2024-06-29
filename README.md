# :earth_americas: Climate Change Dash

A simple Streamlit dashboard app displaying the Climate Change Analysis

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appapppy-pahkgl8dd2b2ve9nrdregs.streamlit.app)

The following project started as I wanted to answer the following questions using data.

Is climate change real, what is actually happening?
Is there countries or regions that are disproportionally affected? What countries had the highest temperature increase?
Is the northern hemisphere changing differently to the southern hemisphere? 
Is there a specific rate of heating and is it accelerating? Can we forecast future rates of heating?
What is the mean temperature change across years? Can we forecast the mean temperature change for future years?

I. Data collection/cleaning

Data: https://www.kaggle.com/datasets/tarunrm09/climate-change-indicators/data ; https://github.com/datasets/geo-countries/blob/master/data/countries.geojson       

Cleaning Summary:
  Handling Missing Values:
  
  Missing ISO2 codes were identified and corrected.
  Missing temperature data were identified, and a combination of linear interpolation followed by forward and backward fill was applied to handle the missing values.
  
  Data Types Conversion:
  Temperature data columns were converted to numeric types to facilitate interpolation and further analysis.
  
  Normalization of Year Columns:
  Year columns were renamed to remove the 'F' prefix for consistency and ease of analysis.
  Verification:
  
  Normalization of country:
  Matched country names by ISO3 identification to match the country names used in the countries.geojson file
  
  The cleaned dataset was verified to ensure no remaining missing values in the temperature data.
