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
  The cleaned dataset was verified to ensure no remaining missing values in the temperature data.
  
  Normalization of country:
  Matched country names by ISO3 identification to match the country names used in the countries.geojson file
  
 


II. Data Analysis

In this climate change analysis project, I conducted a comprehensive investigation into global temperature trends, regional impacts, and future projections using a robust dataset covering the years 1961 to 2022. Here’s what I primarily focused on and discovered:

Temperature Trends Over Time:
I calculated the mean temperature change for each year and visualized the data, which revealed a clear upward trend in global temperatures over the past decades. This analysis highlighted the ongoing reality and severity of climate change.

Regional Temperature Increases:
By identifying the top 10 countries with the highest temperature increases in 2022, I demonstrated which regions are experiencing the most significant warming. This regional analysis is crucial for understanding the disproportionate impacts of climate change.

Hemispherical Comparisons:
I compared temperature trends between the northern and southern hemispheres. Both showed upward trends, with the northern hemisphere experiencing more pronounced increases. This comparison helps in understanding the geographical variability in climate change effects.

Rate of Heating and Acceleration:
Using linear and quadratic regression models, I analyzed the rate of global temperature increase and checked for signs of acceleration. The results indicated not only a continuous rise in temperatures but also an accelerating rate of increase, which underscores the urgency of addressing climate change.

Future Temperature Projections:
I employed exponential smoothing techniques to forecast future temperature changes. My projections suggest that the global temperature will surpass the critical 1.5°C threshold by 2025 if current trends continue. This forecast serves as a stark warning of the potential future impacts of climate change.

Visualizations:
I created a variety of visualizations to effectively communicate my findings. These included line graphs of temperature trends, bar plots of regional temperature increases, and interactive globe heatmaps showing overall temperature changes. These visual tools are essential for conveying complex data in an accessible and engaging manner.
Through these analyses, I provided a detailed and multifaceted view of how global temperatures are changing, which regions are most affected, and what the future might hold if current trends persist. This project underscores the critical importance of taking immediate and substantial action to mitigate the impacts of climate change.
