# :earth_americas: Climate Change Dash

A simple Streamlit dashboard app displaying the Climate Change Analysis

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://climatedashboard.streamlit.app)

# Climate Change Dashboard Project

## Overview

This project was initiated to address key questions about climate change using data analysis:

- Is climate change real, and what is happening?
- Are certain countries or regions disproportionately affected?
- How does the temperature change compare between the northern and southern hemispheres?
- What is the rate of heating, and is it accelerating?
- Can we forecast future rates of heating and mean temperature changes?

## I. Data Collection and Cleaning

**Data Sources**:
- [Climate Change Indicators](https://www.kaggle.com/datasets/tarunrm09/climate-change-indicators/data)
- [Geo Countries](https://github.com/datasets/geo-countries/blob/master/data/countries.geojson)

**Cleaning Summary**:
- **Handling Missing Values**:
  - Corrected missing ISO2 codes.
  - Applied linear interpolation and forward/backward fill to handle missing temperature data.
- **Data Types Conversion**:
  - Converted temperature data columns to numeric types.
- **Normalization of Year Columns**:
  - Renamed year columns to remove the 'F' prefix for consistency.
- **Verification**:
  - Ensured no remaining missing values in the temperature data.
- **Normalization of Country Names**:
  - Matched country names by ISO3 identification to align with the geojson file.

## II. Data Analysis

### Temperature Trends Over Time
- **Analysis**:
  - Calculated and visualized mean temperature changes per year.
  - Results showed a clear upward trend in global temperatures over the past decades.

### Regional Temperature Increases
- **Analysis**:
  - Identified the top 10 countries with the highest temperature increases in 2022.
  - Highlighted regions experiencing the most significant warming.

### Hemispherical Comparisons
- **Analysis**:
  - Compared temperature trends between the northern and southern hemispheres.
  - Both showed upward trends, with the northern hemisphere experiencing more pronounced increases.

### Rate of Heating and Acceleration
- **Analysis**:
  - Used linear and quadratic regression models to analyze the rate of global temperature increase.
  - Found evidence of both a continuous rise and an accelerating rate of increase.

### Future Temperature Projections
- **Analysis**:
  - Employed exponential smoothing to forecast future temperature changes.
  - Projections suggest surpassing the 1.5°C threshold by 2025 if current trends continue.

### Visualizations
- **Tools Used**:
  - Line graphs for temperature trends.
  - Bar plots for regional temperature increases.
  - Interactive globe heatmaps for overall temperature changes.
- **Purpose**:
  - Effectively communicated findings and complex data in an accessible and engaging manner.

Through these analyses, the project provided a detailed view of global temperature changes, identified the most affected regions, and forecasted future trends. This work emphasizes the urgent need for substantial action to address climate change.
