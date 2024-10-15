import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from PIL import Image
import codecs
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# about = st.Page("About.py", title="About", icon="📝")
# visualizations = st.Page("Visualizations.py", title="Visualizations ", icon="📊")
# predictions = st.Page("Predictions.py", title="Predictions", icon="🤖")

df = pd.read_csv("weather.csv")
df = df.drop("datetime", axis = 1)
df_cleaned = df.drop("temp", axis = 1)
cols = df_cleaned.columns

st.header("Scatterplot")
variable = st.radio("Pick one", cols)
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data = df, x = variable, y = "temp")
st.pyplot(fig)

with st.spinner('Loading page...'):
    st.title(":blue[Tempature Prediction]")
    st.subheader("Weather data and analysis")

    image_path = Image.open("cloud.jpg") 
    st.image(image_path)

    st.write("Details about our project, explanation of database, etc")

    df = pd.read_csv("weather.csv")
    df = df.drop(["datetime","temp"], axis = 1)  
    columns = df.columns
    columns

    st.markdown('##### WHY THIS TOPIC❓')
    st.markdown('Accurate temperature prediction is critical for weather forecasting, agriculture, and various industries. Understanding the key factors influencing temperature helps in improving prediction accuracy, enhancing decision-making for farmers, city planners, and climate researchers.')
    st.markdown("##### OUR GOAL 🎯")
    st.markdown("Our goal is to predict the temperature using multiple meteorological features and identify the most influential factors driving temperature changes. We focus on weather-related features like humidity, wind speed, pressure, and more, while aiming for an accurate forecast model.")

    st.markdown('##### OUR DATA 📊')
    st.markdown("Our dataset contains historical weather data including variables such as temperature, humidity, wind speed, and atmospheric pressure. The dataset consists of thousands of daily observations from multiple weather stations.")

    st.markdown("##### Explanation of KEY VARIABLES 📓")
    st.markdown("- HUMIDITY: The amount of water vapor in the air.")
    st.markdown("- WIND SPEED: Speed of the wind at a particular time.")
    st.markdown("- PRESSURE: Atmospheric pressure.")
    st.markdown("- CLOUD COVER: Percentage of the sky covered by clouds.")
    st.markdown("- PRECIPITATION: The amount of liquid or frozen water that falls to the Earth's surface, measured over a specific period. It includes rain, snow, sleet, and hail.")
    st.markdown("- SEA LEVEL PRESSURE: The atmospheric pressure at sea level. It's a standardized measure used in meteorology to compare pressures from different elevations and is crucial for understanding weather patterns and forecasting.")
    st.markdown("- SOLAR RADIATION: The amount of solar energy hitting the Earth’s surface.")

    st.markdown("### Description of Data")


