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

st.title("Tempature Prediction")
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
st.markdown("Our goal is to predict the temperature using multiple meteorological features and identify the most influential factors driving temperature changes.")
st.markdown("We focus on weather-related features like humidity, wind speed, pressure, and more, while aiming for an accurate forecast model.")

st.markdown('##### OUR DATA 📊')

st.markdown("##### Explanation of KEY VARIABLES 📓")
st.markdown("- HUMIDITY: The amount of water vapor in the air.")
st.markdown("- WIND_SPEED: Speed of the wind at a particular time.")
st.markdown("- PRESSURE: Atmospheric pressure.")
st.markdown("- CLOUD_COVER: Percentage of the sky covered by clouds.")
st.markdown("- P: Percentage of the sky covered by clouds.")
st.markdown("- SEA LEVEL PRESSURE: Percentage of the sky covered by clouds.")
st.markdown("- SOLAR_RADIATION: The amount of solar energy hitting the Earth’s surface.
sealevelpressure
")

st.markdown("### Description of Data")
df = pd.read_csv("weather_data.csv")
st.dataframe(df.describe())
st.markdown("🔍 **Observation**: The dataset provides a comprehensive set of weather-related features across different regions and dates, offering valuable insights into temperature variations.")
    

