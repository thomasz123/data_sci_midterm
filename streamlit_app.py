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

st.title("Tempature :blue[Prediction]")
st.subheader("Weather data and analysis")

st.markdown('##### WHY THIS TOPIC❓')
st.markdown('Tempature prediction and understanding tempature patterns are crucial for agriculture, travel, safety, and everyday life. The increasing unpredictability due to climate change makes precise weather forecasting even more important.')
st.markdown("##### OUR GOAL 🎯")
st.markdown("With this project, we aim to predict tempature  ")

image_path = Image.open("cloud.jpg") 
st.image(image_path)

st.write("Details about our project, explanation of database, etc")

df = pd.read_csv("weather.csv")
df = df.drop(["datetime","temp"], axis = 1)  
columns = df.columns
columns