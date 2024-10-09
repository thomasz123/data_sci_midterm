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

st.title("Data Science App")

# image_path = Image.open("wine.jpeg")
# st.image(image_path)

# app_page = st.sidebar.selectbox("Select Page", ['Data Exploration', 'Visualization', 'Prediction'])
df = pd.read_csv("weather.csv")

df = df.drop("datetime", axis = 1)  

st.pyplot(sns.pairplot(df))
df2 = df.drop("temp", axis = 1)



X = df2
y = df["temp"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
lr = LinearRegression()

lr.fit(X_train, y_train)

prediction = lr.predict(X_test)

mae = metrics.mean_absolute_error(prediction, y_test)
st.write("Mean Absolute Error:", mae)

r2 = metrics.r2_score(prediction,y_test)
st.write("R2:", r2)
