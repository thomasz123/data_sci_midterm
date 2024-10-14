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

df = pd.read_csv("weather.csv")
df = df.drop(["datetime"], axis = 1)  

df2 = df.drop("temp", axis = 1)
columns = df2.columns
input = st.multiselect("Select variables:",columns,["dew"])
df2 = df2[input]

st.title("Prediction")
X = df2
y = df["temp"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2)
lr = LinearRegression()

lr.fit(X_train, y_train)

prediction = lr.predict(X_test)

#Print MAE and R2 values
mae = metrics.mean_absolute_error(prediction, y_test)
st.write("Mean Absolute Error:", mae)
r2 = metrics.r2_score(prediction,y_test)
st.write("R2:", r2)

with st.spinner('Loading scatterplot...'):
    fig, ax = plt.subplots(figsize = (10,6))
    plt.title("Actual vs Predicted Temperatures",fontsize=25)
    plt.xlabel("Actual Temperatures",fontsize=18)
    plt.ylabel("Predicted Temperatures", fontsize=18)
    plt.scatter(x=y_test,y=prediction)

    st.pyplot(fig)