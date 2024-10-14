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
df = df.drop("datetime", axis = 1)
df_cleaned = df.drop("temp", axis = 1)

st.title("Visualization")

with st.spinner('Loading visualization charts. Please wait...'):
    #pairplot
    st.header('Pairplot')
    st.pyplot(sns.pairplot(df))

    #heatmap
    st.header('Heatmap')
    corr_matrix= df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
    st.pyplot(fig)

    #scatterplot
    st.header("Scatterplot")
    #sns.scatterplot(data = df, x = "", y = "Temperature Forecast", hue = "Category")
# st.success("Done!")


