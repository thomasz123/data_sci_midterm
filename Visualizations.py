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
cols = df_cleaned.columns

st.title(":blue[Visualizations]")

tab1, tab2, tab3, tab4= st.tabs(["Pairplot", "Correlation Heatmap", "Scatterplot", "Histogram"])

@st.cache_data 
def pairplot():
    st.pyplot(sns.pairplot(df))

@st.cache_data 
def heatmap():
    corr_matrix= df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
    st.pyplot(fig)
    #return fig

@st.cache_data
def histogram():
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['temp'], kde=True)
    st.pyplot(fig)


with tab1: #pairplot
    with st.spinner("Loading visualizations..."):
        st.header('Pairplot')
        pairplot()

with tab2: #heatmap
    with st.spinner("Loading visualizations..."):
        st.header('Heatmap')
        heatmap()

with tab3: #scatterplot
    st.header("Scatterplot")
    variable = st.radio("Pick one", cols)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data = df, x = variable, y = "temp")
    st.pyplot(fig)

with tab4: #histogram
    st.header("Histogram")
    histogram()
    



