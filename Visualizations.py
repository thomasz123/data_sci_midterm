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

st.title("Visualizations")

tab1, tab2, tab3, tab4= st.tabs(["Pairplot", "Correlation Heatmap", "Scatterplot", "Histogram"])

with st.spinner("Loading visualizations..."):
    with tab1: 
        #with st.spinner('Loading pairplot. Please wait...'):
            #pairplot
            st.header('Pairplot')
            st.pyplot(sns.pairplot(df))
    with tab2: 
        with st.spinner('Loading heatmap. Please wait...'):
            #heatmap
            st.header('Heatmap')
            corr_matrix= df.corr()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
            st.pyplot(fig)

    with tab3: 
        with st.spinner('Loading scatterplot. Please wait...'):
            #scatterplot
            st.header("Scatterplot")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(data = df, x = "dew", y = "temp")
            st.pyplot(fig)

    with tab4: 
        with st.spinner('Loading histogram. Please wait...'):
            #histogram
            st.header("Histogram")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['temp'], kde=True)
            st.pyplot(fig)
            # plt.title("Distribution of Temperatures")



