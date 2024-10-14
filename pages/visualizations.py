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
import time

df = pd.read_csv("weather.csv")
df_cleaned = df.drop(["temp", "datetime"], axis = 1)

st.header("Visualization")

with st.spinner('Loading visualization charts...'):
    #pairplot
    st.pyplot(sns.pairplot(df_cleaned))

    #heatmap
    corr_matrix= df_cleaned.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, linewidths=0.5)
    st.pyplot(fig)
# st.success("Done!")


