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

st.markdown("##### Explanation of KEY VARIABLES 📓")
st.markdown("- HUMIDITY: The amount of water vapor in the air.")
st.markdown("- WIND SPEED: Speed of the wind at a particular time.")
st.markdown("- PRESSURE: Atmospheric pressure.")
st.markdown("- CLOUD COVER: Percentage of the sky covered by clouds.")
st.markdown("- PRECIPITATION: The amount of liquid or frozen water that falls to the Earth's surface, measured over a specific period. It includes rain, snow, sleet, and hail.")
st.markdown("- SEA LEVEL PRESSURE: The atmospheric pressure at sea level. It's a standardized measure used in meteorology to compare pressures from different elevations and is crucial for understanding weather patterns and forecasting.")
st.markdown("- SOLAR RADIATION: The amount of solar energy hitting the Earth’s surface.")

st.markdown("### Description of Data")
st.markdown("### Description of Data")
df = pd.read_csv("weather_data.csv")
st.dataframe(df.describe())
st.markdown("🔍 **Observation**: The dataset provides a comprehensive set of weather-related features across different regions and dates, offering valuable insights into temperature variations.")
    
st.markdown("### Missing Values")
dfnull = df.isnull().sum()/len(df)*100
totalmiss = dfnull.sum().round(2)
st.write("Percentage of total missing values:", totalmiss)
st.write(dfnull)
if totalmiss == 0.0:
    st.success("✅ No missing values, which enhances the reliability of our data analysis.")
 else:
    st.warning("Data quality may be affected by missing values greater than 30%.")
    
st.markdown("### Completeness")
nonmissing = (df.notnull().sum().round(2))
completeness = round(sum(nonmissing)/len(df), 2)
st.write("Completeness ratio:", completeness)
    if completeness >= 0.85:
        st.success("✅ High completeness ratio ensures sufficient data for analysis.")
    else:
        st.warning("Low completeness ratio could impact the analysis accuracy.")
    
elif app_mode == '02 Visualization':
    df = pd.read_csv("weather_data.csv")
    variables = st.sidebar.radio("Pick the variable", ["HUMIDITY", "WIND_SPEED", "PRESSURE", "CLOUD_COVER", "SOLAR_RADIATION"])

    if variables == "HUMIDITY":
        st.header("Humidity and Temperature")
        crosstab = pd.crosstab(df['HUMIDITY'], df['TEMPERATURE'])
        fig, ax = plt.subplots()
        crosstab.plot(kind='bar', width=0.8, ax=ax)
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        st.pyplot(fig)

    elif variables == "WIND_SPEED":
        st.header("Wind Speed and Temperature")
        crosstab = pd.crosstab(df['WIND_SPEED'], df['TEMPERATURE'])
        fig, ax = plt.subplots()
        crosstab.plot(kind='bar', width=0.8, ax=ax)
        for p in ax.patches:
            ax.annotate(str(p.get_height()), (p.get_x() * 1.005, p.get_height() * 1.005))
        st.pyplot(fig)

    # Additional visualizations can be added similarly for other variables

elif app_mode == '03 Prediction':
    image_2 = Image.open('prediction_image.png')
    st.image(image_2, width=300)
    
    # Data Preprocessing
    df = pd.read_csv("weather_data.csv")
    X = df.drop('TEMPERATURE', axis=1)
    y = df['TEMPERATURE']
    
    # Convert categorical columns using get_dummies (one-hot encoding)
    X = pd.get_dummies(X)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model_choice = st.sidebar.selectbox('Select to see:', ['KNN', 'Random Forest', 'Comparison Analysis'])
    
    if model_choice == 'KNN':
        knn = KNeighborsRegressor(n_neighbors=3)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = knn.score(X_test_scaled, y_test)
        st.title("Prediction - k-nearest neighbors Model:")
        st.write(f"Model Accuracy: {accuracy*100:.2f}%")
    
    elif model_choice == 'Random Forest':
        st.title("Prediction - Random Forest Regressor Model:")
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        accuracy = rf.score(X_test, y_test)
        st.write(f"Model Accuracy: {accuracy*100:.2f}%")

    

