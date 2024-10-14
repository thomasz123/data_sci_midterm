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

#Linear Regression
with st.spinner('Loading scatterplot...'):
    fig, ax = plt.subplots(figsize = (10,6))
    plt.title("Actual vs Predicted Temperatures",fontsize=25)
    plt.xlabel("Actual Temperatures",fontsize=18)
    plt.ylabel("Predicted Temperatures", fontsize=18)
    plt.scatter(x=y_test,y=prediction)

    st.pyplot(fig)




#Linear Regression (Temp vs. Dew)
st.title("Predicting Temperature based on Dew")
list_columns = df.columns

# Step 1: Splitting the dataset into X and y
X1 = df[["dew"]]  # Use double brackets to make X1 a 2D array
y1 = df["temp"]

# Step 2: Splitting into 4 chunks: X_train, X_test, y_train, y_test
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2)

# Step 3: Initialize the LinearRegression model
lr = LinearRegression()

# Step 4: Train the model
lr.fit(X1_train, y1_train)

# Step 5: Predictions
predictions = lr.predict(X1_test)

# Step 6: Plot the results
with st.spinner('Loading scatterplot...'):
    # fig, ax = plt.subplots(figsize=(10, 6))
    # plt.title("Temperature and Dew", fontsize=25)
    # plt.xlabel("Dew", fontsize=18)
    # plt.ylabel("Temp", fontsize=18)
    
    # # Correct the axes for the scatter plot
    # plt.scatter(x=X1_test, y=y1_test, color='blue', label='Actual')
    # #plt.scatter(x=X1_test, y=predictions, color='red', label='Predicted')
    # plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    
    # plt.legend()
    

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
    plt.xlabel('Actual CO2 Emissions')
    plt.ylabel('Predicted CO2 Emissions')
    plt.title('Actual vs Predicted CO2 Emissions')
    plt.show()
    
    st.pyplot(fig)

#Ranking variables
coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
coeff_df
feature_names = [f'Feature_{i}' for i in list(X.columns)]
df_X = pd.DataFrame(X, columns=feature_names)
# Coefficients represent the importance in linear regression
coefficients = lr.coef_

# Making the coefficients positive to compare magnitude
importance = np.abs(coefficients)

# Plotting feature importance with feature names
plt.figure(figsize=(10, 8))
plt.barh(feature_names, importance)
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance (Linear Regression)')
plt.show()


#Stp6 Evaluation

mae=metrics.mean_absolute_error(predictions,y_test)
r2=metrics.r2_score(predictions,y_test)

st.write("Mean Absolute Error:",mae)
st.write("R2 output:",r2)

#Logistic Regression
