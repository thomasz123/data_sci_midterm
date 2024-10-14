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


# coeff_df = pd.DataFrame(lin_reg.coef_, X.columns, columns=['Coefficient'])
# coeff_df
# feature_names = [f'Feature_{i}' for i in list(X.columns)]
# df_X = pd.DataFrame(X, columns=feature_names)
# # Coefficients represent the importance in linear regression
# coefficients = lin_reg.coef_

# # Making the coefficients positive to compare magnitude
# importance = np.abs(coefficients)

# # Plotting feature importance with feature names
# plt.figure(figsize=(10, 8))
# plt.barh(feature_names, importance)
# plt.xlabel('Absolute Coefficient Value')
# plt.title('Feature Importance (Linear Regression)')
# plt.show()

#linear regression w variables (not expected vs predicted)

st.title("Predicting Temperature based on Dew")
list_columns = df.columns
#input_lr = st.multiselect("Select variables:",list_columns,["___", "___"])

#df2 = df[input_lr]

# Step 1 splitting the dataset into X and y
X1= df["dew"]
# target variable
y= df["temp"]

# Step 2 splitting into 4 chuncks X_train X_test y_train y_test
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# Step3 Initialize the LinearRegression
lr = LinearRegression()

# Step4 Train model
lr.fit(X_train,y_train)

#Step5 Prediction 
predictions = lr.predict(X_test)

# Plot the original data points
plt.scatter(x, y, label='Data points')

# Plot the linear regression line
plt.plot(x, predictions, label='Regression line')


#Stp6 Evaluation

mae=metrics.mean_absolute_error(predictions,y_test)
r2=metrics.r2_score(predictions,y_test)

st.write("Mean Absolute Error:",mae)
st.write("R2 output:",r2)

#Logistic Regression
