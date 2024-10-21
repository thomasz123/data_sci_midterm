import streamlit as st

st.title(":blue[Conclusion]🔬")
st.subheader("Data Quality")
st.write("Our dataset set was pulled from a website that only allowed the download of 1000 columns. As a result, our models were trained on only 1000 days, equating to less than 3 years of data.  Ideally, we would have had much more data to have a more accurate model.")
st.subheader("Model Related Improvements")
"For the linear regression, not all variables are helpful to predict temperature. Ex: percipitation has a negative R2 value, which means the model predicts temperature with less accuracy than a constant value (horizontal line). However, the accuracy of the model's predictions improves as more and more variables are included.   "
"For the logistic regression, the model predicts temperature with an accuracy of about 60%"
st.subheader("Next Steps")
st.markdown("This is a very simple analysis of weather data.  Based on our temperature predictions using different variables, it's clear to see that our predictions became more accurate the more variables were involved.  There are also countless more qualitative variables that we were not able to use in our analysis.  Meteorology is an extremely complex field focused on weather forecasting, and their predictions are much more accurate than ours will be.")