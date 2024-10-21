import streamlit as st

st.title(":blue[Conclusion]🔬")
st.subheader("Data Quality")
st.write("Our dataset set was pulled from a website that only allowed the download of 1000 columns. As a result, our models were trained on only 1000 days, equating to less than 3 years of data.  Ideally, we would have had much more data to have a more accurate model.")
st.subheader("Model Related Improvements")
"For the linear regression, not all variables are helpful to predict temperature. Ex: Cloud cover has a low feature importance or coefficient value and a low r2 value. We can infer from that cloud cover is not that useful for prediction the temperature. "
"For the logistic regression, we noticed that more variables does not necessarily mean a more accurate model. From this, we suggest removing some variables in order to create a more accurate value. "
st.subheader("Next Steps")
st.markdown("This is a very simple analysis of weather data.  Based on our temperature predictions using different variables, it's clear to see that our predictions became more accurate the more variables were involved.  There are also countless more qualitative variables that we were not able to use in our analysis.  Meteorology is an extremely complex field focused on weather forecasting, and their predictions are much more accurate than ours will be.")