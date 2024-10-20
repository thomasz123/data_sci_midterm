import streamlit as st

st.title(":blue[Conclusion]🔬")
st.subheader("Data Quality")
st.write("Our dataset set was pulled from a website that only allowed the download of 1000 columns. As a result, our models were trained on only 1000 days, equating to less than 3 years of data.")
st.subheader("Model Related Improvements")
"For each variable considered in the linear regression, the correlation coefficient was around 0.6. This means that, given our independent variables like dew, , we can only predict the temperature"
""
st.markdown("stuff")
st.subheader("Next Steps")
st.markdown("This is a very simple analysis of weather data.  Based on our temperature predictions using different variables, it's clear to see that our predictions became more accurate the more variables were involved.  There are also countless more qualitative variables that we were not able to use in our analysis.  Meteorology as as field ")