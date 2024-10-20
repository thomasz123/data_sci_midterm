import streamlit as st
from PIL import Image
import codecs
import streamlit.components.v1 as components

#page navigations

about = st.Page("About.py", title="About", icon="📝")
visualizations = st.Page("Visualizations.py", title="Visualizations ", icon="📊")
predictions = st.Page("Predictions.py", title="Predictions", icon="🤖")
conclusion = st.Page("conclusion.py", title ="Conclusion", icon = "🤔")

pg = st.navigation([about, visualizations, predictions, conclusion])
#st.set_page_config(page_title="About", page_icon="📝")
pg.run()
