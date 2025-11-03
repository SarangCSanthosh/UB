# app.py


import streamlit as st
from modules import primary_eda, secondary_eda, competitor_eda

# Set Streamlit app config
st.set_page_config(page_title="UBL Dashboard", layout="wide")

# Sidebar navigation
st.sidebar.title("UBL Dashboard")
page = st.sidebar.radio("Choose Dataset", ["Primary Dataset", "Secondary Dataset","Comparative Analysis"])

# Render selected page
if page == "Primary Dataset":
    primary_eda.run()
elif page == "Secondary Dataset":
    secondary_eda.run()
elif page == "Comparative Analysis":
    competitor_eda.run()
    
