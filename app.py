import streamlit as st
from modules import primary_eda, secondary_eda, competitor_eda

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(page_title="UBL Dashboard", layout="wide")

# ===============================
# SIDEBAR
# ===============================
with st.sidebar:
    # --- Add company logo ---
    st.image("assets/images.jpg", use_container_width=True)  # ← path to your logo

    # Optional: add a line separator
    st.markdown("---")

    # Sidebar title
    st.title("UBL Dashboard")

    # Navigation
    page = st.radio(
        "Choose Dataset",
        ["Primary Dataset", "Secondary Dataset", "Comparative Analysis"]
    )

# ===============================
# PAGE ROUTING
# ===============================
if page == "Primary Dataset":
    primary_eda.run()
elif page == "Secondary Dataset":
    secondary_eda.run()
elif page == "Comparative Analysis":
    competitor_eda.run()
