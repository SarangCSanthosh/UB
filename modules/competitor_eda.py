import streamlit as st
import pandas as pd
import plotly.express as px
import re
import gdown
import os

# ===============================
# Utility functions
# ===============================
@st.cache_data
def load_csv(path_or_file):
    return pd.read_csv(path_or_file)

def prepare_dates(df, date_col="ACTUAL_DATE"):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year
    df["YearMonth"] = df[date_col].dt.to_period("M")
    df["Quarter"] = df[date_col].dt.to_period("Q")
    return df, date_col

# ===============================
# Exact SKU to Brand mapping
# ===============================
SKU_GROUP_MAPPING = {
    # KF
    "KF 330 ML.": "KF LAGER",
    "KF 330 ML. CANS": "KF LAGER",
    "KF 500 ML. CANS": "KF LAGER",
    "KF 650 ML.": "KF LAGER",

    # KF STORM
    "KF STORM 500 ML. CANS": "KF STORM",
    "KF STORM 650 ML.": "KF STORM",

    # KF ULTRA
    "KF ULTRA 330 ML.": "KF ULTRA",
    "KF ULTRA 500 ML. CAN": "KF ULTRA",
    "KF ULTRA 650 ML.": "KF ULTRA",

    # KF ULTRA MAX
    "KF ULTRA MAX 330 ML.": "KF ULTRA MAX",
    "KF ULTRA MAX 500 ML. CANS": "KF ULTRA MAX",
    "KF ULTRA MAX 650 ML.": "KF ULTRA MAX",

    # KF ULTRA WITBIER
    "KF ULTRA WITBIER 330 ML.": "KF ULTRA WITBIER",
    "KF ULTRA WITBIER 500 ML. CANS": "KF ULTRA WITBIER",
    "KF ULTRA WITBIER 650 ML.": "KF ULTRA WITBIER",

    # KFS
    "KFS 330 ML.": "KF STRONG",
    "KFS 330 ML. CANS": "KF STRONG",
    "KFS 500 ML. CANS": "KF STRONG",
    "KFS 650 ML.": "KF STRONG",

    # Bullet
    "BSSB 300 ML.": "Bullet",
    "BSSB 330 ML.": "Bullet",
    "BSSB 330 ML. CANS": "Bullet",
    "BSSB 650 ML.": "Bullet",
}

def map_sku_to_brand(sku):
    return SKU_GROUP_MAPPING.get(sku.strip().upper(), "OTHER")

# ===============================
# Main app
# ===============================
def run():
    st.title("Competitor Analysis Dashboard")

    # --------------------------
    # LOAD DATA via gdown
    # --------------------------

    # Google Drive file ID (replace with your actual file ID)
    file_id = "1m5pCj46_eVXIiYOJdxcKeYhCO8qWfSR7"

    # Destination filename
    output_file = "data.csv"

    # Download file from Google Drive if not already downloaded
    if not os.path.exists(output_file):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_file, quiet=False)

    # Load CSV from local file
    df = load_csv(output_file)
    VOLUME_COL = "VOLUME"
    SKU_COL = "DBF_SKU"

    df, DATE_COL = prepare_dates(df)
    # Map SKU → Brand before any tab uses it
    df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)


    # --------------------------
    # Sidebar filters
    # --------------------------
    st.sidebar.header("Filters")

    # Filter type selection: Year(s) or Date Range
    filter_type = st.sidebar.radio("Filter Data By:", ["Year(s)", "Date Range"])

    if filter_type == "Year(s)":
        year_choice = st.sidebar.multiselect(
            "Select Year(s)",
            options=sorted(df["Year"].dropna().unique()),
            default=sorted(df["Year"].dropna().unique())
        )
        if year_choice:
            df = df[df["Year"].isin(year_choice)]

    else:  # Date Range
        min_date = df[DATE_COL].min()
        max_date = df[DATE_COL].max()

        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=[min_date, max_date],
            min_value=min_date,
            max_value=max_date
        )

        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df[DATE_COL] >= start_date) & (df[DATE_COL] <= end_date)]

    # --------------------------
    # TABS
    # --------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Brand Distribution",
        "Pack Size Wise Analysis",
        "Bottle v/s Cans",
        "Top SKUs",
        "Monthly Trend"
    ])

    # ---- Tab 1: Brand Distribution ----
    with tab1:
        st.markdown("### Question: Which brands dominate in shipment volume?")
        st.subheader("Brand Distribution")
    
        # --- Aggregate volume by Brand ---
        brand_sales = df.groupby("Brand")[VOLUME_COL].sum().round(0).reset_index()
        brand_sales = brand_sales[brand_sales["Brand"] != "OTHER"]
        brand_sales["Percentage"] = (brand_sales[VOLUME_COL] / brand_sales[VOLUME_COL].sum() * 100).round(0)
    
        # --- Group brands with <1% as OTHERS ---
        major_brands = brand_sales[brand_sales["Percentage"] >= 3]
        minor_brands = brand_sales[brand_sales["Percentage"] < 3]
    
        if not minor_brands.empty:
            others_sum = minor_brands[VOLUME_COL].sum().round(0)
            others_pct = minor_brands["Percentage"].sum().round(0)
            others_row = pd.DataFrame({"Brand": ["OTHERS"], VOLUME_COL: [others_sum], "Percentage": [others_pct]})
            brand_sales = pd.concat([major_brands, others_row], ignore_index=True)
    
        brand_sales = brand_sales.sort_values(by=VOLUME_COL, ascending=False)
    
        # --- Toggle between Absolute and Percentage view ---
        granularity = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab1")
        y_col = "Percentage" if granularity == "Percentage" else VOLUME_COL
        y_title = "Volume Share (%)" if y_col == "Percentage" else "Volume"
    
        # --- Bar Chart for Main Brands ---
        fig = px.bar(
            brand_sales,
            x="Brand",
            y=y_col,
            text=brand_sales[y_col].round(2),
            title="Volume Distribution Across Brands (Grouped by OTHERS < 1%)",
            color="Brand",
            labels={y_col: y_title}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
        # --- Pie Chart: Breakdown of OTHERS ---
        if not minor_brands.empty:
            st.markdown("#### Breakdown of Brands Under 'OTHERS' (< 1%)")
            fig_pie = px.pie(
                minor_brands,
                names="Brand",
                values=VOLUME_COL,
                title="Distribution of Brands Grouped Under 'OTHERS'",
                color="Brand",
                hole=0.4
            )
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(height=400, margin=dict(t=50, b=50, l=50, r=50))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("No brands fall under the 'OTHERS' (<1%) category for the current selection.")
    
        # --- Data Table ---
        st.dataframe(brand_sales.set_index("Brand")[[VOLUME_COL, "Percentage"]].round(0))
       
        with st.container():
            st.markdown("""
### **Answer:**
All efforts must be focused on protecting, supporting, and potentially growing KFS. This brand is the core of the entire operation. Bullet and KF are the only two other brands that matter. Resources should be allocated to these two to increase their share and slightly diversify the revenue base away from the KFS dependency.
""")


    # ---- Tab 2: Pack Size Wise Analysis ----
    with tab2:
        st.markdown("###  Question: What are the top-selling SKUs?")
        st.subheader("Pack Size Wise Volume Distribution")
    
        # Extract segment including packaging
        def extract_segment(sku):
            sku = str(sku).upper().strip()
            match = re.search(r'(\d+\s?ML(?:\.?\s?CANS?)?)', sku)
            segment = match.group(1) if match else "Other Segment"
            
            # Normalize segment names: remove trailing '.' and unify 'CAN'/'CANS'
            segment = segment.replace(".", "").replace("CANS", "CAN").strip()
            return segment
            #sku = str(sku).upper().strip()
            #match = re.search(r'(\d+\s?ML(?:\.?\s?CANS?)?)', sku)
            #return match.group(1) if match else "Other Segment"

        df["Segment"] = df[SKU_COL].apply(extract_segment)

        # Brand selection via radio buttons (include "All")
        brands = ["All"] + sorted(df["Brand"].unique())
        selected_brand = st.radio("Select Brand", options=brands, index=0, horizontal=True)

        if selected_brand == "All":
            df_brand = df[df["Brand"] != "OTHER"]
        else:
            df_brand = df[df["Brand"] == selected_brand]

        pack_sales = df_brand.groupby("Segment")[VOLUME_COL].sum().round(0).reset_index().sort_values(by=VOLUME_COL, ascending=False)
        pack_sales["Percentage"] = (pack_sales[VOLUME_COL] / pack_sales[VOLUME_COL].sum() * 100).round(0)

        granularity = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab2")
        y_col = "Percentage" if granularity == "Percentage" else VOLUME_COL
        y_title = "Volume Share (%)" if y_col == "Percentage" else "Volume"

        fig_pack = px.bar(
            pack_sales,
            x="Segment",
            y=y_col,
            text=pack_sales[y_col].round(2),
            title=f"{selected_brand} Pack Size Distribution",
            color="Segment",
            labels={y_col: y_title}
        )
        fig_pack.update_traces(textposition="outside")
        fig_pack.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        fig_pack.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_pack, use_container_width=True)
        st.dataframe(pack_sales.set_index("Segment")[[VOLUME_COL, "Percentage"]].round(0))
        st.markdown("""
### **Answer:**
The 650 ML pack size (light blue bar) is the undisputed leader. Due to its immense volume, every effort should be made to optimise the production, filling, distribution, and marketing of the 650 ML pack for maximum efficiency and cost savings. Marginal improvements here will yield massive absolute returns.The reliance on a single pack size presents a high risk. Strategies to boost the 330 ML CANS and 500 ML CANS should be explored to gradually diversify the volume base, providing resilience against potential market shifts, targeting the 650 ML format.
""")


    # ---- Tab 5: Bottle vs Can Distribution ----
    with tab3:
        st.markdown("###  Question: How is shipment volume split between bottles and cans?")
        st.subheader("Bottle vs Can Distribution")
    
        # Helper function to classify pack type
        def classify_pack_type(sku):
            sku = str(sku).upper()
            if "CAN" in sku or "CANS" in sku:
                return "CAN"
            elif "ML" in sku:
                return "BOTTLE"
            else:
                return "OTHER"
    
        df["Pack_Type"] = df[SKU_COL].apply(classify_pack_type)
    
        pack_type_sales = df.groupby("Pack_Type")[VOLUME_COL].sum().round(0).reset_index()
        pack_type_sales = pack_type_sales.sort_values(by=VOLUME_COL, ascending=False)
    
        # Pie chart for intuitive visualization
        fig_packtype = px.pie(
            pack_type_sales.round(0),
            names="Pack_Type",
            values=VOLUME_COL,
            title="Bottle vs Can Volume Distribution",
            hole=0.4,
            color="Pack_Type"
        )
        fig_packtype.update_traces(textinfo="percent+label")
        fig_packtype.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        st.plotly_chart(fig_packtype, use_container_width=True)
    
        # Data table below the chart
        st.dataframe(pack_type_sales.set_index("Pack_Type")[[VOLUME_COL]].round(0))
        st.markdown("""
### **Answer:**
BOTTLE (light blue) is the primary packaging format, accounting for a massive 83.5% of the total volume. The business's entire supply chain, from manufacturing and filling to inventory, logistics, and recycling/returns, is overwhelmingly structured around the Bottle format. Operational efficiency efforts should naturally be centered here, as marginal improvements in the bottle process will yield the greatest overall volume impact.
""")

  
    


    # ---- Tab 2: Top SKUs with Granularity ----
    with tab4:
        st.markdown("###  Question: What are the top-performing SKUs by volume?")
        st.subheader("Top SKUs by Volume")
        sku_sales = df.groupby(SKU_COL)[VOLUME_COL].sum().round(0).reset_index()
        sku_sales = sku_sales.sort_values(by=VOLUME_COL, ascending=False)

        top_n = st.slider("Show Top-N SKUs", 5, 20, 10)

        # Granularity toggle for Top SKUs
        granularity_sku = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab4")
        sku_data = sku_sales.head(top_n).copy()
        if granularity_sku == "Percentage":
            total = sku_sales[VOLUME_COL].sum()
            sku_data["Value"] = ((sku_data[VOLUME_COL] / total) * 100).round(0)
            y_col = "Value"
            y_title = "Volume Share (%)"
        else:
            sku_data["Value"] = sku_data[VOLUME_COL].round(0)
            y_col = "Value"
            y_title = "Volume"

        fig_sku = px.bar(
            sku_data,
            x=SKU_COL,
            y=y_col,
            text=y_col,
            title=f"Top {top_n} SKUs",
        )
        fig_sku.update_traces(textposition="outside")
        fig_sku.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        fig_sku.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_sku, use_container_width=True)
        st.dataframe(sku_data.set_index(SKU_COL)[["Value"]].round(0))
        st.markdown("""
### **Answer:**
The company’s performance is highly dependent on the stability and success of the KFS 650ML SKU. Protecting this product from competition is required. While KFS 650ML is the leader, the three largest SKUs (KFS 650ML, Bullet 650ML, and KFS 330 ML Cans) should receive the most detailed operational focus, as they represent the foundation of the total volume.
""")


        
    # ---- Tab 3: Monthly Trend (Absolute Only) ----
    with tab5:
        st.markdown("###  Question: How do shipment volumes change month by month for each brand?")
        st.subheader("Monthly Trend by Brand")

        df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)
        trend = df.groupby(["YearMonth", "Brand"])[VOLUME_COL].sum().round(0).reset_index()
        trend["YearMonth"] = trend["YearMonth"].astype(str)

        trend["Value"] = trend[VOLUME_COL].round(0)
        y_title = "Volume"

        fig_trend = px.line(
            trend,
            x="YearMonth",
            y="Value",
            color="Brand",
            markers=True,
            title="Monthly Brand Trend (Absolute)"
        )
        fig_trend.update_yaxes(title_text=y_title)
        fig_trend.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("""
### **Answer:**
- The KFS brand (yellow line) dominates all other brands in absolute volume, consistently operating at a level 2 to 3 times higher than the nearest competitor. Peaks occur in the late Spring/Early Summer reaching volumes of approximately 5,000 to 5,500 units . The 2024 season appears weaker than 2023. The 2024 peak was lower than the 2023 peak, and the post-peak declines were noticeably lower than the equivalent period in 2023.
- The overall volume for Bullet shows no sustained growth and appears stagnant year-over-year.
""")

# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
