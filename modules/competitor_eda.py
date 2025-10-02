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
    "KF 330 ML.": "KF",
    "KF 330 ML. CANS": "KF",
    "KF 500 ML. CANS": "KF",
    "KF 650 ML.": "KF",

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
    "KFS 330 ML.": "KFS",
    "KFS 330 ML. CANS": "KFS",
    "KFS 500 ML. CANS": "KFS",
    "KFS 650 ML.": "KFS",

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
            default=[df["Year"].max()]
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "Brand Distribution",
        "Pack Size Wise Analysis",
        "Top SKUs",
        "Monthly Trend"
    ])

    # ---- Tab 1: Brand Distribution ----
    with tab1:
        st.subheader("Brand Distribution")

        df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)

        # Extract segment including packaging
        def extract_segment(sku):
            sku = sku.upper().strip()
            match = re.search(r'(\d+\s?ML(?:\.?\s?CANS?)?)', sku)
            return match.group(1) if match else "Other Segment"

        df["Segment"] = df[SKU_COL].apply(extract_segment)

        # Aggregate volume by Brand and Segment
        brand_segment_sales = df.groupby(["Brand", "Segment"])[VOLUME_COL].sum().reset_index()
        brand_segment_sales = brand_segment_sales[brand_segment_sales["Brand"] != "OTHER"]

        # Brand selection buttons
        st.write("### Select Brand to View Segments")
        brands = sorted(brand_segment_sales["Brand"].unique())
        selected_brand = st.radio("Click a Brand", options=["All"] + brands, index=0, horizontal=True)

        # Granularity selection
        granularity = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab1")

        if selected_brand == "All":
            brand_sales = brand_segment_sales.groupby("Brand")[VOLUME_COL].sum().reset_index()
            if granularity == "Percentage":
                total = brand_sales[VOLUME_COL].sum()
                brand_sales["Value"] = (brand_sales[VOLUME_COL] / total) * 100
                y_col = "Value"
                y_title = "Volume Share (%)"
            else:
                brand_sales["Value"] = brand_sales[VOLUME_COL]
                y_col = "Value"
                y_title = "Volume"

            fig = px.bar(
                brand_sales,
                x="Brand",
                y=y_col,
                title="Volume Distribution Across Brands",
                text=y_col,
                labels={y_col: y_title},
                color="Brand"
            )
            st.dataframe(brand_sales.set_index("Brand")[[VOLUME_COL, "Value"]].round(2))
        else:
            brand_data = brand_segment_sales[brand_segment_sales["Brand"] == selected_brand].copy()
            if granularity == "Percentage":
                total = brand_data[VOLUME_COL].sum()
                brand_data["Value"] = (brand_data[VOLUME_COL] / total) * 100
                y_col = "Value"
                y_title = "Volume Share (%)"
            else:
                brand_data["Value"] = brand_data[VOLUME_COL]
                y_col = "Value"
                y_title = "Volume"

            fig = px.bar(
                brand_data,
                x="Segment",
                y=y_col,
                title=f"{selected_brand} Segment Distribution",
                text=y_col,
                labels={y_col: y_title},
                color="Segment"
            )
            st.dataframe(brand_data.set_index("Segment")[[VOLUME_COL, "Value"]].round(2))

        # Canvas size, margins, rotate labels
        fig.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        fig.update_xaxes(tickangle=-45)
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 2: Pack Size Wise Analysis ----
    with tab2:
        st.subheader("Pack Size Wise Volume Distribution")

        df["Segment"] = df[SKU_COL].apply(extract_segment)

        pack_sales = df.groupby("Segment")[VOLUME_COL].sum().reset_index()
        pack_sales = pack_sales.sort_values(by=VOLUME_COL, ascending=False)

        # Granularity toggle
        granularity_pack = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab_pack")

        if granularity_pack == "Percentage":
            total = pack_sales[VOLUME_COL].sum()
            pack_sales["Value"] = (pack_sales[VOLUME_COL] / total) * 100
            y_col = "Value"
            y_title = "Volume Share (%)"
        else:
            pack_sales["Value"] = pack_sales[VOLUME_COL]
            y_col = "Value"
            y_title = "Volume"

        fig_pack = px.bar(
            pack_sales,
            x="Segment",
            y=y_col,
            text=y_col,
            title="Pack Size Distribution",
            color="Segment"
        )
        fig_pack.update_traces(textposition="outside")
        fig_pack.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        fig_pack.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_pack, use_container_width=True)

        st.dataframe(pack_sales.set_index("Segment")[[VOLUME_COL, "Value"]].round(2))


    # ---- Tab 2: Top SKUs with Granularity ----
    with tab3:
        st.subheader("Top SKUs by Volume")
        sku_sales = df.groupby(SKU_COL)[VOLUME_COL].sum().reset_index()
        sku_sales = sku_sales.sort_values(by=VOLUME_COL, ascending=False)

        top_n = st.slider("Show Top-N SKUs", 5, 20, 10)

        # Granularity toggle for Top SKUs
        granularity_sku = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab2")
        sku_data = sku_sales.head(top_n).copy()
        if granularity_sku == "Percentage":
            total = sku_sales[VOLUME_COL].sum()
            sku_data["Value"] = (sku_data[VOLUME_COL] / total) * 100
            y_col = "Value"
            y_title = "Volume Share (%)"
        else:
            sku_data["Value"] = sku_data[VOLUME_COL]
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
        st.dataframe(sku_data.set_index(SKU_COL)[[VOLUME_COL, "Value"]].round(2))

    # ---- Tab 3: Monthly Trend (Absolute Only) ----
    with tab4:
        st.subheader("Monthly Trend by Brand")

        df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)
        trend = df.groupby(["YearMonth", "Brand"])[VOLUME_COL].sum().reset_index()
        trend["YearMonth"] = trend["YearMonth"].astype(str)

        trend["Value"] = trend[VOLUME_COL]
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

# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
