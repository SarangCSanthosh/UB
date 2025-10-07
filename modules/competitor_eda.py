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
    file_id = "1m5pCj46_eVXIiYOJdxcKeYhCO8qWfSR7"
    output_file = "data.csv"

    if not os.path.exists(output_file):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_file, quiet=False)

    df = load_csv(output_file)
    VOLUME_COL = "VOLUME"
    SKU_COL = "DBF_SKU"

    df, DATE_COL = prepare_dates(df)
    df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)

    # --------------------------
    # Sidebar filters
    # --------------------------
    st.sidebar.header("Filters")
    filter_type = st.sidebar.radio("Filter Data By:", ["Year(s)", "Date Range"])

    if filter_type == "Year(s)":
        year_choice = st.sidebar.multiselect(
            "Select Year(s)",
            options=sorted(df["Year"].dropna().unique()),
            default=sorted(df["Year"].dropna().unique())
        )
        if year_choice:
            df = df[df["Year"].isin(year_choice)]
    else:
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

    # ---- Tab 1: Brand Distribution (no selection) ----
    with tab1:
        st.subheader("Brand Distribution")
        brand_sales = df.groupby("Brand")[VOLUME_COL].sum().reset_index()
        brand_sales = brand_sales[brand_sales["Brand"] != "OTHER"]
        brand_sales["Percentage"] = (brand_sales[VOLUME_COL] / brand_sales[VOLUME_COL].sum() * 100).round(2)

        granularity = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab1")
        y_col = "Percentage" if granularity == "Percentage" else VOLUME_COL
        y_title = "Volume Share (%)" if y_col == "Percentage" else "Volume"

        fig = px.bar(
            brand_sales,
            x="Brand",
            y=y_col,
            text=brand_sales[y_col].round(2),
            title="Volume Distribution Across Brands",
            color="Brand",
            labels={y_col: y_title}
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(brand_sales.set_index("Brand")[[VOLUME_COL, "Percentage"]].round(2))

    # ---- Tab 2: Pack Size Wise Analysis ----
    with tab2:
        st.subheader("Pack Size Wise Distribution per Brand")

        # Extract segment including packaging
        def extract_segment(sku):
            sku = str(sku).upper().strip()
            match = re.search(r'(\d+\s?ML(?:\.?\s?CANS?)?)', sku)
            return match.group(1) if match else "Other Segment"

        df["Segment"] = df[SKU_COL].apply(extract_segment)

        # Brand selection via radio buttons
        brands = sorted(df["Brand"].unique())
        selected_brand = st.radio("Select Brand", options=brands, index=0, horizontal=True)

        df_brand = df[df["Brand"] == selected_brand]
        pack_sales = df_brand.groupby("Segment")[VOLUME_COL].sum().reset_index().sort_values(by=VOLUME_COL, ascending=False)
        pack_sales["Percentage"] = (pack_sales[VOLUME_COL] / pack_sales[VOLUME_COL].sum() * 100).round(2)

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
        st.dataframe(pack_sales.set_index("Segment")[[VOLUME_COL, "Percentage"]].round(2))

    # ---- Tab 3: Bottle vs Can ----
    with tab3:
        st.subheader("Bottle vs Can Distribution")
        def classify_pack_type(sku):
            sku = str(sku).upper()
            if "CAN" in sku or "CANS" in sku:
                return "CAN"
            elif "ML" in sku:
                return "BOTTLE"
            else:
                return "OTHER"

        df["Pack_Type"] = df[SKU_COL].apply(classify_pack_type)
        pack_type_sales = df.groupby("Pack_Type")[VOLUME_COL].sum().reset_index().sort_values(by=VOLUME_COL, ascending=False)

        fig_packtype = px.pie(
            pack_type_sales,
            names="Pack_Type",
            values=VOLUME_COL,
            title="Bottle vs Can Volume Distribution",
            hole=0.4,
            color="Pack_Type"
        )
        fig_packtype.update_traces(textinfo="percent+label")
        fig_packtype.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        st.plotly_chart(fig_packtype, use_container_width=True)
        st.dataframe(pack_type_sales.set_index("Pack_Type")[[VOLUME_COL]].round(2))

    # ---- Tab 4: Top SKUs ----
    with tab4:
        st.subheader("Top SKUs by Volume")
        sku_sales = df.groupby(SKU_COL)[VOLUME_COL].sum().reset_index().sort_values(by=VOLUME_COL, ascending=False)

        top_n = st.slider("Show Top-N SKUs", 5, 20, 10)
        granularity = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab4")
        sku_data = sku_sales.head(top_n).copy()
        if granularity == "Percentage":
            sku_data["Value"] = (sku_data[VOLUME_COL] / sku_sales[VOLUME_COL].sum() * 100).round(2)
            y_col = "Value"
        else:
            sku_data["Value"] = sku_data[VOLUME_COL].round(2)
            y_col = "Value"

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
        st.dataframe(sku_data.set_index(SKU_COL)[["Value"]].round(2))

    # ---- Tab 5: Monthly Trend ----
    with tab5:
        st.subheader("Monthly Trend by Brand")
        trend = df.groupby(["YearMonth", "Brand"])[VOLUME_COL].sum().reset_index()
        trend["YearMonth"] = trend["YearMonth"].astype(str)
        trend["Value"] = trend[VOLUME_COL]

        fig_trend = px.line(
            trend,
            x="YearMonth",
            y="Value",
            color="Brand",
            markers=True,
            title="Monthly Brand Trend (Absolute)"
        )
        fig_trend.update_yaxes(title_text="Volume")
        fig_trend.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        st.plotly_chart(fig_trend, use_container_width=True)


if __name__ == "__main__":
    run()
