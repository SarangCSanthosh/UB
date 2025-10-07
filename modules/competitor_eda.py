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

    # ---- Tab 1: Brand Distribution ----
    with tab1:
        st.markdown("###  Question: Which brands dominate in shipment volume?")
        st.subheader("Brand Distribution")

        df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)
        brand_segment_sales = df.groupby(["Brand"])[VOLUME_COL].sum().reset_index()
        brand_segment_sales = brand_segment_sales[brand_segment_sales["Brand"] != "OTHER"]

        # Brand selection buttons
        st.write("### View Brand Totals")
        brands = sorted(brand_segment_sales["Brand"].unique())
        selected_brand = st.radio("Click a Brand", options=["All"] + brands, index=0, horizontal=True)

        if selected_brand == "All":
            brand_sales = brand_segment_sales.copy()
            total_volume = brand_sales[VOLUME_COL].sum()
            brand_sales["Percentage"] = (brand_sales[VOLUME_COL] / total_volume * 100).round(2)

            y_col = "Percentage" if st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab1") == "Percentage" else VOLUME_COL
            y_title = "Volume Share (%)" if y_col == "Percentage" else "Volume"

            fig = px.bar(
                brand_sales,
                x="Brand",
                y=y_col,
                title="Volume Distribution Across Brands",
                text=brand_sales[y_col].round(2),
                labels={y_col: y_title},
                color="Brand"
            )
            st.dataframe(brand_sales.set_index("Brand")[[VOLUME_COL, "Percentage"]].round(2))
        else:
            st.info("Select a brand in Tab 2 to see pack-wise distribution.")

        fig.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        fig.update_xaxes(tickangle=-45)
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    # ---- Tab 2: Pack Size Wise Analysis ----
    with tab2:
        st.markdown("###  Question: What are the top-selling SKUs / Pack Sizes?")
        st.subheader("Pack Size Wise Volume Distribution")

        # Extract segment including packaging
        def extract_segment(sku):
            sku = sku.upper().strip()
            match = re.search(r'(\d+\s?ML(?:\.?\s?CANS?)?)', sku)
            return match.group(1) if match else "Other Segment"

        # Brand selection for Tab 2
        brands_tab2 = sorted(df["Brand"].unique())
        selected_brand_tab2 = st.selectbox("Select Brand", ["All"] + brands_tab2, index=0)

        df_pack = df.copy()
        if selected_brand_tab2 != "All":
            df_pack = df_pack[df_pack["Brand"] == selected_brand_tab2]

        df_pack["Segment"] = df_pack[SKU_COL].apply(extract_segment)
        pack_sales = df_pack.groupby("Segment")[VOLUME_COL].sum().reset_index().sort_values(by=VOLUME_COL, ascending=False)

        granularity_pack = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab_pack")
        if granularity_pack == "Percentage":
            total = pack_sales[VOLUME_COL].sum()
            pack_sales["Value"] = ((pack_sales[VOLUME_COL] / total) * 100).round(2)
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
            text=pack_sales[y_col].round(2),
            title=f"Pack Size Distribution - {selected_brand_tab2}",
            color="Segment"
        )
        fig_pack.update_traces(textposition="outside")
        fig_pack.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        fig_pack.update_xaxes(tickangle=-45)
        st.plotly_chart(fig_pack, use_container_width=True)
        st.dataframe(pack_sales.set_index("Segment")[[VOLUME_COL, "Value"]].round(2))

    # ---- Tab 3: Bottle vs Can Distribution ----
    with tab3:
        st.markdown("###  Question: How is shipment volume split between bottles and cans?")
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

    # ---- Tab 4: Top SKUs with Granularity ----
    with tab4:
        st.markdown("###  Question: What are the top-performing SKUs by volume?")
        st.subheader("Top SKUs by Volume")
        sku_sales = df.groupby(SKU_COL)[VOLUME_COL].sum().reset_index().sort_values(by=VOLUME_COL, ascending=False)

        top_n = st.slider("Show Top-N SKUs", 5, 20, 10)
        granularity_sku = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="granularity_tab4")
        sku_data = sku_sales.head(top_n).copy()

        if granularity_sku == "Percentage":
            total = sku_sales[VOLUME_COL].sum()
            sku_data["Value"] = ((sku_data[VOLUME_COL] / total) * 100).round(2)
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

    # ---- Tab 5: Monthly Trend (Absolute Only) ----
    with tab5:
        st.markdown("###  Question: How do shipment volumes change month by month for each brand?")
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

# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
