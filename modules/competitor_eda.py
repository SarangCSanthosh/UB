import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
import gdown
import re

# ===============================
# Utilities
# ===============================

@st.cache_data(show_spinner=False)
def download_and_convert_parquet(file_id: str, csv_filename: str, parquet_filename: str):
    """Download CSV from Google Drive and convert to Parquet for faster loading."""
    if not os.path.exists(parquet_filename):
        if not os.path.exists(csv_filename):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, csv_filename, quiet=False)
        df_csv = pd.read_csv(csv_filename)
        df_csv.to_parquet(parquet_filename, index=False)
    return pd.read_parquet(parquet_filename)

@st.cache_data
def prepare_dates(df, date_col="ACTUAL_DATE"):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year
    df["YearMonth"] = df[date_col].dt.to_period("M")
    df["Quarter"] = df[date_col].dt.to_period("Q")
    return df

# SKU → Brand mapping
SKU_GROUP_MAPPING = {
    "KF 330 ML.": "KF LAGER", "KF 330 ML. CANS": "KF LAGER", "KF 500 ML. CANS": "KF LAGER", "KF 650 ML.": "KF LAGER",
    "KF STORM 500 ML. CANS": "KF STORM", "KF STORM 650 ML.": "KF STORM",
    "KF ULTRA 330 ML.": "KF ULTRA", "KF ULTRA 500 ML. CAN": "KF ULTRA", "KF ULTRA 650 ML.": "KF ULTRA",
    "KF ULTRA MAX 330 ML.": "KF ULTRA MAX", "KF ULTRA MAX 500 ML. CANS": "KF ULTRA MAX", "KF ULTRA MAX 650 ML.": "KF ULTRA MAX",
    "KF ULTRA WITBIER 330 ML.": "KF ULTRA WITBIER", "KF ULTRA WITBIER 500 ML. CANS": "KF ULTRA WITBIER", "KF ULTRA WITBIER 650 ML.": "KF ULTRA WITBIER",
    "KFS 330 ML.": "KF STRONG", "KFS 330 ML. CANS": "KF STRONG", "KFS 500 ML. CANS": "KF STRONG", "KFS 650 ML.": "KF STRONG",
    "BSSB 300 ML.": "Bullet", "BSSB 330 ML.": "Bullet", "BSSB 330 ML. CANS": "Bullet", "BSSB 650 ML.": "Bullet"
}

def map_sku_to_brand(sku):
    return SKU_GROUP_MAPPING.get(str(sku).strip().upper(), "OTHER")

# ===============================
# Main App
# ===============================

def run():
    st.title("Comparative Analysis Dashboard")

    # --------------------------
    # Load data from Drive + Parquet
    # --------------------------
    FILE_ID = "1hwjURmEeUS3W_-72KnmraIlAjd1o1zDl"
    CSV_FILE = "comparative_data.csv"
    PARQUET_FILE = "comparative_data.parquet"

    df = download_and_convert_parquet(FILE_ID, CSV_FILE, PARQUET_FILE)
    st.success("✅ Data loaded successfully!")

    # Prepare dates
    df = prepare_dates(df)
    VOLUME_COL = "VOLUME"
    SKU_COL = "DBF_SKU"

    # Map SKU → Brand
    df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)

    # --------------------------
    # Sidebar Filters
    # --------------------------
    st.sidebar.header("Filters")
    filter_type = st.sidebar.radio("Filter Data By:", ["Year(s)", "Date Range"])
    if filter_type == "Year(s)":
        years = sorted(df["Year"].dropna().unique())
        selected_years = st.sidebar.multiselect("Select Year(s)", options=years, default=years)
        if selected_years:
            df = df[df["Year"].isin(selected_years)]
    else:
        min_date = df["ACTUAL_DATE"].min()
        max_date = df["ACTUAL_DATE"].max()
        date_range = st.sidebar.date_input("Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df["ACTUAL_DATE"] >= start_date) & (df["ACTUAL_DATE"] <= end_date)]

    # --------------------------
    # Tabs
    # --------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Brand Distribution", "Pack Size Wise Analysis", "Bottle v/s Cans",
        "Top SKUs", "Monthly Trend", "Company"
    ])

    # ---- Tab 1: Brand Distribution ----
    with tab1:
        st.subheader("Brand-wise YoY Shipment Change (2023 → 2024)")
        df_filtered = df[df["Year"].isin([2023, 2024])]
        if not df_filtered.empty:
            brand_yearly = df_filtered.groupby(["Brand", "Year"])[VOLUME_COL].sum().reset_index()
            pivot_df = brand_yearly.pivot(index="Brand", columns="Year", values=VOLUME_COL).fillna(0)
            if 2023 in pivot_df.columns and 2024 in pivot_df.columns:
                pivot_df["YoY_Change"] = pivot_df[2024] - pivot_df[2023]
                pivot_df["YoY_Percentage"] = ((pivot_df["YoY_Change"] / pivot_df[2023].replace(0, np.nan)) * 100).round(2)
                pivot_df = pivot_df[pivot_df.index.str.upper() != "OTHER"].sort_values("YoY_Change", ascending=False)

                fig = go.Figure(go.Waterfall(
                    name="YoY Change", orientation="v", measure=["relative"] * len(pivot_df),
                    x=pivot_df.index, y=pivot_df["YoY_Change"],
                    text=pivot_df["YoY_Change"].apply(lambda x: f"{x:,.0f}"), textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    customdata=pivot_df["YoY_Percentage"],
                    hovertemplate="<b>%{x}</b><br>Change: %{y:,.0f}<br>YoY: %{customdata:.2f}%<extra></extra>"
                ))
                fig.update_layout(title="Brand-wise Shipment Growth/Decline (2023 → 2024)", yaxis=dict(title="Change in Volume"), xaxis=dict(title="Brand", tickangle=-45), height=600, margin=dict(b=150), template="plotly_white")
                st.plotly_chart(fig, width='stretch')
                st.dataframe(pivot_df[[2023, 2024, "YoY_Change", "YoY_Percentage"]].round(0))
            else:
                st.warning("Data for both 2023 and 2024 is required to compute YoY change.")
        else:
            st.info("No records found for 2023 or 2024.")

    # ---- Tab 6: Company-wise ----
    with tab6:
        st.subheader("Company-wise YoY Shipment Change (2023 → 2024)")
        df_filtered = df[df["Year"].isin([2023, 2024])]
        if not df_filtered.empty and "DBF_COMPANY" in df_filtered.columns:
            df_filtered["DBF_COMPANY"] = df_filtered["DBF_COMPANY"].astype(str).str.strip().str.upper()
            company_yearly = df_filtered.groupby(["DBF_COMPANY", "Year"])[VOLUME_COL].sum().reset_index()
            pivot_df = company_yearly.pivot(index="DBF_COMPANY", columns="Year", values=VOLUME_COL).fillna(0)
            if 2023 in pivot_df.columns and 2024 in pivot_df.columns:
                pivot_df["YoY_Change"] = pivot_df[2024] - pivot_df[2023]
                pivot_df["YoY_Percentage"] = ((pivot_df["YoY_Change"] / pivot_df[2023].replace(0, np.nan)) * 100).round(2)
                pivot_df = pivot_df[pivot_df.index.str.upper() != "OTHER"].sort_values("YoY_Change", ascending=False)
                pivot_df = pivot_df[pivot_df["YoY_Change"] != 0]

                fig = go.Figure(go.Waterfall(
                    name="YoY Change", orientation="v", measure=["relative"] * len(pivot_df),
                    x=pivot_df.index, y=pivot_df["YoY_Change"],
                    text=pivot_df["YoY_Change"].apply(lambda x: f"{x:,.0f}"), textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    customdata=pivot_df["YoY_Percentage"],
                    hovertemplate="<b>%{x}</b><br>Change: %{y:,.0f}<br>YoY: %{customdata:.2f}%<extra></extra>"
                ))
                fig.update_layout(title="Company-wise Shipment Growth/Decline (2023 → 2024)", yaxis=dict(title="Change in Volume"), xaxis=dict(title="Company", tickangle=-45), height=600, margin=dict(b=150), template="plotly_white")
                st.plotly_chart(fig, width='stretch')
                st.dataframe(pivot_df[[2023, 2024, "YoY_Change", "YoY_Percentage"]].round(0))
            else:
                st.warning("Data for both 2023 and 2024 is required to compute YoY change.")
        else:
            st.info("No records found for 2023 or 2024, or 'DBF_COMPANY' missing.")

# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    run()
