import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import gdown
import os

# ===============================
# Utility Functions
# ===============================

@st.cache_data
def download_csv(file_id, csv_filename="comparative_data.csv"):
    if not os.path.exists(csv_filename):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, csv_filename, quiet=False)
    # Read all as string to avoid type issues
    df_csv = pd.read_csv(csv_filename, dtype=str)
    return df_csv


@st.cache_data
def prepare_dates(df, date_col="ACTUAL_DATE"):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year
    df["YearMonth"] = df[date_col].dt.to_period("M")
    df["Quarter"] = df[date_col].dt.to_period("Q")
    return df, date_col

# SKU → Brand mapping
SKU_GROUP_MAPPING = {
    "KF 330 ML.": "KF LAGER",
    "KF 330 ML. CANS": "KF LAGER",
    "KF 500 ML. CANS": "KF LAGER",
    "KF 650 ML.": "KF LAGER",
    "KF STORM 500 ML. CANS": "KF STORM",
    "KF STORM 650 ML.": "KF STORM",
    "KF ULTRA 330 ML.": "KF ULTRA",
    "KF ULTRA 500 ML. CAN": "KF ULTRA",
    "KF ULTRA 650 ML.": "KF ULTRA",
    "KF ULTRA MAX 330 ML.": "KF ULTRA MAX",
    "KF ULTRA MAX 500 ML. CANS": "KF ULTRA MAX",
    "KF ULTRA MAX 650 ML.": "KF ULTRA MAX",
    "KF ULTRA WITBIER 330 ML.": "KF ULTRA WITBIER",
    "KF ULTRA WITBIER 500 ML. CANS": "KF ULTRA WITBIER",
    "KF ULTRA WITBIER 650 ML.": "KF ULTRA WITBIER",
    "KFS 330 ML.": "KF STRONG",
    "KFS 330 ML. CANS": "KF STRONG",
    "KFS 500 ML. CANS": "KF STRONG",
    "KFS 650 ML.": "KF STRONG",
    "BSSB 300 ML.": "Bullet", 
    "BSSB 330 ML.": "Bullet", 
    "BSSB 330 ML. CANS": "Bullet", 
    "BSSB 650 ML.": "Bullet",
}

def map_sku_to_brand(sku):
    return SKU_GROUP_MAPPING.get(str(sku).strip().upper(), "OTHER")

# ===============================
# Main App
# ===============================

def run():
    st.title("Comparative Analysis Dashboard")

    FILE_ID = "1hwjURmEeUS3W_-72KnmraIlAjd1o1zDl"

    # Load data from CSV
    df = download_csv(FILE_ID)
    st.success("✅ Data loaded successfully!")

    # Prepare dates
    VOLUME_COL = "VOLUME"
    SKU_COL = "DBF_SKU"
    df, DATE_COL = prepare_dates(df)

    # Map SKU → Brand
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
    # Tabs
    # --------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Brand Distribution",
        "Pack Size Wise Analysis",
        "Bottle v/s Cans",
        "Top SKUs",
        "Monthly Trend",
        "Company"
    ])

    # --------------------------
    # Tab 1: Brand Distribution
    # --------------------------
    with tab1:
        st.subheader("Brand-wise YoY Shipment Change (2023 → 2024)")
        df_filtered_years = df[df["Year"].isin([2023, 2024])]
        if not df_filtered_years.empty:
            brand_yearly = df_filtered_years.groupby(["Brand", "Year"])[VOLUME_COL].sum().reset_index()
            pivot_df = brand_yearly.pivot(index="Brand", columns="Year", values=VOLUME_COL).fillna(0)
            if 2023 in pivot_df.columns and 2024 in pivot_df.columns:
                pivot_df["YoY_Change"] = pivot_df[2024] - pivot_df[2023]
                pivot_df["YoY_Percentage"] = ((pivot_df["YoY_Change"]/pivot_df[2023].replace(0, np.nan))*100).round(2)
                pivot_df = pivot_df[pivot_df.index.str.upper() != "OTHER"].sort_values("YoY_Change", ascending=False)

                fig = go.Figure(go.Waterfall(
                    name="YoY Change",
                    orientation="v",
                    measure=["relative"]*len(pivot_df),
                    x=pivot_df.index,
                    y=pivot_df["YoY_Change"],
                    text=pivot_df["YoY_Change"].apply(lambda x:f"{x:,.0f}"),
                    textposition="outside",
                    connector={"line":{"color":"rgb(63,63,63)"}},
                    customdata=pivot_df["YoY_Percentage"],
                    hovertemplate="<b>%{x}</b><br>Change: %{y:,.0f}<br>YoY: %{customdata:.2f}%<extra></extra>"
                ))

                fig.update_layout(
                    title="Brand-wise Shipment Growth/Decline (2023 → 2024)",
                    yaxis=dict(title="Change in Volume"),
                    xaxis=dict(title="Brand", tickangle=-45),
                    height=600,
                    margin=dict(b=150),
                    template="plotly_white"
                )
                st.plotly_chart(fig, width='stretch')

                summary_df = pivot_df[[2023, 2024, "YoY_Change", "YoY_Percentage"]].round(0)
                st.markdown("#### Summary Table")
                st.dataframe(summary_df)
            else:
                st.warning("Data for both 2023 and 2024 required.")
        else:
            st.info("No records found for 2023 or 2024.")

    # --------------------------
    # Tab 2: Pack Size Wise Analysis
    # --------------------------
    with tab2:
        st.subheader("Pack Size Wise Volume Distribution")
        def extract_segment(sku):
            sku = str(sku).upper().strip()
            match = re.search(r'(\d+\s?ML(?:\.?\s?CANS?)?)', sku)
            segment = match.group(1) if match else "Other Segment"
            segment = segment.replace(".", "").replace("CANS","CAN").strip()
            return segment
        df["Segment"] = df[SKU_COL].apply(extract_segment)
        df["Date"] = pd.to_datetime(df[DATE_COL])

        time_granularity = st.radio("Time Granularity", ["Yearly","Quarterly","Monthly","Weekly"], horizontal=True)
        if time_granularity=="Yearly":
            df["Period"]=df["Date"].dt.to_period("Y").astype(str)
        elif time_granularity=="Quarterly":
            df["Period"]=df["Date"].dt.to_period("Q").astype(str)
        elif time_granularity=="Monthly":
            df["Period"]=df["Date"].dt.to_period("M").astype(str)
        else:
            df["Period"]=df["Date"].dt.to_period("W").astype(str)

        brands = ["All"] + sorted(df["Brand"].unique())
        selected_brand = st.radio("Select Brand", options=brands, index=0, horizontal=True)
        df_brand = df if selected_brand=="All" else df[df["Brand"]==selected_brand]

        pack_sizes = ["All"] + sorted(df_brand["Segment"].unique())
        selected_pack = st.radio("Select Pack Size", options=pack_sizes, index=0, horizontal=True)
        df_filtered = df_brand if selected_pack=="All" else df_brand[df_brand["Segment"]==selected_pack]

        pack_sales_time = df_filtered.groupby(["Period","Segment"])[VOLUME_COL].sum().reset_index()
        all_periods = sorted(df["Period"].unique())
        all_segments = sorted(df_filtered["Segment"].unique())
        full_index = pd.MultiIndex.from_product([all_periods, all_segments], names=["Period","Segment"])
        pack_sales_time = pack_sales_time.set_index(["Period","Segment"]).reindex(full_index, fill_value=0).reset_index()
        period_totals = pack_sales_time.groupby("Period")[VOLUME_COL].sum().rename("Total").reset_index()
        pack_sales_time = pack_sales_time.merge(period_totals, on="Period", how="left")

        granularity = st.radio("View Mode", ["Absolute","Percentage"], horizontal=True)
        if granularity=="Percentage":
            pack_sales_time["Share"] = (pack_sales_time[VOLUME_COL]/pack_sales_time["Total"]*100).fillna(0)
            y_col="Share"; y_title="Volume Share (%)"
            chart_type="area"
        else:
            y_col=VOLUME_COL; y_title="Volume"
            chart_type="line"

        if chart_type=="area":
            fig = px.area(pack_sales_time, x="Period", y=y_col, color="Segment", 
                          title=f"{selected_brand} {selected_pack} Pack Size Share Over Time ({time_granularity})",
                          labels={y_col:y_title,"Period":time_granularity})
        else:
            fig = px.line(pack_sales_time, x="Period", y=y_col, color="Segment", markers=True,
                          title=f"{selected_brand} {selected_pack} Pack Size Volume Trend ({time_granularity})",
                          labels={y_col:y_title,"Period":time_granularity})
        st.plotly_chart(fig, width='stretch')

    # --------------------------
    # Tab 3: Bottle vs Cans
    # --------------------------
    with tab3:
        st.subheader("Bottle vs Can Volume Distribution")
        df["Container_Type"] = df[SKU_COL].apply(lambda x: "CAN" if "CAN" in str(x).upper() else "BOTTLE")
        container_sales = df.groupby(["Brand","Container_Type"])[VOLUME_COL].sum().reset_index()
        fig = px.bar(container_sales, x="Brand", y=VOLUME_COL, color="Container_Type", barmode="group",
                     title="Bottle vs Can Volume by Brand")
        st.plotly_chart(fig, width='stretch')

    # --------------------------
    # Tab 4: Top SKUs
    # --------------------------
    with tab4:
        st.subheader("Top 20 SKUs by Volume")
        top_skus = df.groupby(SKU_COL)[VOLUME_COL].sum().sort_values(ascending=False).head(20).reset_index()
        st.dataframe(top_skus)

    # --------------------------
    # Tab 5: Monthly Trend
    # --------------------------
    with tab5:
        st.subheader("Monthly Volume Trend by Brand")
        monthly_sales = df.groupby(["YearMonth","Brand"])[VOLUME_COL].sum().reset_index()
        fig = px.line(monthly_sales, x="YearMonth", y=VOLUME_COL, color="Brand", markers=True,
                      title="Monthly Trend of Volumes by Brand")
        st.plotly_chart(fig, width='stretch')

    # --------------------------
    # Tab 6: Company
    # --------------------------
    with tab6:
        st.subheader("Company-wise Volume Analysis")
        df["Company"] = df["DBF_CUSTOMER"].str.upper()
        # Combine GRANO69 variants
        df["Company"] = df["Company"].replace({
            "GRANO69": "GRANO69",
            "GRANO 69 BEVERAGES PVT LTD": "GRANO69"
        })
        company_sales = df.groupby("Company")[VOLUME_COL].sum().sort_values(ascending=False).reset_index()
        st.dataframe(company_sales)

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    run()
