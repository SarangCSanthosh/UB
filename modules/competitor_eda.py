import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gdown
import os
import re

# ===============================
# Data Loading
# ===============================
@st.cache_data(show_spinner=False)
def load_data_from_drive():
    file_id = "1hwjURmEeUS3W_-72KnmraIlAjd1o1zDl"
    output_file = "data.csv"

    if not os.path.exists(output_file):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_file, quiet=False)

    df = pd.read_csv(output_file)
    df.columns = df.columns.map(str)  # Ensure all column names are strings
    return df

df = load_data_from_drive()
st.success("✅ Data loaded successfully!")

# ===============================
# Utility Functions
# ===============================
def prepare_dates(df, date_col="ACTUAL_DATE"):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year
    df["YearMonth"] = df[date_col].dt.to_period("M")
    df["Quarter"] = df[date_col].dt.to_period("Q")
    return df, date_col

# SKU → Brand Mapping
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
    return SKU_GROUP_MAPPING.get(str(sku).strip().upper(), "OTHER")

def extract_segment(sku):
    sku = str(sku).upper().strip()
    match = re.search(r'(\d+\s?ML(?:\.?\s?CANS?)?)', sku)
    segment = match.group(1) if match else "Other Segment"
    segment = segment.replace(".", "").replace("CANS", "CAN").strip()
    return segment

def classify_pack_type(sku):
    sku = str(sku).upper()
    if "CAN" in sku or "CANS" in sku:
        return "CAN"
    elif "ML" in sku:
        return "BOTTLE"
    else:
        return "OTHER"

# ===============================
# Main App
# ===============================
def run():
    st.title("Comparative Analysis Dashboard")

    VOLUME_COL = "VOLUME"
    SKU_COL = "DBF_SKU"
    COMPANY_COL = "DBF_COMPANY"

    # Prepare date columns
    df_prepared, DATE_COL = prepare_dates(df)
    df_prepared["Brand"] = df_prepared[SKU_COL].apply(map_sku_to_brand)
    df_prepared["Segment"] = df_prepared[SKU_COL].apply(extract_segment)
    df_prepared["Pack_Type"] = df_prepared[SKU_COL].apply(classify_pack_type)
    df_prepared[COMPANY_COL] = df_prepared[COMPANY_COL].astype(str).str.strip().str.upper()

    # --------------------------
    # Sidebar Filters
    # --------------------------
    st.sidebar.header("Filters")
    filter_type = st.sidebar.radio("Filter Data By:", ["Year(s)", "Date Range"])

    df_filtered = df_prepared.copy()
    if filter_type == "Year(s)":
        years = sorted(df_prepared["Year"].dropna().unique())
        year_choice = st.sidebar.multiselect("Select Year(s)", options=years, default=years)
        if year_choice:
            df_filtered = df_filtered[df_filtered["Year"].isin(year_choice)]
    else:
        min_date, max_date = df_prepared[DATE_COL].min(), df_prepared[DATE_COL].max()
        date_range = st.sidebar.date_input("Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)
        if len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df_filtered = df_filtered[(df_filtered[DATE_COL] >= start_date) & (df_filtered[DATE_COL] <= end_date)]

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

    # ---- Tab 1: Brand Distribution ----
    with tab1:
        st.subheader("Brand-wise YoY Shipment Change (2023 → 2024)")
        df_years = df_filtered[df_filtered["Year"].isin([2023, 2024])]
        if not df_years.empty:
            brand_yearly = df_years.groupby(["Brand", "Year"])[VOLUME_COL].sum().reset_index()
            pivot_df = brand_yearly.pivot(index="Brand", columns="Year", values=VOLUME_COL).fillna(0)
            if 2023 in pivot_df.columns and 2024 in pivot_df.columns:
                pivot_df["YoY_Change"] = pivot_df[2024] - pivot_df[2023]
                pivot_df["YoY_Percentage"] = (pivot_df["YoY_Change"] / pivot_df[2023].replace(0, np.nan) * 100).round(2)
                pivot_df = pivot_df[pivot_df.index.str.upper() != "OTHER"]
                pivot_df = pivot_df.sort_values("YoY_Change", ascending=False)
                pivot_df = pivot_df[pivot_df["YoY_Change"] != 0]

                if not pivot_df.empty:
                    fig = go.Figure(go.Waterfall(
                        name="YoY Change",
                        orientation="v",
                        measure=["relative"] * len(pivot_df),
                        x=pivot_df.index,
                        y=pivot_df["YoY_Change"],
                        text=pivot_df["YoY_Change"].apply(lambda x: f"{x:,.0f}"),
                        textposition="outside",
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        customdata=pivot_df["YoY_Percentage"],
                        hovertemplate="<b>%{x}</b><br>Change: %{y:,.0f}<br>YoY: %{customdata:.2f}%<extra></extra>"
                    ))
                    fig.update_layout(title="Brand-wise Shipment Growth/Decline (2023 → 2024)",
                                      yaxis=dict(title="Change in Volume"),
                                      xaxis=dict(title="Brand", tickangle=-45),
                                      height=600,
                                      margin=dict(b=150),
                                      template="plotly_white")
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(pivot_df[[2023, 2024, "YoY_Change", "YoY_Percentage"]].round(0))
                else:
                    st.warning("No YoY change data found for brands.")
            else:
                st.warning("Data for both 2023 and 2024 is required to compute YoY change.")
        else:
            st.info("No records for 2023 or 2024.")

    # ---- Tab 2: Pack Size Analysis ----
    with tab2:
        st.subheader("Pack Size Wise Volume Distribution")
        df_pack = df_filtered.copy()
        time_granularity = st.radio("Select Time Granularity", ["Yearly", "Quarterly", "Monthly", "Weekly"], horizontal=True)
        if time_granularity == "Yearly":
            df_pack["Period"] = df_pack[DATE_COL].dt.to_period("Y").astype(str)
        elif time_granularity == "Quarterly":
            df_pack["Period"] = df_pack[DATE_COL].dt.to_period("Q").astype(str)
        elif time_granularity == "Monthly":
            df_pack["Period"] = df_pack[DATE_COL].dt.to_period("M").astype(str)
        else:
            df_pack["Period"] = df_pack[DATE_COL].dt.to_period("W").astype(str)

        brands = ["All"] + sorted(df_pack["Brand"].unique())
        selected_brand = st.radio("Select Brand", options=brands, index=0, horizontal=True)
        if selected_brand != "All":
            df_pack = df_pack[df_pack["Brand"] == selected_brand]

        pack_sizes = ["All"] + sorted(df_pack["Segment"].unique())
        selected_pack = st.radio("Select Pack Size", options=pack_sizes, index=0, horizontal=True)
        if selected_pack != "All":
            df_pack = df_pack[df_pack["Segment"] == selected_pack]

        pack_sales_time = df_pack.groupby(["Period", "Segment"])[VOLUME_COL].sum().reset_index()
        all_periods = sorted(df_pack["Period"].unique())
        all_segments = sorted(df_pack["Segment"].unique())
        full_index = pd.MultiIndex.from_product([all_periods, all_segments], names=["Period", "Segment"])
        pack_sales_time = pack_sales_time.set_index(["Period", "Segment"]).reindex(full_index, fill_value=0).reset_index()

        period_totals = pack_sales_time.groupby("Period")[VOLUME_COL].sum().rename("Total").reset_index()
        pack_sales_time = pack_sales_time.merge(period_totals, on="Period", how="left")

        view_mode = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True)
        if view_mode == "Percentage":
            pack_sales_time["Share"] = (pack_sales_time[VOLUME_COL] / pack_sales_time["Total"] * 100).fillna(0)
            y_col, y_title = "Share", "Volume Share (%)"
            fig = px.area(pack_sales_time, x="Period", y=y_col, color="Segment",
                          title=f"{selected_brand} {selected_pack} Pack Size Share Over Time ({time_granularity})",
                          labels={y_col: y_title, "Period": time_granularity})
        else:
            y_col, y_title = VOLUME_COL, "Volume"
            fig = px.line(pack_sales_time, x="Period", y=y_col, color="Segment", markers=True,
                          title=f"{selected_brand} {selected_pack} Pack Size Volume Trend ({time_granularity})",
                          labels={y_col: y_title, "Period": time_granularity})

        fig.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50), legend_title_text="Segment", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        summary_base = df_pack.groupby("Segment")[VOLUME_COL].sum().reset_index()
        total_brand_volume = summary_base[VOLUME_COL].sum()
        summary_base["Percentage"] = (summary_base[VOLUME_COL] / total_brand_volume * 100).round(1)
        st.dataframe(summary_base.set_index("Segment")[[VOLUME_COL, "Percentage"]].round(0))

    # ---- Tab 3: Bottle vs Can ----
    with tab3:
        st.subheader("Bottle vs Can Distribution")
        df_type = df_filtered.groupby("Pack_Type")[VOLUME_COL].sum().reset_index().sort_values(by=VOLUME_COL, ascending=False)
        fig_type = px.pie(df_type, names="Pack_Type", values=VOLUME_COL, hole=0.4, color="Pack_Type", title="Bottle vs Can Volume Distribution")
        fig_type.update_traces(texttemplate="%{label}<br>%{percent:.0%}", hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Share: %{percent:.0%}", insidetextorientation='auto')
        fig_type.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        st.plotly_chart(fig_type, width="stretch")
        st.dataframe(df_type.set_index("Pack_Type")[[VOLUME_COL]].round(0))

    # ---- Tab 4: Top SKUs ----
    with tab4:
        st.subheader("Top SKUs by Volume")
        sku_sales = df_filtered.groupby(SKU_COL)[VOLUME_COL].sum().reset_index().sort_values(by=VOLUME_COL, ascending=False)
        top_n = st.slider("Show Top-N SKUs", 5, 20, 10)
        granularity_sku = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="tab4_gran")
        sku_data = sku_sales.head(top_n).copy()
        if granularity_sku == "Percentage":
            total = sku_sales[VOLUME_COL].sum()
            sku_data["Value"] = ((sku_data[VOLUME_COL] / total) * 100).round(0)
        else:
            sku_data["Value"] = sku_data[VOLUME_COL].round(0)

        fig_sku = px.bar(sku_data, x=SKU_COL, y="Value", text="Value", title=f"Top {top_n} SKUs")
        fig_sku.update_traces(textposition="outside")
        fig_sku.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50), xaxis=dict(tickangle=-45))
        st.plotly_chart(fig_sku, width="stretch")
        st.dataframe(sku_data.set_index(SKU_COL)[["Value"]].round(0))

    # ---- Tab 5: Monthly Trend ----
    with tab5:
        st.subheader("Monthly Trend by Brand")
        trend = df_filtered.groupby(["YearMonth", "Brand"])[VOLUME_COL].sum().reset_index()
        trend["YearMonth"] = trend["YearMonth"].astype(str)
        trend["Value"] = trend[VOLUME_COL].round(0)
        fig_trend = px.line(trend, x="YearMonth", y="Value", color="Brand", markers=True, title="Monthly Brand Trend (Absolute)")
        fig_trend.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        st.plotly_chart(fig_trend, width="stretch")

    # ---- Tab 6: Company YoY ----
    with tab6:
        st.subheader("Company-wise YoY Shipment Change (2023 → 2024)")
        df_years = df_filtered[df_filtered["Year"].isin([2023, 2024])]
        if COMPANY_COL in df_years.columns and not df_years.empty:
            company_yearly = df_years.groupby([COMPANY_COL, "Year"])[VOLUME_COL].sum().reset_index()
            pivot_df = company_yearly.pivot(index=COMPANY_COL, columns="Year", values=VOLUME_COL).fillna(0)
            if 2023 in pivot_df.columns and 2024 in pivot_df.columns:
                pivot_df["YoY_Change"] = pivot_df[2024] - pivot_df[2023]
                pivot_df["YoY_Percentage"] = (pivot_df["YoY_Change"] / pivot_df[2023].replace(0, np.nan) * 100).round(2)
                pivot_df = pivot_df[pivot_df.index.str.upper() != "OTHER"]
                pivot_df = pivot_df.sort_values("YoY_Change", ascending=False)
                pivot_df = pivot_df[pivot_df["YoY_Change"] != 0]

                if not pivot_df.empty:
                    fig = go.Figure(go.Waterfall(
                        name="YoY Change",
                        orientation="v",
                        measure=["relative"] * len(pivot_df),
                        x=pivot_df.index,
                        y=pivot_df["YoY_Change"],
                        text=pivot_df["YoY_Change"].apply(lambda x: f"{x:,.0f}"),
                        textposition="outside",
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        customdata=pivot_df["YoY_Percentage"],
                        hovertemplate="<b>%{x}</b><br>Change: %{y:,.0f}<br>YoY: %{customdata:.2f}%<extra></extra>"
                    ))
                    fig.update_layout(title="Company-wise Shipment Growth/Decline (2023 → 2024)",
                                      yaxis=dict(title="Change in Volume"),
                                      xaxis=dict(title="Company", tickangle=-45),
                                      height=600,
                                      margin=dict(b=150),
                                      template="plotly_white")
                    st.plotly_chart(fig, width="stretch")
                    st.dataframe(pivot_df[[2023, 2024, "YoY_Change", "YoY_Percentage"]].round(0))
                else:
                    st.warning("No valid YoY changes found for companies.")
            else:
                st.warning("Data for both 2023 and 2024 is required for company YoY change.")
        else:
            st.info("No company data found.")



df = pd.read_csv("data.csv")
print(df.columns.tolist())


# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
