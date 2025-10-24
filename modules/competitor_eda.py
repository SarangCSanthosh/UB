import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
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


COMPANY_GROUP_MAPPING = {
    # UB Group
    "UB": "UB",

    # SOM Group
    "SOM BREWERIES": "SOM BREWERIES",

    # AB-INBEV
    "AB-INBEV": "AB-INBEV",

    # CIPL
    "CIPL": "CIPL",

    # SNJ
    "SNJ": "SNJ",

    # BIRA
    "BIRA": "BIRA",

    # BIO
    "BIO": "BIO",

    # LILA
    "LILA": "LILA",

    # GRANO
    "GRANO 69 BEVERAGES PVT LTD ": "GRANO 69 BEVERAGES",
    "GRANO69": "GRANO 69 BEVERAGES",
}

def map_company_group(company):
    if not isinstance(company, str):
        return "OTHER"
    cleaned = company.strip().upper()
    return COMPANY_GROUP_MAPPING.get(cleaned, "OTHER")



# ===============================
# Main app
# ===============================
def run():
    st.title("Comparative Analysis Dashboard")

    # --------------------------
    # LOAD DATA via gdown
    # --------------------------

    # Google Drive file ID (replace with your actual file ID)
    file_id = "1hwjURmEeUS3W_-72KnmraIlAjd1o1zD"

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
    tab1, tab2, tab3, tab4, tab5,tab6 = st.tabs([
        "Brand Distribution",
        "Pack Size Wise Analysis",
        "Bottle v/s Cans",
        "Top SKUs",
        "Monthly Trend",
        "COmpany"
    ])

    # ---- Tab 1: Brand Distribution ----
    with tab1:
        st.markdown("### Question: Which brands contributed most to shipment growth or decline?")
        st.subheader("Brand-wise YoY Shipment Change (2023 → 2024)")
    
        # --- Ensure date column exists ---
        if "ACTUAL_DATE" in df.columns:
            df["ACTUAL_DATE"] = pd.to_datetime(df["ACTUAL_DATE"], errors="coerce")
            df["Year"] = df["ACTUAL_DATE"].dt.year
    
            # --- Filter only 2023 and 2024 ---
            df_filtered_years = df[df["Year"].isin([2023, 2024])]
    
            if not df_filtered_years.empty:
                # --- Aggregate volume by brand & year ---
                brand_yearly = (
                    df_filtered_years.groupby(["Brand", "Year"])[VOLUME_COL]
                    .sum()
                    .reset_index()
                )
    
                # --- Pivot: brands as rows, years as columns ---
                pivot_df = brand_yearly.pivot(index="Brand", columns="Year", values=VOLUME_COL).fillna(0)
    
                # --- Compute YoY Change ---
                if 2023 in pivot_df.columns and 2024 in pivot_df.columns:
                    pivot_df["YoY_Change"] = pivot_df[2024] - pivot_df[2023]
                    pivot_df["YoY_Percentage"] = (
                        (pivot_df["YoY_Change"] / pivot_df[2023].replace(0, np.nan)) * 100
                    ).round(2)
    
                    # --- Remove "OTHER" brand if present ---
                    pivot_df = pivot_df[pivot_df.index.str.upper() != "OTHER"]
    
                    # --- Sort by YoY change ---
                    pivot_df = pivot_df.sort_values("YoY_Change", ascending=False)
    
                    # --- Waterfall Chart ---
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
    
                    # --- Layout Styling ---
                    fig.update_layout(
                        title="Brand-wise Shipment Growth/Decline (2023 → 2024)",
                        yaxis=dict(title="Change in Volume"),
                        xaxis=dict(title="Brand", tickangle=-45),
                        height=600,
                        margin=dict(b=150),
                        template="plotly_white"
                    )
    
                    st.plotly_chart(fig, use_container_width=True)
    
                    # --- Show summary table ---
                    summary_df = pivot_df[[2023, 2024, "YoY_Change", "YoY_Percentage"]].round(0)
                    st.markdown("#### Summary Table: Brand YoY Comparison")
                    st.dataframe(summary_df)
                else:
                    st.warning("Data for both 2023 and 2024 is required to compute YoY change.")
            else:
                st.info("No records found for 2023 or 2024.")
        else:
            st.error("The dataset must include an 'ACTUAL_DATE' column to compute YoY change.")

       
        #with st.container():
            #st.markdown("""
### **Insights:**
#- KFS is the brand which has the highest shipment amongst all other brands under UNITED BREWERIES.
#- It has given a strong lead comprising 72 percent of the total volume. 
#- Bullet and Lager are the other two brands contributing highly to the shipment volume. 2% of the shipment comprises various other brands like KF Ultra , KF Ultra Max etc.
#""")


    # ---- Tab 2: Pack Size Wise Analysis ----
    with tab2:
        st.markdown("###  Question: What are the top-selling SKUs?")
        st.subheader("Pack Size Wise Volume Distribution")
    
        def extract_segment(sku):
            sku = str(sku).upper().strip()
            match = re.search(r'(\d+\s?ML(?:\.?\s?CANS?)?)', sku)
            segment = match.group(1) if match else "Other Segment"
            segment = segment.replace(".", "").replace("CANS", "CAN").strip()
            return segment
    
        df["Segment"] = df[SKU_COL].apply(extract_segment)
        df["Date"] = pd.to_datetime(df[DATE_COL])
    
        # --- Time Granularity ---
        time_granularity = st.radio(
            "Select Time Granularity",
            ["Yearly", "Quarterly", "Monthly", "Weekly"],
            horizontal=True
        )
    
        if time_granularity == "Yearly":
            df["Period"] = df["Date"].dt.to_period("Y").astype(str)
        elif time_granularity == "Quarterly":
            df["Period"] = df["Date"].dt.to_period("Q").astype(str)
        elif time_granularity == "Monthly":
            df["Period"] = df["Date"].dt.to_period("M").astype(str)
        else:
            df["Period"] = df["Date"].dt.to_period("W").astype(str)
    
        # --- Brand Filter ---
        brands = ["All"] + sorted(df["Brand"].unique())
        selected_brand = st.radio("Select Brand", options=brands, index=0, horizontal=True)
    
        if selected_brand == "All":
            df_brand = df[df["Brand"] != "OTHER"]
        else:
            df_brand = df[df["Brand"] == selected_brand]
    
        # --- Pack Size Filter (NEW) ---
        pack_sizes = ["All"] + sorted(df_brand["Segment"].unique())
        selected_pack = st.radio("Select Pack Size", options=pack_sizes, index=0, horizontal=True)
    
        if selected_pack == "All":
            df_filtered = df_brand.copy()
        else:
            df_filtered = df_brand[df_brand["Segment"] == selected_pack]
    
        # --- Aggregate by Period + Segment ---
        pack_sales_time = (
            df_filtered.groupby(["Period", "Segment"])[VOLUME_COL]
            .sum()
            .reset_index()
        )
    
        # --- Fill missing combinations (to avoid spikes) ---
        all_periods = sorted(df["Period"].unique())
        all_segments = sorted(df_filtered["Segment"].unique())
        full_index = pd.MultiIndex.from_product([all_periods, all_segments], names=["Period", "Segment"])
        pack_sales_time = (
            pack_sales_time.set_index(["Period", "Segment"])
            .reindex(full_index, fill_value=0)
            .reset_index()
        )
    
        # --- Compute total per period ---
        period_totals = pack_sales_time.groupby("Period")[VOLUME_COL].sum().rename("Total").reset_index()
        pack_sales_time = pack_sales_time.merge(period_totals, on="Period", how="left")
    
        # --- View Mode Toggle ---
        granularity = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True)
    
        if granularity == "Percentage":
            pack_sales_time["Share"] = (
                (pack_sales_time[VOLUME_COL] / pack_sales_time["Total"]) * 100
            ).fillna(0)
            y_col = "Share"
            y_title = "Volume Share (%)"
            chart_type = "area"
        else:
            y_col = VOLUME_COL
            y_title = "Volume"
            chart_type = "line"
    
        # --- Visualization ---
        if chart_type == "area":
            fig = px.area(
                pack_sales_time,
                x="Period",
                y=y_col,
                color="Segment",
                title=f"{selected_brand} {selected_pack} Pack Size Share Over Time ({time_granularity})",
                labels={y_col: y_title, "Period": time_granularity},
            )
        else:
            fig = px.line(
                pack_sales_time,
                x="Period",
                y=y_col,
                color="Segment",
                markers=True,
                title=f"{selected_brand} {selected_pack} Pack Size Volume Trend ({time_granularity})",
                labels={y_col: y_title, "Period": time_granularity},
            )
    
        fig.update_layout(
            height=600,
            margin=dict(t=100, b=100, l=50, r=50),
            legend_title_text="Segment",
            hovermode="x unified",
        )
    
        st.plotly_chart(fig, use_container_width=True)
    
        # --- Summary Table ---
        summary_base = (
            df_brand.groupby("Segment")[VOLUME_COL]
            .sum()
            .reset_index()
        )
        total_brand_volume = summary_base[VOLUME_COL].sum()
        summary_base["Percentage"] = (summary_base[VOLUME_COL] / total_brand_volume * 100).round(1)
    
        if selected_pack != "All":
            # Highlight only selected pack, but show all for context
            summary = summary_base.copy()
            #summary["Highlight"] = summary["Segment"].apply(lambda x: "✅ Selected" if x == selected_pack else "")
        else:
            summary = summary_base.copy()
            #summary["Highlight"] = ""
    
        st.dataframe(
            summary.set_index("Segment")[[VOLUME_COL, "Percentage"]].round(0),
            use_container_width=True
        )

        #st.dataframe(summary.set_index("Segment")[[VOLUME_COL, "Percentage"]].round(0))
        st.markdown("""
### **Insights:**
The 650 ML pack size (light blue bar) is the undisputed leader. It contributed to 81 percent of the total shipment volume. The 330 ml can comes 2nd with just 11% contribution followed by 550 ml can with 6%  and 330 ml with 3%.""")


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
        fig_packtype.update_traces(
            texttemplate="%{label}<br>%{percent:.0%}",  # Rounded percent inside the chart
            hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Share: %{percent:.0%}<extra></extra>",
            insidetextorientation='auto'
        )
        fig_packtype.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
        st.plotly_chart(fig_packtype, use_container_width=True)
    
        # Data table below the chart
        st.dataframe(pack_type_sales.set_index("Pack_Type")[[VOLUME_COL]].round(0))
        st.markdown("""
### **Insights:**
BOTTLE (light blue) is accounting for a massive 83.5% of the total volume. Whereas cans fall massively behind with just 16.5 % contribution.""")

  
    


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
### **Insights:**
The company’s performance is highly dependent on the stability and success of the KF 650ML SKU. While KF 650ML is the leader with about 55% of total volume, the ones that follow closely behind are Bullet’s BSSB 650ML with 16% and KFS 330ML cans with 10%.""")


        
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
### **Insights:**
- The KFS brand (yellow line) dominates all other brands in absolute volume, consistently operating at a level 2 to 3 times higher than the nearest competitor. Peaks occur in the late Spring/Early Summer reaching volumes of approximately 5,000 to 5,500 units . The 2024 season appears weaker than 2023. The 2024 peak was lower than the 2023 peak, and the post-peak declines were noticeably lower than the equivalent period in 2023.
- Although BULLET had seen a rise in the year 2024 , with a spike in MAY 2024 , it has fallen down again by August 2024.

""")


    with tab6:
        st.markdown("### Question: Which companies contributed most to shipment growth or decline?")
        st.subheader("Company-wise YoY Shipment Change (2023 → 2024)")
    
        if "ACTUAL_DATE" in df.columns and "DBF_COMPANY" in df.columns:
            df["ACTUAL_DATE"] = pd.to_datetime(df["ACTUAL_DATE"], errors="coerce")
            df["Year"] = df["ACTUAL_DATE"].dt.year
    
            # --- Clean company names before mapping ---
            df["DBF_COMPANY"] = df["DBF_COMPANY"].astype(str).str.strip().str.upper()
            df["Company_Group"] = df["DBF_COMPANY"].apply(map_company_group)
    
            # --- Filter only 2023 and 2024 ---
            df_filtered_years = df[df["Year"].isin([2023, 2024])]
    
            if not df_filtered_years.empty:
                # Debug: Check what companies exist
                st.write("🧩 Unique Company Groups Present:", df_filtered_years["Company_Group"].unique())
    
                # --- Aggregate ---
                company_yearly = (
                    df_filtered_years.groupby(["Company_Group", "Year"])[VOLUME_COL]
                    .sum()
                    .reset_index()
                )
                st.write("📊 Aggregated Company-Year Data:")
                st.dataframe(company_yearly)
    
                # --- Pivot ---
                pivot_df = company_yearly.pivot(index="Company_Group", columns="Year", values=VOLUME_COL).fillna(0)
                st.write("📈 Pivoted Table:")
                st.dataframe(pivot_df)
    
                # --- Compute YoY ---
                if 2023 in pivot_df.columns and 2024 in pivot_df.columns:
                    pivot_df["YoY_Change"] = pivot_df[2024] - pivot_df[2023]
                    pivot_df["YoY_Percentage"] = (
                        (pivot_df["YoY_Change"] / pivot_df[2023].replace(0, np.nan)) * 100
                    ).round(2)
    
                    pivot_df = pivot_df[pivot_df.index.str.upper() != "OTHER"]
                    pivot_df = pivot_df.sort_values("YoY_Change", ascending=False)
    
                    # Filter out zero-change companies
                    pivot_df = pivot_df[pivot_df["YoY_Change"] != 0]
    
                    if pivot_df.empty:
                        st.warning("No valid YoY changes found — check if other companies have shipment data in both 2023 and 2024.")
                    else:
                        # --- Plot ---
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
    
                        fig.update_layout(
                            title="Company-wise Shipment Growth/Decline (2023 → 2024)",
                            yaxis=dict(title="Change in Volume"),
                            xaxis=dict(title="Company", tickangle=-45),
                            height=600,
                            margin=dict(b=150),
                            template="plotly_white"
                        )
    
                        st.plotly_chart(fig, use_container_width=True)
    
                        st.markdown("#### Summary Table: Company YoY Comparison")
                        st.dataframe(pivot_df[[2023, 2024, "YoY_Change", "YoY_Percentage"]].round(0))
                else:
                    st.warning("Data for both 2023 and 2024 is required to compute YoY change.")
            else:
                st.info("No records found for 2023 or 2024.")
        else:
            st.error("The dataset must include both 'ACTUAL_DATE' and 'DBF_COMPANY' columns.")


# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
