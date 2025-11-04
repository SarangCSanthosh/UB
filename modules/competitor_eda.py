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


# ===============================
# Main app
# ===============================
def run():
    st.title("Comparative Analysis Dashboard")

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
	# --- Ensure 'Brand' column exists ---
    if "DBF_BRAND" in df.columns:
	    df["Brand"] = df["DBF_BRAND"]
    else:
	    st.warning("⚠️ 'DBF_BRAND' column not found — some brand-based charts may not work.")

    # Map SKU → Brand before any tab uses it
    #df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)


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
		"Monthly Trend",
        "Bottle v/s Cans",
		"Pack Wise Salience",
        "Top SKUs",
        
    ])

    # ---- Tab 1: Brand Distribution ----
    with tab1:
        st.markdown("### Question: Which companies and brands contributed most to shipment growth or decline?")
        st.subheader("Company-wise YoY Shipment Change (2023 → 2024)")
    
        # --- Display the image before the chart ---
        st.image("assets/comparison.png", caption="Volume Change by Company (2023–2024)", use_container_width=True)
        st.subheader("Brand-wise YoY Shipment Change (2023 → 2024)")

    
        # --- Ensure date column exists ---
        if "ACTUAL_DATE" in df.columns:
            df["ACTUAL_DATE"] = pd.to_datetime(df["ACTUAL_DATE"], errors="coerce")
            df["Year"] = df["ACTUAL_DATE"].dt.year
    
            # --- Filter only 2023 and 2024 ---
            df_filtered_years = df[(df["Year"].isin([2023, 2024])) & (df["DBF_COMPANY"].str.upper() == "UB")]

		if not df_filtered_years.empty:
			# --- Display available brands per year ---
			brands_2023 = sorted(
				df_filtered_years[df_filtered_years["Year"] == 2023]["DBF_BRAND"].dropna().unique()
			)
			brands_2024 = sorted(
				df_filtered_years[df_filtered_years["Year"] == 2024]["DBF_BRAND"].dropna().unique()
			)
		
			col1, col2 = st.columns(2)
			with col1:
				st.markdown("#### Brands present in **2023**")
				st.write(brands_2023 if brands_2023 else "No brands found for 2023")
			with col2:
				st.markdown("#### Brands present in **2024**")
				st.write(brands_2024 if brands_2024 else "No brands found for 2024")
		
			# --- Aggregate volume by brand & year ---
			brand_yearly = (
				df_filtered_years.groupby(["DBF_BRAND", "Year"])[VOLUME_COL]
				.sum()
				.reset_index()
			)


            #if not df_filtered_years.empty:
                # --- Aggregate volume by brand & year ---
                #brand_yearly = (
                    #df_filtered_years.groupby(["DBF_BRAND", "Year"])[VOLUME_COL]
                    #.sum()
                    #.reset_index()
               #)
    
                # --- Pivot: brands as rows, years as columns ---
                pivot_df = brand_yearly.pivot(index="DBF_BRAND", columns="Year", values=VOLUME_COL).fillna(0)
    
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

       
        with st.container():
            st.markdown("""
### **Insights:**
	
	1. EXTERNAL ANALYSIS
	    • Growth of shipments -> SOM BREWERIES > UBL > AB-INBEV
	
    2. INTERNAL ANALYSIS 
	    • Bullet ⬆️ but KFS ⬇️
    
    KFS might not be the key performing brand of UBL
""")


    # ---- Tab 2: Pack Size Wise Analysis ----
    with tab4:
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

	1. 650 ML - 81% of total shipments
	2. 330 ML - 11% of total shipments
	3. KFS - 650 ML has been undisputed leader
	4. KF storm has seen instances where 330ML has taken over 650 ML sales
	
""")


    # ---- Tab 5: Bottle vs Can Distribution ----
    with tab3:
	    st.markdown("### Question: How is shipment volume split between bottles and cans?")
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
	
	    # Overall Bottle vs Can Split
	    pack_type_sales = (
	        df.groupby("Pack_Type")[VOLUME_COL]
	        .sum()
	        .round(0)
	        .reset_index()
	        .sort_values(by=VOLUME_COL, ascending=False)
	    )
	
	    # Pie chart for intuitive visualization
	    fig_packtype = px.pie(
	        pack_type_sales,
	        names="Pack_Type",
	        values=VOLUME_COL,
	        title="Bottle vs Can Volume Distribution",
	        hole=0.4,
	        color="Pack_Type"
	    )
	    fig_packtype.update_traces(
	        texttemplate="%{label}<br>%{percent:.0%}",
	        hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Share: %{percent:.0%}<extra></extra>",
	        insidetextorientation="auto"
	    )
	    fig_packtype.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
	    st.plotly_chart(fig_packtype, use_container_width=True)
	
	    # Data table below the chart
	    st.dataframe(pack_type_sales.set_index("Pack_Type")[[VOLUME_COL]].round(0))
	
	    # -----------------------------------------
	    # NEW SECTION: Clustered Bar Chart by Depot
	    # -----------------------------------------
	    st.markdown("### Depot-wise Comparison: Bottle vs Can")
	
	    if "DBF_DEPOT" not in df.columns:
	        st.warning("⚠️ 'DBF_DEPOT' column not found in dataset.")
	    else:
	        # Clean depot names (remove 'KSBCL - ' prefix if present)
	        df["Depot_Clean"] = df["DBF_DEPOT"].astype(str).str.replace(r"^KSBCL\s*-\s*", "", regex=True).str.strip()
	
	        # Aggregate Bottle vs Can volumes by depot
	        depot_pack = (
	            df.groupby(["Depot_Clean", "Pack_Type"])[VOLUME_COL]
	            .sum()
	            .reset_index()
	        )
	
	        # Keep only Bottle and Can (remove 'OTHER' if any)
	        depot_pack = depot_pack[depot_pack["Pack_Type"].isin(["BOTTLE", "CAN"])]
	
	        # ---- Sort depots by Bottle volume (descending) ----
	        bottle_order = (
	            depot_pack[depot_pack["Pack_Type"] == "BOTTLE"]
	            .sort_values(by=VOLUME_COL, ascending=False)["Depot_Clean"]
	            .tolist()
	        )
	
	        # Apply this order to the x-axis
	        depot_pack["Depot_Clean"] = pd.Categorical(depot_pack["Depot_Clean"], categories=bottle_order, ordered=True)
	        depot_pack = depot_pack.sort_values("Depot_Clean")
	
	        # Clustered bar chart
	        fig_cluster = px.bar(
	            depot_pack,
	            x="Depot_Clean",
	            y=VOLUME_COL,
	            color="Pack_Type",
	            barmode="group",  # side-by-side bars
	            text_auto=".2s",
	            title="Depot-wise Shipment Volume: Bottle vs Can (Sorted by Bottle Volume)",
	            labels={"Depot_Clean": "Depot", VOLUME_COL: "Shipment Volume"}
	        )
	
	        fig_cluster.update_traces(textposition="outside")
	        fig_cluster.update_layout(
	            height=600,
	            xaxis_tickangle=-45,
	            margin=dict(b=200, t=100),
	            showlegend=True,
	            legend_title_text="Pack Type",
	        )
	        st.plotly_chart(fig_cluster, use_container_width=True)
	
	        # Optional table for exact values
	        st.dataframe(
	            depot_pack.pivot(index="Depot_Clean", columns="Pack_Type", values=VOLUME_COL)
	            .fillna(0)
	            .round(0)
	        )
	
	        # Insights Section
	        st.markdown("""
	        ### **Insights:**
	        🧴 **Bottle:**
	        - 🔼 High: Kalaburagi, Bidar  
	        - 🔽 Low: Chikodi, Chitradurga  
	
	        🥫 **Can:**
	        - 🔼 High: Raichur, Koppal  
	        - 🔽 Low: Jamakhandi, Chikodi  
	        """)

  
    


    # ---- Tab 2: Top SKUs with Granularity ----
    with tab5:
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
    with tab2:
        st.markdown("###  Question: How do shipment volumes change month by month for each brand?")
        st.subheader("Monthly Trend by Brand")

        # --- Map brand names ---
        #df["Brand"] = df[SKU_COL].apply(map_sku_to_brand)
    
        # --- Prepare trend data ---
        df["YearMonth"] = pd.to_datetime(df["ACTUAL_DATE"], errors="coerce").dt.to_period("M").astype(str)
        trend = (
            df.groupby(["YearMonth", "Brand"])[VOLUME_COL]
            .sum()
            .round(0)
            .reset_index()
        )
        trend["Value"] = trend[VOLUME_COL].round(0)
        y_title = "Volume"
    
        # ===============================
        # 📅 EVENT CALENDAR INTEGRATION
        # ===============================
    
        @st.cache_data
        def load_event_calendar(sheet_url: str):
            """Loads and cleans event calendar from Google Sheets."""
            try:
                df_events = pd.read_excel(sheet_url)
                df_events.columns = df_events.columns.str.strip()
    
                if "Date" not in df_events.columns:
                    st.error("❌ 'Date' column not found in event sheet.")
                    return pd.DataFrame(columns=["Date", "Event / Task"])
    
                df_events["Date"] = pd.to_datetime(df_events["Date"], errors="coerce")
                df_events.dropna(subset=["Date"], inplace=True)
    
                # Clean and normalize event names
                def clean_event_name(text):
                    if pd.isna(text):
                        return None
                    text = str(text).strip()
                    if not text or text.lower() in ["nan", "none"]:
                        return None
                    text = (
                        text.replace("Against", "")
                        .replace("Friendly", "BFC")
                        .replace("Footll", "Football")
                        .replace("Pro Ka", "Pro Kabbadi")
                        .replace("C ", " ")
                        .replace("IND World cup", "IND World Cup")
                        .replace("RCB Match", "RCB Match")
                        .replace("Week end", "Weekend")
                        .replace("INDependence", "Independence")
                        .replace("Ni8", "Night")
                    )
                    text = " ".join(text.split())
                    text = text.title().replace("Ipl", "IPL").replace("Bfc", "BFC")
                    return text
    
                df_events["Event / Task"] = df_events["Event / Task"].apply(clean_event_name)
    
                # Group events by YearMonth
                df_events["YearMonth"] = df_events["Date"].dt.to_period("M").astype(str)
    
                def summarize_events(x):
                    counts = x.value_counts()
                    lines = []
                    for event, count in counts.items():
                        if count > 1:
                            lines.append(f"{event} (x{count})")
                        else:
                            lines.append(event)
                    return "<br>".join(lines)
    
                events_agg = (
                    df_events.groupby("YearMonth")["Event / Task"]
                    .apply(summarize_events)
                    .reset_index()
                )
                return events_agg
    
            except Exception as e:
                st.error(f"❌ Could not load event calendar: {e}")
                return pd.DataFrame(columns=["YearMonth", "Event / Task"])
    
        # --- Load Event Calendar ---
        EVENT_XLSX_URL = "https://docs.google.com/spreadsheets/d/1GxgGo6waZV7WDsF50v_nYSu2mxEX6bmj/export?format=xlsx"
        df_events = load_event_calendar(EVENT_XLSX_URL)
    
        # --- Merge Trend with Events ---
        trend = trend.merge(df_events, on="YearMonth", how="left")
    
        # ===============================
        # 📈 LINE CHART
        # ===============================
        fig_trend = px.line(
            trend,
            x="YearMonth",
            y="Value",
            color="Brand",
            markers=True,
            title="Monthly Brand Trend (Absolute)",
            hover_data={"Event / Task": True, "Brand": True, "Value": True},
        )
    
        fig_trend.update_traces(
            hovertemplate="<b>%{x}</b><br>Brand: %{customdata[1]}<br>Volume: %{y:,.0f}<br><br><b>Events:</b><br>%{customdata[0]}<extra></extra>"
        )
    
        fig_trend.update_yaxes(title_text=y_title)
        fig_trend.update_layout(height=600, margin=dict(t=100, b=100, l=50, r=50))
    
        st.plotly_chart(fig_trend, use_container_width=True)
        st.markdown("""
### **Insights:**

	KFS
		1. 2023 - MAY - 🔽 , JUNE - 🔼
		2. 2024 - MAY - 🔼 , JUNE - 🔽 
	Overall -  Dip in KFS from 2023 to 2024 , Spike in bullet from 2023 to 2024

""")

# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
