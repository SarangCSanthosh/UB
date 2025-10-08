import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# HELPERS
# --------------------------
@st.cache_data
def load_excel(path_or_file, sheet_name=0):
    return pd.read_excel(path_or_file, sheet_name=sheet_name)

@st.cache_data
def load_normalized_data(path):
    df_norm = pd.read_excel(path)

    # Rename first column to ACTUAL_DATE
    df_norm.rename(columns={df_norm.columns[0]: "ACTUAL_DATE"}, inplace=True)

    # Ensure datetime
    df_norm["ACTUAL_DATE"] = pd.to_datetime(df_norm["ACTUAL_DATE"], errors="coerce")

    # Sum across all numeric columns (excluding ACTUAL_DATE)
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    df_norm["VOLUME"] = df_norm[numeric_cols].sum(axis=1)

    # Extract YearMonth
    df_norm["YearMonth"] = df_norm["ACTUAL_DATE"].dt.to_period("M").astype(str)

    return df_norm[["ACTUAL_DATE", "YearMonth", "VOLUME"]]

def prepare_dates(df, date_col="ACTUAL_DATE"):
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["Year"] = df[date_col].dt.year
    df["YearMonth"] = df[date_col].dt.to_period("M")
    df["Quarter"] = df[date_col].dt.to_period("Q")
    return df, date_col

@st.cache_data
def load_event_calendar(sheet_id: str):
    """
    Loads the event calendar from a public Google Sheet.
    Input: Google Sheet ID
    Output: DataFrame
    """
    try:
        # Build a direct download link
        download_url = f"https://docs.google.com/spreadsheets/d/1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO/export?format=xlsx"

        df = pd.read_excel(download_url)

        # Clean and standardize columns
        df.columns = df.columns.str.strip()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        # Drop rows without valid dates
        df = df.dropna(subset=["Date"])

        return df

    except Exception as e:
        st.error(f"❌ Could not load event calendar: {e}")
        return pd.DataFrame(columns=["Date", "Day", "Month", "Week Number", "Event / Task", "Remarks"])


# --------------------------
# MAIN APP
# --------------------------
def run():
    st.title("Secondary Dataset Dashboard")

    # --------------------------
    # LOAD DATA
    # --------------------------
    default_path = "https://docs.google.com/spreadsheets/d/1te1MVxSoO3EWwg_9akooxKxIEgI4KDna/export?format=xlsx"
    df = load_excel(default_path)
    SHEET_ID = "1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO"
    df_events = load_event_calendar(SHEET_ID)

    VOLUME_COL = "VOLUME"
    OUTLET_COL = "DBF_OUTLET_NAME"

    df, DATE_COL = prepare_dates(df)

    # --------------------------
    # FIXED KPIs
    # --------------------------
    yearly_data = df.groupby("Year").agg(
        Total_Volume=(VOLUME_COL, "sum"),
        Unique_Outlets=(OUTLET_COL, "nunique"),
        Total_Shipments=(DATE_COL, "count"),
    ).sort_index()

    latest_year = yearly_data.index.max()
    prev_year = latest_year - 1 if latest_year - 1 in yearly_data.index else None

    def pct_delta(cur, prev):
        if prev is None or prev == 0:
            return None
        return ((cur - prev) / prev) * 100

    if prev_year:
        kpi_volume = yearly_data.loc[latest_year, "Total_Volume"]
        kpi_outlets = yearly_data.loc[latest_year, "Unique_Outlets"]
        kpi_shipments = yearly_data.loc[latest_year, "Total_Shipments"]

        delta_volume = pct_delta(kpi_volume, yearly_data.loc[prev_year, "Total_Volume"])
        delta_outlets = pct_delta(kpi_outlets, yearly_data.loc[prev_year, "Unique_Outlets"])
        delta_shipments = pct_delta(kpi_shipments, yearly_data.loc[prev_year, "Total_Shipments"])
    else:
        kpi_volume = kpi_outlets = kpi_shipments = 0
        delta_volume = delta_outlets = delta_shipments = None

    def format_delta(val):
        return f"{val:+.2f}%" if val is not None and pd.notna(val) else "N/A"

    # --------------------------
    # SHOW FIXED KPIs
    # --------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Volume", f"{kpi_volume:,}", format_delta(delta_volume))
    col2.metric("Unique Outlets", f"{kpi_outlets}", format_delta(delta_outlets))
    col3.metric("Total Shipments", f"{kpi_shipments}", format_delta(delta_shipments))
    st.caption(f"YoY change: {latest_year} vs {prev_year}")
    st.markdown("---")

    # --------------------------
    # SIDEBAR FILTERS
    # --------------------------
    st.sidebar.header("Filters")

    filter_mode = st.sidebar.radio("Filter by:", ["Year", "Date Range"], horizontal=True)
    df_filtered = df.copy()

    if filter_mode == "Year":
        year_choice = st.sidebar.multiselect(
            "Select Year(s)", options=sorted(df["Year"].dropna().unique()), default=sorted(df["Year"].dropna().unique())
        )
        if year_choice:
            df_filtered = df_filtered[df_filtered["Year"].isin(year_choice)]
    else:
        start_date = st.sidebar.date_input("Start Date", df[DATE_COL].min().date())
        end_date = st.sidebar.date_input("End Date", df[DATE_COL].max().date())
        mask = (df_filtered[DATE_COL].dt.date >= start_date) & (
            df_filtered[DATE_COL].dt.date <= end_date
        )
        df_filtered = df_filtered.loc[mask]

    # --------------------------
    # VISUALIZATIONS (tabs)
    # --------------------------
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Shipment Trends",
        "Top Outlets",
        "Depot Analysis",
        "Region Donut",
        "Region Stacked",
        "Special Outlets",
    ])

    # ---- Shipment Trends ----
    # ---- Shipment Trends ----
    # ---- Shipment Trends ----
    with tab1:
        st.markdown("###  Question: Do shipment trends look different by year, quarter, or month?")
        st.subheader("Shipment Trends and Event Calendar")
    
        # --- Chart Selection ---
        chart_type = st.radio(
            "Select Chart Type:",
            ["Shipment Trend", "Event Calendar"],
            horizontal=True
        )
    
        # --- Controls (Granularity and View Mode) ---
        granularity = st.radio(
            "Granularity", ["Yearly", "Quarterly", "Monthly"],  # ✅ Removed "Daily"
            horizontal=True, key="trend_granularity"
        )
        view_mode = st.radio(
            "Display Mode", ["Absolute", "Percentage"],
            horizontal=True, key="trend_view"
        )
    
        # ===============================
        # 📊 IF SHIPMENT TREND SELECTED
        # ===============================
        if chart_type == "Shipment Trend":
            # --- Shipment Trend (Filtered Main Data) ---
            if granularity == "Yearly":
                df_filtered["Label"] = df_filtered["Year"].astype(int).astype(str)
            elif granularity == "Quarterly":
                df_filtered["Label"] = df_filtered["Quarter"].astype(str)
            else:  # Monthly granularity
                df_filtered["Label"] = df_filtered["YearMonth"].astype(str)
    
            trend_df = df_filtered.groupby("Label")[VOLUME_COL].sum().reset_index()
    
            if view_mode == "Percentage":
                total_sum = trend_df[VOLUME_COL].sum()
                trend_df["Value"] = (trend_df[VOLUME_COL] / total_sum) * 100
                y_title = "Percentage (%)"
            else:
                trend_df["Value"] = trend_df[VOLUME_COL]
                y_title = "Volume"
    
            # --- Load and Filter Normalised Data ---
            df_normalised = load_normalized_data(
                r"https://docs.google.com/spreadsheets/d/1lg0iIgKx9byQj7d2NZO-k1gKdFuNxxQe/export?format=xlsx"
            )
    
            if filter_mode == "Year":
                if year_choice:
                    df_normalised = df_normalised[df_normalised["ACTUAL_DATE"].dt.year.isin(year_choice)]
            else:
                df_normalised = df_normalised[
                    (df_normalised["ACTUAL_DATE"].dt.date >= start_date)
                    & (df_normalised["ACTUAL_DATE"].dt.date <= end_date)
                ]
    
            df_normalised["Year"] = df_normalised["ACTUAL_DATE"].dt.year
            df_normalised["Quarter"] = df_normalised["ACTUAL_DATE"].dt.to_period("Q").astype(str)
            df_normalised["YearMonth"] = df_normalised["ACTUAL_DATE"].dt.to_period("M").astype(str)
    
            if granularity == "Yearly":
                norm_df = df_normalised.groupby("Year")["VOLUME"].sum().reset_index()
                norm_df["Label"] = norm_df["Year"].astype(str)
            elif granularity == "Quarterly":
                norm_df = df_normalised.groupby("Quarter")["VOLUME"].sum().reset_index()
                norm_df["Label"] = norm_df["Quarter"].astype(str)
            else:
                norm_df = df_normalised.groupby("YearMonth")["VOLUME"].sum().reset_index()
                norm_df["Label"] = norm_df["YearMonth"].astype(str)
    
            if view_mode == "Percentage":
                total_norm = norm_df["VOLUME"].sum()
                norm_df["Normalized_Value"] = (norm_df["VOLUME"] / total_norm) * 100
                norm_y_title = "Normalized Volume (%)"
            else:
                norm_df["Normalized_Value"] = norm_df["VOLUME"]
                norm_y_title = "Normalized Volume"
    
            # --- Load Event Calendar ---
            EVENT_CSV_URL = "https://docs.google.com/spreadsheets/d/1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO/export?format=xlsx"
            df_events = load_event_calendar(EVENT_CSV_URL)
            df_events["Date"] = pd.to_datetime(df_events["Date"], errors="coerce")
    
            if granularity == "Yearly":
                df_events["Label"] = df_events["Date"].dt.year.astype(str)
            elif granularity == "Quarterly":
                df_events["Label"] = df_events["Date"].dt.to_period("Q").astype(str)
            else:
                df_events["Label"] = df_events["Date"].dt.to_period("M").astype(str)
    
            events_agg = df_events.groupby("Label")["Event / Task"].apply(lambda x: "\n".join(x.dropna())).reset_index()
            trend_df = trend_df.merge(events_agg, on="Label", how="left")
    
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trend_df["Label"],
                    y=trend_df["Value"],
                    mode="lines+markers",
                    name=f"Shipment Trend ({granularity}, {view_mode})",
                    fill="tozeroy",
                    yaxis="y1",
                    hovertext=trend_df["Event / Task"],
                    hoverinfo="x+y+text",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=norm_df["Label"],
                    y=norm_df["Normalized_Value"],
                    mode="lines+markers",
                    name=f"Normalized {granularity} Volume",
                    line=dict(color="red"),
                    yaxis="y2",
                    hoverinfo="x+y",
                )
            )
    
            fig.update_layout(
                title=f"Shipment Trend vs Normalized Volume ({granularity})",
                xaxis=dict(title=granularity, type="category"),
                yaxis=dict(title=y_title, side="left"),
                yaxis2=dict(title=norm_y_title, overlaying="y", side="right"),
                legend_title="Metrics",
                height=800,
                width=1600,
                template="plotly_dark",
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)
    

        # ===============================
        # 📅 IF EVENT CALENDAR SELECTED
        # ===============================
        else:
            st.subheader("Event Calendar (Month & Year Selector)")
        
            # --- Load Event Calendar from Google Sheet ---
            EVENT_XLSX_URL = "https://docs.google.com/spreadsheets/d/1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO/export?format=xlsx"
            df_events = load_event_calendar(EVENT_XLSX_URL)
            df_events["Date"] = pd.to_datetime(df_events["Date"], errors="coerce")
        
            # --- Ensure expected columns exist ---
            for col in ["Event / Task", "Remarks"]:
                if col not in df_events.columns:
                    df_events[col] = ""
        
            # --- Extract date components ---
            df_events["Year"] = df_events["Date"].dt.year
            df_events["Month"] = df_events["Date"].dt.month
            df_events["MonthName"] = df_events["Date"].dt.strftime("%B")
            df_events["Day"] = df_events["Date"].dt.day
        
            # --- User selections ---
            selected_year = st.selectbox("Select Year", sorted(df_events["Year"].dropna().unique()))
            selected_month_name = st.selectbox(
                "Select Month",
                sorted(df_events["MonthName"].unique(), key=lambda x: pd.to_datetime(x, format="%B").month)
            )
        
            # --- Filter data for chosen month ---
            df_selected = df_events[
                (df_events["Year"] == selected_year) &
                (df_events["MonthName"] == selected_month_name)
            ].copy()
        
            # --- Merge with shipment data (optional) ---
            df_ship = df.copy()
            df_ship["Date"] = pd.to_datetime(df_ship["ACTUAL_DATE"], errors="coerce")
            ship_day = df_ship.groupby(df_ship["Date"].dt.date)[VOLUME_COL].sum().reset_index()
            ship_day.rename(columns={VOLUME_COL: "VOLUME"}, inplace=True)
            ship_day["Date"] = pd.to_datetime(ship_day["Date"], errors="coerce")
        
            df_selected = pd.merge(df_selected, ship_day[["Date", "VOLUME"]], on="Date", how="left")
            df_selected["VOLUME"] = pd.to_numeric(df_selected["VOLUME"], errors="coerce").fillna(0)
        
            # --- Calendar Grid ---
            df_selected["Weekday"] = df_selected["Date"].dt.day_name()
            df_selected["WeekOfMonth"] = ((df_selected["Date"].dt.day - 1) // 7) + 1
            df_selected["DayNum"] = df_selected["Date"].dt.day
        
            # --- Tooltip text (Event + Remarks) ---
            df_selected["Tooltip"] = (
                "<b>" + df_selected["Date"].dt.strftime("%d %b %Y") + "</b><br>" +
                "Event: " + df_selected["Event / Task"].fillna("") + "<br>" +
                "Remarks: " + df_selected["Remarks"].fillna("") + "<br>" +
                "Volume: " + df_selected["VOLUME"].astype(str)
            )
        
            # --- Pivot tables for grid display ---
            pivot_volume = df_selected.pivot(index="WeekOfMonth", columns="Weekday", values="VOLUME")
            text_matrix = df_selected.pivot(index="WeekOfMonth", columns="Weekday", values="DayNum")
            hover_matrix = df_selected.pivot(index="WeekOfMonth", columns="Weekday", values="Tooltip")

            # --- Remove rows and columns that are completely NaN ---
            pivot_volume = pivot_volume.dropna(how="all")        # Drop empty weeks
            pivot_volume = pivot_volume.dropna(axis=1, how="all")  # Drop empty weekdays
            
            text_matrix = text_matrix.loc[pivot_volume.index, pivot_volume.columns]
            hover_matrix = hover_matrix.loc[pivot_volume.index, pivot_volume.columns]
            
            # Reorder weekdays (optional, only keeps existing days)
            ordered_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            pivot_volume = pivot_volume.reindex(columns=[d for d in ordered_days if d in pivot_volume.columns])
            text_matrix = text_matrix.reindex(columns=[d for d in ordered_days if d in text_matrix.columns])
            hover_matrix = hover_matrix.reindex(columns=[d for d in ordered_days if d in hover_matrix.columns])
                    
            # Reorder weekdays
            #ordered_days = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            #pivot_volume = pivot_volume.reindex(columns=ordered_days)
            #text_matrix = text_matrix.reindex(columns=ordered_days)
            #hover_matrix = hover_matrix.reindex(columns=ordered_days)
        
            # --- Plotly Heatmap ---
            fig = go.Figure(data=go.Heatmap(
                z=pivot_volume.values,
                x=pivot_volume.columns,
                y=pivot_volume.index,
                text=text_matrix.values,
                texttemplate="%{text}",
                colorscale="Viridis",
                hoverinfo="text",
                hovertext=hover_matrix.values
            ))
        
            fig.update_layout(
                title=f"Shipment Calendar — {selected_month_name} {selected_year}",
                xaxis_title="Day of Week",
                yaxis_title="Week of Month",
                yaxis=dict(autorange="reversed"),
                height=650,
                width=1000,
                template="plotly_dark",
                coloraxis_colorbar=dict(title="Shipment Volume")
            )
        
            st.plotly_chart(fig, use_container_width=True)


        st.markdown("""
### **Answer:**
- While the absolute volume peaked in Q2, the normalized volume hit one of its lowest points. This implies that the normalizing factor (e.g., number of workdays, capacity, or seasonal adjustment) was very high in Q2. In other words, the high absolute volume in Q2 did not meet expectations or capacity, resulting in a poor normalized performance.
- Shipments in 2024 did not reach the high peaks seen in 2023, indicating a minor year-over-year volume challenge. The sharp normalized volume drop in May (2023 and 2024) is a critical recurring issue. This suggests that the normalizing factor (e.g., number of workdays, capacity, or seasonal adjustment) for May is either set too high or the operational system fails to cope with the demands of that specific month. The highly efficient performance seen in August 2023 was not repeated in 2024. While the absolute volume was high in May 2024, the normalized score was low, pointing to a failure to translate high volume into high efficiency in 2024.

""")



    with tab2:
        st.markdown("###  Question: Where is shipment activity the highest among outlets?")
        st.subheader("Top Outlets by Volume")
        
        # --- Controls ---
        view_mode_tab2 = st.radio("Display Mode", ["Absolute", "Percentage"], horizontal=True, key="top_outlets_view_mode")
        top_n = st.slider("Top-N Outlets", 5, 25, 10)
        
        outlet_volume = df_filtered.groupby(OUTLET_COL)[VOLUME_COL].sum().round(0).reset_index()
        top_outlets = outlet_volume.sort_values(by=VOLUME_COL, ascending=False).head(top_n)
        
        if view_mode_tab2 == "Percentage":
            top_outlets["Value"] = (top_outlets[VOLUME_COL] / top_outlets[VOLUME_COL].sum().round(0)) * 100
            value_col = "Value"
            title_suffix = " (%)"
        else:
            top_outlets["Value"] = top_outlets[VOLUME_COL]
            value_col = "Value"
            title_suffix = ""
        
        # Treemap
        fig = px.treemap(top_outlets, path=[OUTLET_COL], values=value_col, 
                         title=f"Top {top_n} Outlets Treemap{title_suffix}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Dataframe
        display_df = top_outlets[[OUTLET_COL, value_col]].set_index(OUTLET_COL).round(0)
        st.dataframe(display_df)
        
        st.markdown("""
    ### **Answer:**
    - The most striking feature is the overwhelming dominance of a single outlet: MANAGING DIRECTOR MSIL. This outlet takes up easily over 75% to 85% of the total volume visualized in the Top 10 list.
    - The business's volume is heavily reliant on this one outlet. Any disruption, change in operations, or loss of business from MANAGING DIRECTOR MSIL would have a massive, detrimental impact on the overall volume and stability of the entire system.
    """)


    # ---- Depot Analysis ----
    with tab3:
        st.markdown("###  Question: Which depots are driving the majority of volume?")
        st.subheader("Depot-wise ABC Analysis")
        if "DBF_DEPOT" in df_filtered.columns:
            depot_volume = df_filtered.groupby("DBF_DEPOT")[VOLUME_COL].sum().round(0).reset_index()
            depot_volume = depot_volume.sort_values(by=VOLUME_COL, ascending=False).reset_index(drop=True)
            depot_volume["Cum_Volume"] = depot_volume[VOLUME_COL].cumsum().round(0)
            depot_volume["Cum_Percentage"] = 100 * depot_volume["Cum_Volume"] / depot_volume[VOLUME_COL].sum().round(0)

            def classify(pct):
                if pct <= 70:
                    return "A"
                elif pct <= 90:
                    return "B"
                else:
                    return "C"

            depot_volume["ABC_Category"] = depot_volume["Cum_Percentage"].apply(classify)

            fig = go.Figure()
            fig.add_bar(
                x=depot_volume["DBF_DEPOT"],
                y=depot_volume[VOLUME_COL],
                name="Depot Volume",
                marker_color=depot_volume["ABC_Category"].map({"A": "green", "B": "orange", "C": "red"}),
            )
            fig.add_trace(
                go.Scatter(
                    x=depot_volume["DBF_DEPOT"],
                    y=depot_volume["Cum_Percentage"],
                    mode="lines+markers",
                    name="Cumulative %",
                    yaxis="y2",
                )
            )
            fig.update_layout(
                title="Depot-wise ABC Analysis",
                yaxis=dict(title="Volume"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 110]),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(depot_volume.round(0))
        else:
            st.info("DBF_DEPOT column not found.")
        st.markdown("""
### **Answer:**
 Ensure operations, supply chain, and support for KALABURAGI, BIDAR, and VIJAYAPURA are top-tier, as they are the primary drivers of volume.  Identify which of the Orange depots (KOPPAL to CHITRADURGA) have the highest growth potential to secure the next wave of A-Class contributors. Analyze the cost-to-serve for the Red depots to ensure the low volume is not resulting in disproportionately high operational costs.

""")

    # ---- Region Donut ----
    with tab4:
        st.markdown("###  Question: Which regions account for the largest share of shipments?")
        st.subheader("Region-wise Volume Share")
        if "DBF_REGION" in df_filtered.columns:
            region_volume = df_filtered.groupby("DBF_REGION")[VOLUME_COL].sum().round(0).reset_index()
            fig = px.pie(region_volume, values=VOLUME_COL, names="DBF_REGION", hole=0.5,
                         title="Volume Distribution by Region")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(region_volume.set_index("DBF_REGION").round(0))
        else:
            st.info("DBF_REGION column not found.")
        st.markdown("""
### **Answer:**
NORTH KARNATAKA 2 is the primary driver of volume in this combined area, accounting for nearly two-thirds of the total volume. Resource allocation (e.g., inventory, sales support, and logistics capacity) should prioritize NORTH KARNATAKA 2 to support its dominant volume and ensure continued high performance.

""")

    # ---- Region Stacked ----
    with tab5:
        st.markdown("###  Question: Which outlets contribute most to regional shipment volume?")
        st.subheader("Outlets & Volume by Region (100% Share)")
        if "DBF_REGION" in df_filtered.columns and "DBF_OUTLET_CODE" in df_filtered.columns:
            region_stats = df_filtered.groupby("DBF_REGION").agg(
                Outlet_Count=("DBF_OUTLET_CODE", "nunique"),
                Total_Volume=(VOLUME_COL, "sum"),
            ).reset_index()

            region_stats["Outlet %"] = (region_stats["Outlet_Count"] / region_stats["Outlet_Count"].sum().round(0)) * 100
            region_stats["Volume %"] = (region_stats["Total_Volume"] / region_stats["Total_Volume"].sum().round(0)) * 100

            melted = region_stats.melt(
                id_vars="DBF_REGION",
                value_vars=["Outlet %", "Volume %"],
                var_name="Metric",
                value_name="Percent",
            )

            fig = px.bar(melted, x="DBF_REGION", y="Percent", color="Metric",
                         barmode="stack", text=melted["Percent"].round(0),
                         title="Outlets vs Volume % by Region")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(region_stats.set_index("DBF_REGION").round(0))
        else:
            st.info("DBF_REGION or DBF_OUTLET_CODE not found.")
        st.markdown("""
### **Answer:**
North Karnataka 2 region is the core volume driver, as confirmed by both the previous donut chart (60.2% volume share) and this efficiency chart. It generates significantly more volume per outlet than its counterpart. NORTH KARNATAKA 1 has a low return on its outlet footprint. The volume is diluted across a large number of outlets. 

""")

    # ---- Special Outlets ----
    with tab6:
        st.markdown("###  Question: How does shipment performance differ between Hubbali and Belagavi?")
        st.subheader("Focused Analysis: Hubbali & Belagavi Depots")

        view_mode = st.radio("View Mode", ["Absolute", "Percentage"], horizontal=True, key="special_view_mode")
        depot_group_choice = st.radio("Select Depot Group", ["All", "Hubbali", "Belagavi"], horizontal=True, key="special_depot_group")

        patterns = {
            "Hubbali": ["HUBBALLI-1", "HUBBALLI-2"],
            "Belagavi": ["BELAGAVI", "BELAGAVI-2"],
            "All": ["HUBBALLI-1", "HUBBALLI-2", "BELAGAVI", "BELAGAVI-2"]
        }

        selected_patterns = patterns[depot_group_choice]

        df_special = df_filtered[
            df_filtered["DBF_DEPOT"].str.upper().str.contains("|".join([p.upper() for p in selected_patterns]), na=False)
        ]

        if not df_special.empty:
            comp = df_special.groupby("DBF_DEPOT")[VOLUME_COL].sum().round(0).reset_index()

            if view_mode == "Percentage":
                comp["Value"] = (comp[VOLUME_COL] / comp[VOLUME_COL].sum().round(0)) * 100
                value_col = "Value"
                title_suffix = " (%)"
            else:
                comp["Value"] = comp[VOLUME_COL]
                value_col = "Value"
                title_suffix = ""

            fig_pie = px.pie(comp, values=value_col, names="DBF_DEPOT",
                            title=f"Volume Distribution (Special Depots){title_suffix}", hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)

            outlet_counts = df_special.groupby("DBF_DEPOT")["DBF_OUTLET_CODE"].nunique().reset_index()
            outlet_counts.rename(columns={"DBF_OUTLET_CODE": "Unique_Outlet_Count"}, inplace=True)

            fig_bar = px.bar(outlet_counts, x="DBF_DEPOT", y="Unique_Outlet_Count", text="Unique_Outlet_Count",
                            title="Number of Unique Outlets (Special Depots)")
            fig_bar.update_traces(textposition="outside")
            st.plotly_chart(fig_bar, use_container_width=True)

            trend = df_special.groupby(["YearMonth", "DBF_DEPOT"])[VOLUME_COL].sum().reset_index()
            trend["YearMonth"] = trend["YearMonth"].astype(str)

            if view_mode == "Percentage":
                trend["Total"] = trend.groupby("YearMonth")[VOLUME_COL].transform("sum")
                trend["Value"] = ((trend[VOLUME_COL] / trend["Total"]) * 100).round(0)
                y_title = "Volume Share (%)"
            else:
                trend["Value"] = trend[VOLUME_COL]
                y_title = "Volume"

            fig_trend = px.line(trend, x="YearMonth", y="Value", color="DBF_DEPOT",
                                title=f"Monthly Shipment Trend (Special Depots){title_suffix}", markers=True)
            fig_trend.update_yaxes(title_text=y_title)
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown("""
### **Answer:**
- All four entities, particularly the top two (BELAGAVI and HUBBALLI-1), contribute a significant portion of the volume. Management should ensure dedicated support and resources are allocated proportionally to maintain their performance.
- HUBBALLI-1 (Light Blue) is the dominant sub-depot, controlling 61.4% of the Hubballi group's total volume.The Hubballi area's volume performance is primarily dependent on the performance and stability of HUBBALLI-1. Any operational issue or decline in sales at HUBBALLI-1 would severely impact the entire Hubballi group's performance.Strategic initiatives should be aimed at boosting HUBBALLI-2's volume to achieve a more even distribution, which would reduce the over-reliance on HUBBALLI-1.
- The highest risk in this focused group lies in the BELAGAVI 1 depot. Any major operational setback or sales decline here would result in a severe, sudden drop in the overall volume for the entire Belagavi area. Focused marketing and sales investment is needed to significantly increase its volume and achieve a more diversified risk profile, perhaps aiming for a 45%-55% split over time.
""")
