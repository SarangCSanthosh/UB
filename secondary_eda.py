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
            "Select Year(s)", options=sorted(df["Year"].dropna().unique()), default=[df["Year"].max()]
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
        st.subheader("Shipment Trends")

        # --- Controls ---
        granularity = st.radio(
            "Granularity", ["Yearly", "Quarterly", "Monthly"], horizontal=True, key="trend_granularity"
        )
        view_mode = st.radio(
            "Display Mode", ["Absolute", "Percentage"], horizontal=True, key="trend_view"
        )

        # --- Shipment Trend (Filtered Main Data) ---
        if granularity == "Yearly":
            df_filtered["Label"] = df_filtered["Year"].astype(str)
        elif granularity == "Quarterly":
            df_filtered["Label"] = df_filtered["Quarter"].astype(str)
        else:
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
        df_normalised = load_normalized_data(r"https://docs.google.com/spreadsheets/d/1lg0iIgKx9byQj7d2NZO-k1gKdFuNxxQe/export?format=xlsx")

        # Apply the same filter logic
        if filter_mode == "Year":
            if year_choice:
                df_normalised = df_normalised[df_normalised["ACTUAL_DATE"].dt.year.isin(year_choice)]
        else:
            df_normalised = df_normalised[
                (df_normalised["ACTUAL_DATE"].dt.date >= start_date) &
                (df_normalised["ACTUAL_DATE"].dt.date <= end_date)
            ]

        # Add granularity columns after filtering
        df_normalised["Year"] = df_normalised["ACTUAL_DATE"].dt.year
        df_normalised["Quarter"] = df_normalised["ACTUAL_DATE"].dt.to_period("Q").astype(str)
        df_normalised["YearMonth"] = df_normalised["ACTUAL_DATE"].dt.to_period("M").astype(str)

        # Aggregate by granularity
        if granularity == "Yearly":
            norm_df = df_normalised.groupby("Year")["VOLUME"].sum().reset_index()
            norm_df["Label"] = norm_df["Year"].astype(str)
        elif granularity == "Quarterly":
            norm_df = df_normalised.groupby("Quarter")["VOLUME"].sum().reset_index()
            norm_df["Label"] = norm_df["Quarter"].astype(str)
        else:
            norm_df = df_normalised.groupby("YearMonth")["VOLUME"].sum().reset_index()
            norm_df["Label"] = norm_df["YearMonth"].astype(str)

        # Normalize the normalized data if in Percentage mode
        if view_mode == "Percentage":
            total_norm = norm_df["VOLUME"].sum()
            norm_df["Normalized_Value"] = (norm_df["VOLUME"] / total_norm) * 100
            norm_y_title = "Normalized Volume (%)"
        else:
            norm_df["Normalized_Value"] = norm_df["VOLUME"]
            norm_y_title = "Normalized Volume"

        # --- Plot both trends ---
        fig = go.Figure()

        # Shipment trend (main data)
        fig.add_trace(
            go.Scatter(
                x=trend_df["Label"],
                y=trend_df["Value"],
                mode="lines+markers",
                name=f"Shipment Trend ({granularity}, {view_mode})",
                fill="tozeroy",
                yaxis="y1"
            )
        )

        # Normalized volume (secondary axis)
        fig.add_trace(
            go.Scatter(
                x=norm_df["Label"],
                y=norm_df["Normalized_Value"],
                mode="lines+markers",
                name=f"Normalized {granularity} Volume",
                line=dict(color="red"),
                yaxis="y2"
            )
        )

        fig.update_layout(
            title=f"Shipment Trend vs Normalized Volume ({granularity})",
            xaxis=dict(title=granularity),
            yaxis=dict(title=y_title, side="left"),
            yaxis2=dict(title=norm_y_title, overlaying="y", side="right"),
            legend_title="Metrics"
        )

        st.plotly_chart(fig, use_container_width=True)




    # ---- Top Outlets ----
    with tab2:
        st.subheader("Top Outlets by Volume")
        outlet_volume = df_filtered.groupby(OUTLET_COL)[VOLUME_COL].sum().reset_index()
        top_n = st.slider("Top-N Outlets", 5, 25, 10)
        top_outlets = outlet_volume.sort_values(by=VOLUME_COL, ascending=False).head(top_n)
        fig = px.treemap(top_outlets, path=[OUTLET_COL], values=VOLUME_COL, title=f"Top {top_n} Outlets Treemap")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(top_outlets.set_index(OUTLET_COL).round(2))

    # ---- Depot Analysis ----
    with tab3:
        st.subheader("Depot-wise ABC Analysis")
        if "DBF_DEPOT" in df_filtered.columns:
            depot_volume = df_filtered.groupby("DBF_DEPOT")[VOLUME_COL].sum().reset_index()
            depot_volume = depot_volume.sort_values(by=VOLUME_COL, ascending=False).reset_index(drop=True)
            depot_volume["Cum_Volume"] = depot_volume[VOLUME_COL].cumsum()
            depot_volume["Cum_Percentage"] = 100 * depot_volume["Cum_Volume"] / depot_volume[VOLUME_COL].sum()

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
            st.dataframe(depot_volume.round(2))
        else:
            st.info("DBF_DEPOT column not found.")

    # ---- Region Donut ----
    with tab4:
        st.subheader("Region-wise Volume Share")
        if "DBF_REGION" in df_filtered.columns:
            region_volume = df_filtered.groupby("DBF_REGION")[VOLUME_COL].sum().reset_index()
            fig = px.pie(region_volume, values=VOLUME_COL, names="DBF_REGION", hole=0.5,
                         title="Volume Distribution by Region")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(region_volume.set_index("DBF_REGION").round(2))
        else:
            st.info("DBF_REGION column not found.")

    # ---- Region Stacked ----
    with tab5:
        st.subheader("Outlets & Volume by Region (100% Share)")
        if "DBF_REGION" in df_filtered.columns and "DBF_OUTLET_CODE" in df_filtered.columns:
            region_stats = df_filtered.groupby("DBF_REGION").agg(
                Outlet_Count=("DBF_OUTLET_CODE", "nunique"),
                Total_Volume=(VOLUME_COL, "sum"),
            ).reset_index()

            region_stats["Outlet %"] = (region_stats["Outlet_Count"] / region_stats["Outlet_Count"].sum()) * 100
            region_stats["Volume %"] = (region_stats["Total_Volume"] / region_stats["Total_Volume"].sum()) * 100

            melted = region_stats.melt(
                id_vars="DBF_REGION",
                value_vars=["Outlet %", "Volume %"],
                var_name="Metric",
                value_name="Percent",
            )

            fig = px.bar(melted, x="DBF_REGION", y="Percent", color="Metric",
                         barmode="stack", text=melted["Percent"].round(2),
                         title="Outlets vs Volume % by Region")
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(region_stats.set_index("DBF_REGION").round(2))
        else:
            st.info("DBF_REGION or DBF_OUTLET_CODE not found.")

    # ---- Special Outlets ----
    with tab6:
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
            comp = df_special.groupby("DBF_DEPOT")[VOLUME_COL].sum().reset_index()

            if view_mode == "Percentage":
                comp["Value"] = (comp[VOLUME_COL] / comp[VOLUME_COL].sum()) * 100
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
                trend["Value"] = (trend[VOLUME_COL] / trend["Total"]) * 100
                y_title = "Volume Share (%)"
            else:
                trend["Value"] = trend[VOLUME_COL]
                y_title = "Volume"

            fig_trend = px.line(trend, x="YearMonth", y="Value", color="DBF_DEPOT",
                                title=f"Monthly Shipment Trend (Special Depots){title_suffix}", markers=True)
            fig_trend.update_yaxes(title_text=y_title)
            st.plotly_chart(fig_trend, use_container_width=True)
