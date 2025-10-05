import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# ===============================
# Utility functions
# ===============================
@st.cache_data
def load_excel(path_or_file, sheet_name="primary working data"):
    return pd.read_excel(path_or_file, sheet_name=sheet_name)

@st.cache_data
def load_normalized_shipments(path_or_file):
    df_norm = pd.read_excel(path_or_file)
    df_norm["ACTUAL_DATE"] = pd.to_datetime(df_norm["ACTUAL_DATE"], errors='coerce')
    return df_norm

def prepare_dates(df, date_col="SHIPMENT_DATE"):
    if date_col not in df.columns:
        date_like = [c for c in df.columns if "date" in c.lower()]
        if date_like:
            date_col = date_like[0]
        else:
            raise ValueError("Could not find a date column.")
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df["Year"] = df[date_col].dt.year
    df["YearMonth"] = df[date_col].dt.to_period("M")
    df["Quarter"] = df[date_col].dt.to_period("Q")
    return df, date_col

def _pct_change(curr, prev):
    if prev is None or pd.isna(prev) or prev == 0:
        return None
    return (curr - prev) / prev * 100.0

# ===============================
# Main app
# ===============================
def run():
    st.title("Primary Dataset Dashboard")

    # --------------------------
    # LOAD DATA
    # --------------------------
    # Use shared Google Sheets export link instead of local file
    google_sheets_url = "https://docs.google.com/spreadsheets/d/19Q2PmPZNkyA9o5wXwlsmXfo70v055Q9h/export?format=xlsx"
    df = load_excel(google_sheets_url, sheet_name="primary working data")

    VOLUME_COL = "VOLUME"
    LOCATION_COL = "LOCATION"

    df, DATE_COL = prepare_dates(df, "SHIPMENT_DATE" if "SHIPMENT_DATE" in df.columns else None)
    if "CY" not in df.columns:
        df["CY"] = df["Year"]

    # --------------------------
    # FIXED KPI METRICS (YOY)
    # --------------------------
    latest_year = df["Year"].max()
    prev_year = latest_year - 1

    df_latest = df[df["Year"] == latest_year]
    df_prev = df[df["Year"] == prev_year]

    kpi_volume = df_latest[VOLUME_COL].sum()
    kpi_outlets = df_latest[LOCATION_COL].nunique()
    kpi_shipments = df_latest.shape[0]

    prev_volume = df_prev[VOLUME_COL].sum() if not df_prev.empty else None
    prev_shipments = df_prev.shape[0] if not df_prev.empty else None

    delta_volume = _pct_change(kpi_volume, prev_volume)
    delta_shipments = _pct_change(kpi_shipments, prev_shipments)

    # Show KPIs (fixed, not linked to filters)
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Volume",
        f"{kpi_volume:,.0f}",
        f"{delta_volume:+.2f}%" if delta_volume is not None else None
    )
    col2.metric("Unique Locations", f"{kpi_outlets}")
    col3.metric(
        "Total Shipments",
        f"{kpi_shipments}",
        f"{delta_shipments:+.2f}%" if delta_shipments is not None else None
    )

    st.caption(f"YoY change: {latest_year} vs {prev_year}")
    st.markdown("---")

    # --------------------------
    # SIDEBAR FILTERS
    # --------------------------
    st.sidebar.header("Filters")
    filter_mode = st.sidebar.radio("Filter by:", ["Year", "Date Range"], horizontal=True)

    df_base = df.copy()

    # ---- Location Filter with Select All ----
    locations = df_base[LOCATION_COL].unique().tolist()
    locations = sorted(locations)

    select_all = st.sidebar.checkbox("Select All Locations", value=True)
    if select_all:
        selected_locations = st.sidebar.multiselect(
            "Select Location(s)", options=locations, default=locations
        )
    else:
        selected_locations = st.sidebar.multiselect(
            "Select Location(s)", options=locations
        )

    if selected_locations and LOCATION_COL in df_base.columns:
        df_loc = df_base[df_base[LOCATION_COL].isin(selected_locations)]
    else:
        df_loc = df_base

    # ---- Year or Date Range Filter ----
    if filter_mode == "Year":
        available_years = sorted(df_loc["Year"].dropna().unique())
        selected_years = st.sidebar.multiselect(
            "Select Year(s)", options=available_years, default=available_years
        )
        df_filtered = df_loc[df_loc["Year"].isin(selected_years)] if selected_years else df_loc
    else:
        start_date = st.sidebar.date_input("Start Date", df_loc[DATE_COL].min().date())
        end_date = st.sidebar.date_input("End Date", df_loc[DATE_COL].max().date())
        mask = (df_loc[DATE_COL].dt.date >= start_date) & (df_loc[DATE_COL].dt.date <= end_date)
        df_filtered = df_loc.loc[mask]

    # --------------------------
    # TABS FOR VISUALS
    # --------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Trends", "Top/Bottom Locations", "Heatmaps", "Clustering"
    ])

    # ---- Tab 1: Trends ----
    with tab1:
        st.markdown("###  Question: What trends do we see in shipments across months, quarters, or years?")
        st.subheader("Shipment Trends")
        granularity = st.radio(
            "Select Granularity", 
            ["Yearly", "Quarterly", "Monthly"], 
            horizontal=True, 
            key="granularity_radio"
        )
        value_type = st.radio(
            "Value Type (Trends)", 
            ["Absolute", "Percentage"], 
            horizontal=True, 
            key="trend_value_type"
        )

        if granularity == "Yearly":
            trend = df_filtered.groupby("Year")[VOLUME_COL].sum().reset_index()
            trend["Year_str"] = trend["Year"].astype(int).astype(str)  # Convert to string for axis
        
            if value_type == "Percentage":
                trend[VOLUME_COL] = (trend[VOLUME_COL] / trend[VOLUME_COL].sum() * 100).round(2)
                y_title = "Volume (%)"
            else:
                y_title = "Volume"
        
            fig = px.line(
                trend,
                x="Year_str",  # Use the string column here
                y=VOLUME_COL,
                markers=True,
                title="Yearly Shipment Volume"
            )
            fig.update_yaxes(title_text=y_title)



        elif granularity == "Quarterly":
            trend = df_filtered.groupby("Quarter")[VOLUME_COL].sum().reset_index()
            trend["Quarter"] = trend["Quarter"].astype(str)
            if value_type == "Percentage":
                trend[VOLUME_COL] = (trend[VOLUME_COL] / trend[VOLUME_COL].sum() * 100).round(2)
                y_title = "Volume (%)"
            else:
                y_title = "Volume"
            fig = px.line(trend, x="Quarter", y=VOLUME_COL, markers=True, title="Quarterly Shipment Volume")
            fig.update_yaxes(title_text=y_title)

        else:  # Monthly
            trend = df_filtered.groupby("YearMonth")[VOLUME_COL].sum().reset_index()
            trend["YearMonth"] = trend["YearMonth"].astype(str)
            if value_type == "Percentage":
                trend[VOLUME_COL] = (trend[VOLUME_COL] / trend[VOLUME_COL].sum() * 100).round(2)
                y_title = "Volume (%)"
            else:
                y_title = "Volume"
            fig = px.line(trend, x="YearMonth", y=VOLUME_COL, markers=True, title="Monthly Shipment Volume")
            fig.update_yaxes(title_text=y_title)

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(trend.round(2))
        st.markdown("""
### **Answer: Shipment Volume Analysis (2023–2024)**

- The company or entity represented by this data experienced a substantial decrease in its yearly shipment volume between 2023 and 2024. This indicates a contraction in shipping activity or demand.
- The company's shipment activity is highly cyclical, with a common drop observed in Q3 of both years (though much more severe in 2024). The period from 2024 Q2 to 2024 Q3 represents a major performance concern due to the unprecedented low volume.
- The shipment volume is highly unstable on a monthly basis. While the first half of 2024 contained the single highest month (May), the subsequent crash to the lowest volume in August suggests a major disruption, either due to internal capacity issues, a massive short-term demand spike followed by exhaustion, or an abrupt external market shock. The recent recovery in Q4 2024 (September and October) is a positive sign, but volume remains far below peak levels.
""")



    # ---- Tab 2: Top/Bottom Locations ----
    with tab2:
        st.markdown("###  Question: Where are shipments highest and where are they lagging?")
        st.subheader("Top/Bottom Locations")
        if VOLUME_COL in df_filtered.columns and LOCATION_COL in df_filtered.columns:
            location_volume = df_filtered.groupby(LOCATION_COL)[VOLUME_COL].sum().reset_index()

            choice = st.radio("Select Type", ["Top", "Bottom"], horizontal=True)
            value_type = st.radio("Value Type", ["Absolute", "Percentage"], horizontal=True)
            n_locations = st.slider("Number of Locations", 5, 25, 10)

            if choice == "Top":
                locs = location_volume.sort_values(by=VOLUME_COL, ascending=False).head(n_locations)
            else:
                locs = location_volume.sort_values(by=VOLUME_COL, ascending=True).head(n_locations)

            if value_type == "Percentage":
                total_volume = df_filtered[VOLUME_COL].sum()
                locs[VOLUME_COL] = (locs[VOLUME_COL] / total_volume * 100).round(2)

            fig = px.bar(
                locs,
                x=VOLUME_COL,
                y=LOCATION_COL,
                orientation="h",
                text=locs[VOLUME_COL].round(2)
            )
            fig.update_traces(textposition="outside")
            if choice == "Top":
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            else:
                fig.update_layout(yaxis=dict(categoryorder="total descending"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
### **Answer:**
The shipment volume is heavily concentrated in the top three locations, particularly Kalaburagi and Bidar. Any strategies to increase overall volume should likely focus on maintaining performance in these top three areas or identifying the factors driving the success of these locations to replicate them elsewhere. The business has a clear tiering of locations based on shipment volume.
Cities with lowest shipment volumes are - Gokak , Jamkhandi , Haveri. Disruptive strategies could be implemented here to improve performance
""")


    # ---- Tab 3: Heatmaps ----
    with tab3:
        st.markdown("###  Question: Where are shipment volumes most concentrated?")
        if VOLUME_COL in df_filtered.columns and LOCATION_COL in df_filtered.columns:
            st.markdown("**Volume Heatmap**")
            agg_loc = df_filtered.groupby(LOCATION_COL)[VOLUME_COL].sum().sort_values(ascending=False)
            top_n_heat = st.slider("Top-N Locations for Heatmap", 5, 50, 15, 1)
            keep_locs = agg_loc.head(top_n_heat).index.tolist()
            df_hm = df_filtered[df_filtered[LOCATION_COL].isin(keep_locs)]
            pivot = pd.pivot_table(df_hm, index="YearMonth", columns=LOCATION_COL,
                                   values=VOLUME_COL, aggfunc="sum", fill_value=0)
            pivot.index = pivot.index.astype(str)
            fig_hm = px.imshow(pivot.T, aspect="auto", labels=dict(color="Volume"))
            st.plotly_chart(fig_hm, use_container_width=True)

    # ---- Tab 4: Clustering ----
    with tab4:
        st.markdown("###  Question: Are there distinct groups of locations based on shipment activity?")
        if VOLUME_COL in df_filtered.columns and LOCATION_COL in df_filtered.columns:
            st.subheader("Clustering")
            granularity = st.radio("Clustering Granularity", ["Month", "Quarter", "Year"], horizontal=True)
            normalize = st.checkbox("Normalize Features", value=True)
            k = st.slider("Clusters (k)", 2, 10, 4)

            time_col = "YearMonth" if granularity == "Month" else "Quarter" if granularity == "Quarter" else "Year"
            pivot = pd.pivot_table(df_filtered, index=LOCATION_COL, columns=time_col,
                                   values=VOLUME_COL, aggfunc="sum", fill_value=0)
            pivot.columns = pivot.columns.astype(str)
            pivot = pivot.loc[pivot.sum(axis=1) > 0]

            if pivot.shape[0] < k:
                st.warning(f"Not enough locations ({pivot.shape[0]}) to form {k} clusters. Please reduce k or select more locations.")
            else:
                X = pivot.values
                if normalize:
                    X = MinMaxScaler().fit_transform(X)

                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(X)

                comps = PCA(n_components=2, random_state=42).fit_transform(X)
                comp_df = pd.DataFrame({LOCATION_COL: pivot.index, "PC1": comps[:, 0],
                                        "PC2": comps[:, 1], "Cluster": labels})

                fig_scatter = px.scatter(comp_df, x="PC1", y="PC2", color="Cluster", hover_name=LOCATION_COL)
                st.plotly_chart(fig_scatter, use_container_width=True)
                st.dataframe(comp_df[[LOCATION_COL, "Cluster"]].round(2))

    
# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
