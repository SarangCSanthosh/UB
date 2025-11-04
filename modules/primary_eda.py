import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from collections import Counter

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

@st.cache_data
def load_pci_from_gsheet(gsheet_url: str, sheet_name: str = "PCI") -> pd.DataFrame:
    # The gsheet_url you gave is already the export-to-xlsx URL
    df = pd.read_excel(gsheet_url, sheet_name=sheet_name)
    # Clean up names
    df.columns = [c.strip() for c in df.columns]
    if "Row Labels" in df.columns:
        df.rename(columns={"Row Labels": "Location"}, inplace=True)
    df["Location"] = df["Location"].astype(str).str.strip().str.upper()
    return df

# Use the URL you provided
gsheet_url = "https://docs.google.com/spreadsheets/d/1Pg0DkCaqQJymbrkIIqAcjbgCa-7MVHJB/export?format=xlsx"
df_pci = load_pci_from_gsheet(gsheet_url, sheet_name="PCI")

@st.cache_data
def load_event_calendar(sheet_id: str):
    """
    Loads the event calendar from a public Google Sheet.
    Input: Google Sheet ID
    Output: DataFrame
    """
    try:
        # Build a direct download link
        #download_url = f"https://docs.google.com/spreadsheets/d/1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO/export?format=xlsx"
        
        download_url = f"https://docs.google.com/spreadsheets/d/1GxgGo6waZV7WDsF50v_nYSu2mxEX6bmj/export?format=xlsx"

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
    #kpi_shipments = df_latest.shape[0]

    prev_volume = df_prev[VOLUME_COL].sum() if not df_prev.empty else None
    #prev_shipments = df_prev.shape[0] if not df_prev.empty else None

    delta_volume = _pct_change(kpi_volume, prev_volume)
    #delta_shipments = _pct_change(kpi_shipments, prev_shipments)

    # Show KPIs (fixed, not linked to filters)
    col1, col2 = st.columns(2)
    col1.metric(
        "Total Volume",
        f"{kpi_volume:,.0f}",
        f"{delta_volume:+.0f}%" if delta_volume is not None else None
    )
    col2.metric("Unique Locations", f"{kpi_outlets}")
    

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
        "Trends", "Top/Bottom Locations", "Clustering", "Heatmaps"
    ])

    # ---- Tab 1: Trends ----
    # ---- Tab 1: Trends ----
    with tab1:
        st.markdown("### Question: What trends do we see in shipments across months, quarters, or years?")
        st.subheader("Shipment Trends")
    
        # --- Chart Type ---
        chart_type = st.radio(
            "Select Chart Type:",
            ["Shipment Trend"],
            horizontal=True
        )
    
        if chart_type == "Shipment Trend":
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
    
            # --- Prepare date features ---
            df_filtered["SHIPMENT_DATE"] = pd.to_datetime(df_filtered["SHIPMENT_DATE"], errors="coerce")
            df_filtered["Year"] = df_filtered["SHIPMENT_DATE"].dt.year.astype(int)
            df_filtered["Quarter"] = "Q" + df_filtered["SHIPMENT_DATE"].dt.quarter.astype(str)
            df_filtered["YearMonth"] = df_filtered["SHIPMENT_DATE"].dt.to_period("M").astype(str)
    
            # Remove any old label column
            if "Label" in df_filtered.columns:
                df_filtered.drop(columns=["Label"], inplace=True)
    
            # --- Label for granularity ---
            if granularity == "Yearly":
                df_filtered["Label"] = df_filtered["Year"].astype(str)
            elif granularity == "Quarterly":
                df_filtered["Label"] = df_filtered["Quarter"] + " " + df_filtered["Year"].astype(str)
            else:  # Monthly
                df_filtered["Label"] = df_filtered["YearMonth"]
    
            # --- Aggregate ---
            trend_df = df_filtered.groupby("Label")[VOLUME_COL].sum().reset_index()
    
            # --- Convert to percentage if needed ---
            if value_type == "Percentage":
                total_sum = trend_df[VOLUME_COL].sum()
                trend_df["Value"] = (trend_df[VOLUME_COL] / total_sum) * 100
                y_title = "Percentage (%)"
            else:
                trend_df["Value"] = trend_df[VOLUME_COL]
                y_title = "Volume"
    
            # --- Plot Trend ---
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trend_df["Label"],
                    y=trend_df["Value"],
                    mode="lines+markers",
                    name=f"Shipment Trend ({granularity}, {value_type})",
                    fill="tozeroy",
                    line=dict(color="mediumvioletred", width=2),
                    marker=dict(size=6)
                )
            )
    
            fig.update_layout(
                xaxis=dict(type='category'),
                yaxis=dict(title=y_title),
                title=f"Shipment Trend ({granularity}, {value_type})",
                template="plotly_white",
                height=450,
                margin=dict(l=40, r=40, t=60, b=40)
            )
    
            st.plotly_chart(fig, use_container_width=True)
    
                  
                                
        st.markdown("""
        ### **Insights: Shipment Volume Analysis (2023–2024)**
        
        - Shipment volume 📉 between 2023 and 2024.
        - Shipment activity cyclical --- 📉 in Q3 of both years
        - 2024 - May 📈 , August 📉    
        """)
    
    
    
        # ---- Tab 2: Top/Bottom Locations ----
        with tab2:
            st.markdown("###  Question: Where are shipments highest and where are they lagging?")
            st.subheader("Top/Bottom Locations")
            if VOLUME_COL in df_filtered.columns and LOCATION_COL in df_filtered.columns:
    
            # --- Normalize location names for consistency ---
                df_filtered[LOCATION_COL] = df_filtered[LOCATION_COL].replace({
                    "HUBBALLI-1": "HUBBALLI",
                    "HUBBALLI-2": "HUBBALLI",
                    "BELAGAVI-2": "BELAGAVI",
                    "CHIKODI": "CHIKKODI"
                })
        
                # --- Group shipments by location ---
                location_volume = (
                    df_filtered.groupby(LOCATION_COL)[VOLUME_COL]
                    .sum()
                    .round(0)
                    .reset_index()
                )
        
                choice = st.radio("Select Type", ["Top", "Bottom"], horizontal=True)
                value_type = st.radio("Value Type", ["Absolute", "Percentage"], horizontal=True)
                n_locations = st.slider("Number of Locations", 5, 25, 10)
        
                if choice == "Top":
                    locs = location_volume.sort_values(by=VOLUME_COL, ascending=False).head(n_locations)
                else:
                    locs = location_volume.sort_values(by=VOLUME_COL, ascending=True).head(n_locations)
        
                if value_type == "Percentage":
                    total_volume = df_filtered[VOLUME_COL].sum().round(0)
                    locs[VOLUME_COL] = (locs[VOLUME_COL] / total_volume * 100).round(0)
        
                # --- Determine PCI column based on selected year(s) ---
                if filter_mode == "Year":
                    years_selected = selected_years  
                else:
                    years_selected = sorted(df_filtered["Year"].dropna().unique())
        
                if set(years_selected) == {2023}:
                    pci_col = "Per capita - 2022-23"
                elif set(years_selected) == {2024}:
                    pci_col = "per capita - 2023-24"
                else:
                    pci_col = "Grand Total"
        
                # --- Load PCI dataset ---
                df_pci = pd.read_excel(
                    "https://docs.google.com/spreadsheets/d/1Pg0DkCaqQJymbrkIIqAcjbgCa-7MVHJB/export?format=xlsx",
                    sheet_name="PCI"
                )
        
                df_pci.columns = [c.strip() for c in df_pci.columns]
                df_pci.rename(columns={"Row Labels": "Location"}, inplace=True)
                df_pci["Location"] = df_pci["Location"].str.strip().str.upper()
        
                # --- Merge HUBBALLI & BELAGAVI variants ---
                df_pci["Location"] = df_pci["Location"].replace({
                    "HUBBALLI-1": "HUBBALLI",
                    "HUBBALLI-2": "HUBBALLI",
                    "BELAGAVI-2": "BELAGAVI",
                    "CHIKODI": "CHIKKODI"
                })
        
                # --- Aggregate PCI values ---
                df_pci = df_pci.groupby("Location", as_index=False)[pci_col].mean()
        
                # --- Merge with shipment data ---
                locs["Location_upper"] = locs[LOCATION_COL].str.strip().str.upper()
                df_merged = pd.merge(
                    locs,
                    df_pci,
                    left_on="Location_upper",
                    right_on="Location",
                    how="left"
                )
        
                df_merged.rename(columns={pci_col: "Per Capita Income"}, inplace=True)
    
                if value_type == "Percentage":
                    total_volume = df_merged[VOLUME_COL].sum()
                    total_pci = df_merged["Per Capita Income"].sum()
                
                    df_merged[VOLUME_COL] = (df_merged[VOLUME_COL] / total_volume * 100).round(1)
                    df_merged["Per Capita Income"] = (df_merged["Per Capita Income"] / total_pci * 100).round(1)
        
                # --- Melt for bar chart ---
                df_melted = df_merged.melt(
                    id_vars=[LOCATION_COL],
                    value_vars=[VOLUME_COL, "Per Capita Income"],
                    var_name="Metric",
                    value_name="Value"
                )
        
                # --- Clustered (side-by-side) bar chart ---
                fig = px.bar(
                    df_melted,
                    x="Value",
                    y=LOCATION_COL,
                    color="Metric",
                    orientation="h",
                    #text=df_melted["Value"].round(0),
                    barmode="group",  # <<< CLUSTERED BARS
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
        
                fig.update_traces(textposition="outside")
        
                if choice == "Top":
                    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
                else:
                    fig.update_layout(yaxis=dict(categoryorder="total descending"))
        
                fig.update_layout(
                    title=f"Shipment Volume vs Per Capita Income ({pci_col})",
                    xaxis_title="Value",
                    yaxis_title="Location",
                    legend_title="Metric",
                    template="plotly_dark",
                    height=600,
                    margin=dict(t=80)
                )
        
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
### **Insights:**
-	🏭 Hubballi -  high shipment volume but low per capita income --- core logistics hub. 
-	⚙️ Belagavi - high shipment activity but low income --- industrial or outbound trade flow rather than local consumption
-	📈 Kalaburagi - highest shipment volume and per capita income --- strong positive correlation between income and logistics activity.
""")


    # ---- Tab 3: Heatmaps ----
    with tab4:
        st.markdown("###  Question: Where are shipment volumes most concentrated?")
        if VOLUME_COL in df_filtered.columns and LOCATION_COL in df_filtered.columns:
            st.markdown("**Volume Heatmap**")
            agg_loc = df_filtered.groupby(LOCATION_COL)[VOLUME_COL].sum().round(0).sort_values(ascending=False)
            top_n_heat = st.slider("Top-N Locations for Heatmap", 5, 50, 15, 1)
            keep_locs = agg_loc.head(top_n_heat).index.tolist()
            df_hm = df_filtered[df_filtered[LOCATION_COL].isin(keep_locs)]
            pivot = pd.pivot_table(df_hm, index="YearMonth", columns=LOCATION_COL,
                                   values=VOLUME_COL, aggfunc="sum", fill_value=0)
            pivot.index = pivot.index.astype(str)
            fig_hm = px.imshow(pivot.T, aspect="auto", labels=dict(color="Volume"))
            st.plotly_chart(fig_hm, use_container_width=True)
            st.markdown("""
### **Insights:**
- Cyclical pattern across most locations. 
- Volume is consistently lower at the beginning (Jan/Feb) and end (Oct/Dec) of the year. 
- Massive volume spike for Vijayapura around April-May 2024. 
- Hubballi’s steady shipment levels across all quarters --- logistics powerhouse with minimal seasonal variation

""")

    # ---- Tab 4: Clustering ----
    with tab3:
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

                # --- Add total shipment volume for bubble size ---
                comp_df["Total_Volume"] = pivot.sum(axis=1).values
    
                # --- Bubble Chart ---
                fig_bubble = px.scatter(
                    comp_df,
                    x="PC1",
                    y="PC2",
                    size="Total_Volume",
                    color="Cluster",
                    hover_name=LOCATION_COL,
                    size_max=45,
                    color_continuous_scale="Turbo",
                    title="Clustering of Locations (Bubble Size = Total Shipment Volume)"
                )
                fig_bubble.update_traces(marker=dict(opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
                st.plotly_chart(fig_bubble, use_container_width=True)
    
                # --- Display Cluster Table ---
                st.dataframe(
                    comp_df[[LOCATION_COL, "Cluster", "Total_Volume"]]
                    .sort_values("Cluster")
                    .round(0)
                    .set_index(LOCATION_COL),
                    use_container_width=True,
                    height=250
                )
                #st.dataframe(comp_df[[LOCATION_COL, "Cluster"]].round(0), width=400, height=200)
                st.markdown("""
### **Insights:**
- 🟣 Cluster 0: Low-demand rural zones — Bagalkot, Sedam, Koppal, Ballari (require shared logistics support).
- 🟢 Cluster 1: Balanced trade regions — Raichur, Hosapete, Belagavi (suitable for scaling operations).
- 🟠 Cluster 2: Emerging hubs — Vijayapura, Bidar, Yadgiri (show increasing shipment potential).
- 🔴 Cluster 3: Major shipment centres — Kalaburagi, Hubballi (strategic for warehousing and route optimisation).

""")

    
# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
