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
    # ---- Tab 1: Trends ----
    with tab1:
        st.markdown("###  Question: What trends do we see in shipments across months, quarters, or years?")
        st.subheader("Shipment Trends and Event Calendar")
    
        # --- Chart Selection ---
        chart_type = st.radio(
            "Select Chart Type:",
            ["Shipment Trend", "Event Calendar"],
            horizontal=True
        )
    
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
    
        # ===============================
        # 📈 SHIPMENT TREND
        # ===============================
        if chart_type == "Shipment Trend":
            # --- Create Label in df_filtered based on granularity ---
            if granularity == "Yearly":
                #df_filtered["Label"] = df_filtered["Year"].astype(int).astype(str)
                df_filtered["Label"]=df_filtered.groupby("Year")[VOLUME_COL].sum().reset_index()
            elif granularity == "Quarterly":
                df_filtered["Label"] = df_filtered["Quarter"].astype(str) + " " + df_filtered["Year"].astype(str)
            else:
                df_filtered["Label"] = df_filtered["YearMonth"].astype(str)
    
            trend_df = df_filtered.groupby("Label")[VOLUME_COL].sum().reset_index()
    
            # --- Convert to percentage if needed ---
            if value_type == "Percentage":
                total_sum = trend_df[VOLUME_COL].sum()
                trend_df["Value"] = (trend_df[VOLUME_COL] / total_sum) * 100
                y_title = "Percentage (%)"
            else:
                trend_df["Value"] = trend_df[VOLUME_COL]
                y_title = "Volume"
    
            # --- Load Event Calendar ---
            EVENT_CSV_URL = "https://docs.google.com/spreadsheets/d/1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO/export?format=xlsx"
            df_events = load_event_calendar(EVENT_CSV_URL)
            df_events["Date"] = pd.to_datetime(df_events["Date"], errors="coerce")
    
            # --- Create Label column in df_events based on granularity ---
            if granularity == "Yearly":
                df_events["Label"] = df_events["Date"].dt.year.astype(int).astype(str)
            elif granularity == "Quarterly":
                df_events["Label"] = "Q" + df_events["Date"].dt.quarter.astype(str) + " " + df_events["Date"].dt.year.astype(str)
            else:  # Monthly
                df_events["Label"] = df_events["Date"].dt.to_period("M").astype(str)
    
            # --- Clean and normalize event names ---
            def clean_event_name(text):
                if pd.isna(text):
                    return None
                text = str(text).strip()
                if not text or text.lower() in ["nan", "none"]:
                    return None
    
                # Fix common corrupt patterns
                text = text.replace("Against", "")
                text = text.replace("Footll", "Football")
                text = text.replace("Pro Ka", "Pro Kabbadi")
                text = text.replace("C ", " ")
                text = text.replace("IND World cup", "IND World Cup")
                text = text.replace("RCB Match", "RCB Match")
                text = text.replace("Week end", "Weekend")
    
                # Remove extra spaces and normalize casing
                text = " ".join(text.split())
                text = text.title().replace("Ipl", "IPL").replace("Ind", "IND")
                return text
    
            df_events["Event / Task"] = df_events["Event / Task"].apply(clean_event_name)
    
            # --- Aggregate events by Label ---
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
                df_events.groupby("Label", dropna=False)["Event / Task"]
                .apply(summarize_events)
                .reset_index()
            )
    
            # --- Merge trend with events ---
            trend_df = trend_df.merge(events_agg, on="Label", how="left")
    
            # --- Plot Shipment Trend with Events ---
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=trend_df["Label"],
                    y=trend_df["Value"],
                    mode="lines+markers",
                    name=f"Shipment Trend ({granularity}, {value_type})",
                    fill="tozeroy",
                    yaxis="y1",
                    hovertext=trend_df["Event / Task"],
                    hoverinfo="x+y+text",
                )
            )
    
            fig.update_yaxes(title_text=y_title)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(trend_df.round(0))
    
            st.markdown("""
            ### **Insights: Shipment Volume Analysis (2023–2024)**
            - There has been a substantial decrease in the yearly shipment volume between 2023 and 2024.  
            - The company's shipment activity is highly cyclical, with a common drop observed in Q3 of both years.  
            - The shipment volume is highly unstable on a monthly basis, showing recovery in Q4 2024 but still below peak levels.
            """)
    
            
            # ===============================
            # 📅 EVENT CALENDAR HEATMAP
            # ===============================
        else:
            st.subheader("Monthly Event Heatmap")
    
            EVENT_XLSX_URL = "https://docs.google.com/spreadsheets/d/1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO/export?format=xlsx"
            df_events = load_event_calendar(EVENT_XLSX_URL)
            df_events["Date"] = pd.to_datetime(df_events["Date"], errors="coerce")
    
            df_events["Year"] = df_events["Date"].dt.year
            df_events["Month"] = df_events["Date"].dt.month
            df_events["MonthName"] = df_events["Date"].dt.strftime("%B")
            df_events["Day"] = df_events["Date"].dt.day
    
            selected_year = st.selectbox("Select Year", sorted(df_events["Year"].dropna().unique()))
            selected_month_name = st.selectbox(
                "Select Month",
                sorted(df_events["MonthName"].unique(), key=lambda x: pd.to_datetime(x, format="%B").month)
            )
    
            df_selected = df_events[
                (df_events["Year"] == selected_year) &
                (df_events["MonthName"] == selected_month_name)
            ].copy()
    
            df_ship = df.copy()
            df_ship["Date"] = pd.to_datetime(df_ship["SHIPMENT_DATE"], errors="coerce")
            ship_day = df_ship.groupby(df_ship["Date"].dt.date)[VOLUME_COL].sum().reset_index()
            ship_day.rename(columns={VOLUME_COL: "VOLUME"}, inplace=True)
            ship_day["Date"] = pd.to_datetime(ship_day["Date"], errors="coerce")
    
            df_selected = pd.merge(df_selected, ship_day[["Date", "VOLUME"]], on="Date", how="left")
            df_selected["VOLUME"] = df_selected["VOLUME"].fillna(0)
    
            month_start = pd.Timestamp(f"{selected_year}-{selected_month_name}-01")
            month_end = month_start + pd.offsets.MonthEnd(1)
            start_day = month_start - pd.Timedelta(days=month_start.weekday())
            end_day = month_end + pd.Timedelta(days=(6 - month_end.weekday()))
            full_range = pd.date_range(start_day, end_day, freq="D")
    
            calendar_df = pd.DataFrame({"Date": full_range})
            calendar_df["Day"] = calendar_df["Date"].dt.day
            calendar_df["DayOfWeek"] = calendar_df["Date"].dt.day_name().str[:3]
            calendar_df["Month"] = calendar_df["Date"].dt.month
            calendar_df["Week"] = ((calendar_df["Date"] - start_day).dt.days // 7) + 1
            calendar_df["VOLUME"] = calendar_df["Date"].map(df_selected.set_index("Date")["VOLUME"]).fillna(0)
    
            calendar_df.loc[calendar_df["Month"] != month_start.month, "VOLUME"] = None
            calendar_df.loc[calendar_df["Month"] != month_start.month, "Day"] = ""
    
            ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            pivot_volume = calendar_df.pivot(index="Week", columns="DayOfWeek", values="VOLUME")[ordered_days]
            text_matrix = calendar_df.pivot(index="Week", columns="DayOfWeek", values="Day")[ordered_days]
    
            df_selected["Tooltip"] = (
                "<b>" + df_selected["Date"].dt.strftime("%d %b %Y") + "</b><br>" +
                "Event: " + df_selected["Event / Task"].fillna("") + "<br>" +
                "Volume: " + df_selected["VOLUME"].round(0).astype(int).astype(str)
            )
    
            calendar_df["Tooltip"] = calendar_df["Date"].map(df_selected.set_index("Date")["Tooltip"])
            calendar_df.loc[calendar_df["Month"] != month_start.month, "Tooltip"] = None
            hover_matrix = calendar_df.pivot(index="Week", columns="DayOfWeek", values="Tooltip")[ordered_days]
    
            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot_volume.values,
                    x=pivot_volume.columns,
                    y=pivot_volume.index,
                    text=text_matrix.values,
                    texttemplate="%{text}",
                    hovertext=hover_matrix.values,
                    hoverinfo="text",
                    colorscale="RdPu",
                    showscale=True
                )
            )
    
            fig.update_layout(
                title=f"{selected_month_name} {selected_year} — Shipment Heatmap",
                xaxis=dict(title="", side="top"),
                yaxis=dict(title="", autorange="reversed"),
                width=600,
                height=450,
                template="simple_white",
                margin=dict(l=20, r=20, t=80, b=20),
                coloraxis_colorbar=dict(title="Volume")
            )
    
            st.plotly_chart(fig, use_container_width=True)

              
                            
            st.markdown("""
    ### **Insights: Shipment Volume Analysis (2023–2024)**
    
    - There has been a substantial decrease in the yearly shipment volume between 2023 and 2024. This indicates a contraction in shipping activity or demand.
    - The company's shipment activity is highly cyclical, with a common drop observed in Q3 of both years (though much more severe in 2024). The period from 2024 Q2 to 2024 Q3 represents a major performance concern due to the unprecedented low volume.
    - The shipment volume is highly unstable on a monthly basis. While the first half of 2024 contained the single highest month (May), the subsequent crash to the lowest volume in August. The recent recovery in Q4 2024 (September and October) is a positive sign, but volume remains far below peak levels.
    
    """)



    # ---- Tab 2: Top/Bottom Locations ----
    with tab2:
        st.markdown("###  Question: Where are shipments highest and where are they lagging?")
        st.subheader("Top/Bottom Locations")
        if VOLUME_COL in df_filtered.columns and LOCATION_COL in df_filtered.columns:
            location_volume = df_filtered.groupby(LOCATION_COL)[VOLUME_COL].sum().round(0).reset_index()

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

            fig = px.bar(
                locs,
                x=VOLUME_COL,
                y=LOCATION_COL,
                orientation="h",
                text=locs[VOLUME_COL].round(0)
            )
            fig.update_traces(textposition="outside")
            if choice == "Top":
                fig.update_layout(yaxis=dict(categoryorder="total ascending"))
            else:
                fig.update_layout(yaxis=dict(categoryorder="total descending"))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""
### **Insights:**
The shipment volume is heavily concentrated in the top three locations, particularly Kalaburagi and Bidar. 
Cities with lowest shipment volumes are - Gokak , Jamkhandi , Haveri. Disruptive strategies could be implemented here to improve performance

""")


    # ---- Tab 3: Heatmaps ----
    with tab3:
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
- There is a clear cyclical pattern across most locations. Volume tends to be highest during the middle of the year, roughly from April to July.
- Conversely, volume is consistently lower at the beginning (Jan/Feb) and end (Oct/Dec) of the year. 
- The most prominent feature is the massive volume spike for Vijayapura around April-May 2024. 

""")

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
                st.dataframe(comp_df[[LOCATION_COL, "Cluster"]].round(0))
                st.markdown("""
### **Insights:**
The clustering reveals four distinct groups of locations based on their shipment activity patterns:

- Cluster 0 - Moderate, steady performers:
Includes locations such as Bagalkot, Chikkodi, Gadag, Gokak, Haveri, Jamakhandi, and Sindhanur.
These show balanced shipment volumes with relatively consistent monthly activity and minimal fluctuations.

- Cluster 1 - High-activity, central hubs:
Includes Ballari, Belagavi, Chitradurga, Davangere, Hosapete, and Koppal.
These centers handle larger shipment volumes and serve as regional distribution nodes.

- Cluster 2 - Emerging or volatile markets:
Includes Bidar, Kalaburagi, and Vijayapura.
Their shipment activity is variable, occasionally spiking due to concentrated dispatches, suggesting developing demand or logistical challenges.

- Cluster 3 - Low-volume or specialized locations:
Includes Hubballi, Raichur, Sedam, and Yadgir.
These sites have comparatively low or niche shipment patterns, possibly influenced by geographic or operational constraints.


The presence of four well-separated clusters indicates that shipment behaviour varies meaningfully across regions.
""")

    
# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
