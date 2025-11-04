import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from plotly.subplots import make_subplots
from streamlit_plotly_events import plotly_events

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

        # download_url = f"https://docs.google.com/spreadsheets/d/1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO/export?format=xlsx"
        download_url = f"https://docs.google.com/spreadsheets/d/1GxgGo6waZV7WDsF50v_nYSu2mxEX6bmj/export?format=xlsx"
        # download_url = f"https://docs.google.com/spreadsheets/d/1PZSyJWB_1iPbARkUOiNOVooF51PjDhxlgGxgCdSCzKk/export?format=xlsx"
        # download_url = f"https://docs.google.com/spreadsheets/d/1GxgGo6waZV7WDsF50v_nYSu2mxEX6bmj/export?format=xlsx"

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
    default_path = "https://docs.google.com/spreadsheets/d/1l69N0xrDbXM7-cP1d9nlwBInaqCy8ftC/export?format=xlsx"
    #default_path = "https://docs.google.com/spreadsheets/d/1te1MVxSoO3EWwg_9akooxKxIEgI4KDna/export?format=xlsx"
    df = load_excel(default_path)
    SHEET_ID = "1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO"
    df_events = load_event_calendar(SHEET_ID)

    VOLUME_COL = "VOLUME"
    OUTLET_COL = "DBF_OUTLET_CODE"
    #df[OUTLET_COL] = df[OUTLET_COL].astype(str).str.strip().str.upper()
    # 1️⃣ Inspect potential hidden characters
    #df['clean_outlet'] = (
        #df[OUTLET_COL]
        #.astype(str)
        #.str.replace(r'[\u200b\u200c\u200d\u00a0\r\n\t]', '', regex=True)  # invisible Unicode chars
        #.str.strip()
        #.str.upper()
    #)
    
    # 2️⃣ Compare unique counts
    #print("Before:", df[OUTLET_COL].nunique())
    #print("After:", df['clean_outlet'].nunique())
    
    # 3️⃣ Replace column finally if count improves
    #df[OUTLET_COL] = df['clean_outlet']
    #df.drop(columns=['clean_outlet'], inplace=True)

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
        return f"{val:+.0f}%" if val is not None and pd.notna(val) else "N/A"

    # --------------------------
    # SHOW FIXED KPIs
    # --------------------------
    col1, col2,col3 = st.columns(3)
    col1.metric("Total Volume", f"{int(kpi_volume):,}", format_delta(delta_volume))
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

        # --------------------------------------------
    # Apply the same sidebar filters to df_events
    # --------------------------------------------
    df_events_filtered = df_events.copy()
    
    # Ensure Date column is datetime
    df_events_filtered["Date"] = pd.to_datetime(df_events_filtered["Date"], errors="coerce")
    df_events_filtered = df_events_filtered.dropna(subset=["Date"])
    
    if filter_mode == "Year":
        if year_choice:
            df_events_filtered = df_events_filtered[df_events_filtered["Date"].dt.year.isin(year_choice)]
    else:
        mask_events = (
            (df_events_filtered["Date"].dt.date >= start_date)
            & (df_events_filtered["Date"].dt.date <= end_date)
        )
        df_events_filtered = df_events_filtered.loc[mask_events]


    # --------------------------
    # VISUALIZATIONS (tabs)
    # --------------------------
    tab1, tab2, tab3, tab4, tab5, tab6, tab7,tab8,tab9 = st.tabs([
        "Shipment Trends",
		"Month on Month Shipment",
		"Map",
		"Region Donut",
		"Region Stacked",
		"Special Outlets",
        "Depot Analysis",
        "Depot-wise YoY Change",
		"Top Outlets"
    ])

    # ---- Shipment Trends ----
    # ---- Shipment Trends ----
    with tab1:
	    st.markdown("### Question: Do shipment trends look different by year, quarter, or month?")
	    st.subheader("Shipment Trends and Event Calendar")
	
	    # --- Chart Selection ---
	    chart_type = st.radio(
	        "Select Chart Type:",
	        ["Shipment Trend", "Event Calendar"],
	        horizontal=True
	    )
	
	    if chart_type == "Shipment Trend":
	
	        # --- Controls (Granularity and View Mode) ---
	        granularity = st.radio(
	            "Granularity", ["Yearly", "Quarterly", "Monthly"],
	            horizontal=True, key="trend_granularity"
	        )
	        view_mode = st.radio(
	            "Display Mode", ["Absolute", "Percentage"],
	            horizontal=True, key="trend_view"
	        )
	
	        # --- Shipment Trend (Filtered Main Data) ---
	        if granularity == "Yearly":
	            df_filtered["Label"] = df_filtered["Year"].astype(int).astype(str)
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
	
	        # --- Load Event Calendar ---
	        EVENT_CSV_URL = "https://docs.google.com/spreadsheets/d/1PZSyJWB_1iPbARkUOiNOVooF51PjDhxlgGxgCdSCzKk/export?format=xlsx"
	        df_events = load_event_calendar(EVENT_CSV_URL)
	        df_events["Date"] = pd.to_datetime(df_events["Date"], errors="coerce")
	
	        if granularity == "Yearly":
	            df_events["Label"] = df_events["Date"].dt.year.astype(str)
	        elif granularity == "Quarterly":
	            df_events["Label"] = df_events["Date"].dt.to_period("Q").astype(str)
	        else:
	            df_events["Label"] = df_events["Date"].dt.to_period("M").astype(str)
	
	        # --- Clean event names ---
	        def clean_event_name(text):
	            if pd.isna(text):
	                return None
	            text = str(text).strip()
	            if not text or text.lower() in ["nan", "none"]:
	                return None
	            replacements = {
	                "Against": "",
	                "Friendly": "BFC",
	                "Footll": "Football",
	                "Pro Ka": "Pro Kabbadi",
	                "C ": " ",
	                "IND World cup": "IND World Cup",
	                "RCB Match": "RCB Match",
	                "Week end": "Weekend",
	                "INDependence": "Independence",
	                "Ni8": "Night"
	            }
	            for k, v in replacements.items():
	                text = text.replace(k, v)
	            text = " ".join(text.split())
	            text = text.title().replace("Ipl", "IPL").replace("Bf", "BFC")
	            return text
	
	        df_events["Event / Task"] = df_events["Event / Task"].apply(clean_event_name)
	
	        # --- Aggregate events by Label ---
	        def summarize_events(x):
	            counts = x.value_counts()
	            lines = []
	            for event, count in counts.items():
	                lines.append(f"{event} (x{count})" if count > 1 else event)
	            return "<br>".join(lines)
	
	        events_agg = (
	            df_events.groupby("Label", dropna=False)["Event / Task"]
	            .apply(summarize_events)
	            .reset_index()
	        )
	
	        trend_df = trend_df.merge(events_agg, on="Label", how="left")
	
	        # --- Layout with col1 and col2 ---
	        col1, col2 = st.columns([2, 1])  # left wider for line chart
	
	        # ------------------------
	        # LEFT: Shipment Trend ONLY
	        # ------------------------
	        with col1:
	            fig_trend = go.Figure()
	
	            # Line: Shipment trend (purple)
	            fig_trend.add_trace(
	                go.Scatter(
	                    x=trend_df["Label"],
	                    y=trend_df["Value"],
	                    mode="lines+markers",
	                    name=f"Shipment Trend ({granularity}, {view_mode})",
	                    fill="tozeroy",
	                    hovertext=trend_df["Event / Task"],
	                    hoverinfo="x+y+text",
	                    line=dict(color="mediumvioletred", width=3),
	                    marker=dict(size=7, color="mediumvioletred", line=dict(width=1, color="darkmagenta"))
	                )
	            )
	
	            fig_trend.update_layout(
	                title=f"Shipment Trend ({granularity})",
	                xaxis_title=granularity,
	                yaxis_title=y_title,
	                height=500,
	                hovermode="x unified",
	                template="plotly_white",
	                legend_title="Metrics",
	                margin=dict(l=40, r=40, t=60, b=40)
	            )
	
	            st.plotly_chart(fig_trend, use_container_width=True)
	
	        # ------------------------
	        # RIGHT: Bubble Chart of Event Bins
	        # ------------------------
	        with col2:
	            st.subheader("Event Type Distribution")
	
	            bin_cols = ["Political", "Festival", "Sports", "Celebrity_Deaths", "Public_Holiday", "Movie", "Weekend"]
	
	            df_events_filtered = df_events.copy()
	            df_events_filtered["Date"] = pd.to_datetime(df_events_filtered["Date"], errors="coerce")
	            df_events_filtered = df_events_filtered.dropna(subset=["Date"])
	
	            # Apply filters
	            if filter_mode == "Year":
	                if year_choice:
	                    df_events_filtered = df_events_filtered[df_events_filtered["Date"].dt.year.isin(year_choice)]
	            else:
	                mask_events = (
	                    (df_events_filtered["Date"].dt.date >= start_date)
	                    & (df_events_filtered["Date"].dt.date <= end_date)
	                )
	                df_events_filtered = df_events_filtered.loc[mask_events]
	
	            if df_events_filtered.empty:
	                st.info("ℹ️ No events found for the selected filter.")
	            else:
	                bubble_counts = df_events_filtered[bin_cols].sum().reset_index()
	                bubble_counts.columns = ["Event_Type", "Count"]
	                bubble_counts = bubble_counts[bubble_counts["Count"] > 0]
	
	                if bubble_counts.empty:
	                    st.info("ℹ️ No events recorded for the selected period.")
	                else:
	                    fig_bubble = go.Figure(
	                        go.Scatter(
	                            x=bubble_counts["Event_Type"],
	                            y=[1] * len(bubble_counts),
	                            mode="markers+text",
	                            marker=dict(
	                                size=bubble_counts["Count"] * 10,
	                                color=bubble_counts["Count"],
	                                colorscale="Viridis",
	                                showscale=True,
	                                colorbar=dict(title="Count"),
	                                sizemode="area"
	                            ),
	                            text=bubble_counts["Count"],
	                            textposition="top center",
	                            hovertemplate="<b>%{x}</b><br>Count: %{text}<extra></extra>"
	                        )
	                    )
	
	                    fig_bubble.update_layout(
	                        title="Event Type Distribution (Filtered)",
	                        xaxis_title="Event Type",
	                        yaxis=dict(visible=False),
	                        height=500,
	                        template="plotly_dark",
	                        margin=dict(t=50)
	                    )
	
	                    st.plotly_chart(fig_bubble, use_container_width=True)
	
	





        # ===============================
        # 📅 IF EVENT CALENDAR SELECTED
        # ===============================
	    
	    else:
			st.subheader("Event-Based Shipment Visualization")
		
			# --- Load Event Calendar ---
			EVENT_XLSX_URL = "https://docs.google.com/spreadsheets/d/1QYN4ZHmB-FpA1wUFlzh5Vp-WtMFPV8jO/export?format=xlsx"
			df_events = load_event_calendar(EVENT_XLSX_URL)
		
			df_events["Date"] = pd.to_datetime(df_events["Date"], errors="coerce")
			df_events["Year"] = df_events["Date"].dt.year
			df_events["Month"] = df_events["Date"].dt.month
			df_events["MonthName"] = df_events["Date"].dt.strftime("%B")
			df_events["Week"] = df_events["Date"].dt.isocalendar().week
		
			# --- Granularity Selector ---
			granularity = st.radio("Select Granularity", ["Monthly", "Weekly"], horizontal=True)
		
			# --- Shipment Data ---
			df_ship = df.copy()
			df_ship["Date"] = pd.to_datetime(df_ship["ACTUAL_DATE"], errors="coerce")
		
			# =====================================================================
			# ======================== MONTHLY VIEW ===============================
			# =====================================================================
			if granularity == "Monthly":
				st.subheader("Monthly Event Heatmap")
		
				selected_year = st.selectbox("Select Year", sorted(df_events["Year"].dropna().unique()))
				selected_month_name = st.selectbox(
					"Select Month",
					sorted(df_events["MonthName"].unique(), key=lambda x: pd.to_datetime(x, format="%B").month)
				)
		
				# --- Filter for selected month ---
				df_selected = df_events[
					(df_events["Year"] == selected_year) &
					(df_events["MonthName"] == selected_month_name)
				].copy()
		
				# --- Merge with shipment data ---
				ship_day = df_ship.groupby(df_ship["Date"].dt.date)[VOLUME_COL].sum().reset_index()
				ship_day.rename(columns={VOLUME_COL: "VOLUME"}, inplace=True)
				ship_day["Date"] = pd.to_datetime(ship_day["Date"], errors="coerce")
		
				df_selected = pd.merge(df_selected, ship_day[["Date", "VOLUME"]], on="Date", how="left")
				df_selected["VOLUME"] = df_selected["VOLUME"].fillna(0)
		
				# --- Prepare calendar grid ---
				month_start = pd.Timestamp(f"{selected_year}-{selected_month_name}-01")
				month_end = (month_start + pd.offsets.MonthEnd(1))
		
				# Create continuous range covering full calendar view (Mon–Sun)
				start_day = month_start - pd.Timedelta(days=month_start.weekday())   # Monday of first week
				end_day = month_end + pd.Timedelta(days=(6 - month_end.weekday()))   # Sunday of last week
				full_range = pd.date_range(start_day, end_day, freq="D")
		
				calendar_df = pd.DataFrame({"Date": full_range})
				calendar_df["Day"] = calendar_df["Date"].dt.day
				calendar_df["DayOfWeek"] = calendar_df["Date"].dt.day_name().str[:3]
				calendar_df["Month"] = calendar_df["Date"].dt.month
				calendar_df["Week"] = ((calendar_df["Date"] - start_day).dt.days // 7) + 1
		
				# Merge shipment volume + tooltip info
				calendar_df["VOLUME"] = calendar_df["Date"].map(df_selected.set_index("Date")["VOLUME"]).fillna(0)
		
				df_selected["Tooltip"] = (
					"<b>" + df_selected["Date"].dt.strftime("%d %b %Y") + "</b><br>" +
					"Event: " + df_selected["Event / Task"].fillna("") + "<br>" +
					"Volume: " + df_selected["VOLUME"].round(0).astype(int).astype(str)
				)
				calendar_df["Tooltip"] = calendar_df["Date"].map(df_selected.set_index("Date")["Tooltip"])
		
				# Hide days outside the selected month
				calendar_df.loc[calendar_df["Month"] != month_start.month, ["VOLUME", "Day", "Tooltip"]] = [None, "", None]
		
				# --- Pivot into heatmap grid ---
				ordered_days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
				pivot_volume = calendar_df.pivot(index="Week", columns="DayOfWeek", values="VOLUME")[ordered_days]
				text_matrix = calendar_df.pivot(index="Week", columns="DayOfWeek", values="Day")[ordered_days]
				hover_matrix = calendar_df.pivot(index="Week", columns="DayOfWeek", values="Tooltip")[ordered_days]
		
				# --- Plotly Heatmap ---
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
					title=f"{selected_month_name} {selected_year} — Shipment Volume Calendar",
					xaxis=dict(title="", side="top"),
					yaxis=dict(title="", autorange="reversed"),
					width=700,
					height=450,
					template="simple_white",
					margin=dict(l=20, r=20, t=80, b=20),
					coloraxis_colorbar=dict(title="Volume")
				)
				st.plotly_chart(fig, use_container_width=True)
		
			# =====================================================================
			# ========================= WEEKLY VIEW ===============================
			# =====================================================================
			else:
				st.subheader("Weekly Event Trend")
		
				df_events["Date"] = pd.to_datetime(df_events["Date"], errors="coerce")
				df_events["Year"] = df_events["Date"].dt.year
				df_events["Week"] = df_events["Date"].dt.isocalendar().week
		
				if "Year" not in df_events.columns or df_events["Year"].dropna().empty:
					st.warning("No valid event dates available to display weekly data.")
				else:
					selected_year = st.selectbox("Select Year", sorted(df_events["Year"].dropna().unique()))
		
					# --- Aggregate shipment volume weekly ---
					df_ship["Year"] = df_ship["Date"].dt.year
					df_ship["Week"] = df_ship["Date"].dt.isocalendar().week
		
					weekly_ship = (
						df_ship[df_ship["Year"] == selected_year]
						.groupby("Week")[VOLUME_COL]
						.sum()
						.reset_index()
						.rename(columns={VOLUME_COL: "VOLUME"})
					)
		
					# --- Compute start & end date for each week ---
					def get_week_range(year, week):
						try:
							start = pd.Timestamp.fromisocalendar(year, int(week), 1)
							end = pd.Timestamp.fromisocalendar(year, int(week), 7)
							return start, end
						except Exception:
							return pd.NaT, pd.NaT
		
					weekly_ship["Start_Date"], weekly_ship["End_Date"] = zip(*weekly_ship.apply(
						lambda x: get_week_range(selected_year, int(x["Week"])), axis=1
					))
		
					weekly_ship["WeekRange"] = (
						weekly_ship["Start_Date"].dt.strftime("%d %b") + " – " +
						weekly_ship["End_Date"].dt.strftime("%d %b")
					)
		
					# --- Merge with events (combine events per week) ---
					df_events_week = (
						df_events[df_events["Year"] == selected_year]
						.groupby("Week")["Event / Task"]
						.apply(lambda x: ", ".join(x.dropna().unique()))
						.reset_index()
					)
		
					df_weekly = pd.merge(weekly_ship, df_events_week, on="Week", how="left")
		
					# --- Tooltip ---
					df_weekly["Tooltip"] = (
						"<b>Week " + df_weekly["Week"].astype(str) + "</b><br>" +
						df_weekly["WeekRange"] + "<br>" +
						"Volume: " + df_weekly["VOLUME"].astype(int).astype(str) + "<br>" +
						"Events: " + df_weekly["Event / Task"].fillna("None")
					)
		
					# --- Plotly Weekly Bar Chart ---
					fig_week = go.Figure()
					fig_week.add_trace(go.Bar(
						x=df_weekly["Week"],
						y=df_weekly["VOLUME"],
						text=df_weekly["VOLUME"],
						hovertext=df_weekly["Tooltip"],
						hoverinfo="text",
						marker_color="mediumvioletred",
						name="Weekly Volume"
					))
		
					fig_week.update_layout(
						title=f"{selected_year} — Weekly Shipment Volume & Events",
						xaxis_title="Week Number",
						yaxis_title="Volume",
						template="simple_white",
						height=450,
						width=800,
						margin=dict(l=40, r=40, t=80, b=40)
					)
		
					st.plotly_chart(fig_week, use_container_width=True)
	
		st.markdown("""
	### **Insights:**
	ODD OCCURENCES - Peaks were observed in normalised trend
	- August , December - 2023
	- March, October - 2024
	""")


	

    with tab2:
	    st.subheader("Month-on-Month Shipment Trends")
	
	    # --- Ensure datetime ---
	    df_filtered["ACTUAL_DATE"] = pd.to_datetime(df_filtered["ACTUAL_DATE"], errors="coerce")
	    df_filtered["Year"] = df_filtered["ACTUAL_DATE"].dt.year
	    df_filtered["Month"] = df_filtered["ACTUAL_DATE"].dt.month
	    df_filtered["Month_Name"] = df_filtered["ACTUAL_DATE"].dt.strftime("%b")
	
	    # --- Group data ---
	    trend_df = (
	        df_filtered.groupby(["Year", "Month", "Month_Name"])[VOLUME_COL]
	        .sum()
	        .reset_index()
	    )
	
	    # --- Order months correctly ---
	    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
	                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
	    trend_df["Month_Name"] = pd.Categorical(trend_df["Month_Name"], categories=month_order, ordered=True)
	    trend_df = trend_df.sort_values(["Month", "Year"])
	
	    # --- Get unique years dynamically ---
	    years = sorted(trend_df["Year"].unique())
	
	    # --- Create clustered chart manually using go.Figure ---
	    fig = go.Figure()
	
	    # Add a separate bar trace for each year
	    for yr in years:
	        df_year = trend_df[trend_df["Year"] == yr]
	        fig.add_trace(go.Bar(
	            x=df_year["Month_Name"],
	            y=df_year[VOLUME_COL],
	            name=str(yr),
	            text=df_year[VOLUME_COL],
	            textposition="auto"
	        ))
	
	    # --- Layout settings for side-by-side (clustered) view ---
	    fig.update_layout(
	        barmode="group",  # <— This makes it clustered (side-by-side)
	        xaxis_title="Month",
	        yaxis_title="Shipment Volume",
	        title="Month-on-Month Shipment Trends (2023 vs 2024)",
	        template="plotly_white",
	        bargap=0.15,
	        bargroupgap=0.05,
	        title_x=0.5,
	        legend_title="Year"
	    )
	
	    st.plotly_chart(fig, use_container_width=True)
	
	    # --- Insights ---
	    st.markdown("""
	    ### *Insights:*
	    📊 **February 2024** - lower shipments than 2023 — Union Budget announcements.  
	    🌧 **June 2024** major drop versus 2023 — intense monsoon floods and industrial disruptions across Karnataka.  
	    🎄 **December 2024** shipments lower than 2023 — weaker festive demand and reduced year-end restocking compared to the previous year.
	    """)

    with tab9:
        st.markdown("###  Question: Where is shipment activity the highest among outlets?")
        st.subheader("Top Outlets by Volume")
        
         # --- Load external data from Google Sheets ---
        sheet_url = "https://docs.google.com/spreadsheets/d/1te1MVxSoO3EWwg_9akooxKxIEgI4KDna/export?format=xlsx"
        try:
            external_df = pd.read_excel(sheet_url)
        except Exception as e:
            st.error(f"Error loading external data: {e}")
            st.stop()
        OUTLET_COL = "DBF_OUTLET_NAME"
    
        # --- Check if required columns exist ---
        if OUTLET_COL in external_df.columns and VOLUME_COL in external_df.columns:
            # --- Controls ---
            view_mode_tab2 = st.radio(
                "Display Mode", ["Absolute", "Percentage"], horizontal=True, key="top_outlets_view_mode"
            )
            top_n = st.slider("Top-N Outlets", 5, 25, 10)
    
            # --- Group and sort by outlet volume ---
            outlet_volume = (
                external_df.groupby(OUTLET_COL)[VOLUME_COL]
                .sum()
                .round(0)
                .reset_index()
                .sort_values(by=VOLUME_COL, ascending=False)
            )
    
            top_outlets = outlet_volume.head(top_n)
    
            # --- Handle Percentage vs Absolute ---
            if view_mode_tab2 == "Percentage":
                top_outlets["Value"] = (
                    top_outlets[VOLUME_COL] / top_outlets[VOLUME_COL].sum() * 100
                ).round(2)
                title_suffix = " (%)"
            else:
                top_outlets["Value"] = top_outlets[VOLUME_COL]
                title_suffix = ""
    
            # --- Treemap Chart ---
            fig = px.treemap(
                top_outlets,
                path=[OUTLET_COL],
                values="Value",
                title=f"Top {top_n} Outlets Treemap{title_suffix}",
                custom_data=[top_outlets[OUTLET_COL], top_outlets["Value"]],
            )
    
            fig.update_traces(
                hovertemplate="<b>Outlet:</b> %{customdata[0]}<br><b>Value:</b> %{customdata[1]:,.2f}<extra></extra>"
            )
    
            st.plotly_chart(fig, use_container_width=True)
    
            # --- Display dataframe ---
            display_df = top_outlets[[OUTLET_COL, "Value"]].set_index(OUTLET_COL).round(2)
            st.dataframe(display_df)
    
        else:
            st.warning("Required columns not found in the external sheet. Please check OUTLET_COL and VOLUME_COL names.")
        
        st.markdown("""
    ### **Insights:**
    The most striking feature is the overwhelming dominance of a single outlet: MANAGING DIRECTOR MSIL. This outlet takes up easily over 75% to 80% of the total volume visualized in the Top 10 list. The business's volume is heavily reliant on this one outlet. 

    """)


    # ---- Depot Analysis ----
    with tab7:
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
                hovertemplate="<b>%{x}</b><br>Volume: %{y:,.0f}<extra></extra>"

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
### **Insights:**
- KALABURAGI, BIDAR, and VIJAYAPURA are top-tier, as they are the primary drivers of volume. 
- Orange depots (KOPPAL to CHITRADURGA) have the highest growth potential to secure the next wave of A-Class contributors. 
- Red depots have lowest contribution to the shipment volume - CHIKKODI AND GOKAK being the most risky depots.

""")

    # ---- Region Donut ----
    with tab4:
        st.markdown("###  Question: Which regions account for the largest share of shipments?")
        st.subheader("Region-wise Volume Share")
        if "DBF_REGION" in df_filtered.columns:
            region_volume = df_filtered.groupby("DBF_REGION")[VOLUME_COL].sum().round(0).reset_index()
            fig = px.pie(region_volume, values=VOLUME_COL, names="DBF_REGION", hole=0.5,
                         title="Volume Distribution by Region")
            fig.update_traces(
                texttemplate="%{label}: %{percent:.0%}",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Share: %{percent:.0%}<extra></extra>",
                insidetextorientation='auto'
            )

            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(region_volume.set_index("DBF_REGION").round(0))
        else:
            st.info("DBF_REGION column not found.")
        st.markdown("""
### **Insights:**
- NORTH KARNATAKA 2 is the primary driver of volume contributing 60% to the shipments.
- Whereas NORTH KARNATAKA 1 is lagging by contributing the rest of 40%.
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
            fig.update_traces(texttemplate="%{text:.0f}%", textposition="inside")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(region_stats.set_index("DBF_REGION").round(0))
        else:
            st.info("DBF_REGION or DBF_OUTLET_CODE not found.")
        with st.container():
            st.markdown("""
### **Insights:**
The chart depicts that 61% of Number of outlets in North Karnataka 1 contribute to just 40% of total shipment volume, whereas only 39% of Number of Outlets in North Karnataka 2 drive for 60% of total shipment volume, indicating that North Karnataka 2 region is the core volume driver

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
            fig_pie.update_traces(
                texttemplate="%{label}: %{percent:.0%}",        # inside slice (rounded percentage)
                hovertemplate="<b>%{label}</b><br>Volume: %{value:,.0f}<br>Share: %{percent:.0%}<extra></extra>",  # hover tooltip
                insidetextorientation='auto'              # prevents awkward rotated text
            )
            
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
### **Insights:**
All four entities, particularly the top two (BELAGAVI and HUBBALLI-1), contribute a significant portion of the volume - 35% and 30% respectively.
BELAGAVI 2 AND HUBALLI 2 are contributing fairly lesser - 17% and 18% respectively.
""")


        # ---- TAB 7: Depot Map View ----
    with tab3:
        st.markdown("### Question: What is the geographic spread of shipment volumes?")
        st.subheader(" Depot Shipment Volume Map")
    
        # --- Approximate and refined coordinates for Karnataka depots ---
        DEPOT_COORDS = {
            "BIDAR": (17.9133, 77.5301),
            "KALABURAGI": (17.3297, 76.8343),
            "SEDAM": (17.1794, 77.2830),
            "YADHAGIRI": (16.7666, 77.1376),
            "RAICHUR": (16.2046, 77.3555),
            "SINDHANUR": (15.7689, 76.7557),
            "BALLARI": (15.1394, 76.9214),
            "HOSAPETE": (15.2843, 76.3771),
            "KOPPAL": (15.3452, 76.1545),
            "BAGALKOT": (16.1867, 75.6962),
            "JAMAKHANDI": (16.5048, 75.2911),
            "VIJAYAPURA": (16.8302, 75.7100),
            "BELAGAVI": (15.8497, 74.4977),
            "BELAGAVI-2": (15.8497, 74.4977),
            "CHIKODI": (16.4295, 74.6005),
            "GOKAK": (16.1691, 74.8231),
            "HUBBALLI-1": (15.3647, 75.1239),
            "HUBBALLI-2": (15.3647, 75.1239),
            "GADAG": (15.4310, 75.6297),
            "HAVERI": (14.7951, 75.3979),
            "DAVANGERE": (14.4663, 75.9238),
            "CHITRADURGA": (14.2290, 76.3980),
        }
    
        if "DBF_DEPOT" in df_filtered.columns:
    # --- Aggregate shipment volume ---
            depot_volume_map = df_filtered.groupby("DBF_DEPOT")[VOLUME_COL].sum().reset_index()
            depot_volume_map["Latitude"] = depot_volume_map["DBF_DEPOT"].map(lambda x: DEPOT_COORDS.get(x, (None, None))[0])
            depot_volume_map["Longitude"] = depot_volume_map["DBF_DEPOT"].map(lambda x: DEPOT_COORDS.get(x, (None, None))[1])
            depot_volume_map = depot_volume_map.dropna(subset=["Latitude", "Longitude"])
        
            # --- Map center ---
            center_lat = depot_volume_map["Latitude"].mean()
            center_lon = depot_volume_map["Longitude"].mean()
        
                       # --- Hover text ---
            depot_volume_map["HoverText"] = depot_volume_map.apply(
                lambda row: f"<b>Location:</b> {row['DBF_DEPOT']}<br><b>Volume:</b> {row[VOLUME_COL]:,.0f}", axis=1
            )
            
            # --- Plot advanced map ---
            fig = px.scatter_mapbox(
                depot_volume_map,
                lat="Latitude",
                lon="Longitude",
                size=VOLUME_COL,
                color=VOLUME_COL,
                hover_name=None,  # remove default hover_name
                hover_data=None,  # remove other hover data
                color_continuous_scale="Viridis",
                size_max=55,
                zoom=6,
                mapbox_style="carto-positron",
            )
            
            # --- Styling ---
            fig.update_traces(
                marker=dict(
                    opacity=0.85,
                    sizemode="area",
                    sizeref=2.0 * max(depot_volume_map[VOLUME_COL]) / (55**2),
                ),
                hovertemplate="%{customdata}<extra></extra>",  # use customdata for hover
                customdata=depot_volume_map["HoverText"]
            )

        
            # --- Layout polish ---
            fig.update_layout(
                mapbox_center={"lat": center_lat, "lon": center_lon},
                coloraxis_colorbar=dict(
                    title="Shipment<br>Volume",
                    thickness=15,
                    len=0.75,
                ),
                margin=dict(l=0, r=0, t=60, b=0),
                font=dict(size=13),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                title="Depot Shipment Volume Heat Map (Bubble Chart View)",
            )
        
            st.plotly_chart(fig, use_container_width=True)

            # --- Data table ---
            #st.markdown("#### 📊 Depot-wise Shipment Summary")
            #st.dataframe(
                #depot_volume_map[["DBF_DEPOT", VOLUME_COL, "Latitude", "Longitude"]]
                #.set_index("DBF_DEPOT")
                #.sort_values(VOLUME_COL, ascending=False)
                #.round(0)
            #)
    
    with tab8:
        st.markdown("### Question: How has depot volume changed YoY?")
        st.subheader("Depot-wise YoY Volume Change")
    
        if "DBF_DEPOT" in df_filtered.columns and "ACTUAL_DATE" in df_filtered.columns:
            # Ensure ACTUAL_DATE is datetime
            df_filtered["ACTUAL_DATE"] = pd.to_datetime(df_filtered["ACTUAL_DATE"], errors="coerce")
            df_filtered["Year"] = df_filtered["ACTUAL_DATE"].dt.year
    
            # Aggregate volume by depot and year
            depot_yearly = df_filtered.groupby(["DBF_DEPOT", "Year"])["VOLUME"].sum().reset_index()
    
            # Pivot to have years as columns
            pivot_df = depot_yearly.pivot(index="DBF_DEPOT", columns="Year", values="VOLUME").fillna(0)
    
            # Only consider 2023 and 2024
            if 2023 in pivot_df.columns and 2024 in pivot_df.columns:
                pivot_df["YoY_Change"] = pivot_df[2024] - pivot_df[2023]
                pivot_df["YoY_Percentage"] = ((pivot_df[2024] - pivot_df[2023]) / pivot_df[2023]) * 100
                pivot_df["YoY_Percentage"] = pivot_df["YoY_Percentage"].round(2)
    
                # Sort depots by YoY change for waterfall
                pivot_df = pivot_df.sort_values("YoY_Change", ascending=False)
    
                # Waterfall chart with hover showing absolute and % change
                fig = go.Figure(go.Waterfall(
                    name="YoY Change",
                    orientation="v",
                    measure=["relative"] * len(pivot_df),
                    x=pivot_df.index,
                    y=pivot_df["YoY_Change"],
                    text=pivot_df["YoY_Change"].apply(lambda x: f"{x:,.0f}"),
                    textposition="outside",
                    connector={"line": {"color": "rgb(63, 63, 63)"}},
                    hovertemplate="<b>%{x}</b><br>Volume Change: %{y:,.0f}<br>Percentage Change: %{customdata:.2f}%<extra></extra>",
                    customdata=pivot_df["YoY_Percentage"]
                ))
    
                fig.update_layout(
                    title="Depot-wise YoY Volume Change (2023 → 2024)",
                    yaxis=dict(title="Volume Change"),
                    height=600,
                    margin=dict(b=150),
                )
    
                st.plotly_chart(fig, use_container_width=True)
    
                # Add percentage change to summary table
                summary_df = pivot_df[[2023, 2024, "YoY_Change", "YoY_Percentage"]].round(0)
                st.dataframe(summary_df)
    
                # -------------------------------------------------------------------------
                # NEW SECTION: YoY Change in Disposable Income (Per Capita Income)
                # -------------------------------------------------------------------------
                st.markdown("### YoY Change in Disposable Income (Per Capita Income)")
    
                # Load PCI data
                df_pci = pd.read_excel(
                    "https://docs.google.com/spreadsheets/d/1Pg0DkCaqQJymbrkIIqAcjbgCa-7MVHJB/export?format=xlsx",
                    sheet_name="PCI"
                )
                df_pci.columns = [c.strip() for c in df_pci.columns]
                df_pci.rename(columns={"Row Labels": "Location"}, inplace=True)
                df_pci["Location"] = df_pci["Location"].str.strip().str.upper()
    
                
    
                # Calculate YoY change in PCI
                if "Per capita - 2022-23" in df_pci.columns and "per capita - 2023-24" in df_pci.columns:
                    df_pci["PCI_YoY_Change"] = df_pci["per capita - 2023-24"] - df_pci["Per capita - 2022-23"]
                    df_pci["PCI_YoY_%Change"] = (
                        df_pci["PCI_YoY_Change"] / df_pci["Per capita - 2022-23"]
                    ) * 100
                    df_pci["PCI_YoY_%Change"] = df_pci["PCI_YoY_%Change"].round(2)
    
                    df_pci_sorted = df_pci.sort_values("PCI_YoY_%Change", ascending=False)
    
                    # Bar chart for PCI YoY change
                    fig_pci = px.bar(
                        df_pci_sorted,
                        x="PCI_YoY_%Change",
                        y="Location",
                        orientation="h",
                        color="PCI_YoY_%Change",
                        color_continuous_scale="Tealgrn",
                        text=df_pci_sorted["PCI_YoY_%Change"].apply(lambda x: f"{x:.1f}%")
                    )
    
                    fig_pci.update_layout(
                        title="YoY % Change in Per Capita Income (2023 → 2024)",
                        xaxis_title="YoY % Change in PCI",
                        yaxis_title="Location",
                        height=600,
                        template="plotly_dark",
                        margin=dict(t=60, b=80)
                    )
    
                    fig_pci.update_traces(textposition="outside")
                    st.plotly_chart(fig_pci, use_container_width=True)
    
                    # Display PCI summary table
                    st.dataframe(df_pci_sorted[["Location", "Per capita - 2022-23", "per capita - 2023-24", "PCI_YoY_%Change"]])
                else:
                    st.warning("PCI columns for 2022-23 or 2023-24 are missing in the dataset.")
    
            else:
                st.info("Data for 2023 and/or 2024 is missing.")
        else:
            st.info("DBF_DEPOT or ACTUAL_DATE column not found.")
        st.markdown("""
            ### **Insights**
	        1. CHIKKODI - Good Kingfisher presence around sugar factories
	        2. KALABURAGI - Disposable Income ⬇️, shipments also ⬆️
	        3. SINDHANUR - Young drinkers turn towards hard drinks 
	        4. CHITRADURGA -  New liquor store licenses rejected by CM 
            5. VIJAYAPURA - Disposable Income ⬆️, but shipments ⬇️
            """)
    


# ===============================
# Entry Point
# ===============================
if __name__ == "__main__":
    run()
