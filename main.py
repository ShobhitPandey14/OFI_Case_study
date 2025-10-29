import io
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(
    page_title="Dynamic Fleet Manager",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def style_app() -> None:
    st.markdown(
        """
        <style>
          .stApp {
            background: linear-gradient(180deg, #0b132b 0%, #1c2541 100%);
            color: #F5F7FA;
          }
          [data-baseweb="input"] input, .stTextInput input, .stSelectbox div, .stFileUploader, .stNumberInput input {
            color: #eaecef !important;
          }
          .block-container { max-width: 1300px; }
          .metric-card { background: #101935; padding: 16px; border-radius: 12px; border: 1px solid #2e3a65; }
          .card { background: #101935; padding: 16px; border-radius: 12px; border: 1px solid #2e3a65; }
          .good { color: #4cd137; }
          .warn { color: #fbc531; }
          .bad  { color: #e84118; }
          table { color: #eaecef; }
        </style>
        """,
        unsafe_allow_html=True,
    )



@st.cache_data(show_spinner=False)
def load_csv_default(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def normalize_orders(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    
    expected_cols = [
        "Order_ID",
        "Order_Date",
        "Customer_Segment",
        "Priority",
        "Product_Category",
        "Order_Value_INR",
        "Origin",
        "Destination",
        "Special_Handling",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    for c in missing:
        df[c] = None
    
    df["Order_Value_INR"] = pd.to_numeric(df["Order_Value_INR"], errors="coerce").fillna(0)
    try:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True, errors="coerce")
    except Exception:
        df["Order_Date"] = pd.NaT
    df["Priority"] = df["Priority"].fillna("Standard")
    df["Special_Handling"] = df["Special_Handling"].fillna("None")
    return df[expected_cols]


def normalize_routes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    expected_cols = [
        "Order_ID",
        "Route",
        "Distance_KM",
        "Fuel_Consumption_L",
        "Toll_Charges_INR",
        "Traffic_Delay_Minutes",
        "Weather_Impact",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    for c in missing:
        df[c] = None
    
    for c in ["Distance_KM", "Fuel_Consumption_L", "Toll_Charges_INR", "Traffic_Delay_Minutes"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Weather_Impact"] = df["Weather_Impact"].fillna("None")
    return df[expected_cols]


def normalize_vehicles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    expected_cols = [
        "Vehicle_ID",
        "Vehicle_Type",
        "Capacity_KG",
        "Fuel_Efficiency_KM_per_L",
        "Current_Location",
        "Status",
        "Age_Years",
        "CO2_Emissions_Kg_per_KM",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    for c in missing:
        df[c] = None
    for c in ["Capacity_KG", "Fuel_Efficiency_KM_per_L", "Age_Years", "CO2_Emissions_Kg_per_KM"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Status"] = df["Status"].fillna("Available")
    return df[expected_cols]


# Constraints and Scoring
PRIORITY_WEIGHT = {"Express": 3.0, "Standard": 2.0, "Economy": 1.0}


def is_feasible(order: pd.Series, vehicle: pd.Series) -> bool:
    special = str(order.get("Special_Handling", "None") or "None").strip()
    vtype = str(vehicle.get("Vehicle_Type", "")).strip()

    # Temperature controlled requires Refrigerated
    if special == "Temperature_Controlled" and vtype != "Refrigerated":
        return False

    # Hazmat limited to Medium_Truck or Large_Truck
    if special == "Hazmat" and vtype not in {"Medium_Truck", "Large_Truck"}:
        return False

    # Vehicles not available are excluded
    if str(vehicle.get("Status", "")).strip() != "Available":
        return False

    # Fragile: avoid Express_Bike and Large_Truck if alternatives exist (handled in scoring as penalty)
    return True


def estimate_vehicle_cost(
    distance_km: float,
    fuel_eff_kmpl: float,
    co2_per_km: float,
    toll_inr: float,
    fuel_price_inr_per_l: float,
) -> Tuple[float, float]:
    if fuel_eff_kmpl and fuel_eff_kmpl > 0:
        liters = distance_km / fuel_eff_kmpl
    else:
        liters = 0.0
    fuel_cost = liters * fuel_price_inr_per_l
    direct_cost = fuel_cost + float(toll_inr or 0.0)
    emissions = co2_per_km * distance_km
    return direct_cost, emissions


def compute_pair_score(
    order: pd.Series,
    route: pd.Series,
    vehicle: pd.Series,
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    distance_km = float(route.get("Distance_KM", 0.0) or 0.0)
    toll_inr = float(route.get("Toll_Charges_INR", 0.0) or 0.0)
    delay_min = float(route.get("Traffic_Delay_Minutes", 0.0) or 0.0)
    weather = str(route.get("Weather_Impact", "None") or "None")

    fuel_eff = float(vehicle.get("Fuel_Efficiency_KM_per_L", 0.0) or 0.0)
    co2_per_km = float(vehicle.get("CO2_Emissions_Kg_per_KM", 0.0) or 0.0)

    fuel_price = weights.get("fuel_price_inr_per_l", 100.0)
    cost_inr, emissions_kg = estimate_vehicle_cost(distance_km, fuel_eff, co2_per_km, toll_inr, fuel_price)

    # Time proxy: distance + traffic delay
    time_score = distance_km + (delay_min / 60.0) * 50.0  

    # Proximity penalty if vehicle not at origin
    origin = str(order.get("Origin", "")).strip()
    v_loc = str(vehicle.get("Current_Location", "")).strip()
    proximity_penalty = 0.0 if origin == v_loc else distance_km * 0.05  # heuristic

    # Weather penalty
    weather_penalty_map = {"None": 0.0, "Light_Rain": 0.02, "Fog": 0.03, "Heavy_Rain": 0.05}
    weather_penalty = weather_penalty_map.get(weather, 0.02) * distance_km

    # Fragile penalty for certain vehicle types
    fragile_penalty = 0.0
    if str(order.get("Special_Handling", "None")) == "Fragile" and vehicle.get("Vehicle_Type") in {"Express_Bike", "Large_Truck"}:
        fragile_penalty = distance_km * 0.05

    # Priority influence: higher priority prefers lower (cost/time/penalties)
    base = (
        weights["cost"] * cost_inr
        + weights["time"] * time_score
        + weights["emissions"] * emissions_kg
        + weights["proximity"] * proximity_penalty
        + weather_penalty
        + fragile_penalty
    )

    details = {
        "cost_inr": cost_inr,
        "emissions_kg": emissions_kg,
        "time_score": time_score,
        "proximity_penalty": proximity_penalty,
        "weather_penalty": weather_penalty,
        "fragile_penalty": fragile_penalty,
    }
    return base, details


def greedy_match(
    orders_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    vehicles_df: pd.DataFrame,
    weights: Dict[str, float],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if orders_df.empty or routes_df.empty or vehicles_df.empty:
        return pd.DataFrame(), orders_df

    routes_by_order = routes_df.set_index("Order_ID")

    # Sort orders by priority and value
    def priority_value(row: pd.Series) -> float:
        return PRIORITY_WEIGHT.get(str(row["Priority"]).strip(), 2.0) * 1_000_000 + float(row["Order_Value_INR"])

    sorted_orders = orders_df.sort_values(
        by=["Priority", "Order_Value_INR"],
        key=lambda s: s.map(
            lambda v: PRIORITY_WEIGHT.get(str(v).strip(), 2.0) if s.name == "Priority" else v
        ),
        ascending=[False, False],
    )

    available_vehicle_ids = set(vehicles_df[vehicles_df["Status"] == "Available"]["Vehicle_ID"].tolist())
    vehicle_by_id = {v["Vehicle_ID"]: v for _, v in vehicles_df.iterrows()}

    assignments: List[Dict[str, object]] = []
    unassigned: List[pd.Series] = []

    for _, order in sorted_orders.iterrows():
        oid = order["Order_ID"]
        if oid not in routes_by_order.index:
            unassigned.append(order)
            continue

        route = routes_by_order.loc[oid]

        best_vehicle_id = None
        best_score = float("inf")
        best_details: Dict[str, float] = {}

        for vid in list(available_vehicle_ids):
            vehicle = vehicle_by_id[vid]
            if not is_feasible(order, vehicle):
                continue
            score, details = compute_pair_score(order, route, vehicle, weights)
            if score < best_score:
                best_score = score
                best_vehicle_id = vid
                best_details = details

        if best_vehicle_id is None:
            unassigned.append(order)
            continue

        available_vehicle_ids.remove(best_vehicle_id)
        vehicle = vehicle_by_id[best_vehicle_id]

        # Derived metrics added during assignment
        cost_inr = round(best_details.get("cost_inr", 0.0), 2)
        emissions_kg = round(best_details.get("emissions_kg", 0.0), 2)
        distance_km = float(routes_by_order.loc[oid]["Distance_KM"]) if oid in routes_by_order.index else 0.0
        cost_per_km = round(cost_inr / distance_km, 4) if distance_km > 0 else 0.0
        emission_intensity_per_1k_inr = round((emissions_kg / max(order["Order_Value_INR"], 1.0)) * 1000.0, 4)
        # crude on-time risk score based on delay and priority
        delay_min = float(routes_by_order.loc[oid]["Traffic_Delay_Minutes"]) if oid in routes_by_order.index else 0.0
        priority = str(order["Priority"]).strip()
        target_hours = {"Express": 12, "Standard": 48, "Economy": 96}.get(priority, 48)
        on_time_risk = round(min(1.0, (delay_min / 60.0) / target_hours), 3)

        assignments.append(
            {
                "Order_ID": oid,
                "Vehicle_ID": best_vehicle_id,
                "Vehicle_Type": vehicle["Vehicle_Type"],
                "Origin": order["Origin"],
                "Destination": order["Destination"],
                "Priority": order["Priority"],
                "Order_Value_INR": order["Order_Value_INR"],
                "Route": routes_by_order.loc[oid]["Route"],
                "Distance_KM": distance_km,
                "Cost_Est_INR": cost_inr,
                "Emissions_kg": emissions_kg,
                "Time_Score": round(best_details.get("time_score", 0.0), 2),
                "Score": round(best_score, 2),
                "Cost_per_KM": cost_per_km,
                "Emission_Intensity_kg_per_1kINR": emission_intensity_per_1k_inr,
                "On_Time_Risk": on_time_risk,
            }
        )

    assignments_df = pd.DataFrame(assignments)
    unassigned_df = pd.DataFrame(unassigned).reset_index(drop=True)
    return assignments_df, unassigned_df


# -----------------------------
# UI
# -----------------------------
def sidebar_controls() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    st.sidebar.title("⚙️ Controls")
    st.sidebar.caption("Load data, set weights and run the matcher.")

    st.sidebar.subheader("Data Sources")
    use_default = st.sidebar.toggle("Use bundled CSVs", value=True)

    if use_default:
        orders = load_csv_default("orders.csv")
        routes = load_csv_default("routes_distance.csv")
        vehicles = load_csv_default("vehicle_fleet.csv")
    else:
        orders_file = st.sidebar.file_uploader("Orders CSV", type=["csv"])
        routes_file = st.sidebar.file_uploader("Routes CSV", type=["csv"])
        vehicles_file = st.sidebar.file_uploader("Vehicles CSV", type=["csv"])
        orders = pd.read_csv(orders_file) if orders_file else pd.DataFrame()
        routes = pd.read_csv(routes_file) if routes_file else pd.DataFrame()
        vehicles = pd.read_csv(vehicles_file) if vehicles_file else pd.DataFrame()

    orders = normalize_orders(orders)
    routes = normalize_routes(routes)
    vehicles = normalize_vehicles(vehicles)

    # Validation and missing-data handling
    issues: List[str] = []
    if orders.empty:
        issues.append("Orders data is empty or failed to load.")
    if routes.empty:
        issues.append("Routes data is empty or failed to load.")
    if vehicles.empty:
        issues.append("Vehicles data is empty or failed to load.")

    if not orders.empty and not routes.empty:
        missing_routes = set(orders["Order_ID"]) - set(routes["Order_ID"])
        if missing_routes:
            issues.append(f"{len(missing_routes)} orders missing in routes table. They won't be assigned.")

    if issues:
        with st.sidebar.expander("Data Quality Notes", expanded=False):
            for msg in issues:
                st.warning(msg)

    st.sidebar.subheader("Filters")
    # Order filters
    prios = sorted([p for p in orders["Priority"].dropna().unique().tolist()]) if not orders.empty else []
    segs = sorted([p for p in orders["Customer_Segment"].dropna().unique().tolist()]) if not orders.empty else []
    cats = sorted([p for p in orders["Product_Category"].dropna().unique().tolist()]) if not orders.empty else []
    specs = sorted([p for p in orders["Special_Handling"].dropna().unique().tolist()]) if not orders.empty else []
    origins = sorted([p for p in orders["Origin"].dropna().unique().tolist()]) if not orders.empty else []
    dests = sorted([p for p in orders["Destination"].dropna().unique().tolist()]) if not orders.empty else []

    colf1, colf2 = st.sidebar.columns(2)
    f_prio = colf1.multiselect("Priority", prios, default=prios)
    f_seg = colf2.multiselect("Segment", segs, default=segs)
    colf3, colf4 = st.sidebar.columns(2)
    f_cat = colf3.multiselect("Category", cats, default=cats)
    f_spec = colf4.multiselect("Special Handling", specs, default=specs)
    colf5, colf6 = st.sidebar.columns(2)
    f_origin = colf5.multiselect("Origin", origins, default=origins[:10] if len(origins) > 10 else origins)
    f_dest = colf6.multiselect("Destination", dests, default=dests[:10] if len(dests) > 10 else dests)

    # Date filter
    min_date = orders["Order_Date"].min() if not orders.empty else None
    max_date = orders["Order_Date"].max() if not orders.empty else None
    if min_date and max_date and pd.notna(min_date) and pd.notna(max_date):
        date_range = st.sidebar.date_input("Order Date Range", value=(min_date.date(), max_date.date()))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            orders = orders[(orders["Order_Date"] >= start_date) & (orders["Order_Date"] <= end_date)]

    # Apply order filters
    if not orders.empty:
        if f_prio:
            orders = orders[orders["Priority"].isin(f_prio)]
        if f_seg:
            orders = orders[orders["Customer_Segment"].isin(f_seg)]
        if f_cat:
            orders = orders[orders["Product_Category"].isin(f_cat)]
        if f_spec:
            orders = orders[orders["Special_Handling"].isin(f_spec)]
        if f_origin:
            orders = orders[orders["Origin"].isin(f_origin)]
        if f_dest:
            orders = orders[orders["Destination"].isin(f_dest)]

    # Vehicle filters
    vstates = sorted([p for p in vehicles["Status"].dropna().unique().tolist()]) if not vehicles.empty else []
    vtypes = sorted([p for p in vehicles["Vehicle_Type"].dropna().unique().tolist()]) if not vehicles.empty else []
    vlocs = sorted([p for p in vehicles["Current_Location"].dropna().unique().tolist()]) if not vehicles.empty else []
    colv1, colv2 = st.sidebar.columns(2)
    f_vstate = colv1.multiselect("Vehicle Status", vstates, default=["Available"] if "Available" in vstates else vstates)
    f_vtype = colv2.multiselect("Vehicle Type", vtypes, default=vtypes)
    f_vloc = st.sidebar.multiselect("Vehicle Location", vlocs, default=vlocs[:10] if len(vlocs) > 10 else vlocs)
    if not vehicles.empty:
        if f_vstate:
            vehicles = vehicles[vehicles["Status"].isin(f_vstate)]
        if f_vtype:
            vehicles = vehicles[vehicles["Vehicle_Type"].isin(f_vtype)]
        if f_vloc:
            vehicles = vehicles[vehicles["Current_Location"].isin(f_vloc)]

    st.sidebar.subheader("Objective Weights")
    col1, col2 = st.sidebar.columns(2)
    cost_w = col1.number_input("Cost", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    time_w = col2.number_input("Time", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    col3, col4 = st.sidebar.columns(2)
    emissions_w = col3.number_input("Emissions", min_value=0.0, max_value=10.0, value=0.5, step=0.1)
    proximity_w = col4.number_input("Proximity", min_value=0.0, max_value=10.0, value=0.3, step=0.1)

    fuel_price = st.sidebar.number_input("Fuel Price (INR/L)", min_value=50.0, max_value=200.0, value=100.0, step=1.0)

    weights = {
        "cost": float(cost_w),
        "time": float(time_w),
        "emissions": float(emissions_w),
        "proximity": float(proximity_w),
        "fuel_price_inr_per_l": float(fuel_price),
    }

    return orders, routes, vehicles, weights


def kpi_row(assignments_df: pd.DataFrame) -> None:
    total_assigned = len(assignments_df)
    total_cost = float(assignments_df["Cost_Est_INR"].sum()) if not assignments_df.empty else 0.0
    total_emissions = float(assignments_df["Emissions_kg"].sum()) if not assignments_df.empty else 0.0
    avg_distance = float(assignments_df["Distance_KM"].mean()) if not assignments_df.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Assigned Orders", f"{total_assigned}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Total Cost (INR)", f"{total_cost:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Emissions (kg)", f"{total_emissions:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Avg Distance (KM)", f"{avg_distance:,.0f}")
        st.markdown("</div>", unsafe_allow_html=True)


def results_section(assignments_df: pd.DataFrame, unassigned_df: pd.DataFrame, vehicles_df: pd.DataFrame) -> None:
    st.subheader("Assignments")
    if assignments_df.empty:
        st.info("No feasible assignments found with current settings.")
    else:
        st.dataframe(assignments_df, use_container_width=True, hide_index=True)

        # Plots (bar, histogram, pie, scatter, line)
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            by_type = assignments_df.groupby("Vehicle_Type")["Order_ID"].count().reset_index(name="Assigned_Count")
            fig = px.bar(by_type, x="Vehicle_Type", y="Assigned_Count", color="Vehicle_Type", title="Assignments by Vehicle Type")
            st.plotly_chart(fig, use_container_width=True)
        with chart_col2:
            fig2 = px.histogram(assignments_df, x="Distance_KM", nbins=20, title="Route Distance Distribution")
            st.plotly_chart(fig2, use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)
        with chart_col3:
            by_priority = assignments_df.groupby("Priority")["Order_ID"].count().reset_index(name="Count")
            fig3 = px.pie(by_priority, names="Priority", values="Count", hole=0.4, title="Share by Priority")
            st.plotly_chart(fig3, use_container_width=True)
        with chart_col4:
            fig4 = px.scatter(
                assignments_df,
                x="Cost_Est_INR",
                y="Emissions_kg",
                size="Distance_KM",
                color="Vehicle_Type",
                hover_data=["Order_ID", "Priority", "Cost_per_KM", "On_Time_Risk"],
                title="Cost vs Emissions (bubble size = Distance)",
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Optional line chart by date if available
        if "Order_Date" in assignments_df.columns or "Order_Date" in st.session_state:
            # Join date from orders if not present
            pass

        # Download
        csv_bytes = assignments_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Assignments CSV", data=csv_bytes, file_name="assignments.csv", mime="text/csv")
        if not unassigned_df.empty:
            st.download_button(
                "Download Unassigned CSV",
                data=unassigned_df.to_csv(index=False).encode("utf-8"),
                file_name="unassigned.csv",
                mime="text/csv",
            )
        combined = pd.concat([
            assignments_df.assign(Status="Assigned"),
            unassigned_df.assign(Status="Unassigned") if not unassigned_df.empty else pd.DataFrame(),
        ], ignore_index=True)
        if not combined.empty:
            st.download_button(
                "Download Combined CSV",
                data=combined.to_csv(index=False).encode("utf-8"),
                file_name="orders_with_assignment.csv",
                mime="text/csv",
            )

    st.subheader("Unassigned Orders")
    if not unassigned_df.empty:
        st.warning(f"{len(unassigned_df)} order(s) unassigned.")
        st.dataframe(unassigned_df[["Order_ID", "Priority", "Origin", "Destination", "Special_Handling", "Order_Value_INR"]], use_container_width=True, hide_index=True)
    else:
        st.success("All orders assigned.")

    st.subheader("Vehicle Availability Snapshot")
    st.dataframe(vehicles_df[["Vehicle_ID", "Vehicle_Type", "Current_Location", "Status", "Fuel_Efficiency_KM_per_L"]], use_container_width=True, hide_index=True)


def main() -> None:
    style_app()
    st.title("🚚 Dynamic Fleet Manager")
    st.caption("Match the right vehicle to the right order under real-world constraints.")

    orders, routes, vehicles, weights = sidebar_controls()

    st.markdown("### Data Preview")
    prev1, prev2, prev3 = st.columns(3)
    with prev1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Orders**")
        st.dataframe(orders.head(10), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with prev2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Routes**")
        st.dataframe(routes.head(10), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with prev3:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Vehicles**")
        st.dataframe(vehicles.head(10), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    run = st.button("Run Matching", type="primary", use_container_width=True)

    if run:
        with st.spinner("Computing best assignments..."):
            assignments_df, unassigned_df = greedy_match(orders, routes, vehicles, weights)
            # Join dates for charts if available
            if not assignments_df.empty and "Order_ID" in orders.columns and "Order_Date" in orders.columns:
                assignments_df = assignments_df.merge(
                    orders[["Order_ID", "Order_Date"]], on="Order_ID", how="left"
                )
        kpi_row(assignments_df)
        results_section(assignments_df, unassigned_df, vehicles)
    else:
        st.info("Adjust weights and click 'Run Matching' to generate assignments.")


if __name__ == "__main__":
    main()


