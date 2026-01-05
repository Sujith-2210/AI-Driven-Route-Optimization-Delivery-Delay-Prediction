# Streamlit Dashboard for Delivery Delay Prediction

import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from prediction_pipeline import DeliveryPredictor
from route_optimizer import optimize_deliveries

st.set_page_config(page_title="Delivery Delay Dashboard", page_icon="🚚", layout="wide")

# Load CSS from external file
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


def load_data():
    return pd.read_csv("prepared_logistics_dataset.csv")


def load_predictor():
    return DeliveryPredictor()


def get_heatmap_data(df_sample):
    weights = {"Low": 0.2, "Medium": 0.5, "High": 0.8, "Very High": 1.0}
    return [[row["dest_lat"], row["dest_lng"], weights.get(row["traffic_level"], 0.5)] for _, row in df_sample.iterrows()]


def create_heatmap(df):
    np.random.seed(42)
    sample = df.iloc[:200]
    heat_data = get_heatmap_data(sample)
    m = folium.Map(location=[sample["dest_lat"].mean(), sample["dest_lng"].mean()], zoom_start=12, tiles="cartodbpositron")
    HeatMap(heat_data, radius=15, blur=10).add_to(m)
    return m


def main():
    st.title("🚚 AI-Driven Delivery Delay Prediction")
    st.markdown("Real-time delay predictions and route optimization")
    
    df = load_data()
    predictor = load_predictor()
    
    page = st.sidebar.radio("Select View", ["📊 Dashboard", "🗺️ Route Optimizer", "🔮 Predict Delay", "📈 Driver Analytics"])
    
    if page == "📊 Dashboard":
        show_dashboard(df)
    elif page == "🗺️ Route Optimizer":
        show_route_optimizer(df)
    elif page == "🔮 Predict Delay":
        show_predictor(predictor)
    else:
        show_driver_analytics(df)


def show_dashboard(df):
    st.header("📊 Delivery Performance Dashboard")
    
    total = len(df)
    delayed = int(df["delayed"].sum())
    on_time = (1 - delayed / total) * 100
    
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Deliveries", f"{total:,}")
    c2.metric("On-Time Rate", f"{on_time:.1f}%")
    c3.metric("Delayed", f"{delayed:,}")
    c4.metric("Avg Distance", f"{df['distance_km'].mean():.1f} km")
    c5.metric("Active Drivers", df["driver_id"].nunique())
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🚦 Delay by Traffic")
        data = df.groupby("traffic_level")["delayed"].mean().reset_index()
        data["delayed"] *= 100
        fig = px.bar(data, x="traffic_level", y="delayed", color="delayed", color_continuous_scale="RdYlGn_r")
        fig.update_layout(showlegend=False, height=300, coloraxis_showscale=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="t1")
    
    with col2:
        st.subheader("🌤️ Delay by Weather")
        data = df.groupby("weather_condition")["delayed"].mean().reset_index()
        data["delayed"] *= 100
        fig = px.pie(data, values="delayed", names="weather_condition")
        fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key="t2")
    
    st.subheader("🔥 Traffic Heatmap")
    st_folium(create_heatmap(df), width=None, height=400, returned_objects=[], key="hm")
    
    st.subheader("🚗 Vehicle Performance")
    v = df.groupby("vehicle_type").agg({"delayed": ["count", "mean"], "distance_km": "mean"}).reset_index()
    v.columns = ["Type", "Count", "Delay%", "Dist"]
    v["Delay%"] = (v["Delay%"] * 100).round(1).astype(str) + "%"
    v["Dist"] = v["Dist"].round(1).astype(str) + " km"
    st.dataframe(v, hide_index=True)


def show_route_optimizer(df):
    st.header("🗺️ Route Optimization")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        n = st.slider("Stops", 3, 15, 8, key="ns")
        if st.button("🔄 Optimize", type="primary", key="ob"):
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            s = df.sample(n)[["delivery_id", "dest_lat", "dest_lng", "traffic_level"]]
            st.session_state["route"] = optimize_deliveries(s)
    
    with c2:
        if "route" in st.session_state:
            r = st.session_state["route"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Distance", f"{r['total_distance_km']} km")
            m2.metric("Time", f"{r['estimated_time_hours']:.2f} hrs")
            m3.metric("Stops", r["num_stops"])
    
    if "route" in st.session_state:
        st.subheader("📍 Route Map")
        rd = st.session_state["route"]["route_details"]
        lats = [d["lat"] for d in rd]
        lons = [d["lon"] for d in rd]
        m = folium.Map(location=[np.mean(lats), np.mean(lons)], zoom_start=13, tiles="cartodbpositron")
        folium.PolyLine([[d["lat"], d["lon"]] for d in rd], weight=4, color="#00ff88").add_to(m)
        for i, s in enumerate(rd):
            c = "green" if s["delivery_id"] == "DEPOT" else "blue"
            folium.Marker([s["lat"], s["lon"]], popup=f"Stop {i}", icon=folium.Icon(color=c)).add_to(m)
        st_folium(m, width=None, height=450, returned_objects=[], key="rm")
        st.dataframe(pd.DataFrame(rd)[["stop_number", "delivery_id", "traffic_level"]], hide_index=True)


def show_predictor(predictor):
    st.header("🔮 Delay Prediction")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("📦 Details")
        dist = st.slider("Distance (km)", 0.5, 50.0, 10.0, key="pd")
        wt = st.slider("Weight (kg)", 0.1, 50.0, 5.0, key="pw")
        veh = st.selectbox("Vehicle", ["Bike", "Van", "Truck", "Electric"], key="pv")
        traf = st.selectbox("Traffic", ["Low", "Medium", "High", "Very High"], key="pt")
        wea = st.selectbox("Weather", ["Clear", "Cloudy", "Rainy", "Foggy"], key="pwe")
        road = st.selectbox("Road", ["City", "Highway", "Rural"], key="pr")
        
        if st.button("🎯 Predict", type="primary", key="pb"):
            st.session_state["pred"] = predictor.predict_delay({"distance_km": dist, "package_weight_kg": wt, "vehicle_type": veh, "traffic_level": traf, "weather_condition": wea, "road_type": road})
    
    with c2:
        st.subheader("📊 Result")
        if "pred" in st.session_state:
            p = st.session_state["pred"]
            icons = {"Low": "🟢", "Medium": "🟡", "High": "🟠", "Critical": "🔴"}
            st.markdown(f"### {icons[p['risk_level']]} Risk: **{p['risk_level']}**")
            fig = go.Figure(go.Indicator(mode="gauge+number", value=p["delay_probability"]*100, number={"suffix": "%"}, title={"text": "Delay Probability"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#3388ff"}, "steps": [{"range": [0, 30], "color": "#2ecc71"}, {"range": [30, 60], "color": "#f1c40f"}, {"range": [60, 80], "color": "#e67e22"}, {"range": [80, 100], "color": "#e74c3c"}]}))
            fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="gc")
            if p["delayed"]:
                st.warning("⚠️ High risk - consider rescheduling")
            else:
                st.success("✅ Likely on time")


def show_driver_analytics(df):
    st.header("📈 Driver Analytics")
    
    ds = df.groupby("driver_id").agg({"delayed": ["count", "sum", "mean"], "distance_km": "sum"}).reset_index()
    ds.columns = ["ID", "Total", "Delayed", "Rate", "Dist"]
    ds["OnTime%"] = ((1 - ds["Rate"]) * 100).round(1)
    ds = ds.sort_values("Total", ascending=False)
    
    c1, c2, c3 = st.columns(3)
    best = ds.loc[ds["OnTime%"].idxmax()]
    busy = ds.loc[ds["Total"].idxmax()]
    c1.metric("🏆 Best", f"#{int(best['ID'])}", f"{best['OnTime%']}% on-time")
    c2.metric("📦 Busiest", f"#{int(busy['ID'])}", f"{int(busy['Total'])} deliveries")
    c3.metric("👥 Total", len(ds))
    
    st.divider()
    st.subheader("📊 Top 10")
    fig = px.bar(ds.head(10), x="ID", y="Total", color="OnTime%", color_continuous_scale="RdYlGn")
    fig.update_layout(height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key="dc")
    
    st.subheader("📋 All Drivers")
    show = ds[["ID", "Total", "Delayed", "OnTime%", "Dist"]].head(20).copy()
    show["Dist"] = show["Dist"].round(1).astype(str) + " km"
    st.dataframe(show, hide_index=True)


if __name__ == "__main__":
    main()
