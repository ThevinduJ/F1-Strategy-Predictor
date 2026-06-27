import streamlit as st
import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import numpy as np

# CRITICAL INSTRUCTION: Enable caching BEFORE making any data requests
fastf1.Cache.enable_cache('./data')

# Setup FastF1 matplotlib styles
fastf1.plotting.setup_mpl()

st.title("🏎️ F1 Strategy Predictor")
st.markdown("Compare the telemetry of two drivers' fastest laps.")

st.sidebar.header("Select Parameters")
year = st.sidebar.selectbox("Year", options=list(range(2018, 2026)), index=5)

@st.cache_data
def get_race_list(year):
    try:
        schedule = fastf1.get_event_schedule(year)
        schedule = schedule[schedule['RoundNumber'] > 0]
        return schedule['EventName'].tolist()
    except Exception:
        return ["Monza"]

with st.spinner("Loading race list..."):
    races_list = get_race_list(year)

race = st.sidebar.selectbox("Race", options=races_list)

@st.cache_data
def get_driver_list(year, race):
    try:
        session = fastf1.get_session(year, race, 'R')
        session.load()
        return session.results['Abbreviation'].tolist()
    except Exception:
        return ["VER", "HAM"]

with st.spinner("Loading driver list..."):
    drivers_list = get_driver_list(year, race)

driver1 = st.sidebar.selectbox("Driver 1", options=drivers_list, index=0)
driver2 = st.sidebar.selectbox("Driver 2", options=drivers_list, index=1 if len(drivers_list) > 1 else 0)

if st.sidebar.button("Load Data & Plot"):
    with st.spinner("Fetching session data... This may take a minute."):
        try:
            # Fetch the Race session data
            session = fastf1.get_session(year, race, 'R')
            session.load()
            
            laps_driver1 = session.laps.pick_driver(driver1)
            laps_driver2 = session.laps.pick_driver(driver2)
            
            if laps_driver1.empty or laps_driver2.empty:
                st.error("Could not find lap data for one or both drivers. Please check the abbreviations.")
            else:
                tab1, tab2 = st.tabs(['Telemetry Analysis', 'Tire Degradation'])
                
                # Fetch colors
                try:
                    color_d1 = fastf1.plotting.driver_color(driver1, session=session)
                except Exception:
                    color_d1 = "cyan"
                    
                try:
                    color_d2 = fastf1.plotting.driver_color(driver2, session=session)
                except Exception:
                    color_d2 = "red"
                    
                # Teammate line styles
                if color_d1 == color_d2:
                    style1 = '-'
                    style2 = '--'
                else:
                    style1 = '-'
                    style2 = '-'
                
                with tab1:
                    fastest_d1 = laps_driver1.pick_fastest()
                    fastest_d2 = laps_driver2.pick_fastest()
                    
                    tel_d1 = fastest_d1.get_telemetry().add_distance()
                    tel_d2 = fastest_d2.get_telemetry().add_distance()
                    
                    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
                    
                    # Speed
                    ax1.plot(tel_d1['Distance'], tel_d1['Speed'], color=color_d1, linestyle=style1, label=f"{driver1} ({fastest_d1['LapTime']})")
                    ax1.plot(tel_d2['Distance'], tel_d2['Speed'], color=color_d2, linestyle=style2, label=f"{driver2} ({fastest_d2['LapTime']})")
                    ax1.set_ylabel('Speed (km/h)')
                    ax1.legend()
                    ax1.set_title(f"Fastest Lap Telemetry: {driver1} vs {driver2} - {race} {year}")
                    
                    # Throttle
                    ax2.plot(tel_d1['Distance'], tel_d1['Throttle'], color=color_d1, linestyle=style1)
                    ax2.plot(tel_d2['Distance'], tel_d2['Throttle'], color=color_d2, linestyle=style2)
                    ax2.set_ylabel('Throttle %')
                    
                    # Brake
                    ax3.plot(tel_d1['Distance'], tel_d1['Brake'], color=color_d1, linestyle=style1)
                    ax3.plot(tel_d2['Distance'], tel_d2['Brake'], color=color_d2, linestyle=style2)
                    ax3.set_ylabel('Brake %')
                    ax3.set_xlabel('Distance (m)')
                    
                    st.pyplot(fig)
                
                with tab2:
                    # Filter for quick laps
                    quick_laps_d1 = laps_driver1.pick_quicklaps().dropna(subset=['LapTime', 'TyreLife']).copy()
                    quick_laps_d2 = laps_driver2.pick_quicklaps().dropna(subset=['LapTime', 'TyreLife']).copy()
                    
                    quick_laps_d1['LapTime_s'] = quick_laps_d1['LapTime'].dt.total_seconds()
                    quick_laps_d2['LapTime_s'] = quick_laps_d2['LapTime'].dt.total_seconds()
                    
                    fig2, ax_td = plt.subplots(figsize=(10, 6))
                    
                    ax_td.scatter(quick_laps_d1['TyreLife'], quick_laps_d1['LapTime_s'], color=color_d1, alpha=0.6, label=f"{driver1} Laps")
                    ax_td.scatter(quick_laps_d2['TyreLife'], quick_laps_d2['LapTime_s'], color=color_d2, alpha=0.6, label=f"{driver2} Laps")
                    
                    if len(quick_laps_d1) > 1:
                        x1 = quick_laps_d1['TyreLife'].values
                        y1 = quick_laps_d1['LapTime_s'].values
                        m1, b1 = np.polyfit(x1, y1, 1)
                        ax_td.plot(x1, m1*x1 + b1, color=color_d1, linestyle=style1, label=f"{driver1} Trend")
                        
                    if len(quick_laps_d2) > 1:
                        x2 = quick_laps_d2['TyreLife'].values
                        y2 = quick_laps_d2['LapTime_s'].values
                        m2, b2 = np.polyfit(x2, y2, 1)
                        ax_td.plot(x2, m2*x2 + b2, color=color_d2, linestyle=style2, label=f"{driver2} Trend")
                    
                    ax_td.set_xlabel('Tyre Life (Laps)')
                    ax_td.set_ylabel('Lap Time (s)')
                    ax_td.set_title(f"Tire Degradation: {driver1} vs {driver2}")
                    ax_td.legend()
                    
                    st.pyplot(fig2)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
