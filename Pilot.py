import streamlit as st
import pandas as pd
import pandas_ta as ta
import time
import numpy as np
import streamlit.components.v1 as components

# --- INITIALIZATION ---
if 'trade_active' not in st.session_state:
    st.session_state.update({
        'trade_active': False,
        'entry_price': 0.0,
        'target_pts': 0,
        'exit_price': 0.0,
        'last_signal': None
    })

# --- BROWSER ALERTS ---
def send_browser_alert(title, msg):
    components.html(f"""
    <script>
    if (Notification.permission === "granted") {{
        new Notification("{title}", {{ body: "{msg}" }});
    }} else {{
        Notification.requestPermission();
    }}
    </script>
    """, height=0)

# --- DATA & LOGIC ---
def get_mock_market_data():
    # Simulated 1-min data for testing
    price = 250.0 + np.random.uniform(-2, 2)
    # Mocking DMI values
    adx = 32.0 + np.random.uniform(-1, 1)
    pdi = 25.0 + np.random.uniform(-2, 2)
    ndi = 15.0 + np.random.uniform(-2, 2)
    return round(price, 2), {"adx": adx, "pdi": pdi, "ndi": ndi}

# --- UI ---
st.title("🎯 Nifty Pilot Test Environment")
live_px, dmi = get_mock_market_data()

# 1. LIVE MONITORING
c1, c2, c3 = st.columns(3)
c1.metric("LTP", f"₹{live_px}")
c2.metric("ADX", f"{dmi['adx']:.2f}")
c3.metric("Direction", "+DI" if dmi['pdi'] > dmi['ndi'] else "-DI")

# 2. SIGNAL DETECTION
current_signal = "BULLISH 🐂" if dmi['pdi'] > dmi['ndi'] else "BEARISH 🐻"
if st.session_state.last_signal != current_signal:
    st.session_state.last_signal = current_signal
    send_browser_alert("SIGNAL CHANGE", f"Market is now {current_signal}")

# 3. TRADE EXECUTION
if not st.session_state.trade_active:
    if st.button("🚀 ENTER TRADE", use_container_width=True):
        st.session_state.trade_active = True
        st.session_state.entry_price = live_px
        
        # GEAR LOGIC
        val = dmi['adx']
        if val >= 40: st.session_state.target_pts = 200
        elif val >= 35: st.session_state.target_pts = 150
        elif val >= 30: st.session_state.target_pts = 100
        else: st.session_state.target_pts = 50
        
        st.session_state.exit_price = live_px + st.session_state.target_pts
        st.rerun()
else:
    st.warning(f"TRADE LIVE: Target +{st.session_state.target_pts} pts")
    m1, m2 = st.columns(2)
    m1.metric("Entry", st.session_state.entry_price)
    m2.metric("Exit Target", st.session_state.exit_price)
    
    # EXIT TRIGGER
    if live_px >= st.session_state.exit_price:
        send_browser_alert("🎯 TARGET HIT", "Exiting position...")
        st.balloons()
        st.session_state.trade_active = False
        time.sleep(2)
        st.rerun()

    if st.button("🛑 EMERGENCY EXIT"):
        st.session_state.trade_active = False
        st.rerun()

time.sleep(1)
st.rerun()
