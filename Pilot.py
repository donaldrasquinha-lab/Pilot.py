import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import requests
import time
import streamlit.components.v1 as components
from datetime import date

# ─────────────────────────────────────────────
# CONFIG — fill these in
# ─────────────────────────────────────────────
UPSTOX_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"   # paste your daily token
INSTRUMENT_KEY      = "NSE_INDEX|Nifty 50"       # Nifty 50 spot
ADX_PERIOD          = 14                          # standard DMI period
CANDLE_INTERVAL     = "1minute"                   # 1-min candles for ADX

HEADERS = {
    "Accept":        "application/json",
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
}

# ─────────────────────────────────────────────
# INITIALIZATION
# ─────────────────────────────────────────────
if "trade_active" not in st.session_state:
    st.session_state.update({
        "trade_active": False,
        "entry_price":  0.0,
        "target_pts":   0,
        "exit_price":   0.0,
        "last_signal":  None,
        "last_error":   None,
    })

# ─────────────────────────────────────────────
# BROWSER ALERTS
# ─────────────────────────────────────────────
def send_browser_alert(title: str, msg: str):
    components.html(f"""
    <script>
    if (Notification.permission === "granted") {{
        new Notification("{title}", {{ body: "{msg}" }});
    }} else if (Notification.permission !== "denied") {{
        Notification.requestPermission().then(p => {{
            if (p === "granted")
                new Notification("{title}", {{ body: "{msg}" }});
        }});
    }}
    </script>
    """, height=0)

# ─────────────────────────────────────────────
# UPSTOX: LIVE PRICE via Market Quote OHLC API
# ─────────────────────────────────────────────
def get_live_price() -> float:
    """
    Endpoint: GET /v2/market-quote/ohlc
    Returns last_price for the instrument.
    Docs: https://upstox.com/developer/api-documentation/get-market-quote-ohlc/
    """
    url = "https://api.upstox.com/v2/market-quote/ohlc"
    params = {"instrument_key": INSTRUMENT_KEY, "interval": "1d"}
    r = requests.get(url, headers=HEADERS, params=params, timeout=5)
    r.raise_for_status()
    data = r.json()
    # Response shape: data["data"]["NSE_INDEX:Nifty 50"]["last_price"]
    feeds = data["data"]
    key = list(feeds.keys())[0]          # e.g. "NSE_INDEX:Nifty 50"
    return float(feeds[key]["last_price"])

# ─────────────────────────────────────────────
# UPSTOX: INTRADAY 1-MIN CANDLES
# ─────────────────────────────────────────────
def get_intraday_candles() -> pd.DataFrame:
    """
    Endpoint: GET /v2/historical-candle/intraday/{instrumentKey}/1minute
    Candle format: [timestamp, open, high, low, close, volume, oi]
    Docs: https://upstox.com/developer/api-documentation/get-intra-day-candle-data/
    """
    encoded_key = INSTRUMENT_KEY.replace("|", "%7C").replace(" ", "%20")
    url = (
        f"https://api.upstox.com/v2/historical-candle/intraday"
        f"/{encoded_key}/{CANDLE_INTERVAL}"
    )
    r = requests.get(url, headers=HEADERS, timeout=5)
    r.raise_for_status()
    candles = r.json()["data"]["candles"]

    df = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "vol", "oi"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)   # oldest first for TA
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float)
    return df

# ─────────────────────────────────────────────
# COMPUTE ADX / +DI / -DI  (via pandas_ta)
# ─────────────────────────────────────────────
def compute_dmi(df: pd.DataFrame) -> dict:
    """
    pandas_ta.adx returns columns: ADX_14, DMP_14 (+DI), DMN_14 (-DI)
    Requires at least ADX_PERIOD+1 rows.
    """
    if len(df) < ADX_PERIOD + 1:
        return None                       # not enough candles yet
    adx_df = ta.adx(df["high"], df["low"], df["close"], length=ADX_PERIOD)
    adx = float(adx_df[f"ADX_{ADX_PERIOD}"].iloc[-1])
    pdi = float(adx_df[f"DMP_{ADX_PERIOD}"].iloc[-1])
    ndi = float(adx_df[f"DMN_{ADX_PERIOD}"].iloc[-1])
    return {"adx": adx, "pdi": pdi, "ndi": ndi}

# ─────────────────────────────────────────────
# COMBINED MARKET DATA FETCH
# ─────────────────────────────────────────────
def get_market_data():
    """
    Returns (price, dmi_dict) or raises on error.
    Falls back to mock values when token is placeholder.
    """
    if UPSTOX_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN_HERE":
        # ── MOCK MODE (no token set) ──
        price = round(24500.0 + np.random.uniform(-50, 50), 2)
        adx   = round(30.0 + np.random.uniform(-2, 15), 2)
        pdi   = round(20.0 + np.random.uniform(-3, 10), 2)
        ndi   = round(10.0 + np.random.uniform(-3, 10), 2)
        return price, {"adx": adx, "pdi": pdi, "ndi": ndi}

    price  = get_live_price()
    df     = get_intraday_candles()
    dmi    = compute_dmi(df)
    if dmi is None:
        raise ValueError("Not enough intraday candles to compute ADX yet")
    return price, dmi

# ─────────────────────────────────────────────
# GEAR LOGIC
# ─────────────────────────────────────────────
def gear_from_adx(adx: float) -> int:
    if adx >= 40: return 4    # 200 pts target
    if adx >= 35: return 3    # 150 pts
    if adx >= 30: return 2    # 100 pts
    return 1                  # 50 pts

GEAR_TARGETS = {4: 200, 3: 150, 2: 100, 1: 50}

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Nifty Pilot", layout="wide")
st.title("🎯 Nifty Pilot — Upstox Live")

if UPSTOX_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN_HERE":
    st.info("ℹ️ Running in **mock mode** — paste your Upstox access token into `UPSTOX_ACCESS_TOKEN` to go live.")

# ── FETCH ──
try:
    live_px, dmi = get_market_data()
    st.session_state.last_error = None
except Exception as e:
    st.error(f"API error: {e}")
    st.stop()

gear  = gear_from_adx(dmi["adx"])
target_pts = GEAR_TARGETS[gear]

# ── METRICS ROW ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("LTP (Nifty 50)",  f"₹{live_px:,.2f}")
c2.metric("ADX",             f"{dmi['adx']:.2f}",  f"Gear {gear}")
c3.metric("+DI",             f"{dmi['pdi']:.2f}")
c4.metric("-DI",             f"{dmi['ndi']:.2f}")

# ── SIGNAL ──
bullish        = dmi["pdi"] > dmi["ndi"]
current_signal = "BULLISH 🐂" if bullish else "BEARISH 🐻"
signal_color   = "green" if bullish else "red"
st.markdown(
    f"**Signal:** :{signal_color}[{current_signal}] &nbsp;|&nbsp; "
    f"Target at this ADX: **+{target_pts} pts**"
)

if st.session_state.last_signal and st.session_state.last_signal != current_signal:
    send_browser_alert("SIGNAL CHANGE", f"Market is now {current_signal}")
st.session_state.last_signal = current_signal

st.divider()

# ── TRADE PANEL ──
if not st.session_state.trade_active:
    if st.button("🚀 ENTER TRADE", use_container_width=True, type="primary"):
        st.session_state.trade_active = True
        st.session_state.entry_price  = live_px
        st.session_state.target_pts   = target_pts
        st.session_state.exit_price   = round(live_px + target_pts, 2)
        st.rerun()
else:
    pnl = round(live_px - st.session_state.entry_price, 2)
    pnl_color = "green" if pnl >= 0 else "red"

    st.warning(f"🟡 TRADE LIVE — Target **+{st.session_state.target_pts} pts**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Entry",        f"₹{st.session_state.entry_price:,.2f}")
    m2.metric("Exit Target",  f"₹{st.session_state.exit_price:,.2f}")
    m3.metric("Live P&L",     f"₹{pnl:+,.2f}", delta_color="normal")

    # Auto-exit on target
    if live_px >= st.session_state.exit_price:
        send_browser_alert("🎯 TARGET HIT", f"Exiting at ₹{live_px:,.2f} | P&L: +{pnl}")
        st.balloons()
        st.success(f"Target hit! P&L: ₹{pnl:+,.2f}")
        st.session_state.trade_active = False
        time.sleep(2)
        st.rerun()

    if st.button("🛑 EMERGENCY EXIT", use_container_width=True, type="secondary"):
        send_browser_alert("EXIT", f"Manual exit at ₹{live_px:,.2f} | P&L: {pnl:+}")
        st.session_state.trade_active = False
        st.rerun()

# ── FOOTER ──
st.caption(f"Instrument: `{INSTRUMENT_KEY}` | ADX period: {ADX_PERIOD} | "
           f"Last refreshed: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%H:%M:%S IST')}")

# ── AUTO-REFRESH (every 5s) ──
time.sleep(5)
st.rerun()
