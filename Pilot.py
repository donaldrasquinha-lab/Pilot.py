import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import streamlit.components.v1 as components

# ─────────────────────────────────────────────
# CONFIG — fill these in
# ─────────────────────────────────────────────
UPSTOX_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"
INSTRUMENT_KEY      = "NSE_INDEX|Nifty 50"
ADX_PERIOD          = 14
CANDLE_INTERVAL     = "1minute"

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
# MANUAL ADX / +DI / -DI
# Pure numpy — no pandas_ta, works on Python 3.14
# ─────────────────────────────────────────────
def compute_dmi(df: pd.DataFrame, period: int = 14):
    n = len(df)
    if n < period + 1:
        return None

    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:]  - close[:-1])
        )
    )

    up   = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    dm_p = np.where((up > down) & (up > 0),   up,   0.0)
    dm_n = np.where((down > up) & (down > 0), down, 0.0)

    def wilder(arr, p):
        out = np.zeros(len(arr))
        out[p - 1] = arr[:p].sum()
        for i in range(p, len(arr)):
            out[i] = out[i - 1] - (out[i - 1] / p) + arr[i]
        return out

    atr  = wilder(tr,   period)
    sdmp = wilder(dm_p, period)
    sdmn = wilder(dm_n, period)

    safe = np.where(atr == 0, 1e-10, atr)
    pdi  = 100 * sdmp / safe
    ndi  = 100 * sdmn / safe
    dx   = 100 * np.abs(pdi - ndi) / np.where((pdi + ndi) == 0, 1e-10, pdi + ndi)
    adx  = wilder(dx, period)

    return {
        "adx": round(float(adx[-1]), 2),
        "pdi": round(float(pdi[-1]), 2),
        "ndi": round(float(ndi[-1]), 2),
    }

# ─────────────────────────────────────────────
# UPSTOX: LIVE PRICE
# ─────────────────────────────────────────────
def get_live_price() -> float:
    url    = "https://api.upstox.com/v2/market-quote/ohlc"
    params = {"instrument_key": INSTRUMENT_KEY, "interval": "1d"}
    r      = requests.get(url, headers=HEADERS, params=params, timeout=5)
    r.raise_for_status()
    feeds  = r.json()["data"]
    key    = list(feeds.keys())[0]
    return float(feeds[key]["last_price"])

# ─────────────────────────────────────────────
# UPSTOX: INTRADAY CANDLES
# ─────────────────────────────────────────────
def get_intraday_candles() -> pd.DataFrame:
    encoded = INSTRUMENT_KEY.replace("|", "%7C").replace(" ", "%20")
    url     = (
        f"https://api.upstox.com/v2/historical-candle/intraday"
        f"/{encoded}/{CANDLE_INTERVAL}"
    )
    r       = requests.get(url, headers=HEADERS, timeout=5)
    r.raise_for_status()
    candles = r.json()["data"]["candles"]
    df = pd.DataFrame(candles, columns=["ts","open","high","low","close","vol","oi"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    for col in ["open","high","low","close"]:
        df[col] = df[col].astype(float)
    return df

# ─────────────────────────────────────────────
# COMBINED FETCH
# ─────────────────────────────────────────────
def get_market_data():
    if UPSTOX_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN_HERE":
        price = round(24500.0 + np.random.uniform(-50, 50), 2)
        adx   = round(30.0 + np.random.uniform(-2, 15), 2)
        pdi   = round(20.0 + np.random.uniform(-3, 10), 2)
        ndi   = round(10.0 + np.random.uniform(-3, 10), 2)
        return price, {"adx": adx, "pdi": pdi, "ndi": ndi}

    price = get_live_price()
    df    = get_intraday_candles()
    dmi   = compute_dmi(df, ADX_PERIOD)
    if dmi is None:
        raise ValueError(f"Need {ADX_PERIOD+1}+ candles — market may have just opened.")
    return price, dmi

# ─────────────────────────────────────────────
# GEAR LOGIC
# ─────────────────────────────────────────────
GEAR_TARGETS = {4: 200, 3: 150, 2: 100, 1: 50}

def gear_from_adx(adx: float) -> int:
    if adx >= 40: return 4
    if adx >= 35: return 3
    if adx >= 30: return 2
    return 1

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Nifty Pilot", layout="wide")
st.title("🎯 Nifty Pilot — Upstox Live")

if UPSTOX_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN_HERE":
    st.info("ℹ️ Running in **mock mode** — paste your Upstox access token to go live.")

try:
    live_px, dmi = get_market_data()
except Exception as e:
    st.error(f"API error: {e}")
    st.stop()

gear       = gear_from_adx(dmi["adx"])
target_pts = GEAR_TARGETS[gear]

c1, c2, c3, c4 = st.columns(4)
c1.metric("LTP (Nifty 50)", f"₹{live_px:,.2f}")
c2.metric("ADX",            f"{dmi['adx']:.2f}", f"Gear {gear} → +{target_pts} pts")
c3.metric("+DI",            f"{dmi['pdi']:.2f}")
c4.metric("-DI",            f"{dmi['ndi']:.2f}")

bullish        = dmi["pdi"] > dmi["ndi"]
current_signal = "BULLISH 🐂" if bullish else "BEARISH 🐻"
color          = "green" if bullish else "red"
st.markdown(f"**Signal:** :{color}[{current_signal}]")

if st.session_state.last_signal and st.session_state.last_signal != current_signal:
    send_browser_alert("SIGNAL CHANGE", f"Market is now {current_signal}")
st.session_state.last_signal = current_signal

st.divider()

if not st.session_state.trade_active:
    if st.button("🚀 ENTER TRADE", use_container_width=True, type="primary"):
        st.session_state.trade_active = True
        st.session_state.entry_price  = live_px
        st.session_state.target_pts   = target_pts
        st.session_state.exit_price   = round(live_px + target_pts, 2)
        st.rerun()
else:
    pnl = round(live_px - st.session_state.entry_price, 2)
    st.warning(f"🟡 TRADE LIVE — Target **+{st.session_state.target_pts} pts**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Entry",       f"₹{st.session_state.entry_price:,.2f}")
    m2.metric("Exit Target", f"₹{st.session_state.exit_price:,.2f}")
    m3.metric("Live P&L",    f"₹{pnl:+,.2f}")

    if live_px >= st.session_state.exit_price:
        send_browser_alert("🎯 TARGET HIT", f"P&L: +{pnl} pts")
        st.balloons()
        st.success(f"Target hit! P&L: ₹{pnl:+,.2f}")
        st.session_state.trade_active = False
        time.sleep(2)
        st.rerun()

    if st.button("🛑 EMERGENCY EXIT", use_container_width=True):
        send_browser_alert("EXIT", f"Manual exit | P&L: {pnl:+}")
        st.session_state.trade_active = False
        st.rerun()

st.caption(
    f"Instrument: `{INSTRUMENT_KEY}` | ADX period: {ADX_PERIOD} | "
    f"Last refresh: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%H:%M:%S IST')}"
)

time.sleep(5)
st.rerun()
