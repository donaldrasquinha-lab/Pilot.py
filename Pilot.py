import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import streamlit.components.v1 as components
from datetime import date, timedelta

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
UPSTOX_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"
INSTRUMENT_KEY      = "NSE_INDEX|Nifty 50"
ADX_PERIOD          = 14
CANDLE_INTERVAL     = "1minute"
OI_PERCENTILE       = 70
TOP_N_STRIKES       = 10

HEADERS = {
    "Accept":        "application/json",
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
}

# ─────────────────────────────────────────────
# SESSION STATE
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
            if (p === "granted") new Notification("{title}", {{ body: "{msg}" }});
        }});
    }}
    </script>
    """, height=0)

# ─────────────────────────────────────────────
# ADX / DMI  (pure numpy)
# ─────────────────────────────────────────────
def compute_dmi(df: pd.DataFrame, period: int = 14):
    if len(df) < period + 1:
        return None
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    tr   = np.maximum(high[1:]-low[1:],
           np.maximum(np.abs(high[1:]-close[:-1]),
                      np.abs(low[1:]-close[:-1])))
    up   = high[1:]-high[:-1]
    down = low[:-1]-low[1:]
    dm_p = np.where((up>down)&(up>0),   up,   0.0)
    dm_n = np.where((down>up)&(down>0), down, 0.0)
    def wilder(arr, p):
        out = np.zeros(len(arr))
        out[p-1] = arr[:p].sum()
        for i in range(p, len(arr)):
            out[i] = out[i-1] - out[i-1]/p + arr[i]
        return out
    atr  = wilder(tr,   period)
    sdmp = wilder(dm_p, period)
    sdmn = wilder(dm_n, period)
    safe = np.where(atr==0, 1e-10, atr)
    pdi  = 100*sdmp/safe
    ndi  = 100*sdmn/safe
    dx   = 100*np.abs(pdi-ndi)/np.where((pdi+ndi)==0,1e-10,pdi+ndi)
    adx  = wilder(dx, period)
    return {"adx": round(float(adx[-1]),2),
            "pdi": round(float(pdi[-1]),2),
            "ndi": round(float(ndi[-1]),2)}

# ─────────────────────────────────────────────
# UPSTOX API HELPERS
# ─────────────────────────────────────────────
def get_live_price() -> float:
    r = requests.get("https://api.upstox.com/v2/market-quote/ohlc",
                     headers=HEADERS,
                     params={"instrument_key": INSTRUMENT_KEY, "interval": "1d"},
                     timeout=5)
    r.raise_for_status()
    feeds = r.json()["data"]
    return float(feeds[list(feeds.keys())[0]]["last_price"])

def get_intraday_candles(instrument_key: str) -> pd.DataFrame:
    enc = instrument_key.replace("|", "%7C").replace(" ", "%20")
    r   = requests.get(
        f"https://api.upstox.com/v2/historical-candle/intraday/{enc}/{CANDLE_INTERVAL}",
        headers=HEADERS, timeout=5)
    r.raise_for_status()
    candles = r.json()["data"]["candles"]
    df = pd.DataFrame(candles, columns=["ts","open","high","low","close","vol","oi"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    for c in ["open","high","low","close"]: df[c] = df[c].astype(float)
    return df

def get_nearest_expiry() -> str:
    r = requests.get("https://api.upstox.com/v2/option/contract",
                     headers=HEADERS,
                     params={"instrument_key": INSTRUMENT_KEY}, timeout=5)
    r.raise_for_status()
    expiries = sorted({item["expiry"] for item in r.json()["data"] if item.get("expiry")})
    today    = date.today().isoformat()
    future   = [e for e in expiries if e >= today]
    return future[0] if future else expiries[-1]

def get_option_chain(expiry: str) -> list:
    r = requests.get("https://api.upstox.com/v2/option/chain",
                     headers=HEADERS,
                     params={"instrument_key": INSTRUMENT_KEY, "expiry_date": expiry},
                     timeout=8)
    r.raise_for_status()
    return r.json()["data"]

# ─────────────────────────────────────────────
# BEST OPTION SCANNER
# ─────────────────────────────────────────────
def find_best_option(spot_price: float) -> dict | None:
    expiry = get_nearest_expiry()
    chain  = get_option_chain(expiry)
    rows   = []
    for strike in chain:
        sp = strike["strike_price"]
        for side, key in [("CE","call_options"),("PE","put_options")]:
            opt  = strike.get(key, {})
            md   = opt.get("market_data", {})
            oi   = md.get("oi", 0) or 0
            ltp  = md.get("ltp", 0) or 0
            prev_oi = md.get("prev_oi", 0) or 0
            iv   = opt.get("option_greeks", {}).get("iv", 0) or 0
            ikey = opt.get("instrument_key", "")
            if ltp > 0 and oi > 0 and ikey:
                rows.append({"strike":sp,"side":side,"ltp":ltp,"oi":oi,
                             "prev_oi":prev_oi,"iv":iv,"ikey":ikey,
                             "dist":abs(sp-spot_price),"expiry":expiry})
    if not rows:
        return None
    df_chain    = pd.DataFrame(rows)
    oi_thresh   = np.percentile(df_chain["oi"], OI_PERCENTILE)
    df_filtered = df_chain[df_chain["oi"] >= oi_thresh]
    if df_filtered.empty:
        df_filtered = df_chain
    df_top = df_filtered.nsmallest(TOP_N_STRIKES, "dist")
    results = []
    prog = st.progress(0, text="Scanning option chain for highest ADX…")
    total = len(df_top)
    for i, (_, row) in enumerate(df_top.iterrows()):
        try:
            candles = get_intraday_candles(row["ikey"])
            dmi     = compute_dmi(candles, ADX_PERIOD)
            if dmi:
                results.append({**row.to_dict(), **dmi})
        except Exception:
            pass
        prog.progress((i+1)/total, text=f"Scanning {row['side']} {int(row['strike'])}… ({i+1}/{total})")
    prog.empty()
    if not results:
        return None
    return max(results, key=lambda x: x["adx"])

# ─────────────────────────────────────────────
# GEAR LOGIC
# ─────────────────────────────────────────────
GEAR_TARGETS = {4:200, 3:150, 2:100, 1:50}

def gear_from_adx(adx: float) -> int:
    if adx>=40: return 4
    if adx>=35: return 3
    if adx>=30: return 2
    return 1

def gear_label(g: int) -> str:
    return {4:"Very strong trend",3:"Strong trend",2:"Moderate trend",1:"Weak trend"}[g]

# ─────────────────────────────────────────────
# MOCK FALLBACK
# ─────────────────────────────────────────────
def mock_market_data():
    price   = round(24500 + np.random.uniform(-50,50), 2)
    adx     = round(30 + np.random.uniform(-2,15), 2)
    pdi     = round(20 + np.random.uniform(-3,10), 2)
    ndi     = round(10 + np.random.uniform(-3,10), 2)
    bullish = pdi > ndi
    side    = "CE" if bullish else "PE"
    strike  = round(price/50)*50 + (50 if bullish else -50)
    opt_oi  = int(np.random.uniform(1e5, 5e5))
    best    = {
        "strike":  strike, "side": side,
        "ltp":     round(np.random.uniform(80,300), 2),
        "oi":      opt_oi,
        "prev_oi": int(opt_oi * np.random.uniform(0.85,1.15)),
        "iv":      round(np.random.uniform(12,30), 2),
        "adx":     round(adx + np.random.uniform(-5,10), 2),
        "pdi":     pdi, "ndi": ndi,
        "expiry":  (date.today() + timedelta(days=3)).isoformat(),
        "ikey":    f"NSE_FO|MOCK{strike}{side}",
    }
    return price, {"adx":adx,"pdi":pdi,"ndi":ndi}, best

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Nifty Pilot", layout="wide")

MOCK = UPSTOX_ACCESS_TOKEN == "YOUR_ACCESS_TOKEN_HERE"

# ── Header ──
col_title, col_mode = st.columns([3,1])
col_title.title("Nifty Pilot")
if MOCK:
    col_mode.warning("Mock mode")
else:
    col_mode.success("Live — Upstox")

if MOCK:
    st.info("Paste your Upstox access token into `UPSTOX_ACCESS_TOKEN` to go live.")

# ── FETCH ──
try:
    if MOCK:
        live_px, dmi, best_opt = mock_market_data()
    else:
        live_px  = get_live_price()
        df_idx   = get_intraday_candles(INSTRUMENT_KEY)
        dmi      = compute_dmi(df_idx, ADX_PERIOD)
        if dmi is None:
            st.warning("Not enough candles yet — market may have just opened.")
            st.stop()
        best_opt = find_best_option(live_px)
except Exception as e:
    st.error(f"API error: {e}")
    st.stop()

gear       = gear_from_adx(dmi["adx"])
target_pts = GEAR_TARGETS[gear]
bullish    = dmi["pdi"] > dmi["ndi"]
signal     = "BULLISH" if bullish else "BEARISH"
sig_color  = "green" if bullish else "red"

# ══════════════════════════════════════════════
# SECTION 1 — INDEX
# ══════════════════════════════════════════════
st.subheader("Nifty 50 — index")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("LTP",    f"₹{live_px:,.2f}")
c2.metric("ADX",    f"{dmi['adx']:.2f}", f"Gear {gear}")
c3.metric("+DI",    f"{dmi['pdi']:.2f}")
c4.metric("-DI",    f"{dmi['ndi']:.2f}")
c5.metric("Signal", signal)

# Gear bar
gear_html = "".join(
    f'<div style="flex:1;height:8px;border-radius:4px;background:'
    f'{"#3B6D11" if i<=gear else "#e5e5e5"}"></div>'
    for i in range(1,5)
)
st.markdown(
    f'<div style="display:flex;gap:6px;margin-top:-8px;margin-bottom:4px">{gear_html}</div>'
    f'<p style="font-size:12px;color:gray">Gear {gear} — {gear_label(gear)} '
    f'| Target: <b>+{target_pts} pts</b></p>',
    unsafe_allow_html=True
)

current_signal = f"{signal}"
if st.session_state.last_signal and st.session_state.last_signal != current_signal:
    send_browser_alert("SIGNAL CHANGE", f"Market is now {current_signal}")
st.session_state.last_signal = current_signal

st.divider()

# ══════════════════════════════════════════════
# SECTION 2 — BEST OPTION TO BUY
# ══════════════════════════════════════════════
st.subheader("Best option to buy")

if best_opt:
    opt_gear  = gear_from_adx(best_opt["adx"])
    oi_chg    = best_opt["oi"] - best_opt.get("prev_oi", best_opt["oi"])
    oi_chg_pct= (oi_chg / best_opt["prev_oi"] * 100) if best_opt.get("prev_oi") else 0

    side_color = "🟢" if best_opt["side"] == "CE" else "🟠"

    # Main buy recommendation box
    buy_bg    = "#f0faf4" if best_opt["side"]=="CE" else "#fff8f0"
    buy_border= "#3B6D11" if best_opt["side"]=="CE" else "#854F0B"

    st.markdown(f"""
    <div style="border:2px solid {buy_border};border-radius:12px;padding:16px 20px;
                background:{buy_bg};margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start">
        <div>
          <span style="font-size:12px;font-weight:500;color:{buy_border};
                letter-spacing:.05em">BUY SIGNAL</span>
          <div style="display:flex;align-items:baseline;gap:10px;margin-top:4px">
            <span style="font-size:28px;font-weight:500;color:{buy_border}">
              {best_opt['side']} {int(best_opt['strike'])}
            </span>
            <span style="font-size:13px;color:gray">exp {best_opt['expiry']}</span>
          </div>
          <div style="font-size:13px;color:gray;margin-top:2px">
            LTP &nbsp;<span style="font-size:20px;font-weight:500;color:#111">
              ₹{best_opt['ltp']:.2f}</span>
          </div>
        </div>
        <div style="text-align:right">
          <div style="font-size:11px;color:gray">Option ADX</div>
          <div style="font-size:32px;font-weight:500;color:{buy_border}">
            {best_opt['adx']:.2f}
          </div>
          <div style="font-size:11px;color:gray">Gear {opt_gear} — {gear_label(opt_gear)}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Detail metrics row
    d1,d2,d3,d4,d5 = st.columns(5)
    d1.metric("OI",        f"{best_opt['oi']:,}")
    d2.metric("OI change", f"{oi_chg:+,}", f"{oi_chg_pct:+.1f}%",
              delta_color="normal" if oi_chg>=0 else "inverse")
    d3.metric("IV",        f"{best_opt['iv']:.1f}%")
    d4.metric("+DI",       f"{best_opt['pdi']:.2f}")
    d5.metric("-DI",       f"{best_opt['ndi']:.2f}")

    # OI interpretation
    if best_opt["side"] == "CE":
        if oi_chg > 0:
            oi_note = "Rising CE OI — writers building positions; potential resistance. Buy only if ADX and +DI are strong."
        else:
            oi_note = "Falling CE OI — short covering; CE sellers retreating. Favourable for CE buyers."
    else:
        if oi_chg > 0:
            oi_note = "Rising PE OI — writers building positions; potential support. Buy only if ADX and -DI are strong."
        else:
            oi_note = "Falling PE OI — short covering; PE sellers retreating. Favourable for PE buyers."

    st.caption(f"OI insight: {oi_note}")
    st.caption(f"Instrument key: `{best_opt['ikey']}`")
else:
    st.warning("No option signal found — market may have just opened or chain is empty.")

st.divider()

# ══════════════════════════════════════════════
# SECTION 3 — TRADE CONTROL
# ══════════════════════════════════════════════
st.subheader("Trade control")

if not st.session_state.trade_active:
    if st.button("Enter trade", use_container_width=True, type="primary"):
        st.session_state.trade_active = True
        st.session_state.entry_price  = live_px
        st.session_state.target_pts   = target_pts
        st.session_state.exit_price   = round(live_px + target_pts, 2)
        st.rerun()
else:
    pnl = round(live_px - st.session_state.entry_price, 2)
    st.warning(f"Trade live — target +{st.session_state.target_pts} pts")
    m1,m2,m3 = st.columns(3)
    m1.metric("Entry",       f"₹{st.session_state.entry_price:,.2f}")
    m2.metric("Exit target", f"₹{st.session_state.exit_price:,.2f}")
    m3.metric("Live P&L",    f"₹{pnl:+,.2f}")

    if live_px >= st.session_state.exit_price:
        send_browser_alert("TARGET HIT", f"P&L: +{pnl} pts")
        st.balloons()
        st.success(f"Target hit! P&L: ₹{pnl:+,.2f}")
        st.session_state.trade_active = False
        time.sleep(2)
        st.rerun()

    if st.button("Emergency exit", use_container_width=True):
        send_browser_alert("EXIT", f"Manual exit | P&L: {pnl:+}")
        st.session_state.trade_active = False
        st.rerun()

# ── FOOTER ──
st.caption(
    f"Index: `{INSTRUMENT_KEY}` | ADX period: {ADX_PERIOD} | "
    f"OI filter: top {100-OI_PERCENTILE}% | "
    f"Refresh: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%H:%M:%S IST')}"
)

time.sleep(5)
st.rerun()
