"""
Nifty Pilot — Upstox Live Trading Dashboard
============================================
Token:  stored in .streamlit/secrets.toml  →  [upstox] access_token
Run:    streamlit run nifty_pilot_upstox.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import streamlit.components.v1 as components
from datetime import date, timedelta
from collections import deque

# ─────────────────────────────────────────────
# TOKEN — read from secrets.toml
# ─────────────────────────────────────────────
try:
    ACCESS_TOKEN = st.secrets["upstox"]["access_token"]
except Exception:
    ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"

MOCK_MODE = ACCESS_TOKEN in ("", "YOUR_ACCESS_TOKEN_HERE")

# ─────────────────────────────────────────────
# INDEX CATALOGUE
# ─────────────────────────────────────────────
INDICES = {
    "Nifty 50":        {"key": "NSE_INDEX|Nifty 50",        "lot": 25},
    "Nifty Bank":      {"key": "NSE_INDEX|Nifty Bank",      "lot": 15},
    "Nifty Fin Svc":   {"key": "NSE_INDEX|Nifty Fin Service","lot": 25},
    "Nifty Midcap 50": {"key": "NSE_INDEX|Nifty Midcap 50", "lot": 50},
    "Sensex":          {"key": "BSE_INDEX|SENSEX",          "lot": 10},
}

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
ADX_PERIOD      = 14
CANDLE_INTERVAL = "1minute"
OI_PERCENTILE   = 70
TOP_N_STRIKES   = 10
HISTORY_LEN     = 15
MONITOR_SECS    = 300
GEAR_PTS        = {4: 200, 3: 150, 2: 100, 1: 50}
ORDER_URL       = "https://api-hft.upstox.com/v2/order/place"
BASE_URL        = "https://api.upstox.com/v2"

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _init():
    defaults = dict(
        selected_index  = "Nifty 50",
        trade_active    = False,
        trade_mode      = "mockup",       # "mockup" | "live"
        entry_price     = 0.0,
        entry_adx       = 0.0,
        entry_pdi       = 0.0,
        entry_ndi       = 0.0,
        entry_side      = "",
        target_pts      = 0,
        trailing_sl_pct = 30.0,
        exit_price      = 0.0,
        sl_price        = 0.0,
        highest_pnl     = 0.0,
        last_signal     = None,
        live_order_id   = None,
        monitor_start   = None,
        signal_log      = [],
        act_log         = [],
        h_times         = deque(maxlen=HISTORY_LEN),
        h_opt_ltp       = deque(maxlen=HISTORY_LEN),
        h_opt_adx       = deque(maxlen=HISTORY_LEN),
        h_idx_adx       = deque(maxlen=HISTORY_LEN),
        h_opt_pdi       = deque(maxlen=HISTORY_LEN),
        h_opt_ndi       = deque(maxlen=HISTORY_LEN),
        h_oi            = deque(maxlen=HISTORY_LEN),
        # live-trade order panel state
        lo_strike       = None,
        lo_side         = None,
        lo_ikey         = None,
        lo_qty          = 1,
        lo_target_pts   = 50,
        lo_tsl_pct      = 30.0,
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()

HEADERS = {
    "Accept":        "application/json",
    "Content-Type":  "application/json",
    "Authorization": f"Bearer {ACCESS_TOKEN}",
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def browser_alert(title, msg):
    components.html(f"""<script>
    if(Notification.permission==="granted"){{new Notification("{title}",{{body:"{msg}"}})}}
    else if(Notification.permission!=="denied"){{Notification.requestPermission().then(p=>{{
    if(p==="granted")new Notification("{title}",{{body:"{msg}"}});}});}}
    </script>""", height=0)

def add_log(store_key, entry):
    st.session_state[store_key].insert(0, entry)
    if len(st.session_state[store_key]) > 30:
        st.session_state[store_key].pop()

def ts():
    return pd.Timestamp.now(tz="Asia/Kolkata").strftime("%H:%M:%S")

def gear(adx):
    return 4 if adx >= 40 else 3 if adx >= 35 else 2 if adx >= 30 else 1

def ha(d):
    return float(np.mean(list(d))) if d else 0.0

# ─────────────────────────────────────────────
# ADX / DMI — pure numpy
# ─────────────────────────────────────────────
def compute_dmi(df, period=14):
    if len(df) < period + 1:
        return None
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    tr   = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    up   = h[1:]-h[:-1]; dn = l[:-1]-l[1:]
    dm_p = np.where((up>dn)&(up>0), up, 0.0)
    dm_n = np.where((dn>up)&(dn>0), dn, 0.0)
    def w(a, p):
        o = np.zeros(len(a)); o[p-1] = a[:p].sum()
        for i in range(p, len(a)): o[i] = o[i-1]-o[i-1]/p+a[i]
        return o
    atr  = w(tr, period); sdmp = w(dm_p, period); sdmn = w(dm_n, period)
    safe = np.where(atr==0, 1e-10, atr)
    pdi  = 100*sdmp/safe; ndi = 100*sdmn/safe
    dx   = 100*np.abs(pdi-ndi)/np.where((pdi+ndi)==0,1e-10,pdi+ndi)
    adx  = w(dx, period)
    return {"adx":round(float(adx[-1]),2),
            "pdi":round(float(pdi[-1]),2),
            "ndi":round(float(ndi[-1]),2)}

# ─────────────────────────────────────────────
# UPSTOX API CALLS
# ─────────────────────────────────────────────
def get_live_price(ikey):
    r = requests.get(f"{BASE_URL}/market-quote/ohlc", headers=HEADERS,
                     params={"instrument_key": ikey, "interval": "1d"}, timeout=5)
    r.raise_for_status()
    feeds = r.json()["data"]
    return float(feeds[list(feeds.keys())[0]]["last_price"])

def get_candles(ikey):
    enc = ikey.replace("|", "%7C").replace(" ", "%20")
    r   = requests.get(f"{BASE_URL}/historical-candle/intraday/{enc}/{CANDLE_INTERVAL}",
                       headers=HEADERS, timeout=5)
    r.raise_for_status()
    c   = r.json()["data"]["candles"]
    df  = pd.DataFrame(c, columns=["ts","open","high","low","close","vol","oi"])
    df["ts"] = pd.to_datetime(df["ts"])
    df  = df.sort_values("ts").reset_index(drop=True)
    for col in ["open","high","low","close"]: df[col] = df[col].astype(float)
    return df

def get_nearest_expiry(ikey):
    r = requests.get(f"{BASE_URL}/option/contract", headers=HEADERS,
                     params={"instrument_key": ikey}, timeout=5)
    r.raise_for_status()
    exp    = sorted({i["expiry"] for i in r.json()["data"] if i.get("expiry")})
    today  = date.today().isoformat()
    future = [e for e in exp if e >= today]
    return future[0] if future else exp[-1]

def get_chain(ikey, expiry):
    r = requests.get(f"{BASE_URL}/option/chain", headers=HEADERS,
                     params={"instrument_key": ikey, "expiry_date": expiry}, timeout=8)
    r.raise_for_status()
    return r.json()["data"]

def find_best_option(spot, ikey):
    expiry = get_nearest_expiry(ikey)
    chain  = get_chain(ikey, expiry)
    rows   = []
    for strike in chain:
        sp = strike["strike_price"]
        for side, key in [("CE","call_options"), ("PE","put_options")]:
            opt     = strike.get(key, {})
            md      = opt.get("market_data", {})
            oi      = md.get("oi", 0) or 0
            ltp     = md.get("ltp", 0) or 0
            prev_oi = md.get("prev_oi", 0) or 0
            iv      = opt.get("option_greeks", {}).get("iv", 0) or 0
            o_ikey  = opt.get("instrument_key", "")
            if ltp > 0 and oi > 0 and o_ikey:
                rows.append({"strike":sp,"side":side,"ltp":ltp,"oi":oi,
                             "prev_oi":prev_oi,"iv":iv,"ikey":o_ikey,
                             "dist":abs(sp-spot),"expiry":expiry})
    if not rows: return None
    df       = pd.DataFrame(rows)
    oi_thr   = np.percentile(df["oi"], OI_PERCENTILE)
    df_top   = df[df["oi"] >= oi_thr].nsmallest(TOP_N_STRIKES, "dist")
    results  = []
    prog     = st.progress(0, text="Scanning option chain for highest ADX…")
    for i, (_, row) in enumerate(df_top.iterrows()):
        try:
            dmi = compute_dmi(get_candles(row["ikey"]), ADX_PERIOD)
            if dmi: results.append({**row.to_dict(), **dmi})
        except Exception:
            pass
        prog.progress((i+1)/len(df_top), text=f"Scanning {row['side']} {int(row['strike'])}…")
    prog.empty()
    return max(results, key=lambda x: x["adx"]) if results else None

# ─────────────────────────────────────────────
# LIVE ORDER PLACEMENT
# ─────────────────────────────────────────────
def place_market_order(instrument_token, qty, transaction_type="BUY"):
    """Place a MARKET order via Upstox HFT endpoint. Returns order_id."""
    payload = {
        "quantity":           qty,
        "product":            "D",          # Intraday
        "validity":           "DAY",
        "price":              0,
        "instrument_token":   instrument_token,
        "order_type":         "MARKET",
        "transaction_type":   transaction_type,
        "disclosed_quantity": 0,
        "trigger_price":      0,
        "is_amo":             False,
        "tag":                "nifty-pilot",
    }
    r = requests.post(ORDER_URL, json=payload, headers=HEADERS, timeout=8)
    r.raise_for_status()
    return r.json()["data"]["order_id"]

def place_sl_order(instrument_token, qty, trigger_price, limit_price, transaction_type="SELL"):
    """Place a SL order for exit leg."""
    payload = {
        "quantity":           qty,
        "product":            "D",
        "validity":           "DAY",
        "price":              limit_price,
        "instrument_token":   instrument_token,
        "order_type":         "SL",
        "transaction_type":   transaction_type,
        "disclosed_quantity": 0,
        "trigger_price":      trigger_price,
        "is_amo":             False,
        "tag":                "nifty-pilot-sl",
    }
    r = requests.post(ORDER_URL, json=payload, headers=HEADERS, timeout=8)
    r.raise_for_status()
    return r.json()["data"]["order_id"]

# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────
def buy_confidence(dmi, opt):
    h_opt_ltp = ha(st.session_state.h_opt_ltp)
    h_opt_adx = ha(st.session_state.h_opt_adx)
    h_oi      = ha(st.session_state.h_oi)
    checks = [
        ("Option ADX vs 15m avg",  opt["adx"] > h_opt_adx and opt["adx"] > 30,           25),
        ("Index ADX strength",     dmi["adx"] >= 30,                                      20),
        ("+DI/-DI direction",
            (opt["side"]=="CE" and opt["pdi"] > opt["ndi"]) or
            (opt["side"]=="PE" and opt["ndi"] > opt["pdi"]),                               25),
        ("OI vs 15m avg",          opt["oi"] >= h_oi if h_oi else True,                   15),
        ("Option price rising",    opt["ltp"] > h_opt_ltp if h_opt_ltp else True,         15),
    ]
    score = sum(p for _, passed, p in checks if passed)
    return score, checks

# ─────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────
def mock_data(idx_name):
    bases = {"Nifty 50":24500,"Nifty Bank":51000,"Nifty Fin Svc":22000,
             "Nifty Midcap 50":12000,"Sensex":80000}
    base = bases.get(idx_name, 24500)
    px   = round(base + np.random.uniform(-80, 80), 2)
    adx  = round(30 + np.random.uniform(-3, 15), 2)
    pdi  = round(20 + np.random.uniform(-4, 10), 2)
    ndi  = round(10 + np.random.uniform(-4, 10), 2)
    bullish = pdi > ndi; side = "CE" if bullish else "PE"
    strike  = round(px/50)*50 + (50 if bullish else -50)
    oi      = int(np.random.uniform(1e5, 5e5))
    best = {
        "strike":  strike, "side": side,
        "ltp":     round(np.random.uniform(80, 300), 2),
        "oi":      oi, "prev_oi": int(oi * np.random.uniform(.85, 1.15)),
        "iv":      round(np.random.uniform(12, 30), 2),
        "adx":     round(adx + np.random.uniform(-5, 10), 2),
        "pdi":     pdi, "ndi": ndi,
        "expiry":  (date.today() + timedelta(days=3)).isoformat(),
        "ikey":    f"NSE_FO|MOCK{strike}{side}",
    }
    return px, {"adx":adx,"pdi":pdi,"ndi":ndi}, best

# ─────────────────────────────────────────────
# TRAILING SL LOGIC
# ─────────────────────────────────────────────
def update_trailing_sl(current_ltp, entry_price, tsl_pct):
    """Returns new SL price based on highest P&L seen so far."""
    pnl = current_ltp - entry_price
    if pnl > st.session_state.highest_pnl:
        st.session_state.highest_pnl = pnl
    trail_lock  = st.session_state.highest_pnl * (1 - tsl_pct / 100)
    sl_abs      = entry_price + max(0, trail_lock)
    return round(sl_abs, 2)

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(page_title="Nifty Pilot", layout="wide", page_icon="")

# ── TOP BAR ──────────────────────────────────
hdr_l, hdr_idx, hdr_px, hdr_mode = st.columns([2, 2, 1.5, 1])
hdr_l.title("Nifty Pilot")

# Index selector
selected = hdr_idx.selectbox(
    "Index",
    options=list(INDICES.keys()),
    index=list(INDICES.keys()).index(st.session_state.selected_index),
    label_visibility="collapsed",
)
if selected != st.session_state.selected_index:
    # Reset history on index change
    for k in ["h_times","h_opt_ltp","h_opt_adx","h_idx_adx",
              "h_opt_pdi","h_opt_ndi","h_oi"]:
        st.session_state[k] = deque(maxlen=HISTORY_LEN)
    st.session_state.selected_index = selected
    st.session_state.trade_active   = False

ikey     = INDICES[selected]["key"]
lot_size = INDICES[selected]["lot"]

# Live spot price
if MOCK_MODE:
    spot_display = round({"Nifty 50":24500,"Nifty Bank":51000,"Nifty Fin Svc":22000,
                          "Nifty Midcap 50":12000,"Sensex":80000}.get(selected,24500)
                         + np.random.uniform(-30,30), 2)
else:
    try:
        spot_display = get_live_price(ikey)
    except Exception:
        spot_display = 0.0

hdr_px.metric("Spot", f"₹{spot_display:,.2f}")
hdr_mode.warning("Mock") if MOCK_MODE else hdr_mode.success("Live")

if MOCK_MODE:
    st.info(
        "Running in **mock mode**. "
        "Add your Upstox access token to `.streamlit/secrets.toml` under `[upstox]` to go live."
    )

# ─────────────────────────────────────────────
# FETCH MARKET DATA
# ─────────────────────────────────────────────
try:
    if MOCK_MODE:
        live_px, dmi, best_opt = mock_data(selected)
    else:
        live_px  = get_live_price(ikey)
        df_idx   = get_candles(ikey)
        dmi      = compute_dmi(df_idx, ADX_PERIOD)
        if not dmi: st.warning("Waiting for candles…"); st.stop()
        best_opt = find_best_option(live_px, ikey)
except Exception as e:
    st.error(f"API error: {e}"); st.stop()

if not best_opt:
    st.warning("No option signal found — chain may be empty or market just opened.")
    st.stop()

# Push rolling history
now_ts = ts()
st.session_state.h_times.append(now_ts)
st.session_state.h_opt_ltp.append(best_opt["ltp"])
st.session_state.h_opt_adx.append(best_opt["adx"])
st.session_state.h_idx_adx.append(dmi["adx"])
st.session_state.h_opt_pdi.append(best_opt["pdi"])
st.session_state.h_opt_ndi.append(best_opt["ndi"])
st.session_state.h_oi.append(best_opt["oi"])

score, checks    = buy_confidence(dmi, best_opt)
idx_g            = gear(dmi["adx"])
opt_g            = gear(best_opt["adx"])
bullish          = dmi["pdi"] > dmi["ndi"]
oi_chg           = best_opt["oi"] - best_opt.get("prev_oi", best_opt["oi"])

# ══════════════════════════════════════════════
# DASHBOARD 1 — INDEX + BUY SIGNAL
# ══════════════════════════════════════════════
st.subheader(f"{selected} — index")
c1, c2, c3, c4 = st.columns(4)
c1.metric("LTP",       f"₹{live_px:,.2f}")
c2.metric("ADX",       f"{dmi['adx']:.2f}", f"Gear {idx_g} → +{GEAR_PTS[idx_g]} pts")
c3.metric("+DI / -DI", f"{dmi['pdi']:.2f} / {dmi['ndi']:.2f}")
c4.metric("Signal",    "BULLISH" if bullish else "BEARISH")

sig = "BULLISH" if bullish else "BEARISH"
if st.session_state.last_signal and st.session_state.last_signal != sig:
    browser_alert("SIGNAL CHANGE", f"{selected} is now {sig}")
st.session_state.last_signal = sig

st.divider()

# Best option buy box
st.subheader("Best option to buy")
bdr = "#3B6D11" if best_opt["side"] == "CE" else "#854F0B"
bbg = "#f0faf4" if best_opt["side"] == "CE" else "#fff8f0"
st.markdown(f"""
<div style="border:2px solid {bdr};border-radius:12px;padding:14px 18px;background:{bbg};margin-bottom:14px">
  <div style="display:flex;justify-content:space-between;align-items:flex-start">
    <div>
      <div style="font-size:11px;font-weight:500;color:{bdr};letter-spacing:.05em;margin-bottom:3px">BUY SIGNAL</div>
      <div style="display:flex;align-items:baseline;gap:8px">
        <span style="font-size:26px;font-weight:500;color:{bdr}">{best_opt['side']} {int(best_opt['strike'])}</span>
        <span style="font-size:12px;color:gray">exp {best_opt['expiry']}</span>
      </div>
      <div style="font-size:12px;color:gray;margin-top:2px">LTP &nbsp;
        <span style="font-size:20px;font-weight:500;color:#111">₹{best_opt['ltp']:.2f}</span>
      </div>
    </div>
    <div style="text-align:right">
      <div style="font-size:10px;color:gray">Option ADX</div>
      <div style="font-size:30px;font-weight:500;color:{bdr}">{best_opt['adx']:.2f}</div>
      <div style="font-size:11px;color:gray">Gear {opt_g}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

d1,d2,d3,d4,d5 = st.columns(5)
d1.metric("OI",        f"{best_opt['oi']:,}")
d2.metric("OI change", f"{oi_chg:+,}", f"{oi_chg/max(best_opt.get('prev_oi',1),1)*100:+.1f}%")
d3.metric("IV",        f"{best_opt['iv']:.1f}%")
d4.metric("+DI",       f"{best_opt['pdi']:.2f}")
d5.metric("-DI",       f"{best_opt['ndi']:.2f}")

oi_note = (
    ("Rising CE OI → resistance building" if oi_chg > 0 else "Falling CE OI → short covering, bullish")
    if best_opt["side"] == "CE" else
    ("Rising PE OI → support building"    if oi_chg > 0 else "Falling PE OI → short covering, bearish")
)
st.caption(f"OI insight: {oi_note}   |   `{best_opt['ikey']}`")

st.divider()

# ── MOCKUP TRADE CONTROL ──────────────────────
st.subheader("Trade control — mockup")
st.caption("Simulates entries and exits locally. No real order is sent.")

if not st.session_state.trade_active or st.session_state.trade_mode != "mockup":
    if st.button("Enter trade (mockup)", use_container_width=True, type="primary"):
        st.session_state.trade_active  = True
        st.session_state.trade_mode    = "mockup"
        st.session_state.entry_price   = best_opt["ltp"]
        st.session_state.entry_adx     = best_opt["adx"]
        st.session_state.entry_pdi     = best_opt["pdi"]
        st.session_state.entry_ndi     = best_opt["ndi"]
        st.session_state.entry_side    = best_opt["side"]
        st.session_state.target_pts    = GEAR_PTS[opt_g]
        st.session_state.exit_price    = round(best_opt["ltp"] + GEAR_PTS[opt_g], 2)
        st.session_state.sl_price      = round(best_opt["ltp"] * 0.85, 2)
        st.session_state.highest_pnl   = 0.0
        st.session_state.monitor_start = time.time()
        add_log("act_log", {"time":now_ts,"event":f"[MOCK] Entered {best_opt['side']} {int(best_opt['strike'])} @ ₹{best_opt['ltp']:.2f}"})
        st.rerun()
elif st.session_state.trade_active and st.session_state.trade_mode == "mockup":
    pnl = round(best_opt["ltp"] - st.session_state.entry_price, 2)
    tsl = update_trailing_sl(best_opt["ltp"], st.session_state.entry_price,
                             st.session_state.trailing_sl_pct)
    st.warning(f"[MOCK] Trade live — target +{st.session_state.target_pts} pts")
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Entry",        f"₹{st.session_state.entry_price:,.2f}")
    m2.metric("Exit target",  f"₹{st.session_state.exit_price:,.2f}")
    m3.metric("Trailing SL",  f"₹{tsl:,.2f}")
    m4.metric("Live P&L",     f"₹{pnl:+,.2f}")
    st.session_state.sl_price = tsl

    if best_opt["ltp"] >= st.session_state.exit_price:
        browser_alert("TARGET HIT", f"[MOCK] P&L: +{pnl}")
        st.balloons(); st.success(f"[MOCK] Target hit! P&L: ₹{pnl:+,.2f}")
        add_log("act_log", {"time":now_ts,"event":f"[MOCK] Target hit @ ₹{best_opt['ltp']:.2f} | P&L ₹{pnl:+.2f}"})
        st.session_state.trade_active = False; time.sleep(2); st.rerun()
    if best_opt["ltp"] <= tsl and st.session_state.highest_pnl > 0:
        browser_alert("TRAILING SL HIT", f"[MOCK] Exiting. P&L: {pnl:+}")
        st.error(f"[MOCK] Trailing SL hit! P&L: ₹{pnl:+,.2f}")
        add_log("act_log", {"time":now_ts,"event":f"[MOCK] TSL hit @ ₹{best_opt['ltp']:.2f} | P&L ₹{pnl:+.2f}"})
        st.session_state.trade_active = False; time.sleep(2); st.rerun()
    if st.button("Emergency exit (mockup)", use_container_width=True):
        add_log("act_log", {"time":now_ts,"event":f"[MOCK] Manual exit @ ₹{best_opt['ltp']:.2f} | P&L ₹{pnl:+.2f}"})
        st.session_state.trade_active = False; st.rerun()

if st.session_state.act_log:
    with st.expander("Activity log", expanded=False):
        st.dataframe(pd.DataFrame(st.session_state.act_log[:20]),
                     hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════
# DASHBOARD 2 — SIGNAL ANALYSIS + 15-MIN
# ══════════════════════════════════════════════
st.markdown("---")
st.subheader("Signal analysis — 15-min comparison & trade monitor")

left, right = st.columns([1.2, 1])
with left:
    def pct(cur, ref): return round((cur-ref)/abs(ref)*100,1) if ref else 0.0
    h15 = {k: ha(st.session_state[f"h_{k}"]) for k in
           ["opt_ltp","opt_adx","idx_adx","opt_pdi","opt_ndi","oi"]}
    cmp = {
        "Metric":  ["Option LTP","Option ADX","Index ADX","+DI","-DI","OI"],
        "Now":     [f"₹{best_opt['ltp']:.2f}",f"{best_opt['adx']:.2f}",
                    f"{dmi['adx']:.2f}",f"{best_opt['pdi']:.2f}",
                    f"{best_opt['ndi']:.2f}",f"{best_opt['oi']:,}"],
        "15m avg": [f"₹{h15['opt_ltp']:.2f}",f"{h15['opt_adx']:.2f}",
                    f"{h15['idx_adx']:.2f}",f"{h15['opt_pdi']:.2f}",
                    f"{h15['opt_ndi']:.2f}",f"{h15['oi']:,.0f}"],
        "Change":  [f"{pct(best_opt['ltp'],h15['opt_ltp']):+.1f}%",
                    f"{pct(best_opt['adx'],h15['opt_adx']):+.1f}%",
                    f"{pct(dmi['adx'],h15['idx_adx']):+.1f}%",
                    f"{pct(best_opt['pdi'],h15['opt_pdi']):+.1f}%",
                    f"{pct(best_opt['ndi'],h15['opt_ndi']):+.1f}%",
                    f"{pct(best_opt['oi'],h15['oi']):+.1f}%"],
    }
    st.dataframe(pd.DataFrame(cmp), hide_index=True, use_container_width=True)

    if len(st.session_state.h_opt_ltp) > 1:
        chart_df = pd.DataFrame({
            "Option LTP": list(st.session_state.h_opt_ltp),
            "Option ADX": list(st.session_state.h_opt_adx),
        }, index=list(st.session_state.h_times))
        st.line_chart(chart_df, height=160)

with right:
    verdict_lbl = "BUY" if score>=75 else "WAIT" if score>=45 else "AVOID"
    if score >= 75:   st.success(f"**BUY** — Confidence {score}/100")
    elif score >= 45: st.warning(f"**WAIT** — Confidence {score}/100")
    else:             st.error(f"**AVOID** — Confidence {score}/100")
    for label, passed, pts in checks:
        st.markdown(f"{'✅' if passed else '❌'} {label} &nbsp; `+{pts if passed else 0}/{pts}`")

st.divider()

# 5-min monitor
st.subheader("5-min trade monitor")
if st.session_state.monitor_start:
    elapsed   = int(time.time() - st.session_state.monitor_start)
    remaining = MONITOR_SECS - elapsed % MONITOR_SECS
    mm, ss    = divmod(remaining, 60)
    st.markdown(f"Next 5-min update in: **{mm}:{ss:02d}**")

n1,n2,n3,n4 = st.columns(4)
n1.metric("Entry",       f"₹{st.session_state.entry_price:,.2f}" if st.session_state.trade_active else "—")
n2.metric("Current LTP", f"₹{best_opt['ltp']:.2f}")

if st.session_state.trade_active:
    pnl2      = round(best_opt["ltp"] - st.session_state.entry_price, 2)
    adx_fall  = best_opt["adx"] < st.session_state.entry_adx - 5
    dir_flip  = ((best_opt["side"]=="CE" and best_opt["ndi"]>best_opt["pdi"]) or
                 (best_opt["side"]=="PE" and best_opt["pdi"]>best_opt["ndi"]))
    if   pnl2 > 20:    rec, rmsg = "EXIT", "Target reached — book profit"
    elif pnl2 < -15:   rec, rmsg = "EXIT", "Stop-loss zone"
    elif adx_fall:     rec, rmsg = "WEAK", "ADX falling — trend weakening"
    elif dir_flip:     rec, rmsg = "FLIP", "DI crossover — direction reversed"
    else:              rec, rmsg = "HOLD", f"Conditions holding (score {score}/100)"
    n3.metric("P&L", f"₹{pnl2:+,.2f}")
    n4.metric("Recommendation", rec)
    add_log("signal_log", {"time":now_ts,"type":rec,"msg":rmsg})
    if rec == "EXIT":   st.error(f"**{rec}** — {rmsg}")
    elif rec == "HOLD": st.success(f"**HOLD** — {rmsg}")
    else:               st.warning(f"**{rec}** — {rmsg}")
else:
    n3.metric("P&L","—")
    n4.metric("Status","Ready" if score>=75 else "Wait" if score>=45 else "Avoid")

if st.session_state.signal_log:
    st.dataframe(pd.DataFrame(st.session_state.signal_log[:10]),
                 hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════
# DASHBOARD 3 — LIVE ORDER PLACEMENT
# ══════════════════════════════════════════════
st.markdown("---")
st.subheader("Live order placement")

if MOCK_MODE:
    st.warning(
        "Live order placement is **disabled in mock mode**. "
        "Add your Upstox access token to `.streamlit/secrets.toml` to enable real trading."
    )
else:
    st.error(
        "Real money trading. Every order below is sent live to Upstox. "
        "Double-check all fields before confirming."
    )

    with st.expander("Configure & place live order", expanded=True):
        st.markdown("**Step 1 — Select the option to trade**")

        # Load full chain for manual selection
        try:
            expiry_sel = get_nearest_expiry(ikey)
            chain_sel  = get_chain(ikey, expiry_sel)
        except Exception as e:
            st.error(f"Could not load chain: {e}"); chain_sel = []

        strikes = sorted({int(s["strike_price"]) for s in chain_sel})
        sides   = ["CE", "PE"]

        col_s, col_side, col_exp = st.columns(3)
        lo_strike_val = col_s.selectbox(
            "Strike", strikes,
            index=strikes.index(int(best_opt["strike"])) if int(best_opt["strike"]) in strikes else 0
        )
        lo_side_val   = col_side.selectbox(
            "Side", sides,
            index=sides.index(best_opt["side"])
        )
        col_exp.text_input("Expiry", value=expiry_sel, disabled=True)

        # Resolve instrument key from selection
        lo_ikey_val = None
        for row in chain_sel:
            if int(row["strike_price"]) == lo_strike_val:
                opt_key = "call_options" if lo_side_val == "CE" else "put_options"
                lo_ikey_val = row.get(opt_key, {}).get("instrument_key")
                break

        # Find LTP for selected option
        lo_ltp = 0.0
        if lo_ikey_val:
            for row in chain_sel:
                if int(row["strike_price"]) == lo_strike_val:
                    opt_key = "call_options" if lo_side_val == "CE" else "put_options"
                    lo_ltp  = row.get(opt_key,{}).get("market_data",{}).get("ltp",0) or 0
                    break
        if lo_ltp:
            st.caption(f"Selected option LTP: ₹{lo_ltp:.2f} | Key: `{lo_ikey_val}`")

        st.markdown("**Step 2 — Set quantity, target & trailing SL**")
        col_q, col_t, col_tsl = st.columns(3)
        lo_qty_val    = col_q.number_input("Lots", min_value=1, max_value=50, value=1, step=1)
        lo_target_val = col_t.number_input("Target (pts)", min_value=10, max_value=500,
                                            value=GEAR_PTS[opt_g], step=10)
        lo_tsl_val    = col_tsl.number_input("Trailing SL (%)", min_value=5.0, max_value=80.0,
                                              value=30.0, step=5.0)

        qty_units     = lo_qty_val * lot_size
        target_px_val = round(lo_ltp + lo_target_val, 2) if lo_ltp else 0.0
        sl_px_val     = round(lo_ltp * (1 - lo_tsl_val / 100), 2) if lo_ltp else 0.0

        st.markdown("**Step 3 — Review & confirm**")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Entry LTP",    f"₹{lo_ltp:.2f}")
        r2.metric("Target price", f"₹{target_px_val:.2f}")
        r3.metric("Init SL",      f"₹{sl_px_val:.2f}")
        r4.metric("Total qty",    f"{qty_units} units ({lo_qty_val} lot{'s' if lo_qty_val>1 else ''})")

        st.markdown("**Step 4 — Place order**")
        col_place, col_cancel = st.columns(2)

        if not st.session_state.trade_active:
            confirm = col_place.checkbox("I confirm this is a live order with real money")
            if confirm:
                if col_place.button("Place BUY order (LIVE)", use_container_width=True, type="primary"):
                    if not lo_ikey_val:
                        st.error("Could not resolve instrument key for the selected strike.")
                    else:
                        try:
                            order_id = place_market_order(lo_ikey_val, qty_units, "BUY")
                            # Also place initial SL order
                            sl_trigger = sl_px_val
                            sl_limit   = round(sl_trigger * 0.995, 2)
                            sl_oid     = place_sl_order(lo_ikey_val, qty_units, sl_trigger, sl_limit, "SELL")

                            st.session_state.trade_active   = True
                            st.session_state.trade_mode     = "live"
                            st.session_state.entry_price    = lo_ltp
                            st.session_state.entry_side     = lo_side_val
                            st.session_state.target_pts     = lo_target_val
                            st.session_state.exit_price     = target_px_val
                            st.session_state.sl_price       = sl_px_val
                            st.session_state.trailing_sl_pct= lo_tsl_val
                            st.session_state.highest_pnl    = 0.0
                            st.session_state.live_order_id  = order_id
                            st.session_state.monitor_start  = time.time()
                            st.session_state.lo_ikey        = lo_ikey_val
                            st.session_state.lo_qty         = qty_units

                            add_log("act_log", {
                                "time": now_ts,
                                "event": f"[LIVE] BUY {lo_side_val} {lo_strike_val} "
                                         f"qty={qty_units} order_id={order_id} | "
                                         f"SL order={sl_oid}"
                            })
                            st.success(f"Order placed! ID: {order_id}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Order failed: {e}")
        else:
            if st.session_state.trade_mode == "live":
                pnl_live  = round(best_opt["ltp"] - st.session_state.entry_price, 2)
                tsl_live  = update_trailing_sl(best_opt["ltp"], st.session_state.entry_price,
                                               st.session_state.trailing_sl_pct)
                st.session_state.sl_price = tsl_live
                l1,l2,l3,l4 = st.columns(4)
                l1.metric("Entry",          f"₹{st.session_state.entry_price:,.2f}")
                l2.metric("Target",         f"₹{st.session_state.exit_price:,.2f}")
                l3.metric("Trailing SL",    f"₹{tsl_live:,.2f}")
                l4.metric("Live P&L",       f"₹{pnl_live:+,.2f}")

                # Auto-exit checks for live trade
                if best_opt["ltp"] >= st.session_state.exit_price:
                    browser_alert("TARGET HIT", f"[LIVE] P&L: +{pnl_live}")
                    st.success(f"[LIVE] Target hit! P&L: ₹{pnl_live:+,.2f}")
                    try:
                        exit_id = place_market_order(st.session_state.lo_ikey,
                                                     st.session_state.lo_qty, "SELL")
                        add_log("act_log", {"time":now_ts,
                            "event":f"[LIVE] Target exit order={exit_id} P&L ₹{pnl_live:+.2f}"})
                    except Exception as e:
                        st.error(f"Exit order failed: {e}")
                    st.session_state.trade_active = False; time.sleep(2); st.rerun()

                if best_opt["ltp"] <= tsl_live and st.session_state.highest_pnl > 0:
                    browser_alert("TSL HIT", f"[LIVE] Trailing SL triggered. P&L: {pnl_live:+}")
                    st.error(f"[LIVE] Trailing SL hit! P&L: ₹{pnl_live:+,.2f}")
                    try:
                        exit_id = place_market_order(st.session_state.lo_ikey,
                                                     st.session_state.lo_qty, "SELL")
                        add_log("act_log", {"time":now_ts,
                            "event":f"[LIVE] TSL exit order={exit_id} P&L ₹{pnl_live:+.2f}"})
                    except Exception as e:
                        st.error(f"TSL exit order failed: {e}")
                    st.session_state.trade_active = False; time.sleep(2); st.rerun()

                if col_cancel.button("Exit position (LIVE)", use_container_width=True):
                    try:
                        exit_id = place_market_order(st.session_state.lo_ikey,
                                                     st.session_state.lo_qty, "SELL")
                        add_log("act_log", {"time":now_ts,
                            "event":f"[LIVE] Manual exit order={exit_id} P&L ₹{pnl_live:+.2f}"})
                        browser_alert("EXIT", f"[LIVE] Manual exit | P&L ₹{pnl_live:+.2f}")
                    except Exception as e:
                        st.error(f"Exit failed: {e}")
                    st.session_state.trade_active = False; st.rerun()

# ── FOOTER ──
st.caption(
    f"Index: `{ikey}` | Lot size: {lot_size} | ADX period: {ADX_PERIOD} | "
    f"History: {len(st.session_state.h_opt_ltp)}/{HISTORY_LEN} bars | "
    f"Refresh: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%H:%M:%S IST')}"
)
time.sleep(5)
st.rerun()
