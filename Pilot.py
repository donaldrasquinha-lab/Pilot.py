import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import streamlit.components.v1 as components
from datetime import date, timedelta
from collections import deque

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
UPSTOX_ACCESS_TOKEN = "YOUR_ACCESS_TOKEN_HERE"
INSTRUMENT_KEY      = "NSE_INDEX|Nifty 50"
ADX_PERIOD          = 14
CANDLE_INTERVAL     = "1minute"
OI_PERCENTILE       = 70
TOP_N_STRIKES       = 10
HISTORY_LEN         = 15          # bars of 1-min history to keep
MONITOR_INTERVAL    = 300         # 5-minute monitor window in seconds

HEADERS = {
    "Accept":        "application/json",
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
}

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
def _init_state():
    defaults = dict(
        trade_active   = False,
        entry_price    = 0.0,
        entry_adx      = 0.0,
        entry_pdi      = 0.0,
        entry_ndi      = 0.0,
        target_pts     = 0,
        exit_price     = 0.0,
        last_signal    = None,
        monitor_start  = None,
        signal_log     = [],
        # rolling 15-min history
        h_times        = deque(maxlen=HISTORY_LEN),
        h_opt_ltp      = deque(maxlen=HISTORY_LEN),
        h_opt_adx      = deque(maxlen=HISTORY_LEN),
        h_idx_adx      = deque(maxlen=HISTORY_LEN),
        h_opt_pdi      = deque(maxlen=HISTORY_LEN),
        h_opt_ndi      = deque(maxlen=HISTORY_LEN),
        h_oi           = deque(maxlen=HISTORY_LEN),
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ─────────────────────────────────────────────
# BROWSER ALERTS
# ─────────────────────────────────────────────
def send_browser_alert(title: str, msg: str):
    components.html(f"""<script>
    if(Notification.permission==="granted"){{new Notification("{title}",{{body:"{msg}"}})}}
    else if(Notification.permission!=="denied"){{Notification.requestPermission().then(p=>{{
    if(p==="granted")new Notification("{title}",{{body:"{msg}"}});}});}}
    </script>""", height=0)

# ─────────────────────────────────────────────
# ADX / DMI
# ─────────────────────────────────────────────
def compute_dmi(df: pd.DataFrame, period: int = 14):
    if len(df) < period + 1:
        return None
    high  = df["high"].values.astype(float)
    low   = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    tr   = np.maximum(high[1:]-low[1:], np.maximum(
           np.abs(high[1:]-close[:-1]), np.abs(low[1:]-close[:-1])))
    up   = high[1:]-high[:-1];  down = low[:-1]-low[1:]
    dm_p = np.where((up>down)&(up>0),   up,   0.0)
    dm_n = np.where((down>up)&(down>0), down, 0.0)
    def wilder(arr, p):
        out = np.zeros(len(arr)); out[p-1] = arr[:p].sum()
        for i in range(p, len(arr)): out[i] = out[i-1]-out[i-1]/p+arr[i]
        return out
    atr  = wilder(tr, period); sdmp = wilder(dm_p, period); sdmn = wilder(dm_n, period)
    safe = np.where(atr==0, 1e-10, atr)
    pdi  = 100*sdmp/safe;  ndi = 100*sdmn/safe
    dx   = 100*np.abs(pdi-ndi)/np.where((pdi+ndi)==0,1e-10,pdi+ndi)
    adx  = wilder(dx, period)
    return {"adx":round(float(adx[-1]),2),"pdi":round(float(pdi[-1]),2),"ndi":round(float(ndi[-1]),2)}

# ─────────────────────────────────────────────
# UPSTOX HELPERS
# ─────────────────────────────────────────────
def get_live_price() -> float:
    r = requests.get("https://api.upstox.com/v2/market-quote/ohlc",
                     headers=HEADERS,
                     params={"instrument_key":INSTRUMENT_KEY,"interval":"1d"},timeout=5)
    r.raise_for_status(); feeds = r.json()["data"]
    return float(feeds[list(feeds.keys())[0]]["last_price"])

def get_intraday_candles(instrument_key: str) -> pd.DataFrame:
    enc = instrument_key.replace("|","%7C").replace(" ","%20")
    r   = requests.get(f"https://api.upstox.com/v2/historical-candle/intraday/{enc}/{CANDLE_INTERVAL}",
                       headers=HEADERS, timeout=5)
    r.raise_for_status(); candles = r.json()["data"]["candles"]
    df = pd.DataFrame(candles,columns=["ts","open","high","low","close","vol","oi"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").reset_index(drop=True)
    for c in ["open","high","low","close"]: df[c] = df[c].astype(float)
    return df

def get_nearest_expiry() -> str:
    r = requests.get("https://api.upstox.com/v2/option/contract",
                     headers=HEADERS,params={"instrument_key":INSTRUMENT_KEY},timeout=5)
    r.raise_for_status()
    expiries = sorted({i["expiry"] for i in r.json()["data"] if i.get("expiry")})
    today    = date.today().isoformat()
    future   = [e for e in expiries if e>=today]
    return future[0] if future else expiries[-1]

def get_option_chain(expiry: str) -> list:
    r = requests.get("https://api.upstox.com/v2/option/chain",headers=HEADERS,
                     params={"instrument_key":INSTRUMENT_KEY,"expiry_date":expiry},timeout=8)
    r.raise_for_status(); return r.json()["data"]

def find_best_option(spot_price: float):
    expiry = get_nearest_expiry(); chain = get_option_chain(expiry)
    rows = []
    for strike in chain:
        sp = strike["strike_price"]
        for side, key in [("CE","call_options"),("PE","put_options")]:
            opt = strike.get(key,{}); md = opt.get("market_data",{})
            oi  = md.get("oi",0) or 0;  ltp = md.get("ltp",0) or 0
            prev_oi = md.get("prev_oi",0) or 0
            iv  = opt.get("option_greeks",{}).get("iv",0) or 0
            ikey = opt.get("instrument_key","")
            if ltp>0 and oi>0 and ikey:
                rows.append({"strike":sp,"side":side,"ltp":ltp,"oi":oi,
                             "prev_oi":prev_oi,"iv":iv,"ikey":ikey,
                             "dist":abs(sp-spot_price),"expiry":expiry})
    if not rows: return None
    df_chain = pd.DataFrame(rows)
    oi_thresh  = np.percentile(df_chain["oi"],OI_PERCENTILE)
    df_top     = df_chain[df_chain["oi"]>=oi_thresh].nsmallest(TOP_N_STRIKES,"dist")
    results = []
    prog = st.progress(0,text="Scanning option chain for highest ADX…")
    for i,(_, row) in enumerate(df_top.iterrows()):
        try:
            dmi = compute_dmi(get_intraday_candles(row["ikey"]),ADX_PERIOD)
            if dmi: results.append({**row.to_dict(),**dmi})
        except: pass
        prog.progress((i+1)/len(df_top),text=f"Scanning {row['side']} {int(row['strike'])}…")
    prog.empty()
    return max(results,key=lambda x:x["adx"]) if results else None

# ─────────────────────────────────────────────
# GEAR / SCORE
# ─────────────────────────────────────────────
GEAR_PTS = {4:200,3:150,2:100,1:50}

def gear_from_adx(adx):
    if adx>=40: return 4
    if adx>=35: return 3
    if adx>=30: return 2
    return 1

def ha(deq): return float(np.mean(list(deq))) if deq else 0.0

def buy_confidence(s, best_opt):
    """Score 0-100 comparing current state to 15-min history."""
    h_opt_ltp = ha(st.session_state.h_opt_ltp)
    h_opt_adx = ha(st.session_state.h_opt_adx)
    h_idx_adx = ha(st.session_state.h_idx_adx)
    h_opt_pdi = ha(st.session_state.h_opt_pdi)
    h_opt_ndi = ha(st.session_state.h_opt_ndi)
    h_oi      = ha(st.session_state.h_oi)

    checks = [
        ("Option ADX vs 15m avg",  best_opt["adx"]>h_opt_adx and best_opt["adx"]>30, 25),
        ("Index ADX strength",     s["adx"]>=30, 20),
        ("+DI/-DI direction",
            (best_opt["side"]=="CE" and best_opt["pdi"]>best_opt["ndi"]) or
            (best_opt["side"]=="PE" and best_opt["ndi"]>best_opt["pdi"]), 25),
        ("OI vs 15m avg",          best_opt["oi"] >= h_oi*1.0 if h_oi else True, 15),
        ("Option price rising",    best_opt["ltp"] > h_opt_ltp if h_opt_ltp else True, 15),
    ]
    score = sum(pts for _, passed, pts in checks if passed)
    return score, checks

def verdict_label(score):
    if score>=75: return "BUY", "green", "success"
    if score>=45: return "WAIT", "orange", "warning"
    return "AVOID", "red", "error"

# ─────────────────────────────────────────────
# MOCK
# ─────────────────────────────────────────────
def mock_data():
    px  = round(24500+np.random.uniform(-50,50),2)
    adx = round(30+np.random.uniform(-3,15),2)
    pdi = round(20+np.random.uniform(-4,10),2)
    ndi = round(10+np.random.uniform(-4,10),2)
    bullish = pdi>ndi; side="CE" if bullish else "PE"
    strike = round(px/50)*50+(50 if bullish else -50)
    oi     = int(np.random.uniform(1e5,5e5))
    best   = {"strike":strike,"side":side,
              "ltp":round(np.random.uniform(80,300),2),
              "oi":oi,"prev_oi":int(oi*np.random.uniform(.85,1.15)),
              "iv":round(np.random.uniform(12,30),2),
              "adx":round(adx+np.random.uniform(-5,10),2),
              "pdi":pdi,"ndi":ndi,
              "expiry":(date.today()+timedelta(days=3)).isoformat(),
              "ikey":f"NSE_FO|MOCK{strike}{side}"}
    return px, {"adx":adx,"pdi":pdi,"ndi":ndi}, best

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Nifty Pilot", layout="wide")

MOCK = UPSTOX_ACCESS_TOKEN=="YOUR_ACCESS_TOKEN_HERE"
c_title, c_mode = st.columns([4,1])
c_title.title("Nifty Pilot")
c_mode.warning("Mock") if MOCK else c_mode.success("Live")

if MOCK:
    st.info("Mock mode active — paste Upstox access token to go live.")

# ── FETCH ──
try:
    if MOCK:
        live_px, dmi, best_opt = mock_data()
    else:
        live_px  = get_live_price()
        df_idx   = get_intraday_candles(INSTRUMENT_KEY)
        dmi      = compute_dmi(df_idx, ADX_PERIOD)
        if not dmi: st.warning("Waiting for candles…"); st.stop()
        best_opt = find_best_option(live_px)
except Exception as e:
    st.error(f"API error: {e}"); st.stop()

if not best_opt:
    st.warning("No option signal — chain may be empty or market just opened."); st.stop()

# ── UPDATE HISTORY ──
ts_now = pd.Timestamp.now(tz="Asia/Kolkata").strftime("%H:%M:%S")
st.session_state.h_times.append(ts_now)
st.session_state.h_opt_ltp.append(best_opt["ltp"])
st.session_state.h_opt_adx.append(best_opt["adx"])
st.session_state.h_idx_adx.append(dmi["adx"])
st.session_state.h_opt_pdi.append(best_opt["pdi"])
st.session_state.h_opt_ndi.append(best_opt["ndi"])
st.session_state.h_oi.append(best_opt["oi"])

score, checks = buy_confidence(dmi, best_opt)
verdict, v_color, v_type = verdict_label(score)
gear       = gear_from_adx(dmi["adx"])
opt_gear   = gear_from_adx(best_opt["adx"])
target_pts = GEAR_PTS[gear]

# ══════════════════════════════════════════════
# DASHBOARD A — COMPARISON + SCORE
# ══════════════════════════════════════════════
st.subheader("Signal analysis — current vs 15-min history")

left, right = st.columns([1.2, 1])

with left:
    # Current option
    st.markdown(f"**Best option now:** {best_opt['side']} {int(best_opt['strike'])} | exp {best_opt['expiry']}")
    a1,a2,a3,a4 = st.columns(4)
    a1.metric("Option LTP",  f"₹{best_opt['ltp']:.2f}")
    a2.metric("Option ADX",  f"{best_opt['adx']:.2f}", f"Gear {opt_gear}")
    a3.metric("OI",          f"{best_opt['oi']:,}")
    a4.metric("IV",          f"{best_opt['iv']:.1f}%")

    st.markdown("**Index (Nifty 50)**")
    b1,b2,b3,b4 = st.columns(4)
    b1.metric("LTP",      f"₹{live_px:,.2f}")
    b2.metric("Index ADX",f"{dmi['adx']:.2f}", f"Gear {gear}")
    b3.metric("+DI",      f"{dmi['pdi']:.2f}")
    b4.metric("-DI",      f"{dmi['ndi']:.2f}")

    # 15-min comparison table
    st.markdown("**15-min comparison**")
    h15 = {
        "opt_ltp": ha(st.session_state.h_opt_ltp),
        "opt_adx": ha(st.session_state.h_opt_adx),
        "idx_adx": ha(st.session_state.h_idx_adx),
        "opt_pdi": ha(st.session_state.h_opt_pdi),
        "opt_ndi": ha(st.session_state.h_opt_ndi),
        "oi":      ha(st.session_state.h_oi),
    }
    def pct_chg(cur, ref):
        if not ref: return 0.0
        return round((cur-ref)/abs(ref)*100, 1)

    cmp_data = {
        "Metric":    ["Option LTP","Option ADX","Index ADX","+DI","-DI","OI"],
        "Now":       [f"₹{best_opt['ltp']:.2f}", f"{best_opt['adx']:.2f}",
                      f"{dmi['adx']:.2f}", f"{best_opt['pdi']:.2f}",
                      f"{best_opt['ndi']:.2f}", f"{best_opt['oi']:,}"],
        "15m avg":   [f"₹{h15['opt_ltp']:.2f}", f"{h15['opt_adx']:.2f}",
                      f"{h15['idx_adx']:.2f}", f"{h15['opt_pdi']:.2f}",
                      f"{h15['opt_ndi']:.2f}", f"{h15['oi']:,.0f}"],
        "Change %":  [f"{pct_chg(best_opt['ltp'],h15['opt_ltp']):+.1f}%",
                      f"{pct_chg(best_opt['adx'],h15['opt_adx']):+.1f}%",
                      f"{pct_chg(dmi['adx'],h15['idx_adx']):+.1f}%",
                      f"{pct_chg(best_opt['pdi'],h15['opt_pdi']):+.1f}%",
                      f"{pct_chg(best_opt['ndi'],h15['opt_ndi']):+.1f}%",
                      f"{pct_chg(best_opt['oi'],h15['oi']):+.1f}%"],
    }
    st.dataframe(pd.DataFrame(cmp_data), hide_index=True, use_container_width=True)

    # Price action chart
    if len(st.session_state.h_opt_ltp) > 1:
        st.markdown("**Option LTP — last 15 min**")
        chart_df = pd.DataFrame({
            "Option LTP": list(st.session_state.h_opt_ltp),
            "ADX (option)": list(st.session_state.h_opt_adx),
        }, index=list(st.session_state.h_times))
        st.line_chart(chart_df, height=160)

with right:
    # Buy confidence verdict
    verd_map = {"BUY": "✅ BUY", "WAIT": "⏳ WAIT", "AVOID": "🚫 AVOID"}
    if v_type == "success":
        st.success(f"**{verd_map[verdict]}** — Confidence: {score}/100")
    elif v_type == "warning":
        st.warning(f"**{verd_map[verdict]}** — Confidence: {score}/100")
    else:
        st.error(f"**{verd_map[verdict]}** — Confidence: {score}/100")

    st.markdown("**Signal breakdown**")
    for label, passed, pts in checks:
        icon = "✅" if passed else "❌"
        pts_str = f"+{pts}" if passed else f"0/{pts}"
        st.markdown(f"{icon} {label} &nbsp; `{pts_str} pts`")

    st.markdown("---")
    oi_chg = best_opt["oi"] - best_opt.get("prev_oi", best_opt["oi"])
    oi_note = (
        ("Rising CE OI → writers building; resistance possible" if oi_chg>0
         else "Falling CE OI → short covering; bullish for buyers")
        if best_opt["side"]=="CE" else
        ("Rising PE OI → writers building; support possible" if oi_chg>0
         else "Falling PE OI → short covering; bearish for buyers")
    )
    st.caption(f"OI change: {oi_chg:+,} | {oi_note}")

st.divider()

# ══════════════════════════════════════════════
# DASHBOARD B — 5-MIN TRADE MONITOR
# ══════════════════════════════════════════════
st.subheader("5-min trade monitor")

# Countdown timer
monitor_secs_elapsed = 0
if st.session_state.monitor_start:
    elapsed = int(time.time() - st.session_state.monitor_start)
    monitor_secs_elapsed = elapsed % MONITOR_INTERVAL
    remaining = MONITOR_INTERVAL - monitor_secs_elapsed
    m, s = divmod(remaining, 60)
    st.markdown(f"Next 5-min update in: **{m}:{s:02d}**")
    if monitor_secs_elapsed == 0 and elapsed > 0:
        st.session_state.signal_log.insert(0, {"time": ts_now, "type": "UPDATE",
                                                "msg": "5-min window elapsed — reassessing"})

# Monitor metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Entry price",   f"₹{st.session_state.entry_price:,.2f}" if st.session_state.trade_active else "—")
m2.metric("Current LTP",   f"₹{best_opt['ltp']:.2f}")

if st.session_state.trade_active:
    pnl = round(best_opt["ltp"] - st.session_state.entry_price, 2)
    m3.metric("Live P&L", f"₹{pnl:+.2f}", delta_color="normal")

    # Exit / hold logic
    adx_falling  = best_opt["adx"] < st.session_state.entry_adx - 5
    dir_flipped  = (best_opt["side"]=="CE" and best_opt["ndi"]>best_opt["pdi"]) or \
                   (best_opt["side"]=="PE" and best_opt["pdi"]>best_opt["ndi"])
    big_loss     = pnl < -15
    good_profit  = pnl > 20

    if good_profit:
        reason = "Target reached — book profit now"
        rec    = "EXIT"
        st.success(f"**EXIT recommended** — {reason}")
        send_browser_alert("TARGET HIT", reason)
    elif big_loss:
        reason = "Stop-loss zone — exit to limit damage"
        rec    = "EXIT"
        st.error(f"**EXIT now** — {reason}")
        send_browser_alert("STOP LOSS", reason)
    elif adx_falling:
        reason = "ADX falling — trend weakening"
        rec    = "EXIT"
        st.warning(f"**Consider EXIT** — {reason}")
    elif dir_flipped:
        reason = "DI crossover — direction reversed"
        rec    = "EXIT"
        st.warning(f"**Consider EXIT** — {reason}")
    else:
        reason = f"Conditions holding — score {score}/100"
        rec    = "HOLD"
        st.success(f"**HOLD** — {reason}")

    m4.metric("Recommendation", rec)
    st.session_state.signal_log.insert(0,{"time":ts_now,"type":rec,"msg":reason})
else:
    entry_note = (f"Ready to enter — {verdict}" if score>=75
                  else "Wait for stronger signals" if score>=45
                  else "Avoid — weak setup")
    m3.metric("Status", "—")
    m4.metric("Recommendation", entry_note)

# Signal log
if st.session_state.signal_log:
    st.markdown("**Signal log**")
    log_df = pd.DataFrame(st.session_state.signal_log[:10])
    st.dataframe(log_df, hide_index=True, use_container_width=True)

# Trade buttons
col_enter, col_exit = st.columns(2)
with col_enter:
    if not st.session_state.trade_active:
        if st.button("Enter trade", use_container_width=True, type="primary"):
            st.session_state.trade_active  = True
            st.session_state.entry_price   = best_opt["ltp"]
            st.session_state.entry_adx     = best_opt["adx"]
            st.session_state.entry_pdi     = best_opt["pdi"]
            st.session_state.entry_ndi     = best_opt["ndi"]
            st.session_state.target_pts    = GEAR_PTS[opt_gear]
            st.session_state.exit_price    = round(best_opt["ltp"]+GEAR_PTS[opt_gear],2)
            st.session_state.monitor_start = time.time()
            st.session_state.signal_log.insert(0,{"time":ts_now,"type":"ENTER",
                "msg":f"Entered {best_opt['side']} {int(best_opt['strike'])} at ₹{best_opt['ltp']:.2f}"})
            st.rerun()

with col_exit:
    if st.session_state.trade_active:
        if st.button("Exit now", use_container_width=True):
            pnl = round(best_opt["ltp"]-st.session_state.entry_price,2)
            st.session_state.signal_log.insert(0,{"time":ts_now,"type":"EXIT",
                "msg":f"Manual exit at ₹{best_opt['ltp']:.2f} | P&L ₹{pnl:+.2f}"})
            send_browser_alert("EXIT","Manual exit executed")
            st.session_state.trade_active  = False
            st.session_state.monitor_start = None
            st.rerun()

# ── FOOTER ──
st.caption(
    f"Instrument: `{INSTRUMENT_KEY}` | ADX period: {ADX_PERIOD} | "
    f"History: {len(st.session_state.h_opt_ltp)}/{HISTORY_LEN} bars | "
    f"Refresh: {pd.Timestamp.now(tz='Asia/Kolkata').strftime('%H:%M:%S IST')}"
)

time.sleep(5)
st.rerun()
