import streamlit as st
import requests
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import datetime

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE OI Live Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1e2329; border-radius: 8px; padding: 10px; }
    .bull { color: #00c853; font-weight: bold; font-size: 1.3rem; }
    .bear { color: #ff1744; font-weight: bold; font-size: 1.3rem; }
    .neutral { color: #ffd600; font-weight: bold; font-size: 1.3rem; }
    .title-bar { background: linear-gradient(90deg, #1a237e, #0d47a1);
                 padding: 1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── NSE headers (mimic browser) ───────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.nseindia.com/",
}

# ── Supported indices ─────────────────────────────────────────────────────────
INDEX_CONFIG = {
    "NIFTY":       {"url_key": "NIFTY",      "range": 100, "step": 50,  "lot": 65},
    "BANKNIFTY":   {"url_key": "BANKNIFTY",  "range": 300, "step": 100, "lot": 30},
    "FINNIFTY":    {"url_key": "FINNIFTY",   "range": 100, "step": 50,  "lot": 60},
    "MIDCPNIFTY":  {"url_key": "MIDCPNIFTY", "range": 100, "step": 25,  "lot": 120},
}

# ── NSE data fetcher ──────────────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def fetch_option_chain(symbol: str) -> dict | None:
    """Fetch live option chain from NSE. Returns None on failure."""
    base_url  = f"https://www.nseindia.com/get-quotes/derivatives?symbol={symbol}"
    chain_url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    try:
        with requests.Session() as s:
            s.get(base_url, headers=HEADERS, timeout=10)
            time.sleep(0.5)
            resp = s.get(chain_url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if len(data) >= 2:
                return data
    except Exception as e:
        st.warning(f"⚠️ NSE fetch error: {e}")
    return None

def parse_chain(data: dict, symbol: str, expiry_idx: int = 0, strike_range: int = 300) -> pd.DataFrame:
    """Convert raw NSE JSON → clean DataFrame filtered by strike range."""
    records     = data["records"]
    expiry_list = records["expiryDates"]
    expiry      = expiry_list[expiry_idx] if expiry_idx < len(expiry_list) else expiry_list[0]

    underlying = records.get("underlyingValue", 0)

    rows = []
    for item in records["data"]:
        if item.get("expiryDate") != expiry:
            continue
        sp = item.get("strikePrice", 0)
        if abs(sp - underlying) > strike_range:
            continue
        ce = item.get("CE", {})
        pe = item.get("PE", {})
        rows.append({
            "Strike":       sp,
            "CE_OI":        ce.get("openInterest", 0),
            "CE_ChgOI":     ce.get("changeinOpenInterest", 0),
            "CE_LTP":       ce.get("lastPrice", 0),
            "CE_IV":        ce.get("impliedVolatility", 0),
            "CE_Vol":       ce.get("totalTradedVolume", 0),
            "PE_OI":        pe.get("openInterest", 0),
            "PE_ChgOI":     pe.get("changeinOpenInterest", 0),
            "PE_LTP":       pe.get("lastPrice", 0),
            "PE_IV":        pe.get("impliedVolatility", 0),
            "PE_Vol":       pe.get("totalTradedVolume", 0),
        })

    df = pd.DataFrame(rows).sort_values("Strike").reset_index(drop=True)
    df["Underlying"] = underlying
    df["Expiry"]     = expiry
    return df

def get_sentiment(df: pd.DataFrame):
    """PCR-based sentiment + max pain."""
    total_ce_oi = df["CE_OI"].sum()
    total_pe_oi = df["PE_OI"].sum()
    pcr = total_pe_oi / total_ce_oi if total_ce_oi else 1.0

    if pcr > 1.2:
        sentiment, label = "BULLISH 🟢", "bull"
    elif pcr < 0.8:
        sentiment, label = "BEARISH 🔴", "bear"
    else:
        sentiment, label = "NEUTRAL 🟡", "neutral"

    # Max Pain: strike where total loss for option buyers is max
    pain = {}
    for sp in df["Strike"]:
        loss = (df.loc[df["Strike"] > sp, "CE_OI"] * (df.loc[df["Strike"] > sp, "Strike"] - sp)).sum() + \
               (df.loc[df["Strike"] < sp, "PE_OI"] * (sp - df.loc[df["Strike"] < sp, "Strike"])).sum()
        pain[sp] = loss
    max_pain = min(pain, key=pain.get) if pain else 0

    return pcr, sentiment, label, max_pain, total_ce_oi, total_pe_oi

def find_support_resistance(df: pd.DataFrame):
    """Highest PE OI = Support, Highest CE OI = Resistance."""
    support    = df.loc[df["PE_OI"].idxmax(), "Strike"] if not df.empty else 0
    resistance = df.loc[df["CE_OI"].idxmax(), "Strike"] if not df.empty else 0
    return support, resistance

# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    # Title
    st.markdown("""
    <div class="title-bar">
        <h2 style="color:white;margin:0;">📊 NSE F&O — Live OI Analysis Dashboard</h2>
        <p style="color:#90caf9;margin:0;">NIFTY · BANKNIFTY · FINNIFTY · MIDCPNIFTY</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    st.sidebar.header("⚙️ Settings")
    symbol      = st.sidebar.selectbox("Index", list(INDEX_CONFIG.keys()), index=0)
    cfg         = INDEX_CONFIG[symbol]
    strike_range = st.sidebar.slider("Strike Range (±)", 50, 500,
                                     cfg["range"], step=cfg["step"])
    expiry_idx  = st.sidebar.number_input("Expiry (0 = nearest)", 0, 5, 0)
    auto_refresh = st.sidebar.checkbox("⏱ Auto-refresh (60s)", value=False)
    show_raw    = st.sidebar.checkbox("Show Raw Data Table", value=False)

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Lot Size:** {cfg['lot']}")
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=5, minutes=30)))
    st.sidebar.markdown(f"**IST:** {now.strftime('%H:%M:%S')}")
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if market_open <= now <= market_close and now.weekday() < 5:
        st.sidebar.success("🟢 Market is OPEN")
    else:
        st.sidebar.error("🔴 Market is CLOSED")

    # ── Fetch ─────────────────────────────────────────────────────────────────
    col_btn, col_status = st.columns([1, 4])
    with col_btn:
        refresh = st.button("🔄 Refresh Now")

    if auto_refresh:
        time.sleep(0)  # streamlit reruns automatically via st.rerun below

    with st.spinner(f"Fetching {symbol} option chain from NSE..."):
        data = fetch_option_chain(symbol)

    if data is None:
        st.error("❌ Could not fetch data from NSE. Market may be closed or NSE blocked the request. Try again in a minute.")
        st.info("💡 NSE data is only available 9:15 AM – 3:30 PM IST on trading days.")
        return

    df = parse_chain(data, symbol, int(expiry_idx), strike_range)
    if df.empty:
        st.warning("No data returned for this selection. Try a wider strike range.")
        return

    underlying = df["Underlying"].iloc[0]
    expiry_str = df["Expiry"].iloc[0]
    pcr, sentiment, s_label, max_pain, total_ce, total_pe = get_sentiment(df)
    support, resistance = find_support_resistance(df)

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown(f"**{symbol}** · Expiry: `{expiry_str}` · Spot: `{underlying:,.2f}`")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Spot Price",    f"₹{underlying:,.2f}")
    k2.metric("PCR",           f"{pcr:.2f}")
    k3.metric("Max Pain",      f"₹{max_pain:,}")
    k4.metric("Support (PE)",  f"₹{support:,}")
    k5.metric("Resistance (CE)", f"₹{resistance:,}")
    k6.metric("Sentiment",     sentiment)

    # ── OI Bar Chart ──────────────────────────────────────────────────────────
    st.markdown("### 📊 Call vs Put — Open Interest by Strike")
    fig_oi = go.Figure()
    fig_oi.add_trace(go.Bar(
        x=df["Strike"], y=df["CE_OI"],
        name="CE OI", marker_color="#ef5350",
        opacity=0.85
    ))
    fig_oi.add_trace(go.Bar(
        x=df["Strike"], y=df["PE_OI"],
        name="PE OI", marker_color="#26a69a",
        opacity=0.85
    ))
    fig_oi.add_vline(x=underlying,   line_color="#fff176", line_width=2,
                     annotation_text=f"Spot {underlying:.0f}", annotation_position="top")
    fig_oi.add_vline(x=max_pain,     line_color="#ff7043", line_dash="dot", line_width=1,
                     annotation_text=f"Max Pain {max_pain}", annotation_position="bottom right")
    fig_oi.add_vline(x=support,      line_color="#26a69a", line_dash="dash", line_width=1,
                     annotation_text=f"Support {support}", annotation_position="top left")
    fig_oi.add_vline(x=resistance,   line_color="#ef5350", line_dash="dash", line_width=1,
                     annotation_text=f"Resistance {resistance}", annotation_position="top right")
    fig_oi.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=420, margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_oi, use_container_width=True)

    # ── Change in OI Chart ────────────────────────────────────────────────────
    st.markdown("### 📈 Change in OI (Current vs Previous)")
    fig_chg = go.Figure()
    fig_chg.add_trace(go.Bar(
        x=df["Strike"], y=df["CE_ChgOI"],
        name="CE Chg OI", marker_color="#e53935", opacity=0.8
    ))
    fig_chg.add_trace(go.Bar(
        x=df["Strike"], y=df["PE_ChgOI"],
        name="PE Chg OI", marker_color="#00897b", opacity=0.8
    ))
    fig_chg.add_vline(x=underlying, line_color="#fff176", line_width=2)
    fig_chg.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=360, margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_chg, use_container_width=True)

    # ── IV Skew Chart ─────────────────────────────────────────────────────────
    st.markdown("### 🌡️ Implied Volatility Skew")
    fig_iv = go.Figure()
    fig_iv.add_trace(go.Scatter(
        x=df["Strike"], y=df["CE_IV"],
        mode="lines+markers", name="CE IV",
        line=dict(color="#ef5350", width=2)
    ))
    fig_iv.add_trace(go.Scatter(
        x=df["Strike"], y=df["PE_IV"],
        mode="lines+markers", name="PE IV",
        line=dict(color="#26a69a", width=2)
    ))
    fig_iv.add_vline(x=underlying, line_color="#fff176", line_width=2)
    fig_iv.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        height=320, margin=dict(l=40, r=40, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    st.plotly_chart(fig_iv, use_container_width=True)

    # ── PCR Gauge ─────────────────────────────────────────────────────────────
    st.markdown("### 🎯 PCR Gauge & Sentiment")
    col_g, col_s = st.columns([1, 2])
    with col_g:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pcr,
            title={"text": "Put-Call Ratio", "font": {"color": "white"}},
            gauge={
                "axis": {"range": [0, 2.5], "tickcolor": "white"},
                "bar":  {"color": "#fff176"},
                "steps": [
                    {"range": [0, 0.8],   "color": "#b71c1c"},
                    {"range": [0.8, 1.2], "color": "#f57f17"},
                    {"range": [1.2, 2.5], "color": "#1b5e20"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75, "value": pcr
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            height=260, margin=dict(l=20, r=20, t=30, b=10)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_s:
        st.markdown(f"""
        <br><br>
        <div style="background:#1e2329;padding:20px;border-radius:10px;">
            <h4 style="color:#90caf9;">Sentiment Analysis</h4>
            <p class="{s_label}">{sentiment}</p>
            <table style="color:white;width:100%;">
                <tr><td>Total CE OI</td><td><b>{total_ce:,.0f}</b></td></tr>
                <tr><td>Total PE OI</td><td><b>{total_pe:,.0f}</b></td></tr>
                <tr><td>PCR</td><td><b>{pcr:.3f}</b></td></tr>
                <tr><td>Max Pain</td><td><b>₹{max_pain:,}</b></td></tr>
                <tr><td>Support</td><td><b style="color:#26a69a;">₹{support:,}</b></td></tr>
                <tr><td>Resistance</td><td><b style="color:#ef5350;">₹{resistance:,}</b></td></tr>
            </table>
            <br>
            <small style="color:#78909c;">
            PCR &gt; 1.2 → Bullish &nbsp;|&nbsp; PCR &lt; 0.8 → Bearish &nbsp;|&nbsp; 0.8–1.2 → Neutral
            </small>
        </div>
        """, unsafe_allow_html=True)

    # ── Raw Table ─────────────────────────────────────────────────────────────
    if show_raw:
        st.markdown("### 📋 Raw Option Chain Data")
        st.dataframe(
            df[["Strike", "CE_OI", "CE_ChgOI", "CE_LTP", "CE_IV",
                "PE_OI", "PE_ChgOI", "PE_LTP", "PE_IV"]].style.highlight_max(
                    subset=["CE_OI", "PE_OI"], color="#1b5e20"
                ).format("{:,.0f}", subset=["CE_OI","CE_ChgOI","PE_OI","PE_ChgOI"])
                .format("{:.2f}", subset=["CE_LTP","CE_IV","PE_LTP","PE_IV"]),
            use_container_width=True,
            height=400
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption("📡 Data sourced live from NSE India · Refreshes every 60s · For educational purposes only · Not financial advice")

    if auto_refresh:
        time.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()
