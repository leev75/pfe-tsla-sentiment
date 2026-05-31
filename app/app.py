"""
TSLA Sentiment & Market Prediction Dashboard
PFE — Khelil Dhiaeddine | University of Blida 1
Supervisor: Nesrine Lahiani

Run locally:
    pip install streamlit pandas plotly pyarrow
    streamlit run app.py

Place features_targets_final_clean.parquet in the same folder as app.py,
or adjust DATA_PATH below.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import warnings, os
warnings.filterwarnings("ignore")

# True-label column (X=1% threshold, matching the thesis)
TRUE_COL = "target_X1.0"

# Plain-English definitions for the ⓘ methodology tooltips
GLOSSARY = {
    "Sharpe Ratio": "Risk-adjusted return: average return divided by its volatility. "
                    "Higher is better; above 1 is good, above 2 is excellent.",
    "Directional Accuracy": "Share of days where the model predicted the correct "
                            "market direction (up vs down). 0.5 = random guessing.",
    "Macro-F1": "Average F1 score across all three classes (BUY/SELL/HOLD), weighting "
                "each class equally regardless of how often it appears.",
    "Walk-forward CV": "Cross-validation that always trains on past data and tests on "
                       "future data, never the reverse — avoids look-ahead bias.",
    "Price-only vs Sent+Price": "Price-only models use just market features; Sent+Price "
                                "models add the sentiment features on top.",
}

# ─────────────────────────────────────────────
# CONFIG — adjust path if needed
# ─────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "features_targets_final_clean.parquet")

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────
NAVY   = "#0B2545"
ROYAL  = "#1B4F8A"
GOLD   = "#F5A623"
GREEN  = "#27AE60"
RED    = "#E74C3C"
NEUTR  = "#95A5A6"
BG     = "#F7F9FC"

SOURCE_COLORS = {
    "news":            "#1B4F8A",
    "reddit":          "#FF4500",
    "twitter_general": "#1DA1F2",
    "twitter_musk":    "#F5A623",
    "all":             "#2C3E50",
}
SIGNAL_COLORS = {"BUY": GREEN, "SELL": RED, "HOLD": NEUTR}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="TSLA Sentinel",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
  html, body, [class*="css"] {{ font-family: 'IBM Plex Sans', sans-serif; background:{BG}; }}
  .main {{ background:{BG}; }}
  section[data-testid="stSidebar"] {{ background:{NAVY} !important; }}
  section[data-testid="stSidebar"] * {{ color:white !important; }}
  .metric-card {{ background:#fff; border-radius:12px; padding:18px 22px;
                  border-left:4px solid {ROYAL}; box-shadow:0 2px 8px rgba(11,37,69,.08);
                  margin-bottom:6px; }}
  .metric-card.gold  {{ border-left-color:{GOLD}; }}
  .metric-card.green {{ border-left-color:{GREEN}; }}
  .metric-value {{ font-size:1.9rem; font-weight:700; color:{NAVY};
                   font-family:'IBM Plex Mono'; line-height:1; }}
  .metric-label {{ font-size:.76rem; color:#7F8C8D; text-transform:uppercase;
                   letter-spacing:.08em; margin-top:4px; }}
  .metric-sub   {{ font-size:.70rem; color:#aaa; margin-top:5px; }}
  .sec-hdr {{ font-size:1rem; font-weight:700; color:{NAVY};
              border-bottom:2px solid {GOLD}; padding-bottom:5px;
              margin:22px 0 14px; text-transform:uppercase; letter-spacing:.06em; }}
  .dash-title {{ font-size:1.75rem; font-weight:700; color:{NAVY}; margin-bottom:2px; }}
  .dash-sub   {{ font-size:.88rem; color:#7F8C8D; margin-bottom:18px; }}
  #MainMenu, footer {{ visibility:hidden; }}
  .stDeployButton {{ display:none; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        st.error(f"Data file not found: {DATA_PATH}\n\nPlace `features_targets_final_clean.parquet` next to `app.py` and restart.")
        st.stop()
    ft = pd.read_parquet(DATA_PATH)
    ft["trading_day"] = pd.to_datetime(ft["trading_day"])

    # Build a long-format sentiment dataframe from features_targets
    sources = ["news", "reddit", "twitter_general", "twitter_musk", "all"]
    rows = []
    for src in sources:
        col = f"{src}_sent_mean_w"
        vol = f"{src}_vol"
        if col in ft.columns:
            tmp = ft[["trading_day","close","split"]].copy()
            tmp["source"]          = src
            tmp["sentiment_score"] = ft[col]
            tmp["doc_count"]       = ft[vol] if vol in ft.columns else np.nan
            rows.append(tmp)
    daily_sentiment = pd.concat(rows, ignore_index=True)

    return ft, daily_sentiment

ft, daily_sentiment = load_data()
train_df = ft[ft["split"] == "train"].copy().reset_index(drop=True)
test_df  = ft[ft["split"] == "test"].copy().reset_index(drop=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def compute_equity(prices, signals, initial=10000):
    equity, position = [initial], 0
    for i in range(1, len(prices)):
        ret = (prices[i] - prices[i-1]) / prices[i-1]
        sig = signals[i-1]
        if sig == "BUY":   position = 1
        elif sig == "SELL": position = 0
        equity.append(equity[-1] * (1 + position * ret))
    return equity

def plotly_cfg():
    return {"displayModeBar": False}

def sec_header(title, glossary_key=None):
    """Section header with an optional ⓘ tooltip pulled from GLOSSARY."""
    if glossary_key and glossary_key in GLOSSARY:
        tip = GLOSSARY[glossary_key].replace('"', "&quot;")
        st.markdown(
            f"<div class='sec-hdr'>{title} "
            f"<span title='{tip}' style='cursor:help;color:{GOLD};font-size:.85rem'>ⓘ</span>"
            f"</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='sec-hdr'>{title}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center;padding:10px 0 20px'>
      <div style='font-size:2rem'>📈</div>
      <div style='font-size:1.05rem;font-weight:700;letter-spacing:.05em'>TSLA SENTINEL</div>
      <div style='font-size:.70rem;opacity:.65;margin-top:3px'>Sentiment · Prediction · Signals</div>
    </div>
    <hr style='border-color:rgba(255,255,255,.15);margin:0 0 14px'/>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation",
        ["🏠  Overview","🧠  Sentiment Analysis","📊  Model Comparison","📅  Trading Signals"],
        label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:14px 0'/>",
                unsafe_allow_html=True)
    st.markdown("<div style='font-size:.78rem;opacity:.75;margin-bottom:5px'>YEAR FILTER</div>",
                unsafe_allow_html=True)
    year_filter = st.selectbox("Year", ["All (2020–2023)","2020","2021","2022","2023"],
                                label_visibility="collapsed")

    st.markdown("<div style='font-size:.78rem;opacity:.75;margin:10px 0 5px'>SOURCE FILTER</div>",
                unsafe_allow_html=True)
    source_filter = st.multiselect("Sources",
        ["news","reddit","twitter_general","twitter_musk"],
        default=["news","reddit","twitter_general","twitter_musk"],
        label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(255,255,255,.15);margin:14px 0'/>",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:.68rem;opacity:.5;line-height:1.7'>
      PFE · University of Blida 1<br>
      Student: Khelil Dhiaeddine<br>
      Supervisor: N. Lahiani<br>
      Dataset: TSLA 2020–2023
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE: OVERVIEW
# ═════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown("<div class='dash-title'>TSLA Sentiment & Market Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-sub'>A Machine Learning Approach to Financial News Sentiment and Market Trend Prediction · PFE 2024</div>", unsafe_allow_html=True)

    # KPIs
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        ("85,649","Documents Collected","4 sources · Jan 2020 – Dec 2023","gold"),
        ("0.403","FinBERT Macro-F1","vs TF-IDF+LR 0.380 · SVM 0.364","blue"),
        ("5.71","Best Sharpe Ratio","XGB_price · price-only features","green"),
        ("53.0%","Best Dir. Accuracy","XGB_price · 2023 test set","blue"),
        ("+113.7%","TSLA Buy & Hold","2023 benchmark return","gold"),
    ]
    for col,(val,lbl,sub,color) in zip([c1,c2,c3,c4,c5],kpis):
        with col:
            st.markdown(f"""
            <div class='metric-card {color}'>
              <div class='metric-value'>{val}</div>
              <div class='metric-label'>{lbl}</div>
              <div class='metric-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'>Corpus & TSLA Price</div>", unsafe_allow_html=True)
    cl, cr = st.columns([1,2])

    with cl:
        corpus = {"news":12500,"reddit":28400,"twitter_general":38200,"twitter_musk":6549}
        fig = go.Figure(go.Pie(
            labels=list(corpus.keys()), values=list(corpus.values()),
            hole=.55, marker_colors=[SOURCE_COLORS[s] for s in corpus],
            textinfo="percent+label", textfont_size=11,
        ))
        fig.update_layout(height=280, margin=dict(t=10,b=10,l=10,r=10),
            showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(text="85,649<br>docs",x=.5,y=.5,
                font_size=13,font_color=NAVY,showarrow=False,font_family="IBM Plex Mono")])
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    with cr:
        price_data = ft.sort_values("trading_day")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price_data["trading_day"], y=price_data["close"],
            fill="tozeroy", fillcolor="rgba(27,79,138,.08)",
            line=dict(color=ROYAL,width=2), name="TSLA Close",
            hovertemplate="%{x|%b %d, %Y}: $%{y:.2f}<extra></extra>"))
        fig.add_vrect(
            x0=str(test_df["trading_day"].min()),
            x1=str(test_df["trading_day"].max()),
            fillcolor=GOLD, opacity=.07,
            annotation_text="2023 Test Period",
            annotation_position="top left",
            annotation_font_size=11, annotation_font_color=NAVY)
        fig.update_layout(height=280, margin=dict(t=20,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False,color="#95A5A6"),
            yaxis=dict(showgrid=True,gridcolor="#E8ECF0",color="#95A5A6",title="Close (USD)"),
            hovermode="x unified",
            title=dict(text="TSLA Stock Price · Jan 2020 – Dec 2023",
                       font=dict(size=13,color=NAVY),x=0))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    # Pipeline steps
    st.markdown("<div class='sec-hdr'>Pipeline Summary</div>", unsafe_allow_html=True)
    steps = [
        ("01","Data Collection","85,649 docs · 4 sources\nJan 2020 – Dec 2023"),
        ("02","NLP Preprocessing","Cleaning · Tokenization\nLemmatization · NER"),
        ("03","Sentiment Analysis","FinBERT (primary)\nTF-IDF+LR/SVM · VADER"),
        ("04","Feature Engineering","237 features · 995 rows\nSentiment + Price + Lags"),
        ("05","Market Prediction","XGB · RF · LSTM · Attention\nSharpe · DA · F1"),
    ]
    for col,(num,title,desc) in zip(st.columns(5),steps):
        with col:
            st.markdown(f"""
            <div style='background:#fff;border-radius:10px;padding:15px 12px;
                        box-shadow:0 2px 8px rgba(11,37,69,.07);text-align:center;
                        border-top:3px solid {GOLD}'>
              <div style='font-size:1.5rem;font-weight:800;color:{GOLD};
                          font-family:"IBM Plex Mono"'>{num}</div>
              <div style='font-size:.80rem;font-weight:700;color:{NAVY};margin:5px 0 3px'>{title}</div>
              <div style='font-size:.70rem;color:#7F8C8D;white-space:pre-line;line-height:1.5'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE: SENTIMENT ANALYSIS
# ═════════════════════════════════════════════
elif page == "🧠  Sentiment Analysis":
    st.markdown("<div class='dash-title'>Sentiment Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-sub'>Daily sentiment scores across 4 sources · overlaid on TSLA price</div>", unsafe_allow_html=True)

    # Model comparison — hardcoded thesis results
    st.markdown("<div class='sec-hdr'>Sentiment Model Comparison · TSLA Test Set (N=200)</div>", unsafe_allow_html=True)
    model_res = pd.DataFrame({
        "Model":    ["FinBERT","TF-IDF+LR","TF-IDF+SVM","VADER"],
        "Macro-F1": [0.403, 0.380, 0.364, 0.355],
        "CI_low":   [0.330, 0.310, 0.295, 0.285],
        "CI_high":  [0.470, 0.450, 0.435, 0.425],
    })
    fig = go.Figure()
    for i, row in model_res.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Model"]], y=[row["Macro-F1"]],
            error_y=dict(type="data", symmetric=False,
                array=[row["CI_high"]-row["Macro-F1"]],
                arrayminus=[row["Macro-F1"]-row["CI_low"]],
                color="#555", thickness=2, width=6),
            marker_color=[GOLD,ROYAL,"#2980B9",NEUTR][i],
            text=[f'{row["Macro-F1"]:.3f}'], textposition="outside",
            textfont=dict(size=11,color=NAVY,family="IBM Plex Mono"), width=.45,
        ))
    fig.add_hline(y=0.333, line_dash="dot", line_color=RED, line_width=1.5,
                   annotation_text="Random baseline (0.333)",
                   annotation_position="bottom right",
                   annotation_font_size=10, annotation_font_color=RED)
    fig.update_layout(height=270, margin=dict(t=10,b=30,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0,.55],showgrid=True,gridcolor="#E8ECF0",
                   title="Macro-F1",color="#95A5A6"),
        xaxis=dict(showgrid=False,color="#95A5A6"),
        showlegend=False, bargap=.3,
        annotations=[dict(x=.5,y=-0.25,xref="paper",yref="paper",showarrow=False,
            text="⚠️  Differences not statistically significant at N=200 (overlapping 95% bootstrap CIs)",
            font=dict(size=10,color="#E67E22"),align="center")])
    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    # Sentiment timeline — real data
    st.markdown("<div class='sec-hdr'>Daily Sentiment Score by Source</div>", unsafe_allow_html=True)

    ds = daily_sentiment.copy()
    if year_filter != "All (2020–2023)":
        ds = ds[ds["trading_day"].dt.year == int(year_filter)]
    if source_filter:
        ds = ds[ds["source"].isin(source_filter)]
    ds = ds.sort_values("trading_day")

    price_line = ft.sort_values("trading_day")
    if year_filter != "All (2020–2023)":
        price_line = price_line[price_line["trading_day"].dt.year == int(year_filter)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                         row_heights=[.35,.65], vertical_spacing=.04)
    fig.add_trace(go.Scatter(
        x=price_line["trading_day"], y=price_line["close"],
        line=dict(color=ROYAL,width=1.5), name="TSLA Close",
        hovertemplate="%{x|%b %d}: $%{y:.2f}<extra></extra>"), row=1, col=1)

    for src in (source_filter if source_filter else
                ["news","reddit","twitter_general","twitter_musk"]):
        sd = ds[ds["source"]==src].sort_values("trading_day")
        if len(sd) == 0: continue
        fig.add_trace(go.Scatter(
            x=sd["trading_day"],
            y=sd["sentiment_score"].rolling(7, min_periods=1).mean(),
            line=dict(color=SOURCE_COLORS.get(src,ROYAL),width=1.8),
            name=src,
            hovertemplate=f"{src}: %{{y:.3f}}<extra></extra>"), row=2, col=1)

    fig.add_hline(y=0, line_dash="dash", line_color="#BDC3C7", line_width=1, row=2, col=1)
    fig.update_layout(height=460, margin=dict(t=10,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        legend=dict(orientation="h",y=-0.08,x=0,font_size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#E8ECF0", color="#95A5A6")
    fig.update_yaxes(title_text="Close (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment (7d MA)", row=2, col=1)
    fig.update_xaxes(showgrid=False, color="#95A5A6")
    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    st.info("🔒 **Musk stream architecturally isolated** — twitter_musk feed treated as a separate input to avoid conflating celebrity-driven noise with market-relevant sentiment. See §6 System Design for rationale.")

    # #4 — Document volume by source (the Twitter coverage gap)
    sec_header("Document Volume by Source — the Twitter Coverage Gap")
    vol_rows = []
    for src in ["news","reddit","twitter_general","twitter_musk"]:
        vcol = f"{src}_vol"
        if vcol in ft.columns:
            monthly = ft.set_index("trading_day")[vcol].resample("MS").sum()
            for d, v in monthly.items():
                vol_rows.append({"month": d, "source": src, "volume": v})
    vol_df = pd.DataFrame(vol_rows)

    fig = go.Figure()
    for src in ["news","reddit","twitter_general","twitter_musk"]:
        sub = vol_df[vol_df["source"]==src]
        fig.add_trace(go.Scatter(
            x=sub["month"], y=sub["volume"],
            mode="lines", name=src,
            line=dict(color=SOURCE_COLORS.get(src,ROYAL), width=2),
            stackgroup="one",
            hovertemplate=f"{src}: %{{y:.0f}} docs<extra></extra>"))
    fig.add_vrect(x0="2023-01-01", x1="2023-12-31",
        fillcolor=RED, opacity=.06,
        annotation_text="2023: twitter_general → 0",
        annotation_position="top left",
        annotation_font_size=10, annotation_font_color=RED)
    fig.update_layout(height=270, margin=dict(t=20,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,color="#95A5A6"),
        yaxis=dict(showgrid=True,gridcolor="#E8ECF0",color="#95A5A6",
                   title="Docs per month (stacked)"),
        hovermode="x unified",
        legend=dict(orientation="h",y=-0.12,x=0,font_size=11))
    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())
    st.markdown(
        f"<div style='font-size:.78rem;color:#888;font-style:italic;margin-top:-6px'>"
        f"twitter_general coverage collapses to zero in 2023 (post-API restriction). "
        f"This degrades the sentiment signal exactly during the test period — the root "
        f"cause behind the Key Finding on the Model Comparison page.</div>",
        unsafe_allow_html=True)

    # Sentiment distribution donuts — real data
    st.markdown("<div class='sec-hdr'>Average Sentiment Score by Source</div>", unsafe_allow_html=True)
    src_list = source_filter if source_filter else ["news","reddit","twitter_general","twitter_musk"]
    cols = st.columns(len(src_list))
    for col, src in zip(cols, src_list):
        with col:
            col_name = f"{src}_sent_mean_w"
            if col_name not in ft.columns: continue
            avg = ft[col_name].mean()
            pos = (ft[col_name] > 0.05).mean()
            neg = (ft[col_name] < -0.05).mean()
            neu = 1 - pos - neg
            fig = go.Figure(go.Pie(
                labels=["Positive","Negative","Neutral"],
                values=[max(pos,0), max(neg,0), max(neu,0)],
                hole=.52, marker_colors=[GREEN,RED,NEUTR],
                textinfo="percent", textfont_size=10))
            fig.update_layout(height=165, margin=dict(t=8,b=0,l=0,r=0),
                paper_bgcolor="rgba(0,0,0,0)", showlegend=False,
                annotations=[dict(text=f"{avg:+.2f}",x=.5,y=.5,
                    font_size=15,font_color=NAVY,showarrow=False,
                    font_family="IBM Plex Mono")])
            st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())
            st.markdown(f"<div style='text-align:center;font-size:.76rem;color:#7F8C8D'>{src}</div>",
                        unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═════════════════════════════════════════════
elif page == "📊  Model Comparison":
    st.markdown("<div class='dash-title'>Prediction Model Comparison</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='dash-sub'>2023 test set · "
        f"<span title='{GLOSSARY['Walk-forward CV']}' style='cursor:help;border-bottom:1px dotted #aaa'>walk-forward cross-validation</span> "
        f"· X=1% threshold</div>", unsafe_allow_html=True)

    results = pd.DataFrame([
        {"Model":"XGB_price",  "Feature Set":"Price-only", "Sharpe":5.71,"DA":0.530,"F1":0.400},
        {"Model":"RF_price",   "Feature Set":"Price-only", "Sharpe":4.65,"DA":0.514,"F1":0.382},
        {"Model":"Attn_price", "Feature Set":"Price-only", "Sharpe":3.65,"DA":0.467,"F1":0.361},
        {"Model":"LSTM_price", "Feature Set":"Price-only", "Sharpe":3.49,"DA":0.449,"F1":0.345},
        {"Model":"XGB_full",   "Feature Set":"Sent+Price", "Sharpe":4.20,"DA":0.490,"F1":0.370},
        {"Model":"RF_full",    "Feature Set":"Sent+Price", "Sharpe":3.87,"DA":0.478,"F1":0.358},
        {"Model":"LSTM_full",  "Feature Set":"Sent+Price", "Sharpe":2.91,"DA":0.430,"F1":0.310},
        {"Model":"Attn_full",  "Feature Set":"Sent+Price", "Sharpe":0.12,"DA":0.000,"F1":0.000},
    ])

    sec_header("Sharpe Ratio — Price-only vs Sentiment+Price", "Sharpe Ratio")
    fig = go.Figure()
    for fset, color in [("Price-only",ROYAL),("Sent+Price",GOLD)]:
        sub = results[results["Feature Set"]==fset]
        fig.add_trace(go.Bar(
            x=sub["Model"], y=sub["Sharpe"], name=fset,
            marker_color=color,
            text=[f'{v:.2f}' for v in sub["Sharpe"]],
            textposition="outside",
            textfont=dict(size=10,family="IBM Plex Mono"), width=.38))
    fig.update_layout(height=310, barmode="group", margin=dict(t=10,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Sharpe Ratio",showgrid=True,gridcolor="#E8ECF0",color="#95A5A6"),
        xaxis=dict(showgrid=False,color="#95A5A6"),
        legend=dict(orientation="h",y=1.05,x=.5,xanchor="center",font_size=11))
    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    c1, c2 = st.columns(2)
    for col, metric, title, yrange in [
        (c1, "DA",  "Directional Accuracy", [0,.65]),
        (c2, "F1",  "Macro F1 Score",        [0,.50]),
    ]:
        with col:
            fig = go.Figure()
            for fset, color, sym in [("Price-only",ROYAL,"circle"),("Sent+Price",GOLD,"diamond")]:
                sub = results[results["Feature Set"]==fset]
                fig.add_trace(go.Scatter(
                    x=sub["Model"], y=sub[metric], mode="markers+lines",
                    marker=dict(size=11,color=color,symbol=sym,
                                line=dict(width=1.5,color="white")),
                    line=dict(color=color,width=1.5,dash="dot"), name=fset))
            if metric == "DA":
                fig.add_hline(y=.5, line_dash="dash", line_color=RED, line_width=1.5,
                               annotation_text="Random (0.5)",
                               annotation_font_size=9, annotation_font_color=RED)
            fig.update_layout(height=270, margin=dict(t=30,b=10,l=10,r=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(title=title,range=yrange,showgrid=True,
                           gridcolor="#E8ECF0",color="#95A5A6"),
                xaxis=dict(showgrid=False,color="#95A5A6",tickangle=-25),
                title=dict(text=title,font=dict(size=12,color=NAVY),x=0),
                legend=dict(font_size=10))
            st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    st.markdown(f"""
    <div style='background:#FEF9EC;border-left:4px solid {GOLD};border-radius:8px;
                padding:15px 18px;margin-top:10px'>
      <div style='font-weight:700;color:{NAVY};margin-bottom:5px'>
        🔍 Key Finding: Why Sentiment Features Underperform in 2023
      </div>
      <div style='font-size:.87rem;color:#555;line-height:1.7'>
        All <b>full-feature models</b> (Sentiment+Price) underperform their <b>price-only
        counterparts</b>. This is a <b>documented, interpretable finding</b> — not a failure.<br><br>
        Two compounding factors: <b>(1) 2023 regime shift</b> — TSLA recovery driven by
        fundamentals, not sentiment narrative; <b>(2) Twitter coverage gap</b> — reduced tweet
        volume post-API restriction degraded twitter_general and twitter_musk signal quality.
        The Attn_full collapse (0 BUY predictions) reflects dataset size vs. feature count
        mismatch with N=249 test samples and 237 features.
      </div>
    </div>""", unsafe_allow_html=True)

# ═════════════════════════════════════════════
# PAGE: TRADING SIGNALS
# ═════════════════════════════════════════════
elif page == "📅  Trading Signals":
    st.markdown("<div class='dash-title'>Trading Signals & Portfolio Equity</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-sub'>2023 test set · BUY / SELL / HOLD predictions vs TSLA price</div>", unsafe_allow_html=True)

    model_options = {
        "XGB_price  (Best — Sharpe 5.71)":  "pred_xgb_price",
        "RF_price   (Sharpe 4.65)":          "pred_rf_price",
        "Attn_price (Sharpe 3.65)":          "pred_attn_price",
        "LSTM_price (Sharpe 3.49)":          "pred_lstm_price",
        "XGB_full   (Sent+Price)":           "pred_xgb_full",
        "RF_full    (Sent+Price)":           "pred_rf_full",
        "LSTM_full  (Sent+Price)":           "pred_lstm_full",
        "Attn_full  (Collapsed)":            "pred_attn_full",
    }

    selected = st.selectbox("Select model", list(model_options.keys()))
    pred_col = model_options[selected]

    plot_df = test_df[["trading_day","close",pred_col]].dropna().copy()
    plot_df.columns = ["date","close","signal"]
    plot_df = plot_df.sort_values("date").reset_index(drop=True)

    # Signal chart
    st.markdown("<div class='sec-hdr'>Predicted Signals on TSLA Price (2023)</div>", unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=plot_df["date"], y=plot_df["close"],
        line=dict(color=ROYAL,width=2), name="TSLA Close",
        hovertemplate="%{x|%b %d}: $%{y:.2f}<extra></extra>"))
    for sig, sym, color in [("BUY","triangle-up",GREEN),("SELL","triangle-down",RED)]:
        sub = plot_df[plot_df["signal"]==sig]
        if len(sub):
            fig.add_trace(go.Scatter(
                x=sub["date"], y=sub["close"], mode="markers",
                marker=dict(size=9,color=color,symbol=sym,
                            line=dict(width=1,color="white")),
                name=sig,
                hovertemplate=f"{sig}: %{{x|%b %d}} $%{{y:.2f}}<extra></extra>"))
    fig.update_layout(height=330, margin=dict(t=10,b=10,l=10,r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False,color="#95A5A6"),
        yaxis=dict(showgrid=True,gridcolor="#E8ECF0",color="#95A5A6",title="Close (USD)"),
        legend=dict(orientation="h",y=1.05,x=.5,xanchor="center",font_size=11),
        hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    cl, cr = st.columns([1,2])

    with cl:
        st.markdown("<div class='sec-hdr'>Signal Distribution</div>", unsafe_allow_html=True)
        # Only valid signals (drop NaN rows already handled by dropna above)
        counts = plot_df["signal"].value_counts()
        fig = go.Figure(go.Bar(
            x=counts.index.tolist(), y=counts.values.tolist(),
            marker_color=[SIGNAL_COLORS.get(s,NEUTR) for s in counts.index],
            text=counts.values.tolist(), textposition="outside",
            textfont=dict(family="IBM Plex Mono",size=11), width=.5))
        fig.update_layout(height=240, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True,gridcolor="#E8ECF0",color="#95A5A6"),
            xaxis=dict(showgrid=False,color="#95A5A6"),
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

    with cr:
        st.markdown("<div class='sec-hdr'>Equity Curve vs Buy-and-Hold</div>", unsafe_allow_html=True)
        prices  = plot_df["close"].tolist()
        signals = plot_df["signal"].tolist()
        strat   = compute_equity(prices, signals)
        bh      = [10000*(p/prices[0]) for p in prices]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=strat,
            line=dict(color=GOLD,width=2.5),
            name=f"Strategy ({selected.split('(')[0].strip()})",
            hovertemplate="Strategy: $%{y:,.0f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=plot_df["date"], y=bh,
            line=dict(color=ROYAL,width=1.5,dash="dash"),
            name="TSLA Buy & Hold",
            hovertemplate="B&H: $%{y:,.0f}<extra></extra>"))
        fig.update_layout(height=240, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False,color="#95A5A6"),
            yaxis=dict(showgrid=True,gridcolor="#E8ECF0",color="#95A5A6",
                       title="Portfolio Value ($)"),
            hovermode="x unified",
            legend=dict(orientation="h",y=1.05,x=.5,xanchor="center",font_size=11))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())

        ret_s = (strat[-1]/10000-1)*100
        ret_b = (bh[-1]/10000-1)*100
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""<div class='metric-card green'>
              <div class='metric-value'>{ret_s:+.1f}%</div>
              <div class='metric-label'>Strategy Return</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class='metric-card gold'>
              <div class='metric-value'>{ret_b:+.1f}%</div>
              <div class='metric-label'>Buy & Hold Return</div>
            </div>""", unsafe_allow_html=True)

        if ret_s < ret_b:
            st.markdown(f"""
            <div style='font-size:.75rem;color:#888;margin-top:6px;font-style:italic'>
              Strategy underperforms buy-and-hold — consistent with the documented
              2023 regime shift finding (see Model Comparison page).
            </div>""", unsafe_allow_html=True)

    # ── #1 Confusion Matrix + #7 Per-class F1 ──────────────────────────────
    st.markdown("<div style='margin-top:24px'></div>", unsafe_allow_html=True)
    cm_col, f1_col = st.columns(2)

    # Build aligned true/pred from the parquet (uses TRUE_COL = target_X1.0)
    eval_df = test_df[[TRUE_COL, pred_col]].dropna()
    y_true  = eval_df[TRUE_COL].tolist()
    y_pred  = eval_df[pred_col].tolist()
    labels  = ["BUY", "HOLD", "SELL"]

    with cm_col:
        sec_header("Confusion Matrix", "Macro-F1")
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        # Heatmap
        fig = go.Figure(go.Heatmap(
            z=cm,
            x=[f"Pred {l}" for l in labels],
            y=[f"True {l}" for l in labels],
            colorscale=[[0, "#F7F9FC"], [1, ROYAL]],
            showscale=False,
            text=cm, texttemplate="%{text}",
            textfont=dict(size=16, family="IBM Plex Mono"),
            hovertemplate="%{y} → %{x}: %{z}<extra></extra>"))
        fig.update_layout(height=300, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(side="bottom", color="#555"),
            yaxis=dict(autorange="reversed", color="#555"))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())
        # Diagnostic note: detect collapsed classes
        never_pred = [l for l in labels if l not in set(y_pred)]
        if never_pred:
            st.markdown(
                f"<div style='font-size:.78rem;color:{RED};font-style:italic'>"
                f"⚠️ Model never predicts: {', '.join(never_pred)} — "
                f"a class collapse, visible as an empty column.</div>",
                unsafe_allow_html=True)

    with f1_col:
        sec_header("Per-Class Performance", "Macro-F1")
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0)
        fig = go.Figure()
        for metric_name, vals, color in [
            ("Precision", p, ROYAL),
            ("Recall",    r, GOLD),
            ("F1",        f, GREEN),
        ]:
            fig.add_trace(go.Bar(
                x=labels, y=vals, name=metric_name, marker_color=color,
                text=[f"{v:.2f}" for v in vals], textposition="outside",
                textfont=dict(size=9, family="IBM Plex Mono")))
        fig.update_layout(height=300, barmode="group",
            margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(range=[0,1.05], showgrid=True, gridcolor="#E8ECF0",
                       color="#95A5A6", title="Score"),
            xaxis=dict(showgrid=False, color="#95A5A6"),
            legend=dict(orientation="h", y=1.08, x=.5, xanchor="center", font_size=10))
        st.plotly_chart(fig, use_container_width=True, config=plotly_cfg())
        macro = f.mean()
        st.markdown(
            f"<div style='font-size:.80rem;color:#7F8C8D;text-align:center'>"
            f"Macro-F1 (avg across classes): "
            f"<b style='color:{NAVY};font-family:\"IBM Plex Mono\"'>{macro:.3f}</b></div>",
            unsafe_allow_html=True)
