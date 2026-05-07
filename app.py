import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from analyzer import analyze_text

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Dashboard",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6C3483, #2E86C1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #888;
        font-size: 1rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .verdict-box {
        text-align: center;
        padding: 24px;
        border-radius: 14px;
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 12px;
    }
    .verdict-positive { background: #d4efdf; color: #1e8449; }
    .verdict-negative { background: #fadbd8; color: #c0392b; }
    .verdict-neutral  { background: #d6eaf8; color: #1a5276; }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #6C3483;
        margin-bottom: 8px;
    }
    .sent-positive { border-left: 4px solid #27ae60; padding: 8px 14px; border-radius: 6px; background: #eafaf1; margin: 4px 0; }
    .sent-negative { border-left: 4px solid #e74c3c; padding: 8px 14px; border-radius: 6px; background: #fdedec; margin: 4px 0; }
    .sent-neutral  { border-left: 4px solid #95a5a6; padding: 8px 14px; border-radius: 6px; background: #f2f3f4; margin: 4px 0; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Options")

    show_sentences = st.toggle("Show sentence breakdown", value=True)
    show_vader_detail = st.toggle("Show VADER detail scores", value=False)

    st.divider()
    st.markdown("### 💡 Tips")
    st.markdown("""
- Works best with **3+ sentences**
- Try **news articles, reviews, tweets, essays**
- Higher subjectivity = more opinionated text
- VADER is better for social media slang
- TextBlob is better for formal text
    """)
    st.divider()
    st.markdown("**Models used:**")
    st.markdown("🔵 VADER — rule-based NLP")
    st.markdown("🟣 TextBlob — statistical NLP")
    st.markdown("\nBuilt with ❤️ using **Streamlit + NLTK**")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎭 Sentiment Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyze the emotional tone of any text using two NLP models — VADER & TextBlob.</p>', unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
col_in, col_ex = st.columns([3, 1])

with col_in:
    user_text = st.text_area(
        "Enter your text",
        placeholder="Paste any text here — a review, article, tweet, essay, or anything you want to analyze...",
        height=180,
        label_visibility="collapsed"
    )

with col_ex:
    st.markdown("**Try an example:**")
    if st.button("😊 Positive example", use_container_width=True):
        st.session_state["example"] = (
            "This product is absolutely fantastic! I loved every single feature. "
            "The customer service was incredibly helpful and responded quickly. "
            "I would highly recommend this to all my friends and family."
        )
    if st.button("😠 Negative example", use_container_width=True):
        st.session_state["example"] = (
            "This was a terrible experience. The product broke after just two days. "
            "Customer support was rude and completely unhelpful. "
            "I will never buy from this company again. Absolute waste of money."
        )
    if st.button("😐 Mixed example", use_container_width=True):
        st.session_state["example"] = (
            "The movie had some great visual effects and the acting was decent. "
            "However, the plot was confusing and the ending felt rushed. "
            "Overall it was an average experience, nothing too special."
        )

# Load example into text area if triggered
if "example" in st.session_state:
    user_text = st.session_state.pop("example")
    st.rerun()

# ── Analyze button ────────────────────────────────────────────────────────────
col_btn, col_info = st.columns([1, 3])
with col_btn:
    analyze_btn = st.button(
        "🔍 Analyze Sentiment",
        type="primary",
        use_container_width=True,
        disabled=not user_text.strip()
    )
with col_info:
    if user_text.strip():
        wc = len(user_text.split())
        st.caption(f"Words: {wc}  |  Characters: {len(user_text)}")

st.divider()

# ── Analysis & Results ────────────────────────────────────────────────────────
if analyze_btn and user_text.strip():
    with st.spinner("Analyzing sentiment..."):
        result = analyze_text(user_text)
    st.session_state["result"] = result
    st.session_state["analyzed_text"] = user_text

if "result" not in st.session_state:
    st.info("👆 Enter some text above and click **Analyze Sentiment** to get started.")
    st.stop()

r = st.session_state["result"]

# ── Row 1: Verdict + Score cards ─────────────────────────────────────────────
c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])

verdict = r["final_verdict"]
verdict_class = f"verdict-{verdict.lower()}"
verdict_emoji = "😊" if verdict == "Positive" else "😠" if verdict == "Negative" else "😐"

with c1:
    st.markdown(
        f'<div class="verdict-box {verdict_class}">'
        f'{verdict_emoji} {verdict}</div>',
        unsafe_allow_html=True
    )
    st.caption("Overall Verdict")

with c2:
    st.metric("Consensus Score", f"{r['consensus_score']:+.3f}", help="Average of VADER + TextBlob. Range: -1 (most negative) to +1 (most positive)")

with c3:
    st.metric("Intensity", r["intensity"], help="How strong the sentiment signal is")

with c4:
    st.metric("Subjectivity", r["subjectivity_desc"], help="How opinionated vs factual the text is")

st.markdown("")

# ── Row 2: Charts ─────────────────────────────────────────────────────────────
chart1, chart2, chart3 = st.columns(3)

# Chart 1 — VADER vs TextBlob comparison bar
with chart1:
    st.markdown("#### 🔵🟣 Model Comparison")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="VADER",
        x=["Score"],
        y=[r["vader_compound"]],
        marker_color="#2E86C1",
        width=0.3
    ))
    fig.add_trace(go.Bar(
        name="TextBlob",
        x=["Score"],
        y=[r["textblob_polarity"]],
        marker_color="#6C3483",
        width=0.3
    ))
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[-1, 1], title="Score", zeroline=True, zerolinewidth=2),
        height=260,
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig, use_container_width=True)

# Chart 2 — VADER pos/neg/neu breakdown donut
with chart2:
    st.markdown("#### 📊 VADER Breakdown")
    fig2 = go.Figure(go.Pie(
        labels=["Positive", "Negative", "Neutral"],
        values=[r["vader_positive"], r["vader_negative"], r["vader_neutral"]],
        hole=0.55,
        marker_colors=["#27ae60", "#e74c3c", "#95a5a6"],
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.3f}<extra></extra>"
    ))
    fig2.update_layout(
        height=260,
        margin=dict(t=20, b=20, l=20, r=20),
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig2, use_container_width=True)

# Chart 3 — Subjectivity gauge
with chart3:
    st.markdown("#### 🎯 Subjectivity Gauge")
    fig3 = go.Figure(go.Indicator(
        mode="gauge+number",
        value=r["textblob_subjectivity"],
        number={"suffix": "", "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, 1]},
            "bar": {"color": "#6C3483"},
            "steps": [
                {"range": [0, 0.3], "color": "#d6eaf8"},
                {"range": [0.3, 0.6], "color": "#aed6f1"},
                {"range": [0.6, 1], "color": "#7fb3d3"},
            ],
            "threshold": {
                "line": {"color": "#1a5276", "width": 3},
                "thickness": 0.75,
                "value": r["textblob_subjectivity"]
            }
        }
    ))
    fig3.update_layout(
        height=260,
        margin=dict(t=30, b=10, l=30, r=30),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(fig3, use_container_width=True)

# ── VADER detail (optional) ───────────────────────────────────────────────────
if show_vader_detail:
    st.markdown("#### 🔵 VADER Raw Scores")
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Compound", f"{r['vader_compound']:+.4f}")
    d2.metric("Positive", f"{r['vader_positive']:.4f}")
    d3.metric("Negative", f"{r['vader_negative']:.4f}")
    d4.metric("Neutral",  f"{r['vader_neutral']:.4f}")

# ── Sentence breakdown ────────────────────────────────────────────────────────
if show_sentences and r["sentences"]:
    st.divider()
    st.markdown("### 🔍 Sentence-by-Sentence Breakdown")
    st.caption(f"{r['sentence_count']} sentences · {r['word_count']} words analyzed")

    # Build a dataframe for the chart
    df = pd.DataFrame(r["sentences"])
    df["index"] = [f"S{i+1}" for i in range(len(df))]

    # Mini bar chart for sentence scores
    fig4 = px.bar(
        df,
        x="index",
        y="vader_compound",
        color="vader_compound",
        color_continuous_scale=["#e74c3c", "#f9f9f9", "#27ae60"],
        range_color=[-1, 1],
        labels={"index": "Sentence", "vader_compound": "VADER Score"},
        height=220
    )
    fig4.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis_title="",
        yaxis=dict(range=[-1, 1], zeroline=True, zerolinewidth=2)
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Sentence cards
    for i, sent in enumerate(r["sentences"]):
        label = sent["vader_label"].lower()
        css_class = f"sent-{label}"
        emoji = "✅" if label == "positive" else "❌" if label == "negative" else "➖"
        st.markdown(
            f'<div class="{css_class}">'
            f'<b>S{i+1}</b> {emoji} &nbsp; {sent["sentence"]}<br>'
            f'<small>VADER: <b>{sent["vader_compound"]:+.3f}</b> ({sent["vader_label"]}) &nbsp;|&nbsp; '
            f'TextBlob: <b>{sent["textblob_polarity"]:+.3f}</b> ({sent["textblob_label"]})</small>'
            f'</div>',
            unsafe_allow_html=True
        )

# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
report = f"""# Sentiment Analysis Report

## Input Text
{st.session_state.get('analyzed_text', '')}

## Overall Results
- **Verdict:** {r['final_verdict']}
- **Consensus Score:** {r['consensus_score']:+.4f}
- **Intensity:** {r['intensity']}
- **Subjectivity:** {r['subjectivity_desc']}

## VADER Scores
- Compound: {r['vader_compound']:+.4f}
- Positive: {r['vader_positive']}
- Negative: {r['vader_negative']}
- Neutral: {r['vader_neutral']}

## TextBlob Scores
- Polarity: {r['textblob_polarity']:+.4f}
- Subjectivity: {r['textblob_subjectivity']:.4f}

## Sentence Breakdown
"""
for i, s in enumerate(r["sentences"]):
    report += f"\n**S{i+1}** [{s['vader_label']}] {s['sentence']}\n- VADER: {s['vader_compound']:+.3f} | TextBlob: {s['textblob_polarity']:+.3f}\n"

col_dl, col_cl = st.columns([1, 1])
with col_dl:
    st.download_button(
        "⬇️ Download Report (.md)",
        data=report,
        file_name="sentiment_report.md",
        mime="text/markdown",
        use_container_width=True
    )
with col_cl:
    if st.button("🔄 Clear & Analyze New Text", use_container_width=True):
        st.session_state.pop("result", None)
        st.session_state.pop("analyzed_text", None)
        st.rerun()
