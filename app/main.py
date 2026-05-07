"""
IPL Auction Analytics 2023 – Streamlit Web App
================================================
Premium multi-page analytics dashboard.

Run with:  streamlit run app/main.py
"""

import streamlit as st

# ── Page config MUST be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="IPL Auction Analytics 2023",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.cleaning.pipeline import run_pipeline
from src.eda.analysis import generate_all_charts, get_key_insights
from src.sql.database import setup_database, get_report, search_player, get_team_players, SQL_REPORTS, run_query
from src.ml.models import run_ml_pipeline, predict_price, predict_sold

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Layout ── */
.main { background-color: #0F1117; }
section[data-testid="stSidebar"] { background-color: #161929; border-right: 1px solid #2E3250; }

/* ── Metric cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1A1D2E 0%, #252840 100%);
    border: 1px solid #2E3250;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    margin-bottom: 10px;
}
.kpi-value { font-size: 2rem; font-weight: 800; color: #2EC4B6; }
.kpi-label { font-size: 0.78rem; color: #A0A0A0; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

/* ── Section headers ── */
.section-header {
    font-size: 1.4rem; font-weight: 700; color: #E0E0E0;
    border-left: 4px solid #2EC4B6; padding-left: 12px; margin: 24px 0 16px 0;
}

/* ── Team badges ── */
.team-badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    background: #252840; border: 1px solid #2EC4B6; color: #2EC4B6;
    margin: 2px;
}

/* ── Prediction result ── */
.pred-sold   { background:#1A3A2A; border:1px solid #2EC4B6; border-radius:10px; padding:16px; text-align:center; }
.pred-unsold { background:#3A1A1A; border:1px solid #E63946; border-radius:10px; padding:16px; text-align:center; }

/* ── Sidebar nav ── */
[data-testid="stSidebarNav"] { display: none; }

/* ── Chat bubbles ── */
.chat-user     { background:#1e2235; border-radius:8px; padding:10px 14px; margin:6px 0; }
.chat-assistant{ background:#162530; border-left:3px solid #2EC4B6; border-radius:8px; padding:10px 14px; margin:6px 0; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner="🏏 Processing dataset …")
def load_data():
    df = run_pipeline(save=True)
    setup_database(df)
    return df

@st.cache_data(show_spinner="📊 Generating charts …")
def load_charts(_df):
    return generate_all_charts(_df)

@st.cache_data(show_spinner="🤖 Training ML models …")
def load_ml(_df):
    return run_ml_pipeline(_df)

def kpi(label, value, suffix=""):
    return f"""<div class="kpi-card"><div class="kpi-value">{value}{suffix}</div><div class="kpi-label">{label}</div></div>"""


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAV
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 IPL Auction 2023")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home & KPIs", "📊 Dashboard", "🏟️ Team Analysis",
         "🔍 Player Search", "🤖 AI Chatbot", "🧠 ML Predictions",
         "🗄️ SQL Reports", "📥 Download"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<small style='color:#666'>Data: IPL 2023 Auction<br>568 players | 10 teams</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
df     = load_data()
charts = load_charts(df)
ml_results = load_ml(df)
insights   = get_key_insights(df)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME & KPIs
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home & KPIs":
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px 0;'>
        <div style='font-size:3rem;'>🏏</div>
        <h1 style='font-size:2.4rem; font-weight:900; color:#E0E0E0; margin:0;'>IPL Auction Analytics</h1>
        <p style='color:#A0A0A0; font-size:1.1rem; margin-top:8px;'>2023 Season · 568 Players · 10 Teams · ₹167 Cr Total Spend</p>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row 1
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Total Players", insights["total_players"]), unsafe_allow_html=True)
    c2.markdown(kpi("Sold", insights["sold_players"]), unsafe_allow_html=True)
    c3.markdown(kpi("Unsold", insights["unsold_players"]), unsafe_allow_html=True)
    c4.markdown(kpi("Retained", insights["retained_players"]), unsafe_allow_html=True)
    c5.markdown(kpi("Total Spent", f"₹{insights['total_spend_cr']}", " Cr"), unsafe_allow_html=True)

    st.markdown("")

    # KPI Row 2
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi("Avg Price", f"₹{insights['avg_price_cr']}", " Cr"), unsafe_allow_html=True)
    c2.markdown(kpi("Max Price", f"₹{insights['max_price_cr']}", " Cr"), unsafe_allow_html=True)
    c3.markdown(kpi("Biggest Spender", insights["top_team"].split()[-1]), unsafe_allow_html=True)
    c4.markdown(kpi("Priciest Player", insights["most_expensive_player"].split()[-1]), unsafe_allow_html=True)
    c5.markdown(kpi("Top Role Sold", insights["most_sold_role"]), unsafe_allow_html=True)

    st.markdown("---")

    # Highlight charts
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Sold vs Unsold</div>', unsafe_allow_html=True)
        st.image(charts["sold_unsold"], use_container_width=True)
    with col2:
        st.markdown('<div class="section-header">Team Spending</div>', unsafe_allow_html=True)
        st.image(charts["team_spending"], use_container_width=True)

    st.markdown('<div class="section-header">🏆 Most Expensive Players</div>', unsafe_allow_html=True)
    st.image(charts["top_expensive"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard":
    st.markdown("## 📊 Full Analytics Dashboard")

    tab1, tab2, tab3, tab4 = st.tabs(["💰 Spending", "👤 Players", "📈 Distributions", "🌏 Origin"])

    with tab1:
        c1, c2 = st.columns(2)
        c1.image(charts["team_spending"], use_container_width=True)
        c2.image(charts["role_spending"], use_container_width=True)
        st.image(charts["team_role_heatmap"], use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        c1.image(charts["top_expensive"], use_container_width=True)
        c2.image(charts["bargain_buys"], use_container_width=True)
        st.image(charts["base_vs_final"], use_container_width=True)

    with tab3:
        c1, c2 = st.columns(2)
        c1.image(charts["price_distribution"], use_container_width=True)
        c2.image(charts["price_buckets"], use_container_width=True)
        st.image(charts["boxplot_role"], use_container_width=True)

    with tab4:
        st.image(charts["origin_comparison"], use_container_width=True)
        st.image(charts["team_composition"], use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TEAM ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🏟️ Team Analysis":
    st.markdown("## 🏟️ Team Analysis")

    teams = sorted(df[df["is_sold"]]["current_team"].unique().tolist())
    teams = [t for t in teams if t != "Unsold"]
    selected_team = st.selectbox("Select Team", teams)

    team_df = get_team_players(selected_team)
    team_spend = df[df["current_team"] == selected_team]["cost_cr"].sum()
    team_players = len(team_df)
    team_overseas = team_df[team_df["player_origin"] == "Overseas"].shape[0] if "player_origin" in team_df.columns else "N/A"

    c1, c2, c3 = st.columns(3)
    c1.markdown(kpi("Total Spend", f"₹{team_spend:.1f}", " Cr"), unsafe_allow_html=True)
    c2.markdown(kpi("Squad Size", team_players), unsafe_allow_html=True)
    c3.markdown(kpi("Overseas", team_overseas), unsafe_allow_html=True)

    st.markdown("### Squad Roster")
    styled = team_df.style.format({"cost_cr": "₹{:.2f}", "base_price_cr": "₹{:.2f}"})
    st.dataframe(team_df, use_container_width=True, height=450)

    # Role breakdown
    if len(team_df) > 0:
        role_counts = df[df["current_team"] == selected_team]["player_role"].value_counts()
        st.markdown("### Role Distribution")
        cols = st.columns(len(role_counts))
        for col, (role, cnt) in zip(cols, role_counts.items()):
            col.metric(role, cnt)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PLAYER SEARCH
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Player Search":
    st.markdown("## 🔍 Player Search")

    query = st.text_input("Search player name", placeholder="e.g. Virat, Curran, Rohit …")

    # Filters
    col1, col2, col3 = st.columns(3)
    role_filter   = col1.multiselect("Role", df["player_role"].unique().tolist(), default=[])
    origin_filter = col2.multiselect("Origin", ["Indian", "Overseas"], default=[])
    status_filter = col3.selectbox("Status", ["All", "Sold", "Unsold", "Retained"])

    filtered = df.copy()
    if query:
        filtered = filtered[filtered["player_name"].str.contains(query, case=False, na=False)]
    if role_filter:
        filtered = filtered[filtered["player_role"].isin(role_filter)]
    if origin_filter:
        filtered = filtered[filtered["player_origin"].isin(origin_filter)]
    if status_filter == "Sold":
        filtered = filtered[filtered["is_sold"] & ~filtered["is_retained"]]
    elif status_filter == "Unsold":
        filtered = filtered[~filtered["is_sold"]]
    elif status_filter == "Retained":
        filtered = filtered[filtered["is_retained"]]

    st.markdown(f"**{len(filtered)} players found**")
    cols_show = ["player_name", "current_team", "player_role", "player_origin",
                 "cost_cr", "base_price_cr", "price_bucket", "price_multiplier", "prev_team_2022"]
    st.dataframe(filtered[cols_show].sort_values("cost_cr", ascending=False), use_container_width=True, height=500)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: AI CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AI Chatbot":
    st.markdown("## 🤖 AI Analyst Chatbot")
    st.markdown("Ask any question about the IPL 2023 Auction dataset. The AI uses your actual data to answer.")

    openai_key = st.text_input("OpenAI API Key (required)", type="password", placeholder="sk-…")
    st.caption("Your key is used only for this session and not stored.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for msg in st.session_state.chat_history:
        cls = "chat-user" if msg["role"] == "user" else "chat-assistant"
        icon = "👤" if msg["role"] == "user" else "🤖"
        st.markdown(f'<div class="{cls}">{icon} {msg["content"]}</div>', unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Your question", placeholder="Which team spent the most? Top 5 overseas players?")
        submitted = st.form_submit_button("Send ➤")

    if submitted and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        if not openai_key:
            reply = "⚠️ Please enter your OpenAI API key above to use the AI chatbot."
        else:
            try:
                import openai
                client = openai.OpenAI(api_key=openai_key)

                # Build compact context from dataset
                context_stats = f"""
IPL 2023 Auction Dataset Summary:
- Total players: {len(df)}, Sold: {df['is_sold'].sum()}, Unsold: {(~df['is_sold']).sum()}, Retained: {df['is_retained'].sum()}
- Total auction spend: ₹{df['cost_cr'].sum():.1f} Cr
- Most expensive player: {df.nlargest(1,'cost_cr').iloc[0]['player_name']} (₹{df['cost_cr'].max()} Cr)
- Biggest spending team: {df.groupby('current_team')['cost_cr'].sum().idxmax()}
- Top 5 expensive players: {df.nlargest(5,'cost_cr')[['player_name','current_team','cost_cr']].to_string(index=False)}
- Team spending: {df[df['is_sold']&~df['is_retained']].groupby('current_team')['cost_cr'].sum().sort_values(ascending=False).to_string()}
- Role distribution: {df['player_role'].value_counts().to_string()}
- Columns: player_name, current_team, player_role, player_origin, cost_cr, base_price_cr, price_bucket, price_multiplier, prev_team_2022, is_sold, is_retained, team_changed
"""
                messages = [
                    {"role": "system", "content": f"You are an expert IPL cricket auction data analyst. Answer questions strictly based on this dataset context:\n\n{context_stats}\n\nBe concise, use numbers from the data, and format nicely with markdown."},
                ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history]

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    max_tokens=600,
                )
                reply = response.choices[0].message.content
            except Exception as e:
                reply = f"❌ Error: {str(e)}"

        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("**💡 Sample Questions:**")
    sample_qs = [
        "Which team spent the most in the auction?",
        "Who are the top 5 most expensive overseas players?",
        "What is the average price of a bowler vs batsman?",
        "Which team has the most overseas players?",
        "Who are the best bargain buys under ₹1 Cr?",
    ]
    for q in sample_qs:
        st.markdown(f"• *{q}*")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ML PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 ML Predictions":
    st.markdown("## 🧠 Machine Learning Predictions")

    tab1, tab2, tab3 = st.tabs(["💰 Price Predictor", "🎯 Sold/Unsold Predictor", "📈 Model Performance"])

    with tab1:
        st.markdown("### Predict Auction Price")
        st.caption("Enter player attributes to get an estimated auction price.")
        col1, col2, col3 = st.columns(3)
        role_p   = col1.selectbox("Player Role", ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"])
        origin_p = col2.selectbox("Origin", ["Indian", "Overseas"])
        base_p   = col3.number_input("Base Price (₹ Cr)", min_value=0.1, max_value=2.5, value=0.5, step=0.1)

        if st.button("Predict Price 💰", type="primary"):
            try:
                predicted = predict_price(role_p, base_p, origin_p)
                st.success(f"### Estimated Auction Price: ₹ {predicted:.2f} Crore")
                multiplier = predicted / base_p if base_p > 0 else 0
                st.metric("Price Multiplier", f"{multiplier:.1f}x", f"Base: ₹{base_p} Cr")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with tab2:
        st.markdown("### Predict: Will This Player Get Sold?")
        col1, col2, col3 = st.columns(3)
        role_c   = col1.selectbox("Player Role ", ["Batsman", "Bowler", "All-Rounder", "Wicket-Keeper"])
        origin_c = col2.selectbox("Origin ", ["Indian", "Overseas"])
        base_c   = col3.number_input("Base Price (₹ Cr) ", min_value=0.1, max_value=2.5, value=0.5, step=0.1)

        if st.button("Predict Outcome 🎯", type="primary"):
            try:
                result = predict_sold(role_c, base_c, origin_c)
                if result["prediction"] == "Sold":
                    st.markdown(f'<div class="pred-sold"><h2 style="color:#2EC4B6">✅ SOLD</h2><p style="font-size:1.3rem">Sold Probability: <b>{result["sold_probability"]}%</b></p></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="pred-unsold"><h2 style="color:#E63946">❌ UNSOLD</h2><p style="font-size:1.3rem">Sold Probability: <b>{result["sold_probability"]}%</b></p></div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                col1.metric("Sold Probability",   f"{result['sold_probability']}%")
                col2.metric("Unsold Probability", f"{result['unsold_probability']}%")
            except Exception as e:
                st.error(f"Prediction error: {e}")

    with tab3:
        st.markdown("### Model Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            pm = ml_results.get("price_model", {})
            st.markdown("#### 💰 Price Regression (Random Forest)")
            if pm:
                st.metric("MAE (₹ Cr)", pm.get("mae", "N/A"))
                st.metric("R² Score",   pm.get("r2", "N/A"))
                st.metric("CV R²",      pm.get("cv_r2", "N/A"))
                st.caption(f"Trained on {pm.get('n_train','?')} players, tested on {pm.get('n_test','?')}")
            if os.path.exists("assets/ml_actual_vs_predicted.png"):
                st.image("assets/ml_actual_vs_predicted.png", use_container_width=True)
            if os.path.exists("assets/ml_feature_importance.png"):
                st.image("assets/ml_feature_importance.png", use_container_width=True)
        with col2:
            sm = ml_results.get("sold_classifier", {})
            st.markdown("#### 🎯 Sold/Unsold Classifier (Gradient Boosting)")
            if sm:
                st.metric("Accuracy",    f"{sm.get('accuracy', 0):.1%}")
                st.metric("CV Accuracy", f"{sm.get('cv_accuracy', 0):.1%}")
                st.caption(f"Trained on {sm.get('n_train','?')} players, tested on {sm.get('n_test','?')}")
            if os.path.exists("assets/ml_confusion_matrix.png"):
                st.image("assets/ml_confusion_matrix.png", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SQL REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🗄️ SQL Reports":
    st.markdown("## 🗄️ SQL Reports & Live Query")

    report_names = {
        "Team Total Spending":       "team_total_spend",
        "Top 10 Expensive Players":  "top_10_expensive",
        "Role Statistics":           "role_stats",
        "Indian vs Overseas":        "origin_summary",
        "Top Bargain Buys":          "bargain_buys",
        "Team Foreign Count":        "team_foreign_count",
        "Price Bucket Distribution": "price_bucket_dist",
        "Players Who Switched Teams":"team_switched_players",
    }
    selected_report = st.selectbox("Select Pre-built Report", list(report_names.keys()))

    if st.button("Run Report ▶", type="primary"):
        result = get_report(report_names[selected_report])
        st.dataframe(result, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💻 Custom SQL Query")
    st.caption("Table: `ipl_auction` — Columns: player_name, current_team, player_role, player_origin, cost_cr, base_price_cr, is_sold, is_retained, price_bucket, price_multiplier, prev_team_2022, team_changed")
    custom_sql = st.text_area("SQL Query", value="SELECT player_name, current_team, cost_cr FROM ipl_auction WHERE cost_cr > 10 ORDER BY cost_cr DESC", height=100)
    if st.button("Execute Query ⚡"):
        try:
            result = run_query(custom_sql)
            st.success(f"Returned {len(result)} rows")
            st.dataframe(result, use_container_width=True)
        except Exception as e:
            st.error(f"SQL Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DOWNLOAD
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📥 Download":
    st.markdown("## 📥 Download Reports & Data")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Cleaned Dataset")
        clean_path = "data/cleaned_ipl_auction_2023.csv"
        if os.path.exists(clean_path):
            with open(clean_path, "rb") as f:
                st.download_button("⬇️ Download Cleaned CSV", f, "ipl_auction_2023_cleaned.csv", "text/csv")

    with col2:
        st.markdown("### 📋 SQL Report CSV")
        report_df = get_report("team_total_spend")
        csv_bytes = report_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Team Spending Report", csv_bytes, "team_spending_report.csv", "text/csv")

    st.markdown("---")
    st.markdown("### 📑 Generate PDF Summary Report")

    if st.button("Generate PDF Report 📑", type="primary"):
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
            from reportlab.lib import colors
            from reportlab.lib.units import cm
            import io

            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
            styles = getSampleStyleSheet()
            story = []

            title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=22, textColor=colors.HexColor("#1A5F7A"), spaceAfter=6)
            h2_style    = ParagraphStyle("h2",    parent=styles["Heading2"], fontSize=14, textColor=colors.HexColor("#1A5F7A"), spaceBefore=12)
            body_style  = ParagraphStyle("body",  parent=styles["Normal"], fontSize=10, leading=14)

            story.append(Paragraph("IPL Auction Analytics 2023", title_style))
            story.append(Paragraph("Comprehensive Analytics Report", styles["Heading3"]))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1A5F7A")))
            story.append(Spacer(1, 0.4*cm))

            story.append(Paragraph("Key Performance Indicators", h2_style))
            kpi_data = [["Metric", "Value"],
                        ["Total Players in Pool", str(insights["total_players"])],
                        ["Players Sold at Auction", str(insights["sold_players"])],
                        ["Players Unsold", str(insights["unsold_players"])],
                        ["Players Retained", str(insights["retained_players"])],
                        ["Total Auction Spend", f"₹{insights['total_spend_cr']} Cr"],
                        ["Average Price", f"₹{insights['avg_price_cr']} Cr"],
                        ["Highest Price", f"₹{insights['max_price_cr']} Cr"],
                        ["Most Expensive Player", insights["most_expensive_player"]],
                        ["Biggest Spending Team", insights["top_team"]],
                        ["Indian Players Sold", str(insights["indian_sold"])],
                        ["Overseas Players Sold", str(insights["overseas_sold"])]]
            t = Table(kpi_data, colWidths=[8*cm, 8*cm])
            t.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1A5F7A")),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",   (0,0), (-1,-1), 10),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F5F5F5"), colors.white]),
                ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
                ("PADDING",    (0,0), (-1,-1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 0.5*cm))

            story.append(Paragraph("Team-Wise Spending", h2_style))
            team_data = get_report("team_total_spend")
            tbl_data  = [list(team_data.columns)] + team_data.values.tolist()
            tbl = Table(tbl_data, colWidths=[5*cm, 3*cm, 3*cm, 2.5*cm, 2.5*cm])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1A5F7A")),
                ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
                ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",   (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#F5F5F5"), colors.white]),
                ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
                ("PADDING",    (0,0), (-1,-1), 5),
            ]))
            story.append(tbl)

            doc.build(story)
            pdf_bytes = buffer.getvalue()
            st.download_button("⬇️ Download PDF Report", pdf_bytes, "ipl_auction_2023_report.pdf", "application/pdf")
            st.success("PDF generated successfully!")
        except Exception as e:
            st.error(f"PDF generation error: {e}")
