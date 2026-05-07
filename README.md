# 🏏 IPL Auction Analytics 2023

> **Production-ready Data Science portfolio project** built on the real IPL 2023 Auction dataset (568 players, 10 teams, ₹167 Cr total spend).

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.27-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📁 Project Structure

```
ipl_auction_analytics/
├── data/
│   ├── raw_ipl_auction_2023.csv        ← Original uploaded dataset
│   └── cleaned_ipl_auction_2023.csv    ← Auto-generated after first run
├── src/
│   ├── cleaning/
│   │   └── pipeline.py                 ← Data cleaning & feature engineering
│   ├── eda/
│   │   └── analysis.py                 ← 12 EDA charts + key insights
│   ├── ml/
│   │   └── models.py                   ← Price regression + Sold classifier
│   ├── sql/
│   │   └── database.py                 ← SQLite integration + 8 SQL reports
│   └── utils/
│       └── logger.py                   ← Centralised logging
├── app/
│   └── main.py                         ← Streamlit 8-page web app
├── models/
│   ├── price_regressor.pkl             ← Trained Random Forest (auto-saved)
│   └── sold_classifier.pkl             ← Trained Gradient Boosting (auto-saved)
├── assets/                             ← Auto-generated chart PNGs (12+)
├── logs/                               ← Daily rotating log files
├── reports/                            ← PDF exports
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/ipl-auction-analytics.git
cd ipl-auction-analytics
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY for chatbot features
```

### 3. Run the App
```bash
streamlit run app/main.py
```

The app auto-runs data cleaning, chart generation, and ML training on first launch.

---

## 📊 Dataset Summary

| Attribute | Value |
|-----------|-------|
| Source | IPL 2023 Auction Squad Data |
| Rows | 568 players |
| Columns | 8 raw → 15 after feature engineering |
| Teams | 10 IPL teams |
| Total Auction Spend | ₹167 Crore |
| Most Expensive Player | Sam Curran (₹18.5 Cr) |
| Biggest Spending Team | Sunrisers Hyderabad (₹35.7 Cr) |

### Raw Columns
| Column | Type | Notes |
|--------|------|-------|
| `Player's List` | str | Player name |
| `Base Price` | str | INR or "Retained" |
| `TYPE` | str | BATSMAN / BOWLER / ALL-ROUNDER / WICKETKEEPER |
| `COST IN ₹ (CR.)` | float | NaN = unsold |
| `Cost IN $ (000)` | float | USD equivalent |
| `2022 Squad` | str | Previous team code (NaN = new) |
| `Team` | str | Current team or "Unsold" |

### Engineered Features
- `base_price_cr` — Base price in Crore
- `price_bucket` — Price range category
- `price_multiplier` — Cost / base price ratio
- `player_origin` — Indian / Overseas (inferred)
- `is_retained`, `is_sold`, `team_changed` — Boolean flags
- `prev_team_2022` — Previous team full name

---

## 🧠 Machine Learning

### 1. Price Regression (Random Forest)
- **Target**: `cost_cr` (auction price in Crore)
- **Features**: `base_price_cr`, `player_role`, `player_origin`
- **Metrics**: MAE ~₹2.2 Cr, R² 0.31

### 2. Sold/Unsold Classifier (Gradient Boosting)
- **Target**: `is_sold` (binary)
- **Features**: `base_price_cr`, `player_role`, `player_origin`
- **Accuracy**: ~80%

> **Note**: With only 3 input features available from the dataset, R² for regression is modest. The classifier performs well given the imbalanced dataset (57% unsold).

---

## 🗄️ SQL Reports

8 pre-built reports available in the app:
1. Team Total Spending
2. Top 10 Most Expensive Players
3. Role Statistics
4. Indian vs Overseas Comparison
5. Top Bargain Buys
6. Team Foreign Player Count
7. Price Bucket Distribution
8. Players Who Switched Teams

Plus a live **custom SQL query** editor.

---

## 📱 App Pages

| Page | Features |
|------|----------|
| 🏠 Home & KPIs | 10 KPI cards, highlight charts |
| 📊 Dashboard | All 12 charts across 4 tabs |
| 🏟️ Team Analysis | Per-team squad + spending breakdown |
| 🔍 Player Search | Multi-filter search across 568 players |
| 🤖 AI Chatbot | OpenAI-powered natural language Q&A |
| 🧠 ML Predictions | Price + Sold/Unsold prediction UI |
| 🗄️ SQL Reports | 8 pre-built + custom SQL editor |
| 📥 Download | CSV + PDF export |

---

## 📈 Key Insights

- **Sam Curran** became the most expensive player in IPL history at ₹18.5 Cr
- **Sunrisers Hyderabad** was the biggest spender (₹35.7 Cr), followed by Mumbai Indians (₹20.5 Cr)
- **57% of players went unsold** — auction is highly competitive
- **All-Rounders** dominated both count (213) and total spend (₹70.75 Cr)
- **Batsmen** commanded highest average price at ₹3.04 Cr/player
- **Shivam Mavi** had the highest price multiplier among non-premium buys (15x his base price)

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn (Random Forest, Gradient Boosting) |
| Database | SQLite (via sqlite3 + SQLAlchemy-compatible) |
| Web App | Streamlit |
| AI Chatbot | OpenAI GPT-3.5-turbo |
| PDF Reports | ReportLab |
| Logging | Python `logging` module |
| Config | python-dotenv |

---

## 📝 Resume Bullet Points

> Copy these directly into your resume / LinkedIn:

- **Built production-ready IPL Auction Analytics platform** processing 568 players across 10 IPL teams with automated data cleaning pipeline, feature engineering (15 derived columns), and SQLite integration
- **Developed dual ML models** (Random Forest price regressor + Gradient Boosting classifier at 80.2% accuracy) to predict player auction outcomes using scikit-learn pipelines
- **Created 8-page Streamlit dashboard** with KPI cards, 12 analytical charts, live SQL query editor, AI-powered chatbot (OpenAI API), and one-click PDF/CSV export
- **Automated EDA pipeline** generating 12 publication-quality charts (heatmaps, boxplots, treemaps, scatter plots) with dark-theme branding using Matplotlib & Seaborn

---

## 🤝 Contributing

Pull requests welcome. For major changes, please open an issue first.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
