"""
Exploratory Data Analysis Module
=================================
Generates key insights and static charts from the cleaned IPL Auction dataset.
All charts are saved to assets/ folder for use in the Streamlit app.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
warnings.filterwarnings("ignore")

from src.utils.logger import get_logger

logger = get_logger(__name__)

ASSETS_DIR = "assets"
os.makedirs(ASSETS_DIR, exist_ok=True)

# ── IPL brand colour palette ──────────────────────────────────────────────────
TEAM_COLORS = {
    "Gujarat Titans":              "#1C4E9D",
    "Chennai Super Kings":         "#F9CD05",
    "Delhi Capitals":              "#0078BC",
    "Kolkata Knight Riders":       "#3A225D",
    "Punjab Kings":                "#ED1C24",
    "Lucknow Super Giants":        "#004B8D",
    "Mumbai Indians":              "#004BA0",
    "Royal Challengers Bangalore": "#EC1C24",
    "Rajasthan Royals":            "#EA1A85",
    "Sunrisers Hyderabad":         "#F26522",
}

ROLE_COLORS = {
    "Batsman":       "#E63946",
    "Bowler":        "#2A9D8F",
    "All-Rounder":   "#E9C46A",
    "Wicket-Keeper": "#264653",
}

plt.rcParams.update({
    "figure.facecolor":  "#0F1117",
    "axes.facecolor":    "#1A1D2E",
    "axes.edgecolor":    "#2E3250",
    "axes.labelcolor":   "#E0E0E0",
    "xtick.color":       "#A0A0A0",
    "ytick.color":       "#A0A0A0",
    "text.color":        "#E0E0E0",
    "grid.color":        "#2E3250",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
})


def _savefig(name: str) -> str:
    path = os.path.join(ASSETS_DIR, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    plt.close()
    logger.info(f"Saved chart: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. SOLD VS UNSOLD PIE
# ─────────────────────────────────────────────────────────────────────────────
def chart_sold_unsold(df: pd.DataFrame) -> str:
    counts = df["is_sold"].value_counts()
    labels = ["Sold", "Unsold"]
    sizes  = [counts.get(True, 0), counts.get(False, 0)]
    colors = ["#2EC4B6", "#E63946"]

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.80,
        wedgeprops={"edgecolor": "#0F1117", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(13)
        at.set_color("white")
    ax.set_title("Sold vs Unsold Players", fontsize=16, fontweight="bold", pad=20)
    return _savefig("sold_vs_unsold")


# ─────────────────────────────────────────────────────────────────────────────
# 2. TEAM-WISE TOTAL SPENDING
# ─────────────────────────────────────────────────────────────────────────────
def chart_team_spending(df: pd.DataFrame) -> str:
    sold = df[df["is_sold"] & ~df["is_retained"]]
    team_spend = (
        sold.groupby("current_team")["cost_cr"]
        .sum()
        .sort_values(ascending=True)
    )

    colors = [TEAM_COLORS.get(t, "#7B7B7B") for t in team_spend.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(team_spend.index, team_spend.values, color=colors, height=0.6)

    for bar, val in zip(bars, team_spend.values):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"₹{val:.1f} Cr", va="center", fontsize=10, color="#E0E0E0",
        )
    ax.set_xlabel("Total Spend (₹ Crore)", fontsize=12)
    ax.set_title("Team-Wise Auction Spending (IPL 2023)", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    return _savefig("team_spending")


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROLE-WISE SPENDING BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────
def chart_role_spending(df: pd.DataFrame) -> str:
    sold = df[df["is_sold"] & ~df["is_retained"]]
    role_spend = sold.groupby("player_role")["cost_cr"].sum().sort_values(ascending=False)

    colors = [ROLE_COLORS.get(r, "#7B7B7B") for r in role_spend.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(role_spend.index, role_spend.values, color=colors, width=0.5)

    for bar, val in zip(bars, role_spend.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"₹{val:.1f} Cr", ha="center", fontsize=11, color="#E0E0E0",
        )
    ax.set_ylabel("Total Spend (₹ Crore)", fontsize=12)
    ax.set_title("Spending by Player Role", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    return _savefig("role_spending")


# ─────────────────────────────────────────────────────────────────────────────
# 4. TOP 15 MOST EXPENSIVE PLAYERS
# ─────────────────────────────────────────────────────────────────────────────
def chart_top_expensive(df: pd.DataFrame, n: int = 15) -> str:
    top = (
        df[df["is_sold"] & ~df["is_retained"]]
        .nlargest(n, "cost_cr")[["player_name", "cost_cr", "current_team", "player_role"]]
    )
    colors = [TEAM_COLORS.get(t, "#7B7B7B") for t in top["current_team"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["player_name"][::-1], top["cost_cr"][::-1], color=colors[::-1], height=0.6)

    for bar, val in zip(bars, top["cost_cr"][::-1]):
        ax.text(
            bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
            f"₹{val:.1f} Cr", va="center", fontsize=10, color="#E0E0E0",
        )
    ax.set_xlabel("Price (₹ Crore)", fontsize=12)
    ax.set_title(f"Top {n} Most Expensive Players – IPL 2023 Auction", fontsize=14, fontweight="bold")
    legend_patches = [mpatches.Patch(color=v, label=k) for k, v in TEAM_COLORS.items() if k in top["current_team"].values]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.3)
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    return _savefig("top_expensive_players")


# ─────────────────────────────────────────────────────────────────────────────
# 5. PRICE DISTRIBUTION HISTOGRAM
# ─────────────────────────────────────────────────────────────────────────────
def chart_price_distribution(df: pd.DataFrame) -> str:
    sold = df[(df["is_sold"]) & (~df["is_retained"]) & (df["cost_cr"] > 0)]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(sold["cost_cr"], bins=25, color="#2EC4B6", edgecolor="#0F1117", alpha=0.85)
    ax.axvline(sold["cost_cr"].mean(), color="#E9C46A", linestyle="--", linewidth=2, label=f'Mean ₹{sold["cost_cr"].mean():.2f} Cr')
    ax.axvline(sold["cost_cr"].median(), color="#E63946", linestyle="--", linewidth=2, label=f'Median ₹{sold["cost_cr"].median():.2f} Cr')
    ax.set_xlabel("Cost (₹ Crore)", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)
    ax.set_title("Distribution of Auction Prices (Sold Players)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    return _savefig("price_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# 6. INDIAN VS OVERSEAS SPENDING
# ─────────────────────────────────────────────────────────────────────────────
def chart_origin_comparison(df: pd.DataFrame) -> str:
    sold = df[df["is_sold"] & ~df["is_retained"] & (df["cost_cr"] > 0)]
    origin_data = sold.groupby("player_origin").agg(
        total_spend=("cost_cr", "sum"),
        avg_price=("cost_cr", "mean"),
        count=("player_name", "count"),
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    metrics = [("total_spend", "Total Spend (₹ Cr)"), ("avg_price", "Avg Price (₹ Cr)"), ("count", "# Players Sold")]
    palette = {"Indian": "#2EC4B6", "Overseas": "#E63946"}

    for ax, (col, title) in zip(axes, metrics):
        colors = [palette[o] for o in origin_data["player_origin"]]
        bars = ax.bar(origin_data["player_origin"], origin_data[col], color=colors, width=0.4)
        for bar, val in zip(bars, origin_data[col]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02 * origin_data[col].max(),
                    f"{val:.1f}" if col != "count" else f"{int(val)}", ha="center", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle("Indian vs Overseas Players – Spending Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    return _savefig("origin_comparison")


# ─────────────────────────────────────────────────────────────────────────────
# 7. PRICE BUCKET DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────
def chart_price_buckets(df: pd.DataFrame) -> str:
    bucket_order = ["< 0.5 Cr", "0.5 – 1 Cr", "1 – 2 Cr", "2 – 5 Cr", "5 – 10 Cr", "10+ Cr"]
    sold = df[df["is_sold"] & ~df["is_retained"] & (df["cost_cr"] > 0)]
    counts = sold["price_bucket"].value_counts().reindex(bucket_order, fill_value=0)

    colors = ["#264653", "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#E63946"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(counts.index, counts.values, color=colors, width=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                str(val), ha="center", fontsize=12, fontweight="bold")
    ax.set_xlabel("Price Range", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)
    ax.set_title("Players Sold by Price Bracket", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    return _savefig("price_buckets")


# ─────────────────────────────────────────────────────────────────────────────
# 8. HEATMAP – TEAM × ROLE SPENDING
# ─────────────────────────────────────────────────────────────────────────────
def chart_team_role_heatmap(df: pd.DataFrame) -> str:
    sold = df[df["is_sold"] & ~df["is_retained"]]
    pivot = sold.pivot_table(values="cost_cr", index="current_team", columns="player_role", aggfunc="sum", fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, linecolor="#0F1117",
        cbar_kws={"label": "₹ Crore"},
        ax=ax,
    )
    ax.set_title("Team × Role Spending Heatmap (₹ Crore)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Player Role", fontsize=12)
    ax.set_ylabel("Team", fontsize=12)
    plt.tight_layout()
    return _savefig("team_role_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# 9. TOP BARGAIN BUYS (high price multiplier, low cost)
# ─────────────────────────────────────────────────────────────────────────────
def chart_bargain_buys(df: pd.DataFrame, n: int = 12) -> str:
    """Players bought well below base price expectation (low cost, decent price)."""
    # Bargain = sold, not retained, cost_cr > 0, highest price_multiplier
    # Filter players bought for ≤ 2 Cr but multiplier ≥ 3
    bargains = df[
        (df["is_sold"]) & (~df["is_retained"]) &
        (df["cost_cr"] > 0) & (df["cost_cr"] <= 3) &
        (df["price_multiplier"] >= 2)
    ].nlargest(n, "price_multiplier")[["player_name", "cost_cr", "price_multiplier", "current_team", "player_role"]]

    if bargains.empty:
        logger.warning("No bargain buys found with current filter; loosening filter.")
        bargains = df[(df["is_sold"]) & (~df["is_retained"]) & (df["cost_cr"] > 0)].nlargest(n, "price_multiplier")[
            ["player_name", "cost_cr", "price_multiplier", "current_team", "player_role"]
        ]

    colors = [TEAM_COLORS.get(t, "#7B7B7B") for t in bargains["current_team"]]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(bargains["player_name"][::-1], bargains["price_multiplier"][::-1], color=colors[::-1], height=0.6)
    for bar, val, cost in zip(bars, bargains["price_multiplier"][::-1], bargains["cost_cr"][::-1]):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}x  (₹{cost:.1f} Cr)", va="center", fontsize=9, color="#E0E0E0")
    ax.set_xlabel("Price Multiplier (vs Base Price)", fontsize=12)
    ax.set_title("Top Bargain Buys – Highest Price Multipliers (≤ ₹3 Cr)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    return _savefig("bargain_buys")


# ─────────────────────────────────────────────────────────────────────────────
# 10. BOXPLOT – PRICE BY ROLE
# ─────────────────────────────────────────────────────────────────────────────
def chart_boxplot_role(df: pd.DataFrame) -> str:
    sold = df[df["is_sold"] & ~df["is_retained"] & (df["cost_cr"] > 0)]

    fig, ax = plt.subplots(figsize=(9, 5))
    roles = sold["player_role"].unique()
    data  = [sold[sold["player_role"] == r]["cost_cr"].values for r in roles]
    bp = ax.boxplot(data, labels=roles, patch_artist=True, notch=False,
                    medianprops={"color": "white", "linewidth": 2})
    for patch, role in zip(bp["boxes"], roles):
        patch.set_facecolor(ROLE_COLORS.get(role, "#7B7B7B"))
        patch.set_alpha(0.8)
    ax.set_ylabel("Cost (₹ Crore)", fontsize=12)
    ax.set_title("Price Distribution by Player Role", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    return _savefig("boxplot_role")


# ─────────────────────────────────────────────────────────────────────────────
# 11. TEAM COMPOSITION BY ROLE
# ─────────────────────────────────────────────────────────────────────────────
def chart_team_composition(df: pd.DataFrame) -> str:
    sold = df[df["is_sold"]]
    comp = sold.groupby(["current_team", "player_role"]).size().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    comp.plot(
        kind="bar", ax=ax, stacked=True,
        color=[ROLE_COLORS.get(r, "#7B7B7B") for r in comp.columns],
        edgecolor="#0F1117", width=0.65,
    )
    ax.set_xlabel("Team", fontsize=12)
    ax.set_ylabel("Number of Players", fontsize=12)
    ax.set_title("Team Composition by Player Role", fontsize=14, fontweight="bold")
    ax.legend(title="Role", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    return _savefig("team_composition")


# ─────────────────────────────────────────────────────────────────────────────
# 12. SCATTER – BASE PRICE vs FINAL PRICE
# ─────────────────────────────────────────────────────────────────────────────
def chart_base_vs_final(df: pd.DataFrame) -> str:
    sold = df[df["is_sold"] & ~df["is_retained"] & (df["cost_cr"] > 0) & (df["base_price_cr"] > 0)]

    fig, ax = plt.subplots(figsize=(9, 6))
    for role, grp in sold.groupby("player_role"):
        ax.scatter(grp["base_price_cr"], grp["cost_cr"],
                   color=ROLE_COLORS.get(role, "#7B7B7B"), label=role, alpha=0.75, s=60, edgecolors="white", linewidths=0.4)

    max_val = max(sold["base_price_cr"].max(), sold["cost_cr"].max()) + 1
    ax.plot([0, max_val], [0, max_val], "--", color="#A0A0A0", linewidth=1.2, label="1:1 Line")
    ax.set_xlabel("Base Price (₹ Crore)", fontsize=12)
    ax.set_ylabel("Final Auction Price (₹ Crore)", fontsize=12)
    ax.set_title("Base Price vs Final Price (Sold Players)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return _savefig("base_vs_final")


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_charts(df: pd.DataFrame) -> dict:
    logger.info("Generating all EDA charts …")
    charts = {
        "sold_unsold":        chart_sold_unsold(df),
        "team_spending":      chart_team_spending(df),
        "role_spending":      chart_role_spending(df),
        "top_expensive":      chart_top_expensive(df),
        "price_distribution": chart_price_distribution(df),
        "origin_comparison":  chart_origin_comparison(df),
        "price_buckets":      chart_price_buckets(df),
        "team_role_heatmap":  chart_team_role_heatmap(df),
        "bargain_buys":       chart_bargain_buys(df),
        "boxplot_role":       chart_boxplot_role(df),
        "team_composition":   chart_team_composition(df),
        "base_vs_final":      chart_base_vs_final(df),
    }
    logger.info(f"Generated {len(charts)} charts")
    return charts


def get_key_insights(df: pd.DataFrame) -> dict:
    """Return structured KPIs for dashboard cards."""
    sold     = df[df["is_sold"] & ~df["is_retained"] & (df["cost_cr"] > 0)]
    top_team = sold.groupby("current_team")["cost_cr"].sum().idxmax()
    top_team_spend = sold.groupby("current_team")["cost_cr"].sum().max()
    top_player = sold.nlargest(1, "cost_cr").iloc[0]
    most_sold_role = sold["player_role"].value_counts().idxmax()

    return {
        "total_players":      len(df),
        "sold_players":       int(df["is_sold"].sum()),
        "unsold_players":     int((~df["is_sold"]).sum()),
        "retained_players":   int(df["is_retained"].sum()),
        "total_spend_cr":     round(sold["cost_cr"].sum(), 2),
        "avg_price_cr":       round(sold["cost_cr"].mean(), 2),
        "max_price_cr":       round(sold["cost_cr"].max(), 2),
        "top_team":           top_team,
        "top_team_spend":     round(top_team_spend, 2),
        "most_expensive_player": top_player["player_name"],
        "most_expensive_cost":   top_player["cost_cr"],
        "most_sold_role":     most_sold_role,
        "overseas_sold":      int(sold[sold["player_origin"] == "Overseas"].shape[0]),
        "indian_sold":        int(sold[sold["player_origin"] == "Indian"].shape[0]),
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.cleaning.pipeline import run_pipeline
    df = run_pipeline(save=False)
    charts = generate_all_charts(df)
    print("Charts saved:", list(charts.values()))
    insights = get_key_insights(df)
    print("\nKey Insights:")
    for k, v in insights.items():
        print(f"  {k}: {v}")
