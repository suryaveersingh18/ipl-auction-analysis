"""
Data Cleaning & Feature Engineering Pipeline
=============================================
Processes raw IPL Auction 2023 CSV into a clean, analysis-ready DataFrame.

Detected raw columns:
  - Unnamed: 0          → row index (dropped)
  - Player's List       → player_name
  - Base Price          → base_price_inr  (numeric, 'Retained' → 0)
  - TYPE                → player_role
  - COST IN ₹ (CR.)     → cost_cr
  - Cost IN $ (000)     → cost_usd_000
  - 2022 Squad          → prev_team_2022
  - Team                → current_team  ('Unsold' → retained as category)

Assumptions:
  - NaN in COST IN ₹ means the player was NOT sold at auction (they appear
    in Team == 'Unsold' rows — confirmed by data inspection).
  - 'Retained' in Base Price means the player was retained pre-auction,
    cost set to 0 for auction purposes; actual retention cost is unknown.
  - One season is represented (IPL 2023 Auction / Squad data).
"""

import pandas as pd
import numpy as np
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

RAW_PATH = os.path.join("data", "raw_ipl_auction_2023.csv")
CLEAN_PATH = os.path.join("data", "cleaned_ipl_auction_2023.csv")

# ---------------------------------------------------------
# Team name normalisation map
# ---------------------------------------------------------
TEAM_NAME_MAP = {
    "Gujarat Titans": "Gujarat Titans",
    "Chennai Super Kings": "Chennai Super Kings",
    "Delhi Capitals": "Delhi Capitals",
    "Lucknow Super Giants": "Lucknow Super Giants",
    "Royal Challengers Banglore": "Royal Challengers Bangalore",
    "Rajasthan Royals": "Rajasthan Royals",
    "Sunrisers Hyderabad": "Sunrisers Hyderabad",
    "Mumbai Indians": "Mumbai Indians",
    "Kolkata Knight Riders": "Kolkata Knight Riders",
    "Punjab Super Kings": "Punjab Kings",
    "Unsold": "Unsold",
}

# Short code → full name for 2022 Squad column
SQUAD_CODE_MAP = {
    "GT": "Gujarat Titans",
    "CSK": "Chennai Super Kings",
    "DC": "Delhi Capitals",
    "LSG": "Lucknow Super Giants",
    "RCB": "Royal Challengers Bangalore",
    "RR": "Rajasthan Royals",
    "SRH": "Sunrisers Hyderabad",
    "MI": "Mumbai Indians",
    "KKR": "Kolkata Knight Riders",
    "PBKS": "Punjab Kings",
}


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    logger.info(f"Loading raw dataset from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={
        "Unnamed: 0": "row_id",
        "Player's List": "player_name",
        "Base Price": "base_price_raw",
        "TYPE": "player_role",
        "COST IN ₹ (CR.)": "cost_cr",
        "Cost IN $ (000)": "cost_usd_000",
        "2022 Squad": "prev_team_code",
        "Team": "current_team_raw",
    })
    logger.info("Columns renamed")
    return df


def clean_base_price(df: pd.DataFrame) -> pd.DataFrame:
    """Convert base_price_raw to numeric INR. 'Retained' → 0."""
    def parse_bp(val):
        if str(val).strip().lower() == "retained":
            return 0
        try:
            return int(val)
        except (ValueError, TypeError):
            return np.nan

    df["base_price_inr"] = df["base_price_raw"].apply(parse_bp)
    df["is_retained"] = df["base_price_raw"].astype(str).str.lower() == "retained"
    logger.info("Base price parsed")
    return df


def clean_teams(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise team names and create sold/unsold flag."""
    df["current_team"] = df["current_team_raw"].map(TEAM_NAME_MAP).fillna(df["current_team_raw"])
    df["prev_team_2022"] = df["prev_team_code"].map(SQUAD_CODE_MAP)
    df["is_sold"] = df["current_team"] != "Unsold"
    logger.info("Teams normalised")
    return df


def handle_cost_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unsold players → cost_cr = NaN.
    Retained players already have cost_cr = 0 from raw data.
    Fill unsold with 0 for aggregation, keep NaN flag in is_sold.
    """
    df["cost_cr"] = df["cost_cr"].fillna(0)
    df["cost_usd_000"] = df["cost_usd_000"].fillna(0)
    logger.info("Cost nulls handled")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived analytical columns."""

    # 1. Price bucket
    def price_bucket(cost):
        if cost == 0:
            return "Unsold / Retained"
        elif cost < 0.5:
            return "< 0.5 Cr"
        elif cost < 1:
            return "0.5 – 1 Cr"
        elif cost < 2:
            return "1 – 2 Cr"
        elif cost < 5:
            return "2 – 5 Cr"
        elif cost < 10:
            return "5 – 10 Cr"
        else:
            return "10+ Cr"

    df["price_bucket"] = df["cost_cr"].apply(price_bucket)

    # 2. Domestic vs Overseas
    # We infer: if a player appeared in ANY IPL 2022 squad they are likely domestic
    # (not fully reliable but best proxy available without nationality column)
    # Using player names + prev squad as proxy; explicitly flagged as inferred.
    overseas_keywords = [
        "Little", "Williamson", "Curran", "Stoinis", "Maxwell", "Warner",
        "Buttler", "Stokes", "Archer", "Root", "Bairstow", "Rabada",
        "Nortje", "de Kock", "Klaasen", "Miller", "Markram", "Ngidi",
        "Russell", "Narine", "Sunil", "Holder", "Pollard", "Hetmyer",
        "Pooran", "Bravo", "Thomas", "Coulter", "Hazlewood", "Cummins",
        "Starc", "Zampa", "Green", "Inglis", "Head", "Malan", "Roy",
        "Jason", "Andre", "Rovman", "Shimron", "Alzarri", "Odean",
        "Josh", "Mathews", "Asalanka", "Hasaranga", "Pathum", "Dhananjaya",
        "Kusal", "Dasun", "Thisara", "Chamika", "Wanidu", "Pramod",
        "Azam", "Shadab", "Iftikhar", "Mohammad", "Nawaz", "Rauf",
        "Naseem", "Shaheen", "Babar", "Fakhar", "Imam", "Sarfaraz",
        "Oshane", "Keemo", "Dominic", "Gudakesh", "Akeal",
    ]
    def is_overseas(name):
        return any(kw.lower() in name.lower() for kw in overseas_keywords)

    df["is_overseas_inferred"] = df["player_name"].apply(is_overseas)
    df["player_origin"] = df["is_overseas_inferred"].map({True: "Overseas", False: "Indian"})

    # 3. Base price in Crore
    df["base_price_cr"] = df["base_price_inr"] / 1e7

    # 4. Price multiplier (how many times base price the player was bought for)
    df["price_multiplier"] = np.where(
        (df["base_price_cr"] > 0) & df["is_sold"],
        (df["cost_cr"] / df["base_price_cr"]).round(2),
        np.nan
    )

    # 5. Team changed flag
    df["team_changed"] = (
        df["prev_team_2022"].notna() &
        df["is_sold"] &
        (df["prev_team_2022"] != df["current_team"])
    )

    # 6. Player role simplified
    role_map = {
        "BATSMAN": "Batsman",
        "BOWLER": "Bowler",
        "ALL-ROUNDER": "All-Rounder",
        "WICKETKEEPER": "Wicket-Keeper",
    }
    df["player_role"] = df["player_role"].map(role_map).fillna(df["player_role"])

    logger.info("Feature engineering complete")
    return df


def drop_unused(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=["row_id", "base_price_raw", "current_team_raw", "prev_team_code"])
    return df


def run_pipeline(path: str = RAW_PATH, save: bool = True) -> pd.DataFrame:
    """Execute full cleaning pipeline and return clean DataFrame."""
    logger.info("=== Starting Data Cleaning Pipeline ===")
    df = load_raw(path)
    df = rename_columns(df)
    df = clean_base_price(df)
    df = clean_teams(df)
    df = handle_cost_nulls(df)
    df = engineer_features(df)
    df = drop_unused(df)
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    if save:
        os.makedirs("data", exist_ok=True)
        df.to_csv(CLEAN_PATH, index=False)
        logger.info(f"Cleaned dataset saved to {CLEAN_PATH}")

    logger.info(f"=== Pipeline Complete: {len(df)} rows, {df.shape[1]} columns ===")
    return df


def dataset_summary(df: pd.DataFrame) -> dict:
    """Return a structured summary of dataset quality."""
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "null_counts": df.isnull().sum().to_dict(),
        "duplicates": df.duplicated().sum(),
        "sold_players": int(df["is_sold"].sum()),
        "unsold_players": int((~df["is_sold"]).sum()),
        "retained_players": int(df["is_retained"].sum()),
        "teams": df[df["is_sold"]]["current_team"].unique().tolist(),
        "roles": df["player_role"].value_counts().to_dict(),
    }


if __name__ == "__main__":
    df = run_pipeline()
    summary = dataset_summary(df)
    print("\n=== DATASET SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nSample rows:")
    print(df.head(10).to_string())
